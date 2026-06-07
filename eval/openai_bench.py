"""OpenAI API benchmarking script for LongShOT evaluation framework.

Standalone generation and evaluation script using the OpenAI API with
cost-saving features:
  - Prompt Caching: automatic 90% discount on cached input tokens (>1024 tokens)
  - Batch API: 50% off all tokens, results within 24h
  - Detail level control: "low" (85 tokens/image) vs "high" (~170/tile + 85)

Supports two use cases:
  1. Generation: send pre-extracted frames to a vision model (e.g. gpt-5.4)
  2. Evaluation: use as a text-only judge for criterion evaluation

Usage:
    # Real-time generation with frames (prompt caching automatic)
    python openai_bench.py --tasks postvalid_v2 --mode realtime --max-frames 128

    # Batch generation (50% off, 24h window)
    python openai_bench.py --tasks postvalid_v2 --mode batch --max-frames 128

    # Check pending batches
    python openai_bench.py --check-batches

    # Resume/retrieve a completed batch
    python openai_bench.py --resume-batch batch_abc123 --tasks postvalid_v2

    # Use as evaluation judge
    python openai_bench.py --evaluate --model gpt-5.4 \\
        --candidate-model Qwen_Qwen3_VL_32B --tasks postvalid_v2
"""

import argparse
import json
import os
import re
import signal
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from filelock import FileLock
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from utils import (
    build_video_path_map,
    convert_to_underscored,
    get_generation_artifact_paths,
    get_judge_artifact_paths,
    load_config,
    load_dataset_with_params,
    load_jsonl,
    save_timing_data,
    strip_think_tags,
)

MODEL = "gpt-5.4"
DETAIL_LEVEL = "low"

# Pricing per 1M tokens (gpt-5.4)
PRICING = {
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5-2025-08-07": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-mini-2025-08-07": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "o3": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40},
}


# ═══════════════════════════════════════════════════════════════════════════
# Cost tracking
# ═══════════════════════════════════════════════════════════════════════════

class CostTracker:

    def __init__(self, tracking_path):
        self.path = tracking_path
        self._lock = threading.Lock()
        self._file_lock = FileLock(f"{tracking_path}.lock")

    def _load_data(self):
        base = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cached_tokens": 0,
            "estimated_cost_usd": 0.0,
            "request_count": 0,
            "batches": {},
        }
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    stored = json.load(f)
                base.update(stored)
            except (json.JSONDecodeError, OSError):
                pass
        return base

    def _save_data(self, data):
        dir_path = os.path.dirname(self.path) or "."
        os.makedirs(dir_path, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def add_usage(self, usage, model, mode="realtime"):
        if not usage:
            return 0.0
        prompt = usage.prompt_tokens or 0
        completion = usage.completion_tokens or 0
        cached = 0
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

        pricing = PRICING.get(model, PRICING.get("gpt-5.4"))
        discount = 0.5 if mode == "batch" else 1.0
        uncached = max(0, prompt - cached)
        cost = (
            (uncached / 1e6) * pricing["input"] * discount
            + (cached / 1e6) * pricing["cached_input"] * discount
            + (completion / 1e6) * pricing["output"] * discount
        )

        with self._lock, self._file_lock:
            data = self._load_data()
            data["total_input_tokens"] += prompt
            data["total_output_tokens"] += completion
            data["total_cached_tokens"] += cached
            data["estimated_cost_usd"] += cost
            data["request_count"] += 1
            self._save_data(data)
        return cost

    def set_batch(self, batch_id, **kwargs):
        with self._lock, self._file_lock:
            data = self._load_data()
            data["batches"][batch_id] = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                **kwargs,
            }
            self._save_data(data)

    def update_batch(self, batch_id, **kwargs):
        with self._lock, self._file_lock:
            data = self._load_data()
            if batch_id in data["batches"]:
                data["batches"][batch_id].update(kwargs)
                self._save_data(data)

    def summary(self):
        with self._lock, self._file_lock:
            data = self._load_data()
            return {k: v for k, v in data.items() if k != "batches"}

    @property
    def data(self):
        with self._file_lock:
            return self._load_data()


# ═══════════════════════════════════════════════════════════════════════════
# Image encoding (frames)
# ═══════════════════════════════════════════════════════════════════════════

_frame_b64_cache = {}


def _encode_frame(frame_path):
    """Encode a JPEG frame as a base64 data URL. Cached in memory."""
    if frame_path in _frame_b64_cache:
        return _frame_b64_cache[frame_path]
    import base64
    with open(frame_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    url = f"data:image/jpeg;base64,{data}"
    _frame_b64_cache[frame_path] = url
    return url


def build_frame_content(video_path, max_frames, detail=DETAIL_LEVEL):
    """Build image_url content list from pre-extracted frames."""
    vdir = os.path.dirname(video_path)
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(vdir, "frames", video_id, f"f{max_frames}")

    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(
            f"No frames at {frames_dir}. Run: python scripts/extract_frames.py --counts {max_frames}")

    jpgs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    content = []
    for f in jpgs:
        url = _encode_frame(os.path.join(frames_dir, f))
        content.append({
            "type": "image_url",
            "image_url": {"url": url, "detail": detail},
        })
    return content


# ═══════════════════════════════════════════════════════════════════════════
# Message builders
# ═══════════════════════════════════════════════════════════════════════════

def build_generation_messages(sample, video_path_map, max_frames, detail=DETAIL_LEVEL):
    """Build message lists for each assistant turn in a sample.

    Returns list of (conv_idx, messages) tuples.
    Frames go first in the first user message to maximize prompt cache hits
    across multiple samples from the same video.
    """
    video_id = sample.get("video_id")
    video_path = video_path_map.get(video_id)
    if not video_path:
        raise FileNotFoundError(f"Video not found: {video_id}")

    frame_content = build_frame_content(video_path, max_frames, detail)
    conversations = sample.get("conversations", [])

    turn_tasks = []
    messages = []

    for i, turn in enumerate(conversations):
        if turn["role"] == "user":
            if i == 0:
                content = list(frame_content) + [{"type": "text", "text": turn["content"]}]
            else:
                content = [{"type": "text", "text": turn["content"]}]
            messages.append({"role": "user", "content": content})
        elif turn["role"] == "assistant":
            turn_tasks.append((i, list(messages)))
            messages.append({"role": "assistant", "content": turn["content"]})

    return turn_tasks


def build_eval_message(criterion, ground_truth, model_response):
    """Build a single evaluation prompt for one criterion."""
    prompt = f"""You are an expert evaluator specializing in video content analysis and multimodal understanding.

Your task is to evaluate the Model Response against the **single evaluation criterion** provided, using the Ground Truth Response as a reference.

Ground Truth Response:
{ground_truth}

Model Response:
{model_response}

Evaluation Criterion:
{criterion['description']}

Instructions:
- Evaluate ONLY the provided criterion in this assessment.
- Compare the Model Response to the Ground Truth Response and determine if the criterion is satisfied.
- If the Model Response satisfies the criterion, set "criteria_met" to true; otherwise, set it to false.
- Focus on video content understanding, temporal relationships, and multimodal analysis."""

    return [{"role": "user", "content": prompt}]


# ═══════════════════════════════════════════════════════════════════════════
# Real-time inference
# ═══════════════════════════════════════════════════════════════════════════

def _create_client():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    import httpx
    return OpenAI(
        api_key=api_key,
        http_client=httpx.Client(
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=64),
            timeout=httpx.Timeout(1800, connect=30.0),
        ),
    )


def run_realtime_generation(args, samples, video_path_map, output_file, model, config, tracker):
    """Run real-time generation with automatic prompt caching."""
    client = _create_client()
    max_tokens = config.get("openai", {}).get("max_tokens", config["generation"]["max_tokens"])
    detail = config.get("openai", {}).get("detail", DETAIL_LEVEL)

    # Sort by video_id to maximize prompt cache hits (same frames prefix)
    samples = sorted(samples, key=lambda x: x.get("video_id", ""))

    all_turn_tasks = []
    for si, sample in enumerate(samples):
        try:
            turns = build_generation_messages(sample, video_path_map, args.max_frames, detail)
        except Exception as e:
            print(f"[WARNING] Build failed for {sample.get('sample_id')}: {e}")
            continue
        for conv_idx, msgs in turns:
            all_turn_tasks.append((si, conv_idx, msgs))

    print(f"Dispatching {len(all_turn_tasks)} turns across {args.num_workers} workers")

    completed_lock = threading.Lock()
    turns_remaining = {}
    for si, _, _ in all_turn_tasks:
        turns_remaining[si] = turns_remaining.get(si, 0) + 1

    pbar = tqdm(total=len(samples), desc="Generating")

    def _on_done(si):
        do_write = False
        with completed_lock:
            turns_remaining[si] -= 1
            if turns_remaining[si] == 0:
                do_write = True
        if do_write:
            s = samples[si]
            has_empty = any(
                t.get("role") == "assistant" and not t.get("candidate_response")
                for t in s.get("conversations", [])
            )
            if has_empty:
                print(f"\n[SKIP] {s.get('sample_id')}: empty response, will retry next run")
            else:
                with FileLock(f"{output_file}.lock"):
                    with open(output_file, "a") as f:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
            pbar.update(1)

    def _process_turn(task):
        si, conv_idx, messages = task
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
            timeout=1800,
        )
        resp = strip_think_tags(completion.choices[0].message.content or "")
        samples[si]["conversations"][conv_idx]["candidate_response"] = resp
        tracker.add_usage(completion.usage, model, "realtime")
        _on_done(si)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_process_turn, t): t for t in all_turn_tasks}
        try:
            for future in as_completed(futures):
                try:
                    future.result(timeout=300)
                except Exception as e:
                    t = futures[future]
                    print(f"\n[ERROR] {samples[t[0]].get('sample_id')} turn {t[1]}: {e}")
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    pbar.close()


def run_realtime_evaluation(args, responses, eval_file, model, config, tracker):
    """Run real-time evaluation as judge."""
    client = _create_client()
    eval_max_tokens = config.get("openai", {}).get("eval_max_tokens", config["evaluation"]["max_tokens"])

    json_schema = {
        "type": "object",
        "properties": {"criteria_met": {"type": "boolean"}},
        "required": ["criteria_met"],
        "additionalProperties": False,
    }

    all_crit_tasks = []
    for si, sample in enumerate(responses):
        for turn in sample.get("conversations", []):
            if turn["role"] != "assistant":
                continue
            gt = turn["content"]
            mr = turn.get("candidate_response", "")
            for criterion in turn.get("criteria", []):
                all_crit_tasks.append((si, criterion, gt, mr))

    print(f"Dispatching {len(all_crit_tasks)} criterion evals across {args.num_workers} workers")

    completed_lock = threading.Lock()
    crits_remaining = {}
    for si, _, _, _ in all_crit_tasks:
        crits_remaining[si] = crits_remaining.get(si, 0) + 1

    pbar = tqdm(total=len(responses), desc="Evaluating")

    def _on_done(si):
        do_write = False
        with completed_lock:
            crits_remaining[si] -= 1
            if crits_remaining[si] == 0:
                do_write = True
        if do_write:
            with FileLock(f"{eval_file}.lock"):
                with open(eval_file, "a") as f:
                    f.write(json.dumps(responses[si], ensure_ascii=False) + "\n")
            pbar.update(1)

    def _process_crit(task):
        si, criterion, gt, mr = task
        messages = build_eval_message(criterion, gt, mr)
        for attempt in range(3):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=eval_max_tokens,
                    timeout=120,
                    extra_body={"reasoning_effort": "minimal"},
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "criteria_evaluation",
                            "schema": json_schema,
                            "strict": True,
                        },
                    },
                )
                content = completion.choices[0].message.content
                if not content:
                    raise ValueError(f"Empty response, finish_reason={completion.choices[0].finish_reason}")
                parsed = json.loads(content)
                criterion["criteria_met"] = parsed["criteria_met"]
                tracker.add_usage(completion.usage, model, "realtime")
                _on_done(si)
                return
            except Exception as e:
                if attempt < 2:
                    time.sleep(1.0 * (2 ** attempt))
                    continue
                criterion["criteria_met"] = None
                criterion["evaluation_error"] = str(e)
                _on_done(si)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_process_crit, t): t for t in all_crit_tasks}
        try:
            for future in as_completed(futures):
                try:
                    future.result(timeout=300)
                except Exception as e:
                    t = futures[future]
                    print(f"\n[ERROR] Criterion eval for {responses[t[0]].get('sample_id')}: {e}")
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    pbar.close()


# ═══════════════════════════════════════════════════════════════════════════
# Batch API
# ═══════════════════════════════════════════════════════════════════════════

def _build_batch_jsonl_generation(samples, video_path_map, max_frames, model, config, detail=DETAIL_LEVEL):
    """Build batch JSONL for generation. Returns (lines, id_map).

    id_map: custom_id -> (sample_idx, conv_idx)
    """
    max_tokens = config.get("openai", {}).get("max_tokens", config["generation"]["max_tokens"])

    lines = []
    id_map = {}

    for si, sample in enumerate(samples):
        try:
            turns = build_generation_messages(sample, video_path_map, max_frames, detail)
        except Exception as e:
            print(f"[WARNING] Build failed for {sample.get('sample_id')}: {e}")
            continue
        for conv_idx, msgs in turns:
            custom_id = f"{sample['sample_id']}__t{conv_idx}"
            entry = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": msgs,
                    "max_completion_tokens": max_tokens,
                },
            }
            lines.append(json.dumps(entry, ensure_ascii=False))
            id_map[custom_id] = (si, conv_idx)

    return lines, id_map


def _build_batch_jsonl_eval(responses, model, config):
    """Build batch JSONL for evaluation. Returns (lines, id_map).

    id_map: custom_id -> (sample_idx, turn_idx, crit_idx)
    """
    eval_max_tokens = config.get("openai", {}).get("eval_max_tokens", config["evaluation"]["max_tokens"])

    json_schema = {
        "type": "object",
        "properties": {"criteria_met": {"type": "boolean"}},
        "required": ["criteria_met"],
        "additionalProperties": False,
    }

    lines = []
    id_map = {}

    for si, sample in enumerate(responses):
        for ti, turn in enumerate(sample.get("conversations", [])):
            if turn["role"] != "assistant":
                continue
            gt = turn["content"]
            mr = turn.get("candidate_response", "")
            for ci, criterion in enumerate(turn.get("criteria", [])):
                custom_id = f"{sample['sample_id']}__t{ti}__c{ci}"
                messages = build_eval_message(criterion, gt, mr)
                entry = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": messages,
                        "max_completion_tokens": eval_max_tokens,
                        "reasoning_effort": "minimal",
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "criteria_evaluation",
                                "schema": json_schema,
                                "strict": True,
                            },
                        },
                    },
                }
                lines.append(json.dumps(entry, ensure_ascii=False))
                id_map[custom_id] = (si, ti, ci)

    return lines, id_map


def submit_batch(lines, id_map, tracker, batch_dir, purpose="generation", target_file=None):
    """Upload JSONL and submit a batch. Returns batch_id."""
    client = _create_client()
    os.makedirs(batch_dir, exist_ok=True)

    input_path = os.path.join(batch_dir, f"batch_input_{purpose}_{len(lines)}_pid{os.getpid()}.jsonl")
    with open(input_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Uploading batch file ({len(lines)} requests, "
          f"{os.path.getsize(input_path) / 1024 / 1024:.1f} MB)...")
    batch_file = client.files.create(file=open(input_path, "rb"), purpose="batch")

    print(f"File uploaded: {batch_file.id}")

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"purpose": purpose},
    )

    print(f"Batch created: {batch.id} (status: {batch.status})")

    final_input_path = os.path.join(batch_dir, f"batch_{batch.id}_input.jsonl")
    os.replace(input_path, final_input_path)
    print(f"Input saved: {final_input_path}")

    tracker.set_batch(batch.id, status=batch.status, purpose=purpose,
                      total_requests=len(lines), file_id=batch_file.id,
                      target_file=target_file)

    map_path = os.path.join(batch_dir, f"batch_{batch.id}_map.json")
    with open(map_path, "w") as f:
        json.dump(id_map, f)
    print(f"ID map saved: {map_path}")

    return batch.id


def wait_for_batch(batch_id, tracker, poll_interval=15):
    """Poll a batch until completion, showing progress with tqdm. Returns results."""
    client = _create_client()
    batch = client.batches.retrieve(batch_id)
    total = batch.request_counts.total if batch.request_counts else 0

    pbar = tqdm(total=total, desc=f"Batch {batch_id[:16]}…", unit=" req")

    try:
        while True:
            batch = client.batches.retrieve(batch_id)
            counts = batch.request_counts
            status = batch.status

            if counts:
                done = counts.completed + counts.failed
                pbar.n = done
                pbar.set_postfix(status=status, failed=counts.failed, refresh=False)
                pbar.refresh()

            tracker.update_batch(batch_id, status=status,
                                 completed=counts.completed if counts else 0,
                                 failed=counts.failed if counts else 0)

            if status == "completed":
                pbar.n = total
                pbar.set_postfix(status="completed", refresh=False)
                pbar.refresh()
                break
            elif status in ("failed", "cancelled", "expired"):
                pbar.set_postfix(status=status, refresh=False)
                pbar.refresh()
                pbar.close()
                if status == "failed" and batch.errors and batch.errors.data:
                    for e in batch.errors.data[:5]:
                        print(f"  Error: {e.code} - {e.message}")
                print(f"Batch {status}.")
                return []

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        pbar.close()
        print(f"\nPolling interrupted. Batch {batch_id} still running on OpenAI.")
        print(f"Resume later: --resume-batch {batch_id}")
        return []

    pbar.close()
    return retrieve_batch_results(batch_id, tracker)


EVAL_BATCH_CHUNK_SIZE = 200  # max samples per eval batch
EVAL_BATCH_MIN_TAIL = 20    # merge last chunk into previous if smaller than this


def _poll_all_batches(pending, labels, tracker, poll_interval=15):
    """Poll multiple batches concurrently with per-batch tqdm bars.

    Args:
        pending: {batch_id: total_requests}
        labels:  {batch_id: short display label}
    Yields (batch_id, results) as each completes.
    """
    client = _create_client()
    active = dict(pending)

    # Create a tqdm bar per batch
    bars = {}
    for pos, (batch_id, total) in enumerate(active.items()):
        label = labels.get(batch_id, batch_id[:16])
        bars[batch_id] = tqdm(
            total=total, desc=f"  {label:<30} waiting",
            unit=" req", position=pos, leave=True,
            bar_format="{desc}  {percentage:3.0f}% {bar:25} {n_fmt}/{total_fmt} [{elapsed}]",
            mininterval=1,
        )

    try:
        while active:
            done_this_round = []
            for batch_id in list(active.keys()):
                batch = client.batches.retrieve(batch_id)
                counts = batch.request_counts
                status = batch.status

                tracker.update_batch(batch_id, status=status,
                                     completed=counts.completed if counts else 0,
                                     failed=counts.failed if counts else 0)

                bar = bars[batch_id]
                done = (counts.completed + counts.failed) if counts else 0
                bar.n = done
                label = labels.get(batch_id, batch_id[:16])
                bar.set_description_str(f"  {label:<30} {status}")
                bar.refresh()

                if status == "completed":
                    done_this_round.append(batch_id)
                elif status in ("failed", "cancelled", "expired"):
                    done_this_round.append(batch_id)

            for batch_id in done_this_round:
                del active[batch_id]
                bar = bars[batch_id]
                batch = client.batches.retrieve(batch_id)
                label = labels.get(batch_id, batch_id[:16])
                if batch.status == "completed":
                    bar.n = bar.total
                    bar.set_description_str(f"  {label:<30} done")
                    bar.refresh()
                    bar.close()
                    results = retrieve_batch_results(batch_id, tracker)
                    yield batch_id, results
                else:
                    bar.set_description_str(f"  {label:<30} {batch.status}")
                    bar.close()
                    if batch.status == "failed" and batch.errors and batch.errors.data:
                        for e in batch.errors.data[:3]:
                            tqdm.write(f"  Error: {e.code}: {e.message}")
                    yield batch_id, []

            if active:
                time.sleep(poll_interval)

    except KeyboardInterrupt:
        for bar in bars.values():
            bar.close()
        remaining = list(active.keys())
        print(f"\nPolling interrupted. {len(remaining)} batches still running.")
        for bid in remaining:
            print(f"  {bid}")
        print("Resume later with --resume-batch")
        return

    for bar in bars.values():
        if not bar.disable:
            bar.close()


def _resume_pending_batches(tracker, tracking_dir, all_responses, eval_file, purpose="evaluation"):
    """Resume any pending batches from a previous run. Returns set of sample_ids that were evaluated."""
    client = _create_client()
    batches = tracker.data.get("batches", {})

    pending = {}
    for batch_id, info in batches.items():
        if info.get("purpose") != purpose:
            continue
        if info.get("results_downloaded"):
            continue
        if info.get("target_file") and info["target_file"] != eval_file:
            continue
        status = info.get("status", "")
        if status in ("failed", "cancelled", "expired"):
            continue
        pending[batch_id] = info

    if not pending:
        return set()

    print(f"Found {len(pending)} pending {purpose} batches from previous run, resuming...")

    response_by_id = {s["sample_id"]: s for s in all_responses}
    evaluated_ids = set()

    for batch_id in list(pending):
        try:
            batch = client.batches.retrieve(batch_id)
        except Exception as e:
            print(f"  {batch_id}: error retrieving - {e}")
            continue

        status = batch.status

        if status in ("validating", "in_progress", "finalizing"):
            print(f"  {batch_id}: {status} - polling...")
            results = wait_for_batch(batch_id, tracker)
        elif status == "completed":
            results = retrieve_batch_results(batch_id, tracker)
        else:
            print(f"  {batch_id}: {status} - skipping")
            tracker.update_batch(batch_id, status=status)
            continue

        if not results:
            continue

        map_path = os.path.join(tracking_dir, f"batch_{batch_id}_map.json")
        if not os.path.exists(map_path):
            print(f"  {batch_id}: id_map not found at {map_path}, skipping apply")
            continue

        with open(map_path) as f:
            id_map = json.load(f)

        sample_ids = set()
        for cid in id_map:
            sid = cid.split("__t")[0]
            sample_ids.add(sid)

        chunk = [response_by_id[sid] for sid in sample_ids if sid in response_by_id]
        if chunk:
            apply_eval_results(results, chunk, eval_file)
            evaluated_ids.update(s["sample_id"] for s in chunk
                                 if any(c.get("criteria_met") is not None
                                        for t in s.get("conversations", []) if t["role"] == "assistant"
                                        for c in t.get("criteria", [])))

    return evaluated_ids


def run_chunked_batch_eval(responses, eval_file, model, config, tracker, tracking_dir):
    """Split evaluation into chunks, submit all batches, then poll and write
    results progressively as each completes."""

    resumed_ids = _resume_pending_batches(tracker, tracking_dir, responses, eval_file, "evaluation")
    if resumed_ids:
        responses = [s for s in responses if s["sample_id"] not in resumed_ids]
        print(f"Resumed {len(resumed_ids)} samples from pending batches, {len(responses)} remaining")
        if not responses:
            print("All samples covered by resumed batches.")
            return

    total = len(responses)
    tail = total % EVAL_BATCH_CHUNK_SIZE
    if 0 < tail < EVAL_BATCH_MIN_TAIL and total > EVAL_BATCH_CHUNK_SIZE:
        total_chunks = total // EVAL_BATCH_CHUNK_SIZE
        merge_tail = True
    else:
        total_chunks = (total + EVAL_BATCH_CHUNK_SIZE - 1) // EVAL_BATCH_CHUNK_SIZE
        merge_tail = False

    # Build and submit all batches
    pending = {}       # batch_id -> total_requests
    labels = {}        # batch_id -> display label
    batch_chunks = {}  # batch_id -> chunk (list of samples)

    chunk_boundaries = []
    pos = 0
    for i in range(total_chunks):
        end = pos + EVAL_BATCH_CHUNK_SIZE
        if merge_tail and i == total_chunks - 1:
            end = total
        end = min(end, total)
        chunk_boundaries.append((pos, end))
        pos = end

    for chunk_num, (start, end) in enumerate(chunk_boundaries, 1):
        chunk = responses[start:end]

        lines, id_map = _build_batch_jsonl_eval(chunk, model, config)
        if not lines:
            continue

        print(f"Submitting batch {chunk_num}/{total_chunks} ({len(chunk)} samples, {len(lines)} criteria)")
        batch_id = submit_batch(lines, id_map, tracker, tracking_dir,
                                purpose="evaluation", target_file=eval_file)
        pending[batch_id] = len(lines)
        labels[batch_id] = f"eval {chunk_num}/{total_chunks}"
        batch_chunks[batch_id] = chunk

    if not pending:
        return

    print(f"\n{len(pending)} batches submitted, polling for results...\n")

    for batch_id, results in _poll_all_batches(pending, labels, tracker):
        if results:
            apply_eval_results(results, batch_chunks[batch_id], eval_file)

    print(f"\nAll eval batches complete.")


def run_chunked_batch_generation(samples, video_path_map, output_file, max_frames, model,
                                  config, tracker, tracking_dir, detail=DETAIL_LEVEL):
    """Split generation by video_id, submit all batches, then poll and write
    results progressively as each completes."""
    from collections import OrderedDict

    video_groups = OrderedDict()
    for sample in samples:
        vid = sample.get("video_id", "unknown")
        if vid not in video_groups:
            video_groups[vid] = []
        video_groups[vid].append(sample)

    total_videos = len(video_groups)

    # Build and submit all batches
    pending = {}       # batch_id -> total_requests
    labels = {}        # batch_id -> display label
    batch_chunks = {}  # batch_id -> chunk samples

    for batch_num, (vid, chunk_samples) in enumerate(video_groups.items(), 1):
        lines, id_map = _build_batch_jsonl_generation(
            chunk_samples, video_path_map, max_frames, model, config, detail)
        if not lines:
            continue

        print(f"Submitting batch {batch_num}/{total_videos}: {vid} ({len(chunk_samples)} samples, {len(lines)} turns)")
        batch_id = submit_batch(lines, id_map, tracker, tracking_dir,
                                purpose="generation", target_file=output_file)
        pending[batch_id] = len(lines)
        labels[batch_id] = vid
        batch_chunks[batch_id] = chunk_samples

    if not pending:
        return

    print(f"\n{len(pending)} batches submitted, polling for results...\n")

    for batch_id, results in _poll_all_batches(pending, labels, tracker):
        if results:
            apply_generation_results(results, batch_chunks[batch_id], output_file)

    print(f"\nAll generation batches complete.")


def check_batches(tracker):
    """Print status of all tracked batches."""
    client = _create_client()
    batches = tracker.data.get("batches", {})
    if not batches:
        print("No tracked batches.")
        return

    for batch_id, info in batches.items():
        try:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            counts = batch.request_counts
            tracker.update_batch(batch_id, status=status,
                                 completed=counts.completed if counts else 0,
                                 failed=counts.failed if counts else 0)
            print(f"  {batch_id}: {status} "
                  f"({counts.completed}/{counts.total} done, {counts.failed} failed)"
                  if counts else f"  {batch_id}: {status}")
        except Exception as e:
            print(f"  {batch_id}: error checking - {e}")


def retrieve_batch_results(batch_id, tracker):
    """Download and parse batch results. Returns list of (custom_id, response_content)."""
    client = _create_client()
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        print(f"Batch {batch_id} status: {batch.status} (not completed yet)")
        if batch.status == "failed":
            errors = batch.errors
            if errors and errors.data:
                for e in errors.data[:5]:
                    print(f"  Error: {e.code} - {e.message}")
        return []

    if not batch.output_file_id:
        for retry in range(6):
            time.sleep(5)
            batch = client.batches.retrieve(batch_id)
            if batch.output_file_id:
                break
        if not batch.output_file_id:
            print(f"No output file for batch {batch_id} after retries")
            return []

    print(f"Downloading results from {batch.output_file_id}...")
    content = client.files.content(batch.output_file_id)
    raw = content.read().decode("utf-8")

    batch_info = tracker.data.get("batches", {}).get(batch_id, {})
    save_dir = os.path.dirname(batch_info.get("target_file", "")) if batch_info.get("target_file") else None
    if not save_dir:
        save_dir = os.path.join(".api_cache", "openai")
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"batch_{batch_id}_output.jsonl")
    with open(output_path, "w") as f:
        f.write(raw)
    print(f"Output saved: {output_path}")

    results = []
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        entry = json.loads(line)
        custom_id = entry["custom_id"]
        response = entry.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content", "")
            results.append((custom_id, text))
            usage_data = body.get("usage")
            if usage_data:
                class _Usage:
                    def __init__(self, d):
                        self.prompt_tokens = d.get("prompt_tokens", 0)
                        self.completion_tokens = d.get("completion_tokens", 0)
                        self.prompt_tokens_details = None
                        cached = d.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                        if cached:
                            class _Details:
                                pass
                            self.prompt_tokens_details = _Details()
                            self.prompt_tokens_details.cached_tokens = cached
                tracker.add_usage(_Usage(usage_data), body.get("model", "gpt-5.4"), "batch")
        else:
            error = entry.get("error", {})
            print(f"[ERROR] {custom_id}: {error.get('code')} - {error.get('message', '')[:100]}")

    tracker.update_batch(batch_id, status="completed",
                         results_downloaded=True,
                         results_count=len(results))
    print(f"Retrieved {len(results)} results")
    return results


def apply_generation_results(results, samples, output_file):
    """Apply batch generation results to samples and write to JSONL."""
    result_map = {}
    for custom_id, text in results:
        result_map[custom_id] = strip_think_tags(text)

    written = 0
    for si, sample in enumerate(samples):
        conversations = sample.get("conversations", [])
        all_filled = True
        for i, turn in enumerate(conversations):
            if turn["role"] == "assistant":
                cid = f"{sample['sample_id']}__t{i}"
                resp = result_map.get(cid)
                if resp:
                    turn["candidate_response"] = resp
                elif not turn.get("candidate_response"):
                    all_filled = False

        if all_filled:
            has_empty = any(
                t.get("role") == "assistant" and not t.get("candidate_response")
                for t in conversations
            )
            if not has_empty:
                with FileLock(f"{output_file}.lock"):
                    with open(output_file, "a") as f:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written}/{len(samples)} samples to {output_file}")


def apply_eval_results(results, responses, eval_file):
    """Apply batch evaluation results to responses and write to JSONL."""
    result_map = {}
    for custom_id, text in results:
        result_map[custom_id] = text

    written = 0
    for si, sample in enumerate(responses):
        conversations = sample.get("conversations", [])
        all_filled = True
        for ti, turn in enumerate(conversations):
            if turn["role"] != "assistant":
                continue
            for ci, criterion in enumerate(turn.get("criteria", [])):
                cid = f"{sample['sample_id']}__t{ti}__c{ci}"
                resp = result_map.get(cid)
                if resp:
                    try:
                        parsed = json.loads(resp)
                        criterion["criteria_met"] = parsed["criteria_met"]
                    except (json.JSONDecodeError, KeyError):
                        criterion["criteria_met"] = None
                        criterion["evaluation_error"] = f"Parse error: {resp[:100]}"
                elif criterion.get("criteria_met") is None:
                    all_filled = False

        if all_filled:
            with FileLock(f"{eval_file}.lock"):
                with open(eval_file, "a") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written}/{len(responses)} evaluations to {eval_file}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI API benchmark for LongShOT")
    parser.add_argument("--tasks", nargs="+", help="Task names")
    parser.add_argument("--config", default="tasks.yaml", help="Task config YAML")
    parser.add_argument("--config-file", default="config.yaml", help="System config YAML")
    parser.add_argument("--output-dir", default="results_postvalid", help="Output directory")
    parser.add_argument("--model", default=MODEL, help="OpenAI model name")
    parser.add_argument("--mode", choices=["realtime", "batch"], default="realtime",
                        help="Inference mode (realtime with caching or batch with 50%% off)")
    parser.add_argument("--max-frames", type=int, default=128,
                        help="Number of pre-extracted frames to send")
    parser.add_argument("--detail", choices=["low", "high", "auto"], default=None,
                        help="Image detail level (low=85 tok/img, high=170/tile+85)")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Concurrent workers for realtime mode")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run as evaluation judge instead of generation")
    parser.add_argument("--candidate-model", type=str, default=None,
                        help="Model name whose outputs to evaluate (underscored format)")
    parser.add_argument("--eval-tag", type=str, default=None,
                        help="Judge tag for eval output naming")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of samples to process (0=unlimited, for test runs)")
    parser.add_argument("--check-batches", action="store_true",
                        help="Check status of pending batches")
    parser.add_argument("--resume-batch", type=str, default=None,
                        help="Retrieve results from a completed batch")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config_file)

    model_underscored = convert_to_underscored(args.model)
    tracking_dir = os.path.join(".api_cache", "openai", model_underscored)
    os.makedirs(tracking_dir, exist_ok=True)
    tracker = CostTracker(os.path.join(tracking_dir, "tracking.json"))

    if args.detail:
        global DETAIL_LEVEL
        DETAIL_LEVEL = args.detail

    # ── Check batches ──
    if args.check_batches:
        check_batches(tracker)
        print(f"\nCost summary: {json.dumps(tracker.summary(), indent=2)}")
        return

    # ── Resume batch ──
    if args.resume_batch:
        if not args.tasks:
            print("--tasks required with --resume-batch")
            return
        results = retrieve_batch_results(args.resume_batch, tracker)
        if not results:
            return

        # Search for map file in tracking dir
        map_path = os.path.join(tracking_dir, f"batch_{args.resume_batch}_map.json")
        if not os.path.exists(map_path):
            # Fallback: check old root location
            old_path = f".openai_batch_{args.resume_batch}_map.json"
            if os.path.exists(old_path):
                map_path = old_path
            else:
                print(f"ID map not found: {map_path}")
                return
        with open(map_path) as f:
            id_map = json.load(f)

        # Determine if this is generation or eval batch
        batch_info = tracker.data["batches"].get(args.resume_batch, {})
        purpose = batch_info.get("purpose", "generation")

        # Load task data
        import yaml
        with open(args.config) as f:
            dataset_configs = yaml.safe_load(f)
        task_configs = {}
        for cat_tasks in dataset_configs.values():
            task_configs.update(cat_tasks)

        if purpose == "evaluation":
            candidate = args.candidate_model
            if not candidate:
                print("--candidate-model required for eval batch resume")
                return
            gen_paths = get_generation_artifact_paths(args.output_dir, candidate)
            responses = load_jsonl(gen_paths["output_file"])
            judge_paths = get_judge_artifact_paths(
                args.output_dir, candidate, args.model, args.eval_tag)
            os.makedirs(judge_paths["judge_dir"], exist_ok=True)
            apply_eval_results(results, responses, judge_paths["eval_file"])
        else:
            samples = []
            for task_name in args.tasks:
                samples += load_dataset_with_params(task_configs[task_name], task_name, config)
            # Filter completed
            gen_paths = get_generation_artifact_paths(args.output_dir, model_underscored)
            completed_ids = set()
            if os.path.exists(gen_paths["output_file"]):
                sid_re = re.compile(r'"sample_id"\s*:\s*"([^"]+)"')
                with open(gen_paths["output_file"], "rb") as f:
                    for line in f:
                        m = sid_re.search(line.decode("utf-8", errors="ignore"))
                        if m:
                            completed_ids.add(m.group(1))
            samples = [s for s in samples if s.get("sample_id") not in completed_ids]
            apply_generation_results(results, samples, gen_paths["output_file"])

        print(f"\nCost summary: {json.dumps(tracker.summary(), indent=2)}")
        return

    # ── Generation or Evaluation ──
    if not args.tasks:
        print("--tasks required")
        return

    import yaml
    with open(args.config) as f:
        dataset_configs = yaml.safe_load(f)
    task_configs = {}
    for cat_tasks in dataset_configs.values():
        task_configs.update(cat_tasks)

    start_time = time.time()

    if args.evaluate:
        # ── Evaluation mode ──
        candidate = args.candidate_model
        if not candidate:
            print("--candidate-model required for evaluation")
            return

        gen_paths = get_generation_artifact_paths(args.output_dir, candidate)
        if not os.path.exists(gen_paths["output_file"]):
            print(f"No generation output found: {gen_paths['output_file']}")
            return

        judge_paths = get_judge_artifact_paths(
            args.output_dir, candidate, args.model, args.eval_tag)
        os.makedirs(judge_paths["judge_dir"], exist_ok=True)
        eval_file = judge_paths["eval_file"]

        responses = load_jsonl(gen_paths["output_file"])

        # Filter already evaluated
        evaluated_ids = set()
        if os.path.exists(eval_file):
            sid_re = re.compile(r'"sample_id"\s*:\s*"([^"]+)"')
            with open(eval_file, "rb") as f:
                for line in f:
                    m = sid_re.search(line.decode("utf-8", errors="ignore"))
                    if m:
                        evaluated_ids.add(m.group(1))
        responses = [s for s in responses if s.get("sample_id") not in evaluated_ids]
        if args.limit > 0:
            responses = responses[:args.limit]

        if not responses:
            print("All samples already evaluated.")
            return

        print(f"Evaluating {len(responses)} samples with {args.model} (mode: {args.mode})")

        if args.mode == "batch":
            run_chunked_batch_eval(responses, eval_file, args.model, config, tracker, tracking_dir)
        else:
            run_realtime_evaluation(args, responses, eval_file, args.model, config, tracker)

        elapsed = time.time() - start_time
        print(f"\nDone in {elapsed:.0f}s")
        print(f"Cost summary: {json.dumps(tracker.summary(), indent=2)}")

    else:
        # ── Generation mode ──
        gen_paths = get_generation_artifact_paths(args.output_dir, model_underscored)
        os.makedirs(gen_paths["model_dir"], exist_ok=True)
        output_file = gen_paths["output_file"]

        samples = []
        for task_name in args.tasks:
            samples += load_dataset_with_params(task_configs[task_name], task_name, config)

        # Filter completed
        completed_ids = set()
        if os.path.exists(output_file):
            sid_re = re.compile(r'"sample_id"\s*:\s*"([^"]+)"')
            with open(output_file, "rb") as f:
                for line in f:
                    m = sid_re.search(line.decode("utf-8", errors="ignore"))
                    if m:
                        completed_ids.add(m.group(1))

        samples = [s for s in samples if s.get("sample_id") not in completed_ids]
        if args.limit > 0:
            samples = samples[:args.limit]

        if not samples:
            print("All samples already processed.")
            return

        print(f"Processing {len(samples)} samples with {args.model} "
              f"(mode: {args.mode}, frames: {args.max_frames}, skipped: {len(completed_ids)})")

        video_path_map = build_video_path_map(config["paths"]["video_path"])
        print(f"Video path map: {len(video_path_map)} videos indexed")

        if args.mode == "batch":
            detail = config.get("openai", {}).get("detail", DETAIL_LEVEL)
            run_chunked_batch_generation(samples, video_path_map, output_file,
                                         args.max_frames, args.model, config,
                                         tracker, tracking_dir, detail)
        else:
            try:
                run_realtime_generation(args, samples, video_path_map, output_file,
                                        args.model, config, tracker)
            except KeyboardInterrupt:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                print("\n[INTERRUPTED]")

        elapsed = time.time() - start_time
        save_timing_data(gen_paths["timing_file"], "generation", {
            "total": elapsed,
            "samples_processed": len(samples) - len(completed_ids),
            "samples_skipped": len(completed_ids),
            "samples_total": len(samples) + len(completed_ids),
            "mode": args.mode,
            "model": args.model,
        })
        print(f"\nDone in {elapsed:.0f}s")
        print(f"Cost summary: {json.dumps(tracker.summary(), indent=2)}")


if __name__ == "__main__":
    main()
