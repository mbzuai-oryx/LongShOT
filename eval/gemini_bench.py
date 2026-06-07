"""Gemini API benchmarking script for LongShOT evaluation framework.

Standalone generation script that uses Google's Gemini API with cost-saving
features: Files API (free upload/reuse), Context Caching (90% off cached
tokens), Flex inference (50% off), and Batch API (50% off, async).

Usage:
    python gemini_bench.py --tasks postvalid_v2 --mode flex --num-workers 8
    python gemini_bench.py --tasks postvalid_v2 --mode batch
    python gemini_bench.py --check-batches
    python gemini_bench.py --resume-batch BATCH_ID --tasks postvalid_v2
"""

import argparse
import copy
import json
import os
import random
import re
import signal
import subprocess
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from filelock import FileLock
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

from utils import (
    build_video_path_map,
    convert_to_underscored,
    load_config,
    load_dataset_with_params,
    save_timing_data,
    strip_think_tags,
)

# ---------------------------------------------------------------------------
# Global defaults — override via CLI --model
# ---------------------------------------------------------------------------
MODEL = "gemini-3.1-pro-preview"

PRICING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_pricing.json")

# Files API retention (Gemini deletes after 48h)
FILE_TTL_HOURS = 48
# Safety margin: treat files as expired 1h before actual expiry
FILE_EXPIRY_MARGIN_HOURS = 1

# Max video duration (seconds) before speedup is applied
MAX_VIDEO_DURATION_SECS = 45 * 60


# ═══════════════════════════════════════════════════════════════════════════
# Pricing helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_pricing():
    with open(PRICING_FILE) as f:
        return json.load(f)


def _select_tier(model_pricing, total_input_tokens):
    for tier in model_pricing["tiers"]:
        cap = tier.get("max_input_tokens")
        if cap is None or total_input_tokens <= cap:
            return tier
    return model_pricing["tiers"][-1]


def compute_request_cost(model, mode, prompt_tokens, output_tokens, cached_tokens, pricing_data):
    model_pricing = pricing_data["models"].get(model)
    if not model_pricing:
        return 0.0

    discount = pricing_data["tier_discounts"].get(mode, 1.0)
    tier = _select_tier(model_pricing, prompt_tokens)

    uncached_input = max(0, prompt_tokens - cached_tokens)
    input_cost = (uncached_input / 1_000_000) * tier["input_per_1m"] * discount
    cached_cost = (cached_tokens / 1_000_000) * tier["cached_input_per_1m"] * discount
    output_cost = (output_tokens / 1_000_000) * tier["output_per_1m"] * discount

    return input_cost + cached_cost + output_cost


# ═══════════════════════════════════════════════════════════════════════════
# GeminiTracker — persistent ID & cost tracking
# ═══════════════════════════════════════════════════════════════════════════

class GeminiTracker:

    def __init__(self, tracking_path):
        self.path = tracking_path
        self._lock = threading.Lock()
        self.data = {
            "files": {},
            "caches": {},
            "batches": {},
            "requests": {},
            "cost": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cached_tokens": 0,
                "estimated_cost_usd": 0.0,
            },
            "request_logs": [],
        }
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    stored = json.load(f)
                for key in self.data:
                    if key in stored:
                        self.data[key] = stored[key]
            except (json.JSONDecodeError, OSError):
                pass

    def _save(self):
        dir_path = os.path.dirname(self.path) or "."
        os.makedirs(dir_path, exist_ok=True)
        try:
            fd, tmp = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(self.data, f, indent=2, default=str)
                os.replace(tmp, self.path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:
            print(f"[WARNING] Failed to save tracker to {self.path}: {e}")

    def save(self):
        with self._lock:
            self._save()

    def get_file(self, video_id):
        with self._lock:
            return self.data["files"].get(video_id)

    def set_file(self, video_id, gemini_name, uri, state="ACTIVE"):
        now = datetime.now(timezone.utc)
        with self._lock:
            self.data["files"][video_id] = {
                "gemini_name": gemini_name,
                "uri": uri,
                "upload_time": now.isoformat(),
                "expires_at": (now + timedelta(hours=FILE_TTL_HOURS)).isoformat(),
                "state": state,
            }
            self._save()

    def is_file_expired(self, video_id):
        entry = self.get_file(video_id)
        if not entry:
            return True
        expires = datetime.fromisoformat(entry["expires_at"])
        margin = timedelta(hours=FILE_EXPIRY_MARGIN_HOURS)
        return datetime.now(timezone.utc) >= (expires - margin)

    def get_cache(self, video_id):
        with self._lock:
            return self.data["caches"].get(video_id)

    def set_cache(self, video_id, cache_name, model, ttl_seconds):
        now = datetime.now(timezone.utc)
        with self._lock:
            self.data["caches"][video_id] = {
                "cache_name": cache_name,
                "model": model,
                "created_at": now.isoformat(),
                "ttl_seconds": ttl_seconds,
                "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
            }
            self._save()

    def is_cache_expired(self, video_id):
        entry = self.get_cache(video_id)
        if not entry:
            return True
        expires = datetime.fromisoformat(entry["expires_at"])
        return datetime.now(timezone.utc) >= expires

    def set_batch(self, batch_id, status, model, display_name, total_requests):
        with self._lock:
            self.data["batches"][batch_id] = {
                "batch_id": batch_id,
                "status": status,
                "model": model,
                "display_name": display_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "total_requests": total_requests,
                "completed_requests": 0,
                "failed_requests": 0,
            }
            self._save()

    def update_batch(self, batch_id, **kwargs):
        with self._lock:
            if batch_id in self.data["batches"]:
                self.data["batches"][batch_id].update(kwargs)
                self._save()

    def get_batch(self, batch_id):
        with self._lock:
            return self.data["batches"].get(batch_id)

    def get_pending_batches(self):
        with self._lock:
            return {
                bid: info for bid, info in self.data["batches"].items()
                if info.get("status") not in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")
            }

    def set_request(self, custom_id, batch_id):
        with self._lock:
            self.data["requests"][custom_id] = {
                "batch_id": batch_id,
                "status": "pending",
            }

    def add_cost(self, model, mode, prompt_tokens, output_tokens, cached_tokens,
                 sample_id, turn_idx, pricing_data):
        cost = compute_request_cost(
            model, mode, prompt_tokens, output_tokens, cached_tokens, pricing_data)
        log_entry = {
            "sample_id": sample_id,
            "turn_idx": turn_idx,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_token_count": prompt_tokens,
            "candidates_token_count": output_tokens,
            "cached_content_token_count": cached_tokens,
            "estimated_cost_usd": round(cost, 6),
        }
        with self._lock:
            self.data["cost"]["total_input_tokens"] += prompt_tokens
            self.data["cost"]["total_output_tokens"] += output_tokens
            self.data["cost"]["total_cached_tokens"] += cached_tokens
            self.data["cost"]["estimated_cost_usd"] += cost
            self.data["request_logs"].append(log_entry)
            self._save()
        return cost

    def get_cost_summary(self):
        with self._lock:
            return dict(self.data["cost"])


# ═══════════════════════════════════════════════════════════════════════════
# Retry helper
# ═══════════════════════════════════════════════════════════════════════════

def _is_retryable(exc):
    msg = str(exc).lower()
    for code in ("429", "503", "500", "resource_exhausted", "unavailable",
                 "internal", "deadline_exceeded", "connection"):
        if code in msg:
            return True
    return False


def _retry_with_backoff(fn, max_retries=5, base_delay=2.0, max_delay=120.0, label=""):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if not _is_retryable(e) or attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay) + random.uniform(0, 1)
            print(f"[RETRY {attempt+1}/{max_retries}] {label}: {type(e).__name__}: "
                  f"{str(e)[:120]}. Waiting {delay:.1f}s...")
            time.sleep(delay)


# ═══════════════════════════════════════════════════════════════════════════
# GeminiFileManager — Files API + Context Caching lifecycle
# ═══════════════════════════════════════════════════════════════════════════

def _get_video_duration(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")
    return float(result.stdout.strip())


def _build_atempo_chain(speed_factor):
    """Build chained atempo filters since each instance only supports 0.5–100.0."""
    filters = []
    remaining = speed_factor
    while remaining > 100.0:
        filters.append("atempo=100.0")
        remaining /= 100.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


def _speedup_video(video_path, target_duration, video_id):
    duration = _get_video_duration(video_path)
    if duration <= target_duration:
        return video_path, None

    speed_factor = duration / target_duration
    print(f"[SPEEDUP] {video_id}: {duration:.0f}s -> {target_duration:.0f}s "
          f"({speed_factor:.2f}x)")

    suffix = os.path.splitext(video_path)[1] or ".mp4"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix=f"speedup_{video_id}_")
    os.close(fd)

    atempo_filter = _build_atempo_chain(speed_factor)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-filter_complex",
        f"[0:v]setpts=PTS/{speed_factor:.6f}[v];[0:a]{atempo_filter}[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264",  "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        tmp_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=True)
    except subprocess.CalledProcessError as e:
        # Retry without audio in case source has no audio stream
        cmd_no_audio = [
            "ffmpeg", "-y", "-i", video_path,
            "-filter:v", f"setpts=PTS/{speed_factor:.6f}",
            "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-movflags", "+faststart",
            tmp_path,
        ]
        try:
            subprocess.run(cmd_no_audio, capture_output=True, text=True,
                           timeout=1800, check=True)
        except subprocess.CalledProcessError as e2:
            os.unlink(tmp_path)
            raise RuntimeError(
                f"ffmpeg speedup failed for {video_id}: {e2.stderr[:300]}") from e2

    return tmp_path, tmp_path


class GeminiFileManager:

    def __init__(self, client, tracker):
        self.client = client
        self.tracker = tracker

    def upload_video(self, video_id, video_path):
        if not self.tracker.is_file_expired(video_id):
            entry = self.tracker.get_file(video_id)
            if entry and entry.get("state") == "ACTIVE":
                # Verify the file still exists on Gemini's side
                try:
                    info = self.client.files.get(name=entry["gemini_name"])
                    state = str(getattr(info, "state", "")).upper()
                    if "ACTIVE" in state:
                        return entry["gemini_name"], entry["uri"]
                except Exception:
                    pass
                # File gone or inaccessible, re-upload

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        upload_path, tmp_file = _speedup_video(
            video_path, MAX_VIDEO_DURATION_SECS, video_id)

        try:
            uploaded = _retry_with_backoff(
                lambda: self.client.files.upload(file=upload_path),
                max_retries=3, base_delay=5.0, label=f"upload {video_id}")
        finally:
            if tmp_file:
                try:
                    os.unlink(tmp_file)
                except OSError:
                    pass

        if not uploaded or not getattr(uploaded, "name", None):
            raise RuntimeError(f"Upload returned invalid response for {video_id}")

        uri = getattr(uploaded, "uri", None) or ""
        self.tracker.set_file(video_id, uploaded.name, uri, state="PROCESSING")
        self.wait_for_processing(uploaded.name, video_id)
        # Re-read URI after processing (may have changed)
        try:
            info = self.client.files.get(name=uploaded.name)
            uri = getattr(info, "uri", uri) or uri
        except Exception:
            pass
        self.tracker.set_file(video_id, uploaded.name, uri, state="ACTIVE")
        return uploaded.name, uri

    def wait_for_processing(self, file_name, video_id, timeout=900):
        start = time.time()
        poll_interval = 10
        while time.time() - start < timeout:
            try:
                info = self.client.files.get(name=file_name)
            except Exception as e:
                print(f"[WARNING] files.get failed for {video_id}: {e}")
                time.sleep(poll_interval)
                continue

            state = str(getattr(info, "state", "UNKNOWN")).upper()
            if "ACTIVE" in state:
                self.tracker.set_file(video_id, info.name, info.uri, state="ACTIVE")
                return
            if "FAILED" in state:
                raise RuntimeError(f"File processing failed for {video_id}: {file_name}")
            time.sleep(poll_interval)

        raise TimeoutError(f"File {video_id} not ACTIVE after {timeout}s")

    def create_cache(self, video_id, model, file_uri, ttl=3600):
        if not self.tracker.is_cache_expired(video_id):
            entry = self.tracker.get_cache(video_id)
            if entry and entry.get("model") == model:
                return entry["cache_name"]

        cache = _retry_with_backoff(
            lambda: self.client.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_uri(
                                file_uri=file_uri, mime_type="video/mp4")],
                        )
                    ],
                    ttl=f"{ttl}s",
                )
            ),
            max_retries=3, base_delay=5.0,
            label=f"cache {video_id}")

        self.tracker.set_cache(video_id, cache.name, model, ttl)
        return cache.name

    def create_all_caches(self, file_map, model, ttl=3600):
        results = {}
        to_create = []

        for vid, (name, uri) in file_map.items():
            if not self.tracker.is_cache_expired(vid):
                entry = self.tracker.get_cache(vid)
                if entry and entry.get("model") == model:
                    results[vid] = entry["cache_name"]
                    continue
            to_create.append((vid, uri))

        if not to_create:
            print(f"All {len(results)} caches already valid.")
            return results

        print(f"Creating {len(to_create)} caches ({len(results)} already valid)...")
        for vid, uri in tqdm(to_create, desc="Creating caches"):
            cache_name = self.create_cache(vid, model, uri, ttl)
            results[vid] = cache_name

        return results

    def delete_file(self, video_id):
        """Best-effort delete; files auto-expire after 48h anyway."""
        entry = self.tracker.get_file(video_id)
        if not entry:
            return
        try:
            _retry_with_backoff(
                lambda: self.client.files.delete(name=entry["gemini_name"]),
                max_retries=2, base_delay=2.0, max_delay=10.0,
                label=f"delete {video_id}")
        except Exception:
            pass  # file will auto-expire in 48h
        with self.tracker._lock:
            self.tracker.data["files"].pop(video_id, None)
            self.tracker._save()

    def delete_all_files(self):
        with self.tracker._lock:
            all_vids = list(self.tracker.data["files"].keys())
        deleted = 0
        for vid in all_vids:
            self.delete_file(vid)
            deleted += 1
        print(f"Deleted {deleted} files from Gemini storage.")
        return deleted

    def list_files(self):
        try:
            files = list(self.client.files.list())
        except Exception as e:
            print(f"[ERROR] Failed to list files: {e}")
            return

        total_bytes = 0
        print(f"\n{'Name':<35} {'Size':>10} {'State':<15} {'Expires'}")
        print("-" * 85)
        for f in files:
            size = getattr(f, "size_bytes", 0) or 0
            state = str(getattr(f, "state", "?"))
            name = getattr(f, "name", "?")
            expire = getattr(f, "expiration_time", None) or getattr(f, "expire_time", "?")
            print(f"  {name:<33} {size/1e6:>8.1f}MB  {state:<15} {expire}")
            total_bytes += size
        print("-" * 85)
        print(f"  Total: {len(files)} files, {total_bytes/1e6:.1f}MB / 20,000MB "
              f"({total_bytes/20e9*100:.1f}% used)")


# ═══════════════════════════════════════════════════════════════════════════
# GeminiBenchmark — main orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class GeminiBenchmark:

    def __init__(self, model, mode, config, args):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. "
                "Set it or add to .env file.")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.mode = mode
        self.config = config
        self.args = args
        self.pricing_data = load_pricing()

        model_underscored = args.alias or convert_to_underscored(model)
        model_dir = os.path.join(args.output_dir, model_underscored)
        os.makedirs(model_dir, exist_ok=True)

        tracking_path = args.tracking_file or os.path.join(
            model_dir, "gemini_tracking.json")
        self.tracker = GeminiTracker(tracking_path)
        self.file_manager = GeminiFileManager(self.client, self.tracker)

        self.output_file = os.path.join(model_dir, f"{model_underscored}.jsonl")
        self.timing_file = os.path.join(model_dir, f"{model_underscored}_timing.json")
        self.model_underscored = model_underscored

    # --- Shared helpers ---

    @staticmethod
    def _build_sample_index(samples):
        """Build {sample_id}__turn_{i} -> (sample, i) mapping for batch result parsing."""
        index = {}
        for sample in samples:
            sid = sample.get("sample_id", "unknown")
            for i, turn in enumerate(sample.get("conversations", [])):
                if turn.get("role") == "assistant":
                    index[f"{sid}__turn_{i}"] = (sample, i)
        return index

    def _load_samples(self):
        """Load and return samples from task YAML config."""
        import yaml
        if not os.path.exists(self.args.task_config):
            raise FileNotFoundError(f"Task config not found: {self.args.task_config}")

        with open(self.args.task_config) as f:
            dataset_configs = yaml.safe_load(f)

        task_configs = {}
        all_tasks = []
        for cat_tasks in dataset_configs.values():
            all_tasks.extend(cat_tasks.keys())
            task_configs.update(cat_tasks)

        tasks_to_load = []
        for task in self.args.tasks:
            if task == "all":
                tasks_to_load = all_tasks
                break
            elif task in all_tasks:
                tasks_to_load.append(task)
            else:
                print(f"Warning: Unknown task '{task}'")

        if not tasks_to_load:
            print("No valid tasks. Available:", ", ".join(all_tasks))
            return []

        test_samples = []
        for task_name in tasks_to_load:
            print(f"Loading task: {task_name}")
            test_samples += load_dataset_with_params(
                task_configs[task_name], task_name, self.config)

        return test_samples

    # --- Turn task building ---

    def _build_turn_tasks(self, sample, cache_name, file_uri):
        """Returns list of (conv_index, contents) tuples for each assistant turn."""
        conversations = sample.get("conversations", [])
        if not conversations or not isinstance(conversations, list):
            return []

        turn_tasks = []
        contents = []

        for i, turn in enumerate(conversations):
            role = turn.get("role")
            text_content = turn.get("content", "")

            if role == "user":
                if not text_content:
                    text_content = " "
                if i == 0 and not cache_name and file_uri:
                    parts = [
                        types.Part.from_uri(file_uri=file_uri, mime_type="video/mp4"),
                        types.Part.from_text(text=text_content),
                    ]
                else:
                    parts = [types.Part.from_text(text=text_content)]
                contents.append(types.Content(role="user", parts=parts))

            elif role == "assistant":
                turn_tasks.append((i, list(contents)))
                gt_text = text_content or " "
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=gt_text)],
                ))

        return turn_tasks

    # Map CLI mode strings to SDK ServiceTier enum
    _SERVICE_TIER_MAP = {
        "flex": types.ServiceTier.FLEX,
        "standard": types.ServiceTier.STANDARD,
    }

    def _call_gemini(self, contents, cache_name=None, service_tier=None):
        """Returns (response_text, usage). Raises on unrecoverable errors."""
        config_kwargs = {
            "temperature": self.config["generation"]["temperature"],
            "max_output_tokens": self.config["generation"]["max_tokens"],
            "top_p": self.config["generation"]["top_p"],
        }
        if cache_name:
            config_kwargs["cached_content"] = cache_name
        if service_tier:
            config_kwargs["service_tier"] = self._SERVICE_TIER_MAP.get(
                service_tier, service_tier)

        system_prompt = self.config["system"].get("system_prompt", "")
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        gen_config = types.GenerateContentConfig(**config_kwargs)

        def _do_call():
            return self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=gen_config,
            )

        response = _retry_with_backoff(
            _do_call, max_retries=5, base_delay=2.0, max_delay=120.0)

        if response is None:
            raise RuntimeError("API returned None response after retries")

        # Check for blocked content / safety filters
        if not response.candidates:
            block_reason = getattr(response, "prompt_feedback", None)
            raise RuntimeError(
                f"No candidates in response. "
                f"Prompt feedback: {block_reason}")

        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)

        # Extract text from parts safely
        text = ""
        content = getattr(candidate, "content", None)
        if content and getattr(content, "parts", None):
            for part in content.parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    text += part_text

        if not text:
            fr = str(finish_reason).upper() if finish_reason else "UNKNOWN"
            raise RuntimeError(f"Empty response (finish_reason={fr})")

        text = strip_think_tags(text)
        usage = getattr(response, "usage_metadata", None)
        return text, usage

    def _process_sample(self, sample, cache_name, file_uri, service_tier=None):
        sample_id = sample.get("sample_id", "unknown")
        turn_tasks = self._build_turn_tasks(sample, cache_name, file_uri)

        if not turn_tasks:
            return True

        for conv_idx, contents in turn_tasks:
            try:
                text, usage = self._call_gemini(contents, cache_name, service_tier)
            except Exception as e:
                print(f"\n[SKIP] {sample_id} turn {conv_idx}: {type(e).__name__}: "
                      f"{str(e)[:150]}")
                return False

            sample["conversations"][conv_idx]["candidate_response"] = text

            if usage:
                prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
                self.tracker.add_cost(
                    self.model, self.mode, prompt_tokens, output_tokens,
                    cached_tokens, sample_id, conv_idx, self.pricing_data)

        return True

    PIPELINE_DEPTH = 10  # max videos uploaded/cached ahead of inference

    def _generate_online(self, samples, output_file, service_tier=None):
        """Streaming pipeline: upload/cache videos ahead, infer per-video, cleanup after."""
        video_base_path = self.config["paths"]["video_path"]
        if not os.path.isabs(video_base_path):
            video_base_path = os.path.abspath(video_base_path)
        video_path_map = build_video_path_map(video_base_path)

        # Group samples by video_id (samples are already sorted by video_id)
        from collections import OrderedDict
        video_groups = OrderedDict()
        for s in samples:
            vid = s.get("video_id", "")
            if not vid or not s.get("conversations"):
                continue
            video_groups.setdefault(vid, []).append(s)

        video_order = list(video_groups.keys())
        num_workers = self.args.num_workers
        tier_label = service_tier or "standard"

        print(f"\nStarting {tier_label} inference: {len(samples)} samples across "
              f"{len(video_order)} videos, {num_workers} workers, "
              f"pipeline depth {self.PIPELINE_DEPTH}")

        # --- Pipeline state ---
        video_ready = {}       # vid -> (cache_name, file_uri) or None on failure
        video_events = {vid: threading.Event() for vid in video_order}
        pipeline_slots = threading.Semaphore(self.PIPELINE_DEPTH)
        pipeline_error = [None]  # mutable container for thread error

        def _uploader():
            """Upload + cache videos ahead of inference, limited by pipeline_slots."""
            for vid in video_order:
                if pipeline_error[0]:
                    break
                pipeline_slots.acquire()
                try:
                    path = video_path_map.get(vid)
                    if not path or not os.path.exists(path):
                        print(f"\n[SKIP VIDEO] {vid}: file not found")
                        video_ready[vid] = None
                        video_events[vid].set()
                        continue

                    _, uri = self.file_manager.upload_video(vid, path)
                    cache_name = None
                    if not self.args.no_cache:
                        cache_name = self.file_manager.create_cache(
                            vid, self.model, uri, self.args.cache_ttl)
                    video_ready[vid] = (cache_name, uri)
                except Exception as e:
                    print(f"\n[PIPELINE ERROR] {vid}: {type(e).__name__}: "
                          f"{str(e)[:150]}")
                    video_ready[vid] = None
                    pipeline_error[0] = e
                finally:
                    video_events[vid].set()

        uploader_thread = threading.Thread(target=_uploader, daemon=True)
        uploader_thread.start()

        # --- Inference loop: process video groups as they become ready ---
        completed = 0
        skipped = 0
        counters_lock = threading.Lock()
        cost_print_interval = 50

        pbar = tqdm(total=len(samples), desc=f"Generating ({tier_label})")
        executor = ThreadPoolExecutor(max_workers=num_workers)

        try:
            for vid in video_order:
                video_events[vid].wait()

                entry = video_ready.get(vid)
                if entry is None:
                    group_size = len(video_groups[vid])
                    skipped += group_size
                    pbar.update(group_size)
                    pipeline_slots.release()
                    if pipeline_error[0]:
                        raise pipeline_error[0]
                    continue

                cache_name, file_uri = entry
                group = video_groups[vid]

                # Submit all samples for this video
                def _process_one(sample_orig, _cache=cache_name, _uri=file_uri):
                    sid = sample_orig.get("sample_id", "unknown")
                    sample = copy.deepcopy(sample_orig)
                    try:
                        success = self._process_sample(
                            sample, _cache, _uri, service_tier)
                    except Exception as e:
                        print(f"\n[ERROR] {sid}: {type(e).__name__}: "
                              f"{str(e)[:200]}")
                        return False

                    if success:
                        try:
                            with FileLock(f"{output_file}.lock"):
                                with open(output_file, "a") as f:
                                    f.write(json.dumps(
                                        sample, ensure_ascii=False) + "\n")
                        except Exception as e:
                            print(f"\n[ERROR] Write failed for {sid}: {e}")
                            return False
                    return success

                futures = [executor.submit(_process_one, s) for s in group]

                for future in as_completed(futures):
                    try:
                        success = future.result(
                            timeout=self.config["generation"].get("timeout", 1800))
                    except Exception as e:
                        print(f"\n[ERROR] Future failed: {e}")
                        success = False

                    with counters_lock:
                        if success:
                            completed += 1
                            if completed % cost_print_interval == 0:
                                summary = self.tracker.get_cost_summary()
                                print(
                                    f"\n[COST] {completed} samples | "
                                    f"${summary['estimated_cost_usd']:.4f} | "
                                    f"input: {summary['total_input_tokens']:,} | "
                                    f"output: {summary['total_output_tokens']:,} | "
                                    f"cached: {summary['total_cached_tokens']:,}")
                        else:
                            skipped += 1
                    pbar.update(1)

                # Video group done — delete file to free storage, release slot
                self.file_manager.delete_file(vid)
                pipeline_slots.release()

        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Cancelling pending tasks...")
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            executor.shutdown(wait=False)
            pbar.close()
            uploader_thread.join(timeout=5)

        return completed, skipped

    def _build_batch_requests(self, samples, cache_map, file_map):
        requests = []
        sample_index = {}

        for sample in samples:
            sid = sample.get("sample_id", "unknown")
            vid = sample.get("video_id")
            file_entry = file_map.get(vid)
            if not file_entry:
                print(f"[SKIP BATCH] {sid}: no uploaded file for video {vid}")
                continue

            _, file_uri = file_entry
            cache_name = cache_map.get(vid)
            turn_tasks = self._build_turn_tasks(sample, cache_name, file_uri)

            for conv_idx, contents in turn_tasks:
                custom_id = f"{sid}__turn_{conv_idx}"
                sample_index[custom_id] = (sample, conv_idx)

                config_kwargs = {
                    "temperature": self.config["generation"]["temperature"],
                    "max_output_tokens": self.config["generation"]["max_tokens"],
                    "top_p": self.config["generation"]["top_p"],
                }
                if cache_name:
                    config_kwargs["cached_content"] = cache_name

                system_prompt = self.config["system"].get("system_prompt", "")
                if system_prompt:
                    config_kwargs["system_instruction"] = system_prompt

                req = types.InlinedRequest(
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                    metadata={"custom_id": custom_id},
                )
                requests.append(req)

        return requests, sample_index

    @staticmethod
    def _normalize_batch_state(raw_state):
        """Normalize SDK batch state strings like 'JOB_STATE_SUCCEEDED' to 'SUCCEEDED'."""
        s = str(raw_state).upper()
        for name in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED", "RUNNING", "PENDING"):
            if name in s:
                return name
        return s

    def _recover_existing_batches(self, samples, output_file):
        """Check tracker for existing batches, retrieve completed results,
        and return (recovered_count, still_running_batch_ids).
        """
        with self.tracker._lock:
            all_batches = dict(self.tracker.data["batches"])

        recoverable = {
            bid: info for bid, info in all_batches.items()
            if info.get("status") not in ("RETRIEVED", "FAILED", "CANCELLED",
                                           "EXPIRED", "POLL_TIMEOUT")
        }

        if not recoverable:
            return 0, []

        sample_index = self._build_sample_index(samples)

        total_recovered = 0
        still_running = []
        print(f"Checking {len(recoverable)} existing batch(es) from previous runs...")

        for batch_id in recoverable:
            try:
                job = self.client.batches.get(name=batch_id)
            except Exception as e:
                print(f"  {batch_id}: failed to fetch ({e})")
                still_running.append(batch_id)
                continue

            state = self._normalize_batch_state(getattr(job, "state", "UNKNOWN"))
            self.tracker.update_batch(batch_id, status=state)

            if state == "SUCCEEDED":
                completed, skipped = self._retrieve_batch_results(
                    job, sample_index, output_file)
                total_recovered += completed
                self.tracker.update_batch(
                    batch_id, status="RETRIEVED",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    completed_requests=completed,
                    failed_requests=skipped)
                print(f"  {batch_id}: retrieved {completed} samples")
            elif state in ("FAILED", "CANCELLED", "EXPIRED"):
                print(f"  {batch_id}: {state}")
            else:
                print(f"  {batch_id}: {state} (will continue polling)")
                still_running.append(batch_id)

        return total_recovered, still_running

    def _generate_batch(self, samples, output_file):
        """Submit one batch per video with concurrent polling for results."""
        video_base_path = self.config["paths"]["video_path"]
        if not os.path.isabs(video_base_path):
            video_base_path = os.path.abspath(video_base_path)
        video_path_map = build_video_path_map(video_base_path)

        from collections import OrderedDict

        video_groups = OrderedDict()
        for s in samples:
            vid = s.get("video_id", "")
            if not vid or not s.get("conversations"):
                continue
            video_groups.setdefault(vid, []).append(s)

        video_order = list(video_groups.keys())

        print(f"\nBatch mode: {len(samples)} samples across {len(video_order)} videos "
              f"(1 batch per video, submit + poll in parallel)")

        # --- Phase 0: Recover results from existing batches ---
        recovered, still_running = self._recover_existing_batches(
            samples, output_file)
        if recovered > 0:
            print(f"Recovered {recovered} samples from previous batch runs.")

        # Shared state — seed with sample index for still-running batches from recovery
        all_sample_indices = dict(self._build_sample_index(samples))
        sample_indices_lock = threading.Lock()
        pending_batches = list(still_running)
        pending_lock = threading.Lock()
        submit_done = threading.Event()
        total_submitted = [0]
        total_completed = [0]
        total_skipped = [0]
        submit_error = [None]

        # --- Submit thread ---
        def _submit_worker():
            submit_pbar = tqdm(
                video_order, desc="Submitting", position=0, leave=True)
            try:
                for vid in submit_pbar:
                    if submit_error[0]:
                        break

                    group = video_groups[vid]
                    submit_pbar.set_postfix_str(
                        f"{vid} ({len(group)} samples)", refresh=True)

                    path = video_path_map.get(vid)
                    if not path or not os.path.exists(path):
                        print(f"\n[SKIP] Video not found: {vid}")
                        continue

                    try:
                        _, uri = self.file_manager.upload_video(vid, path)

                        cache_map = {}
                        if not self.args.no_cache:
                            cache_name = self.file_manager.create_cache(
                                vid, self.model, uri, self.args.cache_ttl)
                            cache_map[vid] = cache_name

                        file_map = {vid: (None, uri)}
                        requests, sample_index = self._build_batch_requests(
                            group, cache_map, file_map)

                        if not requests:
                            self.file_manager.delete_file(vid)
                            continue

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        display_name = (
                            f"longshot-{self.model_underscored}-{vid}-{timestamp}")

                        batch_job = self.client.batches.create(
                            model=self.model,
                            src=requests,
                            config={"display_name": display_name},
                        )
                        batch_id = batch_job.name

                        self.tracker.set_batch(
                            batch_id, "PENDING", self.model,
                            display_name, len(requests))
                        for req in requests:
                            self.tracker.set_request(
                                req.metadata["custom_id"], batch_id)
                        self.tracker.save()

                        with sample_indices_lock:
                            all_sample_indices.update(sample_index)
                        with pending_lock:
                            pending_batches.append(batch_id)
                        total_submitted[0] += 1

                        self.file_manager.delete_file(vid)

                    except Exception as e:
                        print(f"\n[SUBMIT ERROR] {vid}: {type(e).__name__}: "
                              f"{str(e)[:150]}")
                        submit_error[0] = e
                        break

            finally:
                submit_pbar.close()
                submit_done.set()

        # --- Poll thread ---
        def _poll_worker():
            poll_pbar = tqdm(
                desc="Polling", position=1, leave=True, unit="batch")
            first_check = True

            while True:
                if not first_check:
                    time.sleep(60)
                first_check = False

                with pending_lock:
                    to_check = list(pending_batches)

                if not to_check and submit_done.is_set():
                    break
                if not to_check:
                    continue

                poll_pbar.set_postfix_str(
                    f"checking {len(to_check)} batches...")

                still_pending = []
                for batch_id in to_check:
                    try:
                        job = self.client.batches.get(name=batch_id)
                    except Exception:
                        still_pending.append(batch_id)
                        continue

                    state = self._normalize_batch_state(
                        getattr(job, "state", "UNKNOWN"))
                    self.tracker.update_batch(batch_id, status=state)

                    if state == "SUCCEEDED":
                        with sample_indices_lock:
                            idx = dict(all_sample_indices)
                        completed, skipped = self._retrieve_batch_results(
                            job, idx, output_file)
                        total_completed[0] += completed
                        total_skipped[0] += skipped
                        self.tracker.update_batch(
                            batch_id, status="SUCCEEDED",
                            completed_at=datetime.now(timezone.utc).isoformat(),
                            completed_requests=completed,
                            failed_requests=skipped)
                        poll_pbar.update(1)
                    elif state in ("FAILED", "CANCELLED", "EXPIRED"):
                        print(f"\n[BATCH {state}] {batch_id}")
                        self.tracker.update_batch(
                            batch_id, status=state,
                            completed_at=datetime.now(timezone.utc).isoformat())
                        poll_pbar.update(1)
                    else:
                        still_pending.append(batch_id)

                with pending_lock:
                    pending_batches[:] = still_pending

                poll_pbar.set_postfix_str(
                    f"submitted={total_submitted[0]} "
                    f"done={total_completed[0]} "
                    f"waiting={len(still_pending)}")

                if not still_pending and submit_done.is_set():
                    break

            poll_pbar.close()

        # Start both threads
        submitter = threading.Thread(target=_submit_worker, daemon=True)
        poller = threading.Thread(target=_poll_worker, daemon=True)
        submitter.start()
        poller.start()

        try:
            submitter.join()
            poller.join()
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Batch IDs saved to tracking file.")
            print("Resume with: --check-batches or --resume-batch <ID>")
            submit_error[0] = KeyboardInterrupt()
            submit_done.set()
            raise

        if submit_error[0] and not isinstance(submit_error[0], KeyboardInterrupt):
            raise submit_error[0]

        return total_completed[0], total_skipped[0]

    def _retrieve_batch_results(self, job, sample_index, output_file):
        completed = 0
        skipped = 0

        # Load already-written sample IDs to prevent duplicates
        written_ids = set()
        if os.path.exists(output_file):
            sid_pattern = re.compile(r'"sample_id"\s*:\s*"([^"]+)"')
            with open(output_file, "rb") as f:
                for line in f:
                    m = sid_pattern.search(line.decode("utf-8", errors="ignore"))
                    if m:
                        written_ids.add(m.group(1))

        # Responses live in job.dest.inlined_responses, not job.inlined_responses
        dest = getattr(job, "dest", None)
        responses = getattr(dest, "inlined_responses", None) if dest else None
        if not responses:
            print(f"[WARNING] No inlined_responses found in batch job.")
            return 0, 0

        sample_turns = {}
        sample_objs = {}

        for resp in responses:
            try:
                metadata = getattr(resp, "metadata", None) or {}
                custom_id = metadata.get("custom_id")
                if not custom_id or custom_id not in sample_index:
                    continue

                sample_orig, conv_idx = sample_index[custom_id]
                sid = sample_orig.get("sample_id", "unknown")

                if sid not in sample_objs:
                    sample_objs[sid] = copy.deepcopy(sample_orig)
                    sample_turns[sid] = set()

                sample = sample_objs[sid]
                text = ""

                resp_obj = getattr(resp, "response", None)
                if resp_obj:
                    candidates = getattr(resp_obj, "candidates", None)
                    if candidates and len(candidates) > 0:
                        content = getattr(candidates[0], "content", None)
                        if content and getattr(content, "parts", None):
                            for part in content.parts:
                                part_text = getattr(part, "text", None)
                                if part_text:
                                    text += part_text

                text = strip_think_tags(text)
                if not text:
                    fr = ""
                    if resp_obj and getattr(resp_obj, "candidates", None):
                        fr = str(getattr(
                            resp_obj.candidates[0], "finish_reason", "")).upper()
                    print(f"\n[EMPTY] {custom_id} finish_reason={fr or 'UNKNOWN'}")
                    continue
                if conv_idx < len(sample.get("conversations", [])):
                    sample["conversations"][conv_idx]["candidate_response"] = text
                sample_turns[sid].add(conv_idx)

                usage = getattr(resp_obj, "usage_metadata", None) if resp_obj else None
                if usage:
                    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                    cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
                    self.tracker.add_cost(
                        self.model, "batch", prompt_tokens, output_tokens,
                        cached_tokens, sid, conv_idx, self.pricing_data)

            except Exception as e:
                cid = (getattr(resp, "metadata", None) or {}).get("custom_id", "?")
                print(f"\n[WARNING] Failed to parse batch response {cid}: {e}")

        for sid, sample in sample_objs.items():
            if sid in written_ids:
                continue  # already in output file

            expected_turns = {
                idx for idx, turn in enumerate(sample.get("conversations", []))
                if turn.get("role") == "assistant"
            }
            if expected_turns <= sample_turns.get(sid, set()):
                try:
                    with FileLock(f"{output_file}.lock"):
                        with open(output_file, "a") as f:
                            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    completed += 1
                except Exception as e:
                    print(f"\n[ERROR] Write failed for {sid}: {e}")
                    skipped += 1
            else:
                missing = expected_turns - sample_turns.get(sid, set())
                print(f"\n[WARNING] Incomplete turns for {sid}, missing: {missing}")
                skipped += 1

        return completed, skipped

    def _poll_and_retrieve_batch(self, batch_id, sample_index, output_file):
        """Poll a single batch until terminal state, then retrieve results. Max: 48h."""
        poll_delays = [60, 60, 120, 120, 300]
        attempt = 0
        max_poll_seconds = 48 * 3600
        poll_start = time.time()

        print(f"Polling batch {batch_id}...")
        while True:
            if time.time() - poll_start > max_poll_seconds:
                print(f"\n[TIMEOUT] Batch {batch_id} polling exceeded 48h limit.")
                self.tracker.update_batch(
                    batch_id, status="POLL_TIMEOUT",
                    completed_at=datetime.now(timezone.utc).isoformat())
                return 0, len(sample_index)
            try:
                job = self.client.batches.get(name=batch_id)
            except Exception as e:
                print(f"[WARNING] batches.get failed: {e}")
                time.sleep(60)
                continue

            state = self._normalize_batch_state(getattr(job, "state", "UNKNOWN"))

            self.tracker.update_batch(batch_id, status=state)

            if state == "SUCCEEDED":
                print(f"\nBatch {batch_id} completed.")
                completed, skipped = self._retrieve_batch_results(
                    job, sample_index, output_file)
                self.tracker.update_batch(
                    batch_id, status="SUCCEEDED",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    completed_requests=completed,
                    failed_requests=skipped)
                return completed, skipped
            elif state in ("FAILED", "CANCELLED", "EXPIRED"):
                print(f"\nBatch {batch_id} ended with state: {state}")
                self.tracker.update_batch(
                    batch_id, status=state,
                    completed_at=datetime.now(timezone.utc).isoformat())
                return 0, len(sample_index)

            elapsed = int(time.time() - poll_start)
            delay = poll_delays[min(attempt, len(poll_delays) - 1)]
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {batch_id}: "
                  f"{state} ({elapsed}s elapsed) — next check in {delay}s",
                  end="\r", flush=True)
            time.sleep(delay)
            attempt += 1

    def resume_batch(self, batch_id, output_file, samples):
        """Resume polling a previously submitted batch job."""
        sample_index = self._build_sample_index(samples)
        print(f"Resuming batch {batch_id} ({len(sample_index)} possible requests)")
        return self._poll_and_retrieve_batch(batch_id, sample_index, output_file)

    def check_pending_batches(self):
        pending = self.tracker.get_pending_batches()
        if not pending:
            print("No pending batch jobs.")
            return

        print(f"\n{'Batch ID':<45} {'Status':<15} {'Created':<22} {'Requests':>8}")
        print("-" * 95)
        for bid, info in pending.items():
            try:
                job = self.client.batches.get(name=bid)
                state = str(getattr(job, "state", "UNKNOWN"))
                self.tracker.update_batch(bid, status=state)
            except Exception as e:
                state = f"ERROR: {e}"
            print(f"  {bid:<43} {state:<15} "
                  f"{info.get('created_at', '?'):<22} "
                  f"{info.get('total_requests', '?'):>8}")

    def generate(self):
        """Run the full generation pipeline."""
        test_samples = self._load_samples()
        if not test_samples:
            return

        test_samples = sorted(test_samples, key=lambda x: x.get("video_id", ""))

        # Load already-completed sample IDs
        completed_ids = set()
        if os.path.exists(self.output_file):
            sid_pattern = re.compile(r'"sample_id"\s*:\s*"([^"]+)"')
            with open(self.output_file, "rb") as f:
                for line in f:
                    m = sid_pattern.search(line.decode("utf-8", errors="ignore"))
                    if m:
                        completed_ids.add(m.group(1))

        pending = [s for s in test_samples if s.get("sample_id") not in completed_ids]

        if self.args.limit and self.args.limit > 0:
            pending = pending[:self.args.limit]

        print(f"\nModel: {self.model} | Mode: {self.mode}")
        print(f"Total samples: {len(test_samples)} | "
              f"Already done: {len(completed_ids)} | "
              f"Pending: {len(pending)}"
              + (f" (limited to {self.args.limit})" if self.args.limit else ""))

        if not pending:
            print("All samples already completed.")
            return

        generation_start = time.time()
        interrupted = False
        completed = 0
        skipped = 0

        try:
            if self.mode == "batch":
                completed, skipped = self._generate_batch(pending, self.output_file)
            elif self.mode == "flex":
                completed, skipped = self._generate_online(
                    pending, self.output_file, service_tier="flex")
            elif self.mode == "standard":
                completed, skipped = self._generate_online(
                    pending, self.output_file, service_tier=None)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        except KeyboardInterrupt:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            print("\n[INTERRUPTED] Saving partial state...")
            interrupted = True
        finally:
            timing_data = {
                "total": time.time() - generation_start,
                "inference": time.time() - generation_start,
                "samples_processed": completed,
                "samples_skipped": skipped + len(completed_ids),
                "samples_total": len(test_samples),
                "interrupted": interrupted,
                "mode": self.mode,
                "model": self.model,
            }
            save_timing_data(self.timing_file, "generation", timing_data)

            cost_summary = self.tracker.get_cost_summary()
            print(f"\n{'='*60}")
            print(f"  Generation Summary")
            print(f"{'='*60}")
            print(f"  Model:     {self.model}")
            print(f"  Mode:      {self.mode}")
            print(f"  Completed: {completed}")
            print(f"  Skipped:   {skipped}")
            print(f"  Duration:  {time.time() - generation_start:.1f}s")
            print(f"  {'─'*56}")
            print(f"  Cost Estimate:")
            print(f"    Input tokens:   {cost_summary['total_input_tokens']:>12,}")
            print(f"    Output tokens:  {cost_summary['total_output_tokens']:>12,}")
            print(f"    Cached tokens:  {cost_summary['total_cached_tokens']:>12,}")
            print(f"    Est. cost:      ${cost_summary['estimated_cost_usd']:>11.4f}")
            print(f"{'='*60}")

            if interrupted:
                print(f"[INTERRUPTED] State saved. Re-run to resume.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Gemini API benchmarking for LongShOT evaluation")
    parser.add_argument("--model", default=MODEL,
                        help=f"Gemini model name (default: {MODEL})")
    parser.add_argument("--mode", default="flex",
                        choices=["flex", "batch", "standard"],
                        help="Inference mode (default: flex)")
    parser.add_argument("--tasks", "-t", nargs="+",
                        help="Task names from tasks.yaml (required for generation)")
    parser.add_argument("--num-workers", "-n", type=int, default=8,
                        help="Concurrent API calls for flex/standard mode")
    parser.add_argument("--output-dir", "-o", default="results_postvalid",
                        help="Output directory")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="System config path")
    parser.add_argument("--task-config", default="tasks.yaml",
                        help="Task definitions YAML")
    parser.add_argument("--cache-ttl", type=int, default=3600,
                        help="Context cache TTL in seconds (default: 3600)")
    parser.add_argument("--resume-batch",
                        help="Resume polling a specific batch job ID")
    parser.add_argument("--check-batches", action="store_true",
                        help="Check status of all pending batch jobs")
    parser.add_argument("--alias", default=None,
                        help="Override output directory name")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable context caching")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of samples to process (0=unlimited)")
    parser.add_argument("--tracking-file", default=None,
                        help="Path to gemini_tracking.json")
    parser.add_argument("--list-files", action="store_true",
                        help="List all files on Gemini with storage usage")
    parser.add_argument("--cleanup-files", action="store_true",
                        help="Delete all tracked files from Gemini storage")

    args = parser.parse_args()
    config = load_config(args.config)

    bench = GeminiBenchmark(args.model, args.mode, config, args)

    if args.list_files:
        bench.file_manager.list_files()
        return

    if args.cleanup_files:
        bench.file_manager.delete_all_files()
        return

    if args.check_batches:
        bench.check_pending_batches()
        return

    if args.resume_batch:
        if not args.tasks:
            parser.error("--tasks required with --resume-batch")
        test_samples = bench._load_samples()
        if not test_samples:
            print("No samples loaded. Check --tasks.")
            return
        completed, skipped = bench.resume_batch(
            args.resume_batch, bench.output_file, test_samples)
        print(f"Resume complete: {completed} completed, {skipped} skipped")
        return

    if not args.tasks:
        parser.error("--tasks is required for generation")

    try:
        bench.generate()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Exiting.")
    except Exception as e:
        print(f"\n[FATAL] {type(e).__name__}: {e}")
        traceback.print_exc()
        # Ensure tracker state is saved even on fatal errors
        try:
            bench.tracker.save()
        except Exception:
            pass
        raise SystemExit(1)


if __name__ == "__main__":
    main()
