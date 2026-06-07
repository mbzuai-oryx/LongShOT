import json
from datasets import load_dataset
from tqdm import tqdm
import requests
import re
from openai import OpenAI
import base64
import io
from filelock import FileLock
import subprocess
import sys
import time
import psutil
import os
from PIL import Image
from datetime import datetime
import yaml
import httpx

def create_openai_client(config):
    """Create an OpenAI client with connection pool sized for high-concurrency workloads."""
    return OpenAI(
        api_key=config['server']['api_key'],
        base_url=config['server']['base_url'],
        http_client=httpx.Client(
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=500),
        ),
    )


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def create_logs_directory(config=None):
    """Create logs directory if it doesn't exist"""
    if config is None:
        config = load_config()
    logs_dir = config['paths']['logs_dir']
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    return logs_dir

def load_jsonl(file_path):
    with open(file_path, 'rb') as f:
        raw = f.read()
    return [json.loads(line) for line in raw.split(b'\n') if line.strip()]

def load_dataset_with_params(params, task_name, config=None):
    """Load a video benchmark task dataset using stored parameters.

    Caches to a local JSONL file to avoid re-downloading from HF Hub on every run.
    """
    if config is None:
        config = load_config()

    path = params["path"]
    name = params.get("name", "default")
    split = params.get("split", "test")

    # Cache directory for downloaded datasets
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dataset_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{task_name}_{split}.jsonl")

    if os.path.exists(cache_file):
        print(f"Loading {task_name} from local cache: {cache_file}")
        dataset_list = load_jsonl(cache_file)
    else:
        print(f"Downloading {task_name} from HF Hub (first time only)...")
        if name == "default":
            ds = load_dataset(path)[split]
        else:
            ds = load_dataset(path, name=name)[split]

        dataset_list = list(tqdm(ds))

        # Save to local cache
        with open(cache_file, 'w') as f:
            for item in dataset_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Cached {len(dataset_list)} samples to {cache_file}")

    # Apply dataset limit from config (0 or None means no limit)
    dataset_limit = config['system'].get('dataset_limit', 0)
    if dataset_limit and dataset_limit > 0:
        return dataset_list[:dataset_limit]

    return dataset_list

_video_path_map_cache = {}

def build_video_path_map(video_base_path):
    """Build video_id -> path map with in-memory caching. Only walks the directory once per path."""
    if video_base_path in _video_path_map_cache:
        return _video_path_map_cache[video_base_path]

    video_path_map = {}
    for root, _, files in os.walk(video_base_path):
        for f in files:
            if f.endswith('.mp4'):
                video_path_map[f[:-4]] = os.path.join(root, f)

    _video_path_map_cache[video_base_path] = video_path_map
    return video_path_map


def get_model_name(config=None):
    """Get the model name from the API"""
    if config is None:
        config = load_config()
    
    url = f"{config['server']['base_url']}models"
    response = requests.get(url)
    response.raise_for_status()
    models_data = response.json()
    
    # Return first model ID if available, otherwise default
    if models_data.get("data"):
        return models_data["data"][0]["id"]
    return "unknown_model"

def convert_to_underscored(model_name):
    """Convert model name to underscored format by replacing non-alphanumeric chars with underscore"""
    return re.sub(r'[^a-zA-Z0-9]', '_', (lambda split: split[-2] if split[-1].strip() == "" else split[-1])(model_name.split('/')))


def normalize_eval_tag(eval_tag=None, eval_model=None):
    """Return a filesystem-safe judge tag."""
    raw_tag = eval_tag if eval_tag else (convert_to_underscored(eval_model) if eval_model else "default")
    normalized = re.sub(r'[^a-zA-Z0-9]+', '_', raw_tag).strip('_').lower()
    return normalized or "default"


def get_model_result_dir(output_dir, model_name_underscored):
    """Return the model's root results directory."""
    return os.path.join(output_dir, model_name_underscored)


def get_generation_artifact_paths(output_dir, model_name_underscored):
    """Return generation artifact paths stored at the model root."""
    model_dir = get_model_result_dir(output_dir, model_name_underscored)
    stem = os.path.join(model_dir, model_name_underscored)
    return {
        "model_dir": model_dir,
        "output_file": f"{stem}.jsonl",
        "timing_file": f"{stem}_timing.json",
    }


def get_judge_artifact_paths(output_dir, model_name_underscored, eval_model=None, eval_tag=None):
    """Return judge-specific artifact paths under model_dir/judges/<tag>/."""
    normalized_tag = normalize_eval_tag(eval_tag, eval_model)
    model_dir = get_model_result_dir(output_dir, model_name_underscored)
    judge_dir = os.path.join(model_dir, "judges", normalized_tag)
    stem = os.path.join(judge_dir, model_name_underscored)
    return {
        "eval_tag": normalized_tag,
        "judge_dir": judge_dir,
        "eval_file": f"{stem}_eval.jsonl",
        "score_file": f"{stem}_score.txt",
        "score_json_file": f"{stem}_score.json",
        "timing_file": f"{stem}_timing.json",
    }

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_ORPHAN_CLOSE_THINK_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)

def strip_think_tags(text):
    """Remove <think>...</think> blocks and orphan think tags from model output."""
    if not text:
        return text
    # Strip paired <think>...</think> blocks
    text = _THINK_RE.sub("", text)
    # Strip orphan </think> where opening <think> is missing or appears after </think>
    while "</think>" in text:
        open_idx = text.find("<think>")
        close_idx = text.find("</think>")
        if open_idx == -1 or open_idx > close_idx:
            # Orphan </think> — strip everything before and including it
            text = text[close_idx + len("</think>"):].lstrip()
        else:
            break
    # Clean up any remaining paired tags created by orphan removal
    text = _THINK_RE.sub("", text)
    return text.strip()

def is_video_agent_model(model_name):
    """Check if the model is a video-agent model"""
    return "video-agent" in model_name.lower()


def get_agent_name(model_name):
    """Extract agent name from model identifier (e.g., 'video-agent-video_explorer' -> 'video_explorer')."""
    if not is_video_agent_model(model_name):
        return None
    # Handle formats like 'video-agent-video_explorer' or 'video-agent:video_explorer'
    lower = model_name.lower()
    for sep in ["-video-agent-", "video-agent-", "video-agent:"]:
        if sep in lower:
            idx = lower.find(sep) + len(sep)
            return model_name[idx:].strip()
    return None


def extract_question_for_agent(sample):
    """Extract the user question from a multi-turn conversation sample."""
    for turn in sample.get("conversations", []):
        if turn.get("role") == "user":
            return turn.get("content", "")
    raise ValueError(f"No user turn found in sample {sample.get('sample_id')}")


def inject_agent_response(sample, response_text, reasoning_steps=None):
    """Inject an agent's response into the sample's assistant turn."""
    for turn in sample.get("conversations", []):
        if turn.get("role") == "assistant":
            turn["candidate_response"] = response_text
            if reasoning_steps:
                turn["agent_reasoning"] = reasoning_steps
            break
    return sample


def get_audio_path(video_path):
    """Return the pre-extracted audio path for a video (same dir, .wav extension).

    Audio files should be pre-extracted using extract_audio.sh.
    Returns the path if it exists, None otherwise.
    """
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    return audio_path if os.path.exists(audio_path) else None


def is_api_model(backend):
    """Check if the backend is a cloud API (no local server needed)."""
    return backend in ("api",)


def is_voxtral_model(model_name):
    return "voxtral" in (model_name or "").lower()


# Voxtral-Small-24B-2507 supports up to 30 minutes of audio context.
VOXTRAL_MAX_AUDIO_SEC = 30 * 60


def _atempo_chain(factor):
    """Decompose a speed factor into a chain of ffmpeg atempo filters (each in [0.5, 2.0])."""
    parts = []
    f = float(factor)
    while f > 2.0:
        parts.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        parts.append("atempo=0.5")
        f /= 0.5
    parts.append(f"atempo={f:.6f}")
    return ",".join(parts)


def get_audio_path_for_model(video_path, model_name):
    """Return audio path tailored to the model.

    For Voxtral, audios longer than 30 minutes are sped up via ffmpeg's
    atempo filter so the output is exactly 30 minutes and fits in context.
    The sped-up file is cached next to the original as `<stem>.vx30.wav`.
    """
    audio_path = get_audio_path(video_path)
    if not audio_path or not is_voxtral_model(model_name):
        return audio_path

    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            text=True,
        ).strip()
        duration = float(out)
    except Exception as e:
        print(f"[WARNING] ffprobe failed for {audio_path}: {e}")
        return audio_path

    if duration <= VOXTRAL_MAX_AUDIO_SEC:
        return audio_path

    stem, ext = os.path.splitext(audio_path)
    cached = f"{stem}.vx30{ext}"
    factor = duration / VOXTRAL_MAX_AUDIO_SEC

    with FileLock(f"{cached}.lock"):
        if os.path.exists(cached):
            return cached
        afilter = _atempo_chain(factor)
        tmp = f"{cached}.tmp{ext}"
        print(f"[Voxtral] Speeding up {os.path.basename(audio_path)} "
              f"({duration:.0f}s → {VOXTRAL_MAX_AUDIO_SEC}s, factor={factor:.3f})")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", audio_path,
                 "-filter:a", afilter, "-ac", "1", tmp],
                check=True,
            )
            os.replace(tmp, cached)
        except Exception as e:
            print(f"[ERROR] ffmpeg speedup failed for {audio_path}: {e}")
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            return audio_path

    return cached

def generate_candidate_response(sample, model_name, config=None, client=None, video_path_map=None, omni=False, max_frames=0, audio_only=False, backend="vllm"):
    """Generate model response for a video benchmark sample.

    Thin wrapper around build_turn_tasks + execute_turn.  Used only by the
    single-worker sequential path; the parallel path dispatches turns
    directly via the thread pool.
    """
    if config is None:
        config = load_config()
    if client is None:
        client = create_openai_client(config)

    if backend == "api":
        from openrouter_client import execute_turn_openrouter
        _exec = execute_turn_openrouter
    else:
        _exec = execute_turn

    conversations = sample.get('conversations', sample.get('question', []))
    turn_tasks = build_turn_tasks(sample, model_name, config, video_path_map, omni=omni, max_frames=max_frames, audio_only=audio_only)

    for conv_idx, msgs, extra_body in turn_tasks:
        conversations[conv_idx]['candidate_response'] = _exec(
            client, model_name, msgs, extra_body, config)

    return conversations


_frame_content_cache = {}  # (video_path, max_frames) -> list of image_url dicts


def _get_frame_content(video_path, max_frames):
    """Return cached list of image_url content dicts for a video's pre-extracted frames.

    Cached per (video_path, max_frames) so repeated samples from the same video
    (typically ~10 samples/video) reuse the same list without re-reading the
    directory or rebuilding dicts.
    """
    key = (video_path, max_frames)
    if key in _frame_content_cache:
        return _frame_content_cache[key]

    vdir = os.path.dirname(video_path)
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(vdir, "frames", video_id, f"f{max_frames}")

    if not os.path.isdir(frames_dir):
        _frame_content_cache[key] = []
        return []

    # os.listdir + sort is faster than glob for large directories
    # Filenames are zero-padded (000000.jpg) so lexicographic sort = numeric sort
    jpgs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))

    content = [
        {"type": "image_url", "image_url": {"url": f"file://{os.path.join(frames_dir, f)}"}}
        for f in jpgs
    ]
    _frame_content_cache[key] = content
    return content


def build_turn_tasks(sample, model_name, config=None, video_path_map=None, omni=False, max_frames=0, audio_only=False):
    """Pre-build API call inputs for every assistant turn in a sample.

    Since ground truth (not model output) is used as context for subsequent
    turns, each turn's messages can be fully constructed upfront.

    Args:
        omni: When True, enable audio extraction from video files.
              Qwen Omni  → use_audio_in_video via mm_processor_kwargs
              MiniCPM-o  → extract audio track, pass as separate audio_url
        max_frames: When > 0, use pre-extracted frames as image_url list
                    instead of video_url. Frames at:
                    {video_dir}/frames/{video_id}/f{max_frames}/*.jpg
        audio_only: When True, send only audio (no video/frames) to the model.
                    Uses pre-extracted .wav files alongside the video files.

    Returns:
        List of (conv_index, messages, extra_body) tuples, one per assistant
        turn.  Empty list when there are no assistant turns.
    """
    if config is None:
        config = load_config()

    video_id = sample.get("video_id")
    is_video_agent = is_video_agent_model(model_name)

    video_path = None
    if not is_video_agent:
        if video_id:
            if video_path_map is not None:
                video_path = video_path_map.get(video_id)
            else:
                video_base_path = config['paths']['video_path']
                if not os.path.isabs(video_base_path):
                    video_base_path = os.path.abspath(video_base_path)
                for root, _, files in os.walk(video_base_path):
                    if f"{video_id}.mp4" in files:
                        video_path = os.path.join(root, f"{video_id}.mp4")
                        break
    else:
        if not video_id:
            raise ValueError("Video ID is required for video-agent models")

    conversations = sample.get('conversations', sample.get('question', []))
    if not isinstance(conversations, list):
        return []

    # System message
    if is_video_agent:
        base_messages = []
        if config['system']['system_prompt']:
            base_messages.append({"role": "system", "content": config['system']['system_prompt']})
    else:
        base_messages = ([{"role": "system", "content": [{"type": "text", "text": config['system']['system_prompt']}]}]
                         if config['system']['system_prompt'] else [])

    turn_tasks = []
    messages = list(base_messages)

    for i, turn in enumerate(conversations):
        if turn['role'] == 'user':
            if is_video_agent:
                messages.append({"role": "user", "content": turn['content']})
            else:
                if i == 0:
                    if not video_path or not os.path.exists(video_path):
                        raise FileNotFoundError(f"Video file not found: {video_path}")

                    if audio_only:
                        # Audio-only mode: send just the audio track, no video/frames
                        audio_path = get_audio_path_for_model(video_path, model_name)
                        if not audio_path:
                            raise FileNotFoundError(
                                f"No pre-extracted audio for {video_id}. "
                                f"Run: bash scripts/extract_audio.sh")
                        content = [
                            {"type": "audio_url", "audio_url": {"url": f"file://{audio_path}"}},
                        ]
                    elif max_frames > 0:
                        # Use pre-extracted frames as image list (cached per video)
                        frame_content = _get_frame_content(video_path, max_frames)
                        if not frame_content:
                            raise FileNotFoundError(
                                f"No pre-extracted frames at f{max_frames} for {video_id}. "
                                f"Run: python scripts/extract_frames.py --counts {max_frames}")
                        content = list(frame_content)
                    else:
                        # Native video input
                        content = [
                            {"type": "video_url", "video_url": {"url": f"file://{video_path}"}},
                        ]
                    if omni:
                        audio_path = get_audio_path_for_model(video_path, model_name)
                        if audio_path:
                            content.append({"type": "audio_url", "audio_url": {"url": f"file://{audio_path}"}})
                    content.append({"type": "text", "text": turn['content']})
                else:
                    content = [{"type": "text", "text": turn['content']}]
                messages.append({"role": "user", "content": content})
        elif turn['role'] == 'assistant':
            turn_tasks.append((i, list(messages)))
            if is_video_agent:
                messages.append({"role": "assistant", "content": turn['content']})
            else:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": turn['content']}]})

    # Build extra_body
    if is_video_agent:
        extra_body = {"video_id": video_id, "stream": False}
    else:
        extra_body = None

    return [(idx, msgs, extra_body) for idx, msgs in turn_tasks]


def execute_turn(client, model_name, messages, extra_body, config):
    """Execute a single LLM API call and return the stripped response text."""
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=config['generation']['max_tokens'],
        temperature=config['generation']['temperature'],
        top_p=config['generation']['top_p'],
        timeout=config['generation']['timeout'],
        extra_body=extra_body,
    )
    return strip_think_tags(completion.choices[0].message.content)


def _evaluate_single_criterion(criterion, ground_truth_response, model_response, eval_model, config, sample_id, client):
    """Evaluate a single criterion against ground truth. Used for concurrent criterion evaluation."""
    json_schema = {
        "type": "object",
        "properties": {
            "criteria_met": {"type": "boolean"}
        },
        "required": ["criteria_met"]
    }

    prompt = f"""You are an expert evaluator specializing in video content analysis and multimodal understanding.

Your task is to evaluate the Model Response against the **single evaluation criterion** provided, using the Ground Truth Response as a reference.

Ground Truth Response:
{ground_truth_response}

Model Response:
{model_response}

Evaluation Criterion:
{criterion['description']}

Instructions:
- Evaluate ONLY the provided criterion in this assessment.
- Compare the Model Response to the Ground Truth Response and determine if the criterion is satisfied.
- If the Model Response satisfies the criterion, set "criteria_met" to true; otherwise, set it to false.
- Focus on video content understanding, temporal relationships, and multimodal analysis. /no_think"""

    max_retries = 3
    retry_delay = 0.5

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=eval_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config['evaluation']['temperature'],
                max_tokens=config['evaluation']['max_tokens'],
                top_p=config['evaluation']['top_p'],
                timeout=config['evaluation']['timeout'],
                extra_body={
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "criteria_evaluation",
                            "schema": json_schema
                        }
                    }
                },
            )

            eval_content = completion.choices[0].message.content
            if eval_content is None:
                finish_reason = completion.choices[0].finish_reason
                raise ValueError(f"API returned None content. Finish reason: {finish_reason}")

            parsed_result = json.loads(eval_content)
            criterion['criteria_met'] = parsed_result['criteria_met']
            return

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[WARNING] Retry {attempt + 1}/{max_retries - 1} for sample {sample_id}: {str(e)[:100]}")
                time.sleep(retry_delay)
            else:
                print(f"[ERROR] Failed after {max_retries} attempts for sample {sample_id}, criterion '{criterion.get('name')}': {e}")
                criterion['criteria_met'] = None
                criterion['evaluation_error'] = str(e)


def evaluate_candidate_response(sample, eval_model="Qwen/Qwen3-14B", config=None, eval_file_path="", client=None):
    """Evaluate a video benchmark response against ground truth criteria"""
    if config is None:
        config = load_config()

    if 'conversations' not in sample:
        return sample

    # Create client if not provided
    if client is None:
        client = create_openai_client(config)

    # Skip samples where no candidate response was generated
    has_response = any(
        t.get('role') == 'assistant' and t.get('candidate_response')
        for t in sample['conversations']
    )
    if not has_response:
        return sample

    # Collect all (criterion, ground_truth, model_response) tuples for concurrent eval
    criterion_tasks = []
    for turn in sample['conversations']:
        if turn['role'] != 'assistant':
            continue
        ground_truth_response = turn['content']
        model_response = turn.get('candidate_response', '')
        for criterion in turn['criteria']:
            criterion_tasks.append((criterion, ground_truth_response, model_response))

    sample_id = sample.get('sample_id')

    if len(criterion_tasks) <= 1:
        # Single criterion — no threading overhead needed
        for criterion, gt, mr in criterion_tasks:
            _evaluate_single_criterion(criterion, gt, mr, eval_model, config, sample_id, client)
    else:
        # Bounded concurrency: up to 8 criteria in parallel per sample
        max_concurrent = min(8, len(criterion_tasks))
        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=max_concurrent) as crit_executor:
            futures = [
                crit_executor.submit(_evaluate_single_criterion, crit, gt, mr, eval_model, config, sample_id, client)
                for crit, gt, mr in criterion_tasks
            ]
            for f in futures:
                f.result()

    return sample


def process_sample(sample, output_file, model_name, config=None, client=None, video_path_map=None, omni=False, max_frames=0, audio_only=False, backend="vllm"):
    """Process a single sample."""
    response = generate_candidate_response(sample, model_name, config, client=client, video_path_map=video_path_map, omni=omni, max_frames=max_frames, audio_only=audio_only, backend=backend)

    sample['conversations'] = response

    has_empty = any(
        t.get('role') == 'assistant' and not t.get('candidate_response')
        for t in (response if isinstance(response, list) else [])
    )
    if has_empty:
        print(f"[SKIP] {sample.get('sample_id')}: empty response, will retry next run")
        return sample

    lock_file = f"{output_file}.lock"
    with FileLock(lock_file):
        with open(output_file, 'a') as f:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    return sample


def process_video_evaluation(sample, eval_file, eval_model, config=None, client=None):
    """Process a single video benchmark evaluation and save result to the evaluation file"""
    evaluated_sample = evaluate_candidate_response(sample, eval_model, config, eval_file, client=client)

    lock_file = f"{eval_file}.lock"
    with FileLock(lock_file):
        with open(eval_file, 'a') as f:
            f.write(json.dumps(evaluated_sample, ensure_ascii=False) + '\n')
    
    return evaluated_sample

def is_moe_model(model_name):
    """Detect if a model uses Mixture-of-Experts architecture by inspecting its config.

    Searches recursively through nested configs (e.g. thinker_config.text_config
    in Qwen3-Omni, text_config in standard VL models, etc.)
    """
    try:
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config_dict = model_config.to_dict()
        moe_keys = {"num_local_experts", "num_experts", "n_routed_experts", "num_experts_per_tok", "moe_num_experts"}

        def _search(d, path=""):
            """Recursively search dict for MoE indicators."""
            if not isinstance(d, dict):
                return False
            for key in moe_keys:
                val = d.get(key)
                if val is None:
                    continue
                # Handle list values like [64, 64] (e.g. ERNIE MoE)
                if isinstance(val, (list, tuple)):
                    val = max(val) if val else 0
                if int(val) > 1:
                    location = f"{path}.{key}" if path else key
                    print(f"Detected MoE model: {model_name} ({location}={val})")
                    return True
            # Recurse into nested config dicts (text_config, thinker_config, etc.)
            # Skip vision_config — vision encoder MoE is not handled by
            # vLLM's --enable-expert-parallel which targets the LLM backbone.
            skip_configs = {"vision_config"}
            for k, v in d.items():
                if isinstance(v, dict) and k.endswith("_config") and k not in skip_configs:
                    if _search(v, path=f"{path}.{k}" if path else k):
                        return True
            return False

        return _search(config_dict)
    except Exception as e:
        print(f"Warning: Could not detect MoE status for {model_name}: {e}")
    return False


def start_vllm(model_name, tensor_parallel_size, config_type="candidate", config=None, omni=False, max_frames=0, audio_only=False, served_model_name=None):
    """Start the VLLM server for video benchmark evaluation with the specified model"""
    if config is None:
        config = load_config()
    
    # Get vLLM configuration based on type (candidate or evaluation)
    vllm_config_key = f"vllm_{config_type}"
    if vllm_config_key not in config:
        vllm_config_key = "vllm_candidate"  # fallback to candidate config
    
    vllm_config = config[vllm_config_key]
    
    # Setup logging
    logs_dir = create_logs_directory(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = convert_to_underscored(model_name)
    log_file_path = os.path.join(logs_dir, f"vllm_{config_type}_{model_name_safe}_{timestamp}.log")

    # Build command
    cmd = [
        "vllm", "serve", model_name,
        "--tensor-parallel-size", str(tensor_parallel_size),
    ]
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])
    
    # Determine max-model-len: use config if specified, otherwise read from model
    max_model_len = vllm_config['max_model_len']
    if not max_model_len:
        try:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config_dict = model_config.to_dict()
            # Check nested configs for multimodal models (text_config, thinker_config, etc.)
            def _find_max_pos(d):
                val = d.get("max_position_embeddings")
                if val:
                    return int(val)
                for k, v in d.items():
                    if isinstance(v, dict) and k.endswith("_config"):
                        found = _find_max_pos(v)
                        if found:
                            return found
                return None
            max_model_len = _find_max_pos(config_dict)
            if max_model_len:
                print(f"Auto-detected max_model_len={max_model_len} from model config")
        except Exception as e:
            print(f"Warning: Could not auto-detect max_model_len: {e}")

    # Audio-only models: don't force larger context, let vLLM use model defaults
    if audio_only and max_model_len:
        print(f"Audio-only mode: skipping auto-detected max_model_len={max_model_len}, letting vLLM use model defaults")
        max_model_len = None

    # Omni mode: video+audio tokens can exceed default context length
    if omni and (not max_model_len or int(max_model_len) < 131072):
        print(f"Omni mode: setting max_model_len → 131072 (audio+video)")
        max_model_len = 131072
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    if "Phi-4-multimodal" in model_name:
        max_model_len = 65536
        print(f"Phi-4-multimodal: capping max_model_len → {max_model_len} (vision encoder OOM at 131K)")

    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])
    
    cmd.extend([
        "--max-num-seqs", str(vllm_config['max_num_seqs']), 
        "--port", config['server']['port'],
    ])
    
    # Add optional flags.
    # Skip --trust-remote-code when the model ships custom config/modeling files
    # for an architecture that vLLM already supports natively — remote code causes
    # config class mismatches (e.g. two different Qwen3VLConfig classes).
    skip_remote_code = False
    if vllm_config.get('trust_remote_code', True):
        try:
            from transformers import AutoConfig as _AC
            native_cfg = _AC.from_pretrained(model_name, trust_remote_code=False)
            remote_cfg = _AC.from_pretrained(model_name, trust_remote_code=True)
            if type(native_cfg).__name__ == type(remote_cfg).__name__ and type(native_cfg) is not type(remote_cfg):
                skip_remote_code = True
                print(f"Skipping --trust-remote-code: native {type(native_cfg).__name__} available, remote code would shadow it")
        except Exception:
            pass

    if vllm_config.get('trust_remote_code', True) and not skip_remote_code:
        cmd.append("--trust-remote-code")
    
    if 'allowed_local_media_path' in vllm_config:
        cmd.extend(["--allowed-local-media-path", vllm_config['allowed_local_media_path']])
    
    if vllm_config.get('enable_prefix_caching', True):
        cmd.append("--enable-prefix-caching")

    # Audio-only mode: send only audio, no video
    if audio_only:
        cmd.extend(["--limit-mm-per-prompt", '{"audio": 1}'])
        print(f"Audio-only mode: added --limit-mm-per-prompt audio=1")
    elif omni and max_frames > 0:
        # Frames + audio (e.g. gemma-3n with :f128:omni)
        cmd.extend(["--limit-mm-per-prompt", f'{{"image": {max_frames}, "audio": 1}}'])
        print(f"Omni+frame mode: added --limit-mm-per-prompt image={max_frames},audio=1")
    elif omni:
        cmd.extend(["--limit-mm-per-prompt", '{"video": 1, "audio": 1}'])
        print(f"Omni mode: added --limit-mm-per-prompt video=1,audio=1")
    elif max_frames == 0:
        # Default: 1 video per prompt — limits profiling memory
        cmd.extend(["--limit-mm-per-prompt", '{"video": 1}'])

    # Frame mode: set image limit (non-omni only; omni+frames handled above)
    if max_frames > 0 and not omni:
        cmd.extend(["--limit-mm-per-prompt", f'{{"image": {max_frames}}}'])

    # Frame mode: extend image fetch timeout for many frames
    if max_frames > 0:
        os.environ["VLLM_IMAGE_FETCH_TIMEOUT"] = "60"

    # Add additional arguments with dynamic adjustments
    if 'additional_args' in vllm_config and vllm_config['additional_args']:
        additional = list(vllm_config['additional_args'])

        # Compute available GPUs and set data-parallel-size dynamically
        tp = int(tensor_parallel_size)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        num_gpus = len(visible.split(",")) if visible else 8
        # DP must divide evenly: tp * dp == num_gpus
        dp_size = num_gpus // tp if num_gpus % tp == 0 else 1

        # Replace --data-parallel-size with computed value
        if "--data-parallel-size" in additional:
            idx = additional.index("--data-parallel-size")
            additional[idx + 1] = str(dp_size)
        elif dp_size > 1:
            additional.extend(["--data-parallel-size", str(dp_size)])

        # MiniCPM-o uses idefics2 vision encoder which:
        # 1. Breaks with --mm-encoder-tp-mode data (skips qkv stacking → KeyError)
        # 2. Has a heavy encoder cache; 65536 batched tokens consumes ~53 GiB/GPU,
        #    leaving 0 KV blocks. 16384 uses ~13 GiB encoder cache, leaves ~68 GiB
        #    for KV, and allows 4-8 video requests batched simultaneously.
        if "minicpm" in model_name.lower():
            if "--mm-encoder-tp-mode" in additional:
                idx = additional.index("--mm-encoder-tp-mode")
                removed_mode = additional[idx + 1]
                del additional[idx:idx + 2]
                print(f"Skipped --mm-encoder-tp-mode {removed_mode} (unsupported by MiniCPM vision encoder)")
            if "--max-num-batched-tokens" in additional:
                idx = additional.index("--max-num-batched-tokens")
                old_val = additional[idx + 1]
                # Omni mode needs larger encoder cache for audio tokens
                cap = "32768" if omni else "16384"
                additional[idx + 1] = cap
                print(f"Capped --max-num-batched-tokens {old_val} → {cap} (MiniCPM encoder cache)")

        # Ovis uses a visual tokenizer with 65536 vocabulary. During profiling,
        # the softmax over [num_visual_tokens, 65536] in float32 is ~30 GiB.
        # Reduce encoder cache budget so profiling creates 1 video item, not 2.
        if "ovis" in model_name.lower():
            if "--max-num-batched-tokens" in additional:
                idx = additional.index("--max-num-batched-tokens")
                old_val = additional[idx + 1]
                additional[idx + 1] = "65536"
                print(f"Capped --max-num-batched-tokens {old_val} → 65536 (Ovis visual tokenizer OOM)")

        # Auto-detect MoE models and enable expert parallelism
        if "--enable-expert-parallel" not in additional and is_moe_model(model_name):
            additional.append("--enable-expert-parallel")
            # With expert parallelism, disable data parallelism to avoid conflicts
            if "--data-parallel-size" in additional:
                idx = additional.index("--data-parallel-size")
                additional[idx + 1] = "1"
            print(f"Enabled --enable-expert-parallel for MoE model (TP={tp}, EP across {num_gpus} GPUs)")

        print(f"GPU layout: {num_gpus} GPUs, TP={tp}, DP={dp_size}")
        cmd.extend(additional)
    
    print(f"Starting VLLM server ({config_type}) for video benchmark with model {model_name}...")
    print(f"Logs will be written to: {log_file_path}")

    # Start server process with logging
    log_file = open(log_file_path, 'w')
    server_process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    # Wait for server to be ready
    print(f"Waiting for {model_name} server to become available...")
    server_url = config['server']['base_url']
    url = f"{server_url}models"
    
    while True:
        try:
            requests.get(url, timeout=config['system']['server_check_timeout']).raise_for_status()
            print(f"\n✅ {model_name} server is up and ready for evaluation.")
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print(".", end="", flush=True)
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"\nUnexpected error: {e}")
            break
    
    return server_process, log_file


_HF_ENV_MODELS = {
    "ming-flash-omni": "python",
    "minicpm-o": "python",
    "salmonn2": "python",
    "videollama3": "python",
    "tempo": "python",
    "longcat-next": "python",
    "omnivinci": "python",
    "baichuan-omni": "python",
}
_HF_SINGLE_GPU_MODELS = ("minicpm", "salmonn2", "videollama3", "tempo", "omnivinci", "baichuan-omni")

# Extra PYTHONPATH entries for HF models whose upstream repo isn't pip-installable.
_HF_PYTHONPATH = {
    "tempo": "./Tempo",
}


def _find_hf_python(model_name):
    """Return the Python interpreter for a model-specific conda env, or sys.executable."""
    lower = model_name.lower().replace("_", "-")
    for pattern, python_path in _HF_ENV_MODELS.items():
        if pattern in lower and os.path.exists(python_path):
            return python_path
    return sys.executable


def _resolve_hf_topology(model_name, allocated_gpus):
    """Choose dp vs tp for transformers_serve.

    Auto mode preserves current throughput for single-GPU custom backends
    while enabling actual tensor-parallel sharding for larger generic models.
    """
    allocated_gpus = max(1, int(allocated_gpus))
    override = os.environ.get("LONGSHOT_HF_TOPOLOGY", "auto").strip().lower()
    if override not in {"auto", "dp", "tp"}:
        override = "auto"

    if override == "dp":
        return 1, allocated_gpus, "dp"
    if override == "tp":
        return allocated_gpus, 1, "tp"

    lower = model_name.lower().replace("_", "-")
    if any(pattern in lower for pattern in _HF_SINGLE_GPU_MODELS):
        return 1, allocated_gpus, "dp"
    return allocated_gpus, 1, "tp"


def start_transformers_server(model_name, tensor_parallel_size, config=None, omni=False, max_frames=0):
    """Start a transformers_serve.py server as an alternative to vLLM.

    Uses HuggingFace Transformers directly for models that lack native vLLM
    support (e.g. MiniCPM-o with audio).  Exposes the same OpenAI-compatible
    API on the configured port.

    Models with strict transformers version requirements (e.g. MiniCPM-o) are
    launched from a dedicated conda env so the main env is unaffected.
    """
    if config is None:
        config = load_config()

    logs_dir = create_logs_directory(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = convert_to_underscored(model_name)
    log_file_path = os.path.join(logs_dir, f"hf_serve_{model_name_safe}_{timestamp}.log")

    port = config["server"]["port"]
    allocated_gpus = int(tensor_parallel_size)
    tp, replicas, topology = _resolve_hf_topology(model_name, allocated_gpus)

    python_bin = _find_hf_python(model_name)
    print(f"Using Python: {python_bin}")

    cmd = [
        python_bin, "transformers_serve.py",
        "--model", model_name,
        "--port", str(port),
        "--tp", str(tp),
        "--replicas", str(replicas),
        "--dtype", "bfloat16",
        "--compile",
    ]
    if omni:
        cmd.append("--omni")
    if max_frames > 0:
        cmd.extend(["--max-video-frames", str(max_frames)])

    print(
        f"Starting Transformers server for {model_name} "
        f"(allocated_gpus={allocated_gpus}, topology={topology}, tp={tp}, replicas={replicas}, port={port})"
    )
    print(f"Command: {' '.join(cmd)}")
    print(f"Logs: {log_file_path}")

    env = os.environ.copy()
    lower = model_name.lower().replace("_", "-")
    extra_paths = [p for pat, p in _HF_PYTHONPATH.items() if pat in lower]
    if extra_paths:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ":".join([*extra_paths, existing]) if existing else ":".join(extra_paths)
        print(f"Extra PYTHONPATH for {model_name}: {env['PYTHONPATH']}")

    log_file = open(log_file_path, "w")
    server_process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)

    # Wait for readiness via /health endpoint
    health_url = f"http://localhost:{port}/health"
    print(f"Waiting for Transformers server to become available...")
    while True:
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                print(f"\n✅ Transformers server for {model_name} is up and ready.")
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print(".", end="", flush=True)
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"\nUnexpected error: {e}")
            break

    return server_process, log_file


def kill_candidate_server(process, log_file=None):
    """Kill the server process (vLLM, transformers_serve, etc.) and its children"""
    if not process:
        return
    
    print("Shutting down video model server...")
    try:
        parent = psutil.Process(process.pid)
        
        # Terminate all children first, then parent
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
        
        # Force kill any processes that didn't terminate gracefully
        _, still_alive = psutil.wait_procs([parent], timeout=10)
        for p in still_alive:
            p.kill()
        
        print("Video model server process terminated successfully")
    except Exception as e:
        print(f"Error killing video model server: {e}")
        process.kill()
    finally:
        # Close log file if it exists and is open
        if log_file and not log_file.closed:
            log_file.close()
            print("Log file closed")

def save_timing_data(timing_file, stage_name, time_data):
    """Save timing data to JSON file.

    For 'generation' and 'evaluation': appends to runs[], rebuilds cumulative.
    For anything else ('scoring', 'total'): overwrites the key.
    """
    timing = {}
    if os.path.exists(timing_file):
        try:
            with open(timing_file, 'r') as f:
                timing = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            timing = {}

    if stage_name in ('generation', 'evaluation') and isinstance(time_data, dict):
        runs = timing.get(stage_name, {}).get('runs', [])
        run_entry = dict(time_data)
        run_entry['run_timestamp'] = datetime.now().isoformat()
        run_id = os.environ.get('LONGSHOT_RUN_ID')
        if run_id:
            run_entry['run_id'] = run_id
        runs.append(run_entry)

        # Sum numeric fields across all runs
        cumulative = {}
        for run in runs:
            for k, v in run.items():
                if isinstance(v, (int, float)):
                    cumulative[k] = cumulative.get(k, 0) + v

        timing[stage_name] = {'cumulative': cumulative, 'runs': runs}
    else:
        timing[stage_name] = time_data

    with open(timing_file, 'w') as f:
        json.dump(timing, f, indent=2)

