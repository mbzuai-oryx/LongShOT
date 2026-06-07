"""
OpenRouter API client for cloud-hosted model evaluation.

Sends pre-extracted frames (JPEG) and compressed audio (MP3) as base64 data
URLs. Raw video is not supported — use the :f{N} flag to select frame count.

Uses the OpenAI SDK since OpenRouter is API-compatible.
"""

import base64
import json
import os
import subprocess
import time
from functools import lru_cache

import httpx
from filelock import FileLock
from openai import OpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Speech is intelligible at 32kbps mono; keeps a 60-min WAV (~600MB) under 15MB
AUDIO_COMPRESS_BITRATE = "48k"


def create_openrouter_client(config):
    """Create an OpenAI client pointed at OpenRouter."""
    api_cfg = config.get("openrouter", {})
    api_key = api_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
            "or openrouter.api_key in config.yaml"
        )
    timeout = api_cfg.get("timeout", 1800)
    max_connections = api_cfg.get("max_connections", 8)
    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        http_client=httpx.Client(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            timeout=httpx.Timeout(timeout, connect=30.0),
        ),
    )


@lru_cache(maxsize=512)
def _encode_image_base64(file_path):
    """Read a JPEG/PNG and return its base64 data URL. Cached per path."""
    ext = os.path.splitext(file_path)[1].lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/jpeg")
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{data}"


@lru_cache(maxsize=256)
def _compress_and_encode_audio(wav_path):
    """Compress WAV to MP3 (cached on disk), then return base64 data URL.

    The MP3 is cached alongside the WAV as <stem>.or.mp3 so subsequent
    samples from the same video skip the ffmpeg step.
    """
    stem, _ = os.path.splitext(wav_path)
    mp3_path = f"{stem}.or.mp3"

    with FileLock(f"{mp3_path}.lock"):
        if not os.path.exists(mp3_path):
            tmp = f"{mp3_path}.tmp"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error",
                     "-i", wav_path,
                     "-ac", "1", "-b:a", AUDIO_COMPRESS_BITRATE,
                     "-f", "mp3", tmp],
                    check=True, timeout=120,
                )
                os.replace(tmp, mp3_path)
            except Exception as e:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                raise RuntimeError(f"Audio compression failed for {wav_path}: {e}") from e

    with open(mp3_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:audio/mpeg;base64,{data}"


def convert_messages_for_openrouter(messages):
    """Convert file:// media references to base64 data URLs for OpenRouter.

    Handles:
    - image_url with file:// → base64 JPEG/PNG (for pre-extracted frames)
    - audio_url with file:// → compressed MP3 base64 (WAV → MP3 → base64)
    - video_url → raises error (use :f{N} flag for frame extraction instead)
    """
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_content = []
        for item in content:
            item_type = item.get("type", "")

            if item_type == "video_url":
                raise ValueError(
                    "Raw video not supported for API models — too large for base64. "
                    "Add :f128 (or :f256) flag to use pre-extracted frames instead. "
                    "Example: \"xiaomi/mimo-v2-omni:0:api:omni:f128\""
                )

            elif item_type == "audio_url":
                url = item["audio_url"]["url"]
                if url.startswith("file://"):
                    path = url[7:]
                    data_url = _compress_and_encode_audio(path)
                    new_content.append({
                        "type": "input_audio",
                        "input_audio": {"data": data_url, "format": "mp3"},
                    })
                else:
                    new_content.append(item)

            elif item_type == "image_url":
                url = item["image_url"]["url"]
                if url.startswith("file://"):
                    path = url[7:]
                    data_url = _encode_image_base64(path)
                    new_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })
                else:
                    new_content.append(item)

            else:
                new_content.append(item)

        msg["content"] = new_content
    return messages


def execute_turn_openrouter(client, model_name, messages, extra_body, config):
    """Execute a single API call against OpenRouter with retry and rate-limit handling."""
    from utils import strip_think_tags

    api_cfg = config.get("openrouter", {})
    max_retries = api_cfg.get("max_retries", 3)
    base_delay = api_cfg.get("retry_delay", 2.0)

    messages = convert_messages_for_openrouter(messages)

    or_extra_body = {}
    provider_prefs = api_cfg.get("provider")
    if provider_prefs:
        or_extra_body["provider"] = provider_prefs

    for attempt in range(max_retries):
        try:
            max_tokens = api_cfg.get("max_tokens", config["generation"]["max_tokens"])
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=config["generation"]["temperature"],
                top_p=config["generation"]["top_p"],
                timeout=config["generation"]["timeout"],
                extra_body=or_extra_body or None,
            )
            response_text = completion.choices[0].message.content or ""

            _track_usage(completion, model_name, config)

            return strip_think_tags(response_text)

        except Exception as e:
            error_str = str(e)
            is_retryable = any(code in error_str for code in ("429", "502", "503", "529"))
            if is_retryable and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            raise


def _track_usage(completion, model_name, config):
    """Append token usage to a cost-tracking JSONL file (best-effort)."""
    try:
        usage = completion.usage
        if not usage:
            return

        api_cfg = config.get("openrouter", {})
        cost_file = api_cfg.get("cost_log", "openrouter_costs.jsonl")

        entry = {
            "timestamp": time.time(),
            "model": model_name,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

        with FileLock(f"{cost_file}.lock"):
            with open(cost_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
