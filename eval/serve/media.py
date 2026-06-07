"""Media cache infrastructure and loading functions (video, audio, images)."""

import base64
import io
import math
import os
import subprocess
import tempfile
import time
import threading
import wave
from collections import OrderedDict
from typing import Any

import numpy as np

# True LRU cache using OrderedDict for O(1) access, move-to-end, and eviction.
_media_cache: OrderedDict[str, Any] = OrderedDict()
_media_cache_lock = threading.Lock()
_MEDIA_CACHE_MAX = 50  # overridden at startup based on replica count

# Per-key deduplication: prevents N threads from decoding the same media
# simultaneously.  First thread sets an Event; others wait on it.
_inflight: dict[str, threading.Event] = {}
_inflight_lock = threading.Lock()

# Stats
_cache_hits = 0
_cache_misses = 0


def _set_media_cache_max(n: int):
    global _MEDIA_CACHE_MAX
    _MEDIA_CACHE_MAX = n


def _cache_evict():
    """Evict oldest entries if cache exceeds max size. O(1) per eviction."""
    while len(_media_cache) > _MEDIA_CACHE_MAX:
        _media_cache.popitem(last=False)


def _cache_get(key: str):
    """Get from cache and move to end (most recently used). Returns None on miss."""
    global _cache_hits, _cache_misses
    if key in _media_cache:
        _media_cache.move_to_end(key)
        _cache_hits += 1
        return _media_cache[key]
    _cache_misses += 1
    return None


def _cache_put(key: str, value: Any):
    """Insert into cache, evicting oldest if full."""
    _media_cache[key] = value
    _media_cache.move_to_end(key)
    _cache_evict()


def cache_stats() -> dict:
    """Return cache hit/miss statistics."""
    total = _cache_hits + _cache_misses
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "hit_rate": round(_cache_hits / total, 3) if total > 0 else 0,
        "size": len(_media_cache),
        "max": _MEDIA_CACHE_MAX,
    }


def _load_with_dedup(key: str, load_fn):
    """Load media with per-key deduplication.

    If another thread is already loading the same key, wait for it instead
    of starting a redundant load.  Returns the cached value.
    """
    # Fast path: already cached
    with _media_cache_lock:
        hit = _cache_get(key)
        if hit is not None:
            return hit

    # Check if another thread is loading this key
    with _inflight_lock:
        if key in _inflight:
            event = _inflight[key]
        else:
            event = threading.Event()
            _inflight[key] = event
            event = None  # we are the loader

    if event is not None:
        # Wait for the other thread to finish loading
        event.wait(timeout=300)
        with _media_cache_lock:
            hit = _cache_get(key)
            if hit is not None:
                return hit
        # Loader failed — fall through to load ourselves

    try:
        result = load_fn()
        with _media_cache_lock:
            _cache_put(key, result)
        return result
    finally:
        with _inflight_lock:
            ev = _inflight.pop(key, None)
            if ev is not None:
                ev.set()  # wake up waiters


def parse_url(url: str) -> str:
    """Strip file:// prefix to get a local path."""
    if url.startswith("file://"):
        return url[7:]
    return url


def _read_wav_mono_16k(audio_path: str) -> np.ndarray:
    """Read a PCM WAV file as float32 mono at 16kHz."""
    with wave.open(audio_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sample_rate != 16000:
        raise ValueError(f"expected 16kHz wav, got {sample_rate}Hz")

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported wav sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


def _load_audio_array(path: str) -> np.ndarray:
    """Load audio or video media into a 16kHz mono float32 waveform."""
    try:
        if path.lower().endswith(".wav"):
            return _read_wav_mono_16k(path)
    except Exception:
        pass

    audio_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", audio_tmp.name],
            capture_output=True,
            check=True,
            timeout=120,
        )
        return _read_wav_mono_16k(audio_tmp.name)
    finally:
        if os.path.exists(audio_tmp.name):
            os.unlink(audio_tmp.name)


def load_video_frames(video_path: str, max_frames: int = 0):
    """Extract uniformly sampled frames from a video as PIL Images (cached, deduped)."""
    cache_key = f"frames:{video_path}:{max_frames}"

    def _load():
        from decord import VideoReader, cpu
        from PIL import Image

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        fps = vr.get_avg_fps()
        if max_frames > 0:
            n = min(total, max_frames)
        else:
            n = max(1, int(total / fps))

        indices = np.linspace(0, total - 1, n, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f) for f in frames]

    return _load_with_dedup(cache_key, _load)


def load_video_omni_chunks(video_path: str, max_frames: int = 0):
    """Extract interleaved frame+audio 1-second chunks for MiniCPM-o omni mode (cached)."""
    cache_key = f"omni:{video_path}:{max_frames}"

    def _load():
        from decord import VideoReader, cpu
        from PIL import Image

        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / fps
        total_seconds = max(1, math.ceil(duration))

        audio_np = None
        audio_path_wav = os.path.splitext(video_path)[0] + ".wav"
        if os.path.exists(audio_path_wav):
            try:
                audio_np = _load_audio_array(audio_path_wav)
            except Exception:
                pass

        if audio_np is None:
            try:
                audio_np = _load_audio_array(video_path)
            except Exception:
                pass

        if max_frames > 0 and total_seconds > max_frames:
            second_indices = np.linspace(0, total_seconds - 1, max_frames, dtype=int).tolist()
        else:
            second_indices = list(range(total_seconds))

        frame_indices = [
            int(min((sec + 0.5) / duration, 1.0) * (total_frames - 1))
            for sec in second_indices
        ]
        all_frames = vr.get_batch(frame_indices).asnumpy()
        contents = []
        for i, sec in enumerate(second_indices):
            contents.append(Image.fromarray(all_frames[i]))
            if audio_np is not None:
                chunk = audio_np[16000 * sec: 16000 * (sec + 1)]
                if len(chunk) > 0:
                    contents.append(chunk)
        return contents

    return _load_with_dedup(cache_key, _load)


def load_minicpm_video_content(
    video_path: str,
    *,
    audio_path: str | None = None,
    use_audio: bool = True,
    stack_frames: int = 1,
    use_ffmpeg: bool = False,
):
    """Load MiniCPM-native video content using the upstream sampling rules."""
    cache_key = (
        f"minicpm_video:{video_path}:{audio_path or ''}:"
        f"{int(use_audio)}:{int(stack_frames)}:{int(use_ffmpeg)}"
    )

    def _load():
        from decord import VideoReader, cpu
        from PIL import Image

        max_num_frames = int(os.getenv("MAX_NUM_FRAMES", "64"))

        def _uniform_sample(seq, n):
            if len(seq) <= n:
                return list(seq)
            idxs = np.linspace(0, len(seq) - 1, n, dtype=int)
            return [seq[i] for i in idxs]

        def _load_audio_waveform():
            if not use_audio:
                return None

            if audio_path is not None:
                return _load_audio_array(audio_path)

            adjacent_wav = os.path.splitext(video_path)[0] + ".wav"
            if os.path.exists(adjacent_wav):
                return _load_audio_array(adjacent_wav)

            return _load_audio_array(video_path)

        vr = VideoReader(str(video_path), ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / fps

        if max_num_frames > 0 and duration > max_num_frames:
            timestamps = [round(i * 0.1, 1) for i in range(int(duration / 0.1))]
            frame_idx = [min(int(ts * fps), total_frames - 1) for ts in timestamps]
            frame_idx = _uniform_sample(frame_idx, max_num_frames)
            timestamps = _uniform_sample(timestamps, max_num_frames)
        else:
            num_seconds = max(1, math.ceil(duration))
            timestamps = list(range(num_seconds))
            frame_idx = [min(int(ts * fps), total_frames - 1) for ts in timestamps]

        video = vr.get_batch(frame_idx).asnumpy()
        video_frames = [Image.fromarray(v.astype("uint8")).convert("RGB") for v in video]
        audio_np = _load_audio_waveform()
        audio_segments = []
        if audio_np is not None:
            for i, start_time in enumerate(timestamps):
                start_sample = int(start_time * 16000)
                end_sample = start_sample + 16000  # 1-second chunk
                segment = audio_np[start_sample:end_sample]
                if len(segment) < 1600:
                    segment = np.concatenate([segment, np.zeros(1600 - len(segment), dtype=segment.dtype)])
                audio_segments.append(segment)
        else:
            audio_segments = None

        stacked_frames = None

        contents = []
        for idx, frame in enumerate(video_frames):
            contents.append(frame)
            if use_audio and audio_segments is not None and idx < len(audio_segments):
                audio_chunk = audio_segments[idx]
                if audio_chunk is not None and len(audio_chunk) > 0:
                    contents.append(audio_chunk)
            if stacked_frames is not None and idx < len(stacked_frames):
                stacked = stacked_frames[idx]
                if stacked is not None:
                    contents.append(stacked)
        return contents

    return _load_with_dedup(cache_key, _load)


def load_audio(audio_path: str):
    """Load audio as 16kHz mono numpy array (cached, deduped)."""
    cache_key = f"audio:{audio_path}"

    def _load():
        return _load_audio_array(audio_path)

    return _load_with_dedup(cache_key, _load)


def load_image(url: str):
    """Load an image from a URL or local path (cached, deduped for file paths)."""
    from PIL import Image
    path = parse_url(url)

    if path.startswith("data:"):
        header, b64 = path.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    cache_key = f"image:{path}"
    return _load_with_dedup(cache_key, lambda: Image.open(path).convert("RGB"))
