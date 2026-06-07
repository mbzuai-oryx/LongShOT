"""Tempo-6B inference via the upstream `tempo` package (Vision-CAIR/Tempo-6B).

Requires the Tempo repo (github.com/FeiElysia/Tempo) to be importable —
either `pip install -e .` from a clone or add the clone to PYTHONPATH.
The model uses a custom routing/compression architecture (Adaptive Token
Allocation) that cannot load via plain `transformers`.
"""

import multiprocessing
import os
import time

import numpy as np
import torch
from decord import cpu, VideoReader

from serve.schemas import ChatRequest
from serve.media import parse_url, _load_with_dedup

# Cache CPU core count to avoid repeated syscalls
try:
    _AVAILABLE_CORES = len(os.sched_getaffinity(0))
except AttributeError:
    _AVAILABLE_CORES = multiprocessing.cpu_count()
_DECORD_THREADS = min(max(1, _AVAILABLE_CORES - 1), 16)


class TempoInferMixin:
    """Mixin for Vision-CAIR/Tempo-6B."""

    def _init_tempo(self, args, dtype):
        from tempo.builder import load_pretrained_model

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # benchmark=True helps when input sizes are consistent (typical for video)
        torch.backends.cudnn.benchmark = os.environ.get("TEMPO_CUDNN_BENCHMARK", "1").lower() in ("1", "true")
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        use_flash_attn = False
        try:
            import flash_attn  # noqa: F401
            use_flash_attn = True
        except ImportError:
            pass

        print(f"Loading Tempo on GPU {self.device_id}: {args.model}", flush=True)
        t0 = time.monotonic()

        self.tokenizer, self.model, self.image_processor = load_pretrained_model(
            args.model,
            device_map=f"cuda:{self.device_id}",
            use_flash_attn=use_flash_attn,
        )

        # Tempo-specific compression knobs — expose via generic args if needed
        self._tempo_visual_token_budget = int(os.environ.get("TEMPO_VISUAL_TOKEN_BUDGET", 8192))
        self._tempo_max_ctx = int(os.environ.get("TEMPO_MAX_CTX", 16384))
        self._tempo_frame_windows = int(os.environ.get("TEMPO_FRAME_WINDOWS", 8))
        self._tempo_frame_stride = int(os.environ.get("TEMPO_FRAME_STRIDE", 8))
        self._tempo_conv_version = os.environ.get("TEMPO_CONV_VERSION", "qwen")
        self._tempo_video_fps = float(os.environ.get("TEMPO_VIDEO_FPS", 2.0))

        self.model.config.visual_token_budget = self._tempo_visual_token_budget
        self.model.config.tokenizer_model_max_length = self._tempo_max_ctx
        self.model.get_vision_tower_aux_list()[0].dynamic_compress = True

        self.model.eval()
        self.model.to(dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16)

        # Set pad_token_id to suppress repeated warnings during generation
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        self._tempo_max_frames = args.max_video_frames or 1024
        self.processor = None  # no HF AutoProcessor

        # Optional torch.compile for faster generation (set TEMPO_COMPILE=1 to enable)
        if os.environ.get("TEMPO_COMPILE", "").lower() in ("1", "true"):
            print("Compiling Tempo model with torch.compile...", flush=True)
            self.model = torch.compile(self.model, mode="reduce-overhead")

        torch.cuda.empty_cache()

        # Validate mm_datautils imports early to fail fast on environment issues
        try:
            from tempo.mm_datautils import (
                compute_segment_timestamp, KeywordsStoppingCriteria,
                tokenizer_image_token, process_qwen_content,
            )
        except (ImportError, AttributeError) as e:
            raise RuntimeError(
                f"Tempo environment error: {e}\n"
                "This is likely a pyarrow/datasets version conflict. "
                "Try: pip install 'pyarrow<15.0.0' 'datasets<2.19.0' in the tempo environment."
            ) from e

        print(f"Tempo loaded on GPU {self.device_id} ({time.monotonic()-t0:.1f}s, "
              f"flash_attn={use_flash_attn})", flush=True)

    def _warmup_tempo(self):
        """Warmup: run tokenizer + short generation to pre-compile CUDA kernels."""
        try:
            # Tokenizer warmup
            self.tokenizer.encode("warmup", return_tensors="pt")
            # Short generation warmup to trigger kernel compilation
            dummy_ids = torch.tensor([[1, 2, 3]], device=f"cuda:{self.device_id}")
            with torch.inference_mode():
                self.model.generate(
                    dummy_ids,
                    max_new_tokens=2,
                    do_sample=False,
                    use_cache=True,
                    vlm_inputs=None,
                    seg_timestamps=None,
                    images=None,
                    image_sizes=None,
                )
            torch.cuda.synchronize(self.device_id)
        except Exception as e:
            print(f"  [GPU {self.device_id}] Tempo warmup skipped: {e}", flush=True)

    # ── Preload ──────────────────────────────────────────────────────────

    def _tempo_preload(self, request: ChatRequest):
        """Preload the full vision prep chain: decode frames + run local compressor.

        Runs on the media thread pool so the GPU thread never blocks on
        ffmpeg decode or `process_qwen_content`.
        """
        from serve.models.base import extract_video_path
        video_path = extract_video_path(request)
        if video_path is None:
            return
        query = _extract_user_text(request) or "Describe the video."
        try:
            self._tempo_get_vision_cpu(video_path, query)
        except Exception as e:
            print(f"  [PRELOAD] Tempo preload failed: {e}", flush=True)

    def _tempo_preload_path(self, path: str, media_type: str):
        path = parse_url(path)
        ext = os.path.splitext(path)[1].lower()
        if media_type == "auto":
            media_type = "video" if ext in (".mp4", ".avi", ".mkv", ".mov", ".webm") else media_type
        if media_type == "video":
            self._tempo_load_video_cached(path)

    def _tempo_load_video_cached(self, video_path: str):
        """Cache decoded frames per video to skip redundant decord work.

        Returns (frames_np_array, real_fps) — same shape as upstream load_video.
        """
        cache_key = f"tempo_frames:{video_path}:{self._tempo_max_frames}:{self._tempo_video_fps}"
        return _load_with_dedup(cache_key, lambda: _tempo_decode_video(
            video_path, self._tempo_video_fps, self._tempo_max_frames,
        ))

    # ── Inference ────────────────────────────────────────────────────────

    def _tempo_get_vision_cpu(self, video_path: str, query: str):
        """Run local-compressor preprocessing on CPU and cache the result.

        `process_qwen_content` depends on query text, so the cache key includes
        the query.  In practice each (video, query) pair is unique per run, so
        this mainly exists to make preload effective — the media thread can
        populate the cache before the GPU thread needs it.
        """
        from tempo.mm_datautils import process_qwen_content

        cache_key = (f"tempo_vision:{video_path}:{query}:"
                     f"{self._tempo_frame_windows}:{self._tempo_frame_stride}:"
                     f"{self._tempo_max_frames}:{self._tempo_video_fps}")

        def _process():
            video_frames, real_fps = self._tempo_load_video_cached(video_path)
            vlm_inputs = process_qwen_content(
                video_frames, "video", query, self.image_processor[0], real_fps,
                self._tempo_frame_windows, self._tempo_frame_stride, is_eval=True,
            )
            # Keep on CPU — GPU thread moves tensors with .to(device).
            return {"vlm_inputs": vlm_inputs, "real_fps": real_fps}

        return _load_with_dedup(cache_key, _process)

    def _prepare_tempo(self, request: ChatRequest) -> dict:
        """CPU-side prep: frame decode, vision preprocessing, tokenization."""
        from serve.models.base import extract_video_path
        from tempo.conversation import conv_templates, SeparatorStyle
        from tempo.constants import (
            DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
        )
        from tempo.mm_datautils import (
            compute_segment_timestamp, KeywordsStoppingCriteria,
            tokenizer_image_token,
        )

        video_path = extract_video_path(request)
        query = _extract_user_text(request) or "Describe the video."
        if video_path is None:
            raise ValueError("Tempo requires a video input")

        vision = self._tempo_get_vision_cpu(video_path, query)
        vlm_inputs = vision["vlm_inputs"]
        real_fps = vision["real_fps"]

        seg_timestamps = compute_segment_timestamp(
            len(vlm_inputs["video_grid_thw"]), self.tokenizer, real_fps,
            self._tempo_frame_stride, self._tempo_frame_windows,
        )

        if getattr(self.model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + query
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + query

        conv = conv_templates[self._tempo_conv_version].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0)
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        return {
            "request": request,
            "input_ids": input_ids,
            "vlm_inputs": vlm_inputs,
            "seg_timestamps": seg_timestamps,
            "stopping_criteria": stopping_criteria,
            "stop_str": stop_str,
        }

    def _generate_tempo_prepared(self, prepared: dict) -> str:
        request = prepared["request"]
        device = self.model.device

        input_ids = prepared["input_ids"].to(device, non_blocking=True)
        attention_mask = torch.ones_like(input_ids, device=device)
        vlm_inputs = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v)
                      for k, v in prepared["vlm_inputs"].items()}

        gen_kwargs = dict(
            max_new_tokens=request.max_tokens,
            attention_mask=attention_mask,
            use_cache=True,
            do_sample=request.temperature > 0,
            temperature=request.temperature if request.temperature > 0 else None,
            stopping_criteria=[prepared["stopping_criteria"]],
            vlm_inputs=vlm_inputs,
            seg_timestamps=prepared["seg_timestamps"],
            images=None,
            image_sizes=None,
        )

        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, **gen_kwargs)
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]

        pred = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        stop_str = prepared["stop_str"]
        if stop_str and pred.endswith(stop_str):
            pred = pred[: -len(stop_str)].strip()
        return pred

    def _infer_tempo(self, request: ChatRequest) -> str:
        return self._generate_tempo_prepared(self._prepare_tempo(request))


def _tempo_decode_video(video_path: str, video_fps: float, max_frames: int):
    """decord-based frame sampler (inlined from upstream Tempo.infer.load_video)."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=_DECORD_THREADS)
    total_frames = len(vr)
    original_fps = vr.get_avg_fps() or video_fps

    clip_duration = total_frames / original_fps if original_fps > 0 else 0
    target = max(1, round(clip_duration * video_fps))
    n = min(max(target, 4), max_frames)
    if total_frames <= 1:
        indices = [0]
    elif n == 1:
        indices = [total_frames - 1]
    else:
        indices = np.round(np.linspace(0, total_frames - 1, n)).astype(int)
        indices = np.clip(indices, 0, total_frames - 1).tolist()

    frames = vr.get_batch(indices).asnumpy()
    real_fps = len(frames) / clip_duration if clip_duration > 0 else video_fps
    return frames, real_fps


def _extract_user_text(request: ChatRequest) -> str:
    for msg in request.messages:
        if msg.get("role") != "user":
            continue
        raw = msg.get("content", "")
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            parts = [p.get("text", "") for p in raw
                     if isinstance(p, dict) and p.get("type") == "text"]
            if parts:
                return "\n".join(parts)
    return ""
