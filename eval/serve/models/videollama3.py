"""VideoLLaMA3 inference via AutoModelForCausalLM + AutoProcessor (trust_remote_code)."""

import os
import torch

from serve.schemas import ChatRequest
from serve.media import parse_url, _load_with_dedup, _media_cache_lock, _cache_get


class VideoLLaMA3InferMixin:
    """Mixin for VideoLLaMA3 models (DAMO-NLP-SG/VideoLLaMA3-*).

    The model's custom processor handles video loading, frame sampling, and
    tokenization when given a conversation with video_path.  We just need
    to convert from the OpenAI-format request to the processor's conversation
    format and pass the right kwargs to generate().
    """

    def _init_videollama3(self, args, dtype):
        """Load VideoLLaMA3 model and processor."""
        from transformers import AutoModelForCausalLM, AutoProcessor

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # cudnn.benchmark must stay False — VideoLLaMA3's patch embedding
        # runs Conv2d with 60K+ batch items; benchmark mode picks algorithms
        # that write out-of-bounds on Blackwell GPUs, causing CUDA illegal
        # memory access.
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        # Shim type aliases removed in transformers 5.x but used by
        # VideoLLaMA3's trust_remote_code files (written for 4.46.3).
        from transformers import image_utils
        if not hasattr(image_utils, "VideoInput"):
            from typing import List, Union
            import numpy as np
            from PIL import Image as _PILImage
            image_utils.VideoInput = Union[
                List["_PILImage.Image"], List[np.ndarray], List[List["_PILImage.Image"]], List[List[np.ndarray]]
            ]

        print(f"Loading VideoLLaMA3 on GPU {self.device_id}: {args.model}", flush=True)
        import time
        t0 = time.monotonic()

        attn_impl = "sdpa"
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            pass

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            device_map=f"cuda:{self.device_id}",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            args.model, trust_remote_code=True,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        self._vl3_logged_batch_token_compression_override = False

        # VideoLLaMA3's NaViT vision encoder uses O(N²) attention over all
        # vision tokens.  Without flash-attn, even 64 frames can OOM on 95GB GPUs.
        # flash_attention_2 reduces this to O(N) and is strongly recommended.
        if attn_impl != "flash_attention_2":
            self._vl3_max_frames = min(args.max_video_frames or 16, 16)
            print(f"  WARNING: flash-attn not installed — limiting to {self._vl3_max_frames} frames "
                  f"(install flash-attn for up to 180 frames)", flush=True)
        else:
            self._vl3_max_frames = args.max_video_frames or 128

        # torch.compile is incompatible with VideoLLaMA3's custom generate()
        # and NaViT vision encoder (variable-size inputs, custom ops).
        if args.compile:
            print("Skipping torch.compile for VideoLLaMA3 (incompatible with custom generate)", flush=True)

        # Monkey-patch the processor's load_video to cache decoded frames.
        # Video decoding via ffmpeg is the main CPU bottleneck — without this,
        # GPUs sit idle waiting for ffmpeg to finish on every request.
        # With this patch, only the first request per video hits ffmpeg;
        # subsequent requests get cached PIL frames instantly.
        _orig_load_video = self.processor.load_video
        def _cached_load_video(*args_lv, **kwargs_lv):
            # Extract video_path from args or kwargs
            video_path = kwargs_lv.get("video_path") or (args_lv[0] if args_lv else None)
            max_frames = kwargs_lv.get("max_frames", self._vl3_max_frames)
            if video_path:
                cache_key = f"vl3_frames:{video_path}:{max_frames}"
                return _load_with_dedup(cache_key,
                                        lambda: _orig_load_video(*args_lv, **kwargs_lv))
            return _orig_load_video(*args_lv, **kwargs_lv)
        self.processor.load_video = _cached_load_video

        torch.cuda.empty_cache()
        print(f"VideoLLaMA3 loaded on GPU {self.device_id}: {args.model} "
              f"({time.monotonic()-t0:.1f}s, attn={attn_impl})", flush=True)

    def _warmup_videollama3(self):
        """Warmup: tokenizer only. Vision encoder warms up on first request."""
        pass

    # ── Preload ──────────────────────────────────────────────────────────

    def _videollama3_preload(self, request: ChatRequest):
        """Pre-process video into vision tensor cache from the media thread pool.

        Runs the full processor pipeline (ffmpeg decode → image processing →
        vision tensors) ahead of time.  When the GPU inference thread later
        calls _vl3_build_inputs, it gets cached vision tensors instantly and
        only needs to run text tokenization.
        """
        from serve.models.base import extract_video_path
        video_path = extract_video_path(request)
        if video_path is None:
            return
        cache_key = f"vl3_vision:{video_path}:{self._vl3_max_frames}"
        with _media_cache_lock:
            if _cache_get(cache_key) is not None:
                return
        try:
            self._vl3_get_vision_tensors(video_path)
        except Exception as e:
            print(f"  [PRELOAD] VL3 preload failed: {e}", flush=True)

    def _videollama3_preload_path(self, path: str, media_type: str):
        """Preload a raw video path into the VideoLLaMA3 tensor caches."""
        path = parse_url(path)
        ext = os.path.splitext(path)[1].lower()
        if media_type == "auto":
            media_type = "video" if ext in (".mp4", ".avi", ".mkv", ".mov", ".webm") else media_type
        if media_type == "video":
            self._vl3_get_vision_tensors(path)

    # ── Conversation building ───────────────────────────────────────────

    def _vl3_build_conversation_with_video(self, request: ChatRequest, video_path: str):
        """Convert OpenAI-format request to VideoLLaMA3 conversation with video."""
        conversation = []
        video_added = False
        for msg in request.messages:
            role = msg.get("role", "user")
            raw = msg.get("content", "")

            if isinstance(raw, str):
                conversation.append({"role": role, "content": [{"type": "text", "text": raw}]})
                continue
            if not isinstance(raw, list):
                continue

            content_parts = []
            for part in raw:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype == "text":
                    content_parts.append({"type": "text", "text": part.get("text", "")})

            # Add video to the first user message
            if not video_added and role == "user" and video_path:
                content_parts.insert(0, {
                    "type": "video",
                    "video": {
                        "video_path": video_path,
                        "fps": 1,
                        "max_frames": self._vl3_max_frames,
                    },
                })
                video_added = True

            if content_parts:
                conversation.append({"role": role, "content": content_parts})

        if not conversation:
            conversation = [{"role": "user", "content": [{"type": "text", "text": "Describe."}]}]
        return conversation

    def _vl3_get_vision_tensors(self, video_path: str) -> dict:
        """Get cached vision tensors (pixel_values, grid_sizes, merge_sizes, modals).

        Two cache layers:
          1. Frame cache (via monkey-patched load_video) — skips ffmpeg
          2. Vision tensor cache (this method) — skips image processing too

        On first call: processor decodes video + processes images → cache tensors.
        On subsequent calls: returns cached tensors instantly.
        """
        cache_key = f"vl3_vision:{video_path}:{self._vl3_max_frames}"

        def _process():
            conversation = [{"role": "user", "content": [
                {"type": "video", "video": {
                    "video_path": video_path,
                    "fps": 1,
                    "max_frames": self._vl3_max_frames,
                }},
                {"type": "text", "text": "."},
            ]}]
            inputs = self.processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            return {
                "pixel_values": inputs["pixel_values"],
                "grid_sizes": inputs["grid_sizes"],
                "merge_sizes": inputs["merge_sizes"],
                "modals": inputs["modals"],
            }

        return _load_with_dedup(cache_key, _process)

    def _vl3_build_inputs(self, request: ChatRequest, video_path: str):
        """Build processor inputs for a request with video.

        Uses two-layer caching:
          1. Vision tensors (pixel_values, grid_sizes) cached per video
          2. Only text tokenization runs per request (fast)

        On cache hit, calls the processor for correct tokenization (image
        token placeholders depend on grid_sizes), then replaces the vision
        tensors with the cached versions — skipping both ffmpeg decode AND
        image processing (resize/normalize of 128 frames).
        """
        # Get cached vision tensors (processes video only on first call)
        vision = self._vl3_get_vision_tensors(video_path)

        # Run processor for text tokenization (video frames cached via
        # monkey-patch, so load_video returns instantly)
        conversation = self._vl3_build_conversation_with_video(request, video_path)
        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Replace vision tensors with cached versions — avoids redundant
        # image processing (resize/normalize) that the processor just did.
        inputs["pixel_values"] = vision["pixel_values"]
        inputs["grid_sizes"] = vision["grid_sizes"]
        inputs["merge_sizes"] = vision["merge_sizes"]
        inputs["modals"] = vision["modals"]
        return inputs

    def _prepare_videollama3(self, request: ChatRequest) -> dict:
        """Prepare VideoLLaMA3 processor inputs on CPU."""
        from serve.models.base import extract_video_path
        video_path = extract_video_path(request)

        if video_path:
            inputs = self._vl3_build_inputs(request, video_path)
        else:
            conversation = [{"role": "user", "content": [{"type": "text", "text": "Describe."}]}]
            inputs = self.processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

        return {
            "mode": "single",
            "request": request,
            "inputs": inputs,
        }

    def _prepare_videollama3_batch(self, requests: list[ChatRequest]):
        """Prepare a same-video VideoLLaMA3 batch on CPU."""
        if len(requests) <= 1:
            return [self._prepare_videollama3(requests[0])] if requests else []

        from serve.models.base import extract_video_path
        video_paths = [extract_video_path(r) for r in requests]
        unique_videos = set(v for v in video_paths if v)
        if len(unique_videos) != 1 or video_paths[0] is None:
            return [self._prepare_videollama3(req) for req in requests]

        video_path = video_paths[0]
        all_inputs = [self._vl3_build_inputs(req, video_path) for req in requests]

        batch_size = len(all_inputs)
        all_ids = [inp["input_ids"].squeeze(0) for inp in all_inputs]
        max_len = max(ids.shape[0] for ids in all_ids)
        pad_id = self.tokenizer.pad_token_id or 0
        padded_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, ids in enumerate(all_ids):
            seq_len = ids.shape[0]
            offset = max_len - seq_len
            padded_ids[i, offset:] = ids
            attention_mask[i, offset:] = 1

        ref = all_inputs[0]
        return {
            "mode": "batch",
            "requests": requests,
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "pixel_values": ref["pixel_values"].repeat(batch_size, 1),
            "grid_sizes": ref["grid_sizes"].repeat(batch_size, 1),
            "merge_sizes": ref["merge_sizes"].repeat(batch_size),
            "modals": ref["modals"] * batch_size,
        }

    # ── Single inference ────────────────────────────────────────────────

    def _generate_videollama3_prepared(self, prepared: dict) -> str:
        """Run VideoLLaMA3 generation from prepared CPU tensors."""
        request = prepared["request"]
        inputs = prepared["inputs"]
        device = self.model.device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        gen_kwargs = dict(max_new_tokens=request.max_tokens, do_sample=False)
        if request.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = request.temperature
            gen_kwargs["top_p"] = request.top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # VideoLLaMA3's custom generate() converts input_ids to inputs_embeds
        # internally, so output_ids contains only the generated tokens.
        text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

        return text

    def _infer_videollama3(self, request: ChatRequest) -> str:
        """Compatibility wrapper for direct inference."""
        return self._generate_videollama3_prepared(self._prepare_videollama3(request))

    # ── Batch inference ─────────────────────────────────────────────────

    def _generate_videollama3_batch_prepared(self, prepared_batch) -> list[str | Exception]:
        """Run batched VideoLLaMA3 generation from prepared CPU tensors."""
        if isinstance(prepared_batch, list):
            results = []
            for prepared in prepared_batch:
                try:
                    results.append(self._generate_videollama3_prepared(prepared))
                except Exception as e:
                    results.append(e)
            return results

        prepared = prepared_batch
        requests = prepared["requests"]
        device = self.model.device
        batch_size = len(requests)

        gen_kwargs = dict(
            max_new_tokens=max(r.max_tokens for r in requests),
            do_sample=False,
        )
        if requests and requests[0].temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = requests[0].temperature
            gen_kwargs["top_p"] = requests[0].top_p

        print(f"  [GPU {self.device_id}] [VL3] BATCH {batch_size} requests, "
              f"max_len={prepared['input_ids'].shape[1]}", flush=True)

        original_token_compression = getattr(self.model.config, "use_token_compression", None)
        disable_token_compression = batch_size > 1 and bool(original_token_compression)
        if disable_token_compression:
            if not self._vl3_logged_batch_token_compression_override:
                print(
                    f"  [GPU {self.device_id}] [VL3] disabling token compression for batched generation",
                    flush=True,
                )
                self._vl3_logged_batch_token_compression_override = True
            self.model.config.use_token_compression = False

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids=prepared["input_ids"].to(device),
                    attention_mask=prepared["attention_mask"].to(device),
                    pixel_values=prepared["pixel_values"].to(device, self.model.dtype),
                    grid_sizes=prepared["grid_sizes"].to(device),
                    merge_sizes=prepared["merge_sizes"].to(device),
                    modals=prepared["modals"],
                    **gen_kwargs,
                )
        finally:
            if disable_token_compression:
                self.model.config.use_token_compression = original_token_compression

        results = []
        for i in range(batch_size):
            text = self.processor.batch_decode(
                [output_ids[i]], skip_special_tokens=True
            )[0].strip()
            results.append(text)
        return results

    def _infer_videollama3_batch(self, requests: list[ChatRequest]) -> list[str | Exception]:
        """Compatibility wrapper for direct batch inference."""
        return self._generate_videollama3_batch_prepared(
            self._prepare_videollama3_batch(requests)
        )
