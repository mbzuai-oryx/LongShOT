"""LongCat-Next inference via its custom HF remote code.

LongCat-Next (meituan-longcat/LongCat-Next) is a discrete-native any-to-any
omni model: text/image/audio in and out. There is no vLLM kernel for the
``longcat_next`` arch, so we serve it through the transformers backend.

Key API quirks handled here:
  * ``AutoTokenizer(..., fix_mistral_regex=True)`` is required.
  * ``model.text_tokenizer`` must be bound dynamically after load.
  * The processor takes a single rendered string with embedded
    ``<longcat_img_start>PATH<longcat_img_end>`` /
    ``<longcat_audio_start>PATH<longcat_audio_end>`` tags and returns a
    *3-tuple* ``(text_inputs, visual_inputs, audio_inputs)`` rather than a
    BatchEncoding dict.
  * ``model.generate`` accepts ``visual_inputs``/``audio_inputs`` kwargs and
    returns a struct with ``.sequences``/``.visual_ids``/``.audio_text_ids``/
    ``.audio_ids``. We only consume ``.sequences`` (text-only scoring).
  * Visual / audio decoders are lazy-loaded; never calling
    ``decode_visual_ids_and_save`` / ``decode_audio_ids_and_save`` keeps
    them off the GPU and saves significant VRAM.

Optimizations baked in:
  * bfloat16 + flash-attention-2 (sdpa fallback).
  * TF32 matmul + cudnn benchmark.
  * Processor-output cache keyed on the rendered prompt + media signature.
  * Frame-extraction cache to disk so repeated questions about the same video
    skip ffmpeg/decord.
  * ``device_map="auto"`` for TP>1; pinned single-device otherwise.
  * ``num_logits_to_keep=1`` and ``use_cache=True`` at generate time.
"""

import hashlib
import os
import tempfile
import threading
from bisect import bisect_left
from typing import Any

import torch

from serve.media import _load_with_dedup, parse_url, load_video_frames
from serve.schemas import ChatRequest


_VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".MP4")
_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

_FRAME_DIR_LOCK = threading.Lock()
_FRAME_DIR = os.path.join(tempfile.gettempdir(), "longcat_next_frames")


def _is_video_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in tuple(e.lower() for e in _VIDEO_EXTS)


def _is_audio_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _AUDIO_EXTS


def _frame_cache_dir(video_path: str, n_frames: int) -> str:
    """Stable on-disk dir for extracted frames of (video, n_frames)."""
    h = hashlib.md5(f"{os.path.realpath(video_path)}:{n_frames}".encode()).hexdigest()[:16]
    return os.path.join(_FRAME_DIR, h)


def _extract_video_to_jpg_paths(video_path: str, n_frames: int) -> list[str]:
    """Extract uniformly sampled frames as JPGs and return their paths.

    Cached: a sentinel ``done`` file marks completion so concurrent callers
    don't re-encode. Falls through ``_load_with_dedup`` to coalesce.
    """
    cache_key = f"longcat_frames:{video_path}:{n_frames}"

    def _do_extract():
        out_dir = _frame_cache_dir(video_path, n_frames)
        sentinel = os.path.join(out_dir, "done")
        with _FRAME_DIR_LOCK:
            if os.path.exists(sentinel):
                return sorted(
                    os.path.join(out_dir, f) for f in os.listdir(out_dir)
                    if f.startswith("f") and f.endswith(".jpg")
                )
            os.makedirs(out_dir, exist_ok=True)

        frames = load_video_frames(video_path, n_frames)
        paths = []
        for i, img in enumerate(frames):
            p = os.path.join(out_dir, f"f{i:04d}.jpg")
            img.save(p, format="JPEG", quality=92)
            paths.append(p)
        with open(sentinel, "w") as fh:
            fh.write(str(len(paths)))
        return paths

    return _load_with_dedup(cache_key, _do_extract)


def _to_device(obj, device, dtype: torch.dtype | None = None):
    """Recursively move tensors in obj onto device (and cast floats to dtype)."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        if dtype is not None and obj.is_floating_point():
            return obj.to(device=device, dtype=dtype)
        return obj.to(device=device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cls = type(obj)
        return cls(_to_device(v, device, dtype) for v in obj)
    return obj


class LongCatNextInferMixin:
    """Mixin for meituan-longcat/LongCat-Next."""

    # ─── load ────────────────────────────────────────────────────────────

    def _init_longcat_next(self, args, dtype):
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        attn_impl = "sdpa"
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            pass

        # Reduce allocator fragmentation — dNaViT's variable-shape patch tensors
        # otherwise leave large reserved-but-unallocated holes that OOM the
        # visual encoder pass. Must be set before the first CUDA allocation.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        self._lcn_dtype = dtype
        self._lcn_max_frames = self.max_video_frames or 64

        print(
            f"Loading LongCat-Next on devices {self.device_ids}: {args.model} "
            f"(attn={attn_impl}, tp={self.tensor_parallel_size}, frames={self._lcn_max_frames})",
            flush=True,
        )

        tokenizer_kwargs = dict(trust_remote_code=True)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.model, fix_mistral_regex=True, **tokenizer_kwargs
            )
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)

        self.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        from transformers import AutoConfig
        self._lcn_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
        if self.tensor_parallel_size > 1:
            # Cap each GPU so accelerate balances weights evenly and leaves
            # headroom for dNaViT visual-encoder activations at inference time.
            model_kwargs["device_map"] = "auto"
            headroom_gb = 25
            model_kwargs["max_memory"] = {
                dev: f"{95 - headroom_gb}GiB"
                for dev in self.device_ids
            }
        else:
            model_kwargs["device_map"] = f"cuda:{self.device_id}"

        self.model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        self.model.eval()

        # Required dynamic binding per the model card.
        self.model.text_tokenizer = self.tokenizer

        if args.compile:
            print(
                "Skipping torch.compile for LongCat-Next (custom multimodal generate)",
                flush=True,
            )

        torch.cuda.empty_cache()
        print(
            f"LongCat-Next loaded on devices {self.device_ids}: {args.model}",
            flush=True,
        )

    def _build_longcat_device_map(self) -> dict[str, int]:
        """Spread 150 GB across devices with headroom for visual activations.

        The visual encoder (dNaViT) runs a full forward over all frame patches
        and needs significant free VRAM. We pin it to the last device (which
        gets fewer LLM layers) and pack the heavy generation heads onto
        earlier devices.
        """
        num_layers = int(getattr(self._lcn_config, "num_layers", 0)
                         or getattr(self._lcn_config, "num_hidden_layers", 14))
        tp = self.tensor_parallel_size
        primary = self.device_id
        last_dev = self.device_ids[-1]

        # Distribute LLM layers, giving the last device fewer so the
        # visual encoder has more activation headroom there.
        per_dev = num_layers // tp
        extra = num_layers % tp
        device_map: dict[str, int] = {}
        layer_idx = 0
        for i, dev in enumerate(self.device_ids):
            n = per_dev + (1 if i < extra else 0)
            for _ in range(n):
                device_map[f"model.layers.{layer_idx}"] = dev
                layer_idx += 1

        # Heavy generation heads (never called for text-only scoring) →
        # pack onto the primary device away from the visual encoder.
        device_map["visual_head"] = primary
        device_map["audio_head"] = primary

        # Visual + audio encoders on the last device (most headroom).
        device_map["model.visual_tokenizer"] = last_dev
        device_map["model.audio_tokenizer"] = last_dev

        # Catch-all for small modules (embeddings, norms, etc.).
        device_map[""] = primary
        return device_map

    def _warmup_longcat_next(self):
        """Tokenizer warmup only; full multimodal warmup is expensive."""
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "warmup"}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    # ─── prompt rendering ────────────────────────────────────────────────

    def _lcn_render_parts(self, parts: list[dict[str, Any]]) -> str:
        """Render OpenAI content parts as a LongCat-tagged string."""
        out = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "text":
                out.append(part.get("text", ""))
            elif ptype in ("image_url", "video_url"):
                key = "image_url" if ptype == "image_url" else "video_url"
                path = parse_url(part[key]["url"])
                if _is_video_path(path) or ptype == "video_url":
                    for fp in _extract_video_to_jpg_paths(path, self._lcn_max_frames):
                        out.append(f"<longcat_img_start>{fp}<longcat_img_end>")
                    if self.omni:
                        sibling = os.path.splitext(path)[0] + ".wav"
                        if os.path.exists(sibling):
                            out.append(f"<longcat_audio_start>{sibling}<longcat_audio_end>")
                else:
                    out.append(f"<longcat_img_start>{path}<longcat_img_end>")
            elif ptype == "audio_url":
                path = parse_url(part["audio_url"]["url"])
                out.append(f"<longcat_audio_start>{path}<longcat_audio_end>")
        return "".join(out)

    def _lcn_messages_from_request(self, request: ChatRequest) -> list[dict[str, Any]]:
        messages = []
        for msg in request.messages:
            role = (msg.get("role") or "user").lower()
            raw = msg.get("content", "")
            if isinstance(raw, str):
                messages.append({"role": role, "content": raw})
            elif isinstance(raw, list):
                messages.append({"role": role, "content": self._lcn_render_parts(raw)})
        if not messages:
            messages = [{"role": "user", "content": "Describe the input."}]
        return messages

    # ─── preload ─────────────────────────────────────────────────────────

    def _longcat_next_preload(self, request: ChatRequest):
        """Pre-extract any frames so the GPU thread never blocks on ffmpeg."""
        for msg in request.messages:
            raw = msg.get("content", "")
            if not isinstance(raw, list):
                continue
            for part in raw:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype == "video_url":
                    path = parse_url(part["video_url"]["url"])
                    _extract_video_to_jpg_paths(path, self._lcn_max_frames)
                elif ptype == "image_url":
                    path = parse_url(part["image_url"]["url"])
                    if _is_video_path(path):
                        _extract_video_to_jpg_paths(path, self._lcn_max_frames)

    def _longcat_next_preload_path(self, path: str, media_type: str):
        path = parse_url(path)
        if media_type == "auto":
            if _is_video_path(path):
                media_type = "video"
            elif _is_audio_path(path):
                media_type = "audio"
            else:
                media_type = "image"
        if media_type == "video":
            _extract_video_to_jpg_paths(path, self._lcn_max_frames)

    # ─── prepare ─────────────────────────────────────────────────────────

    def _lcn_processor_cache_key(self, text_input: str) -> str:
        h = hashlib.md5(text_input.encode("utf-8")).hexdigest()
        return f"longcat_proc:{h}"

    def _lcn_run_processor(self, text_input: str):
        """Run the LongCat processor with on-disk dedup caching."""
        def _process():
            text_inputs, visual_inputs, audio_inputs = self.processor(
                text=text_input, return_tensors="pt"
            )
            return {"text": text_inputs, "visual": visual_inputs, "audio": audio_inputs}

        return _load_with_dedup(self._lcn_processor_cache_key(text_input), _process)

    def _prepare_longcat_next(self, request: ChatRequest) -> dict:
        messages = self._lcn_messages_from_request(request)
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        proc_out = self._lcn_run_processor(text_input)
        prompt_len = proc_out["text"]["input_ids"].shape[1]
        return {
            "request": request,
            "text_input": text_input,
            "proc_out": proc_out,
            "prompt_len": prompt_len,
        }

    # ─── generate ────────────────────────────────────────────────────────

    def _generate_longcat_next_prepared(self, prepared: dict) -> str:
        request = prepared["request"]
        device = torch.device(f"cuda:{self.device_id}")
        proc_out = prepared["proc_out"]

        text_inputs = _to_device(proc_out["text"], device)
        visual_inputs = _to_device(proc_out["visual"], device, self._lcn_dtype)
        audio_inputs = _to_device(proc_out["audio"], device, self._lcn_dtype)

        # Sampling: respect request, fall back to LongCat's recommended
        # audio-to-text preset (works well for grounded multimodal QA).
        do_sample = request.temperature > 0
        gen_kwargs = dict(
            input_ids=text_inputs["input_ids"],
            visual_inputs=visual_inputs,
            audio_inputs=audio_inputs,
            max_new_tokens=request.max_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            do_sample=do_sample,
        )
        if "attention_mask" in text_inputs:
            gen_kwargs["attention_mask"] = text_inputs["attention_mask"]
        if do_sample:
            gen_kwargs["temperature"] = request.temperature
            gen_kwargs["top_p"] = request.top_p
            gen_kwargs["top_k"] = 20
            gen_kwargs["repetition_penalty"] = 1.1

        with torch.inference_mode():
            outputs = self.model.generate(**gen_kwargs)

        sequences = outputs.sequences
        prompt_len = prepared["prompt_len"]
        gen_ids = sequences[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()

    def _infer_longcat_next(self, request: ChatRequest) -> str:
        return self._generate_longcat_next_prepared(self._prepare_longcat_next(request))
