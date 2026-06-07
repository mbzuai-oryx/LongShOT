"""ModelBackend — model replica or shard group pinned to specific GPU(s)."""

import os
from typing import Any

import torch

from serve.schemas import ChatRequest
from serve.media import (
    parse_url,
    load_audio,
    load_image,
    load_video_frames,
    load_video_omni_chunks,
)
from serve.models.minicpm import MiniCPMInferMixin
from serve.models.generic import GenericInferMixin
from serve.models.ming import MingFlashOmniInferMixin
from serve.models.longcat_next import LongCatNextInferMixin
from serve.models.salmonn2 import Salmonn2InferMixin
from serve.models.videollama3 import VideoLLaMA3InferMixin
from serve.models.tempo import TempoInferMixin
from serve.models.omnivinci import OmniVinciInferMixin
from serve.models.baichuan_omni import BaichuanOmniInferMixin


# ── Common video-path extraction ────────────────────────────────────────
# Shared by all model mixins to avoid duplicating the same parsing logic.

_VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".MP4")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def extract_video_path(request: ChatRequest) -> str | None:
    """Extract the video file path from an OpenAI-format ChatRequest.

    Handles three input patterns used by the eval pipeline:
      1. video_url  — direct video path
      2. image_url pointing to a video file (by extension)
      3. image_url list of pre-extracted frames — recovers original video
         from frame directory structure: {video_dir}/frames/{id}/f{N}/*.jpg
    """
    video_path = None
    frame_paths = []

    for msg in request.messages:
        raw = msg.get("content", "")
        if not isinstance(raw, list):
            continue
        for part in raw:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "video_url":
                video_path = parse_url(part["video_url"]["url"])
            elif ptype == "image_url":
                url = parse_url(part["image_url"]["url"])
                if any(url.endswith(e) for e in _VIDEO_EXTS):
                    video_path = url
                else:
                    frame_paths.append(url)

    # Recover video path from pre-extracted frame paths
    if video_path is None and frame_paths:
        frame_dir = os.path.dirname(frame_paths[0])
        video_id_dir = os.path.dirname(frame_dir)
        video_id = os.path.basename(video_id_dir)
        video_dir = os.path.dirname(os.path.dirname(video_id_dir))
        for ext in _VIDEO_EXTS:
            candidate = os.path.join(video_dir, video_id + ext)
            if os.path.exists(candidate):
                video_path = candidate
                break

    return video_path


def extract_audio_path(request: ChatRequest) -> str | None:
    """Extract the first audio path from an OpenAI-format ChatRequest."""
    for msg in request.messages:
        raw = msg.get("content", "")
        if not isinstance(raw, list):
            continue
        for part in raw:
            if isinstance(part, dict) and part.get("type") == "audio_url":
                return parse_url(part["audio_url"]["url"])
    return None


def extract_media_signature(request: ChatRequest) -> tuple[Any, ...]:
    """Build a stable media signature for batching and cache locality."""
    video_path = extract_video_path(request)
    images = []
    audios = []
    for msg in request.messages:
        raw = msg.get("content", "")
        if not isinstance(raw, list):
            continue
        for part in raw:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "image_url":
                images.append(parse_url(part["image_url"]["url"]))
            elif ptype == "audio_url":
                audios.append(parse_url(part["audio_url"]["url"]))

    if video_path is not None:
        return ("video", video_path, tuple(audios))
    if images or audios:
        return ("media", tuple(images), tuple(audios))
    return ("text",)


def iter_request_media(request: ChatRequest):
    """Yield normalized media entries from an OpenAI-format request."""
    for msg in request.messages:
        raw = msg.get("content", "")
        if not isinstance(raw, list):
            continue
        for part in raw:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "video_url":
                yield "video", parse_url(part["video_url"]["url"])
            elif ptype == "image_url":
                path = parse_url(part["image_url"]["url"])
                if any(path.endswith(ext) for ext in _VIDEO_EXTS):
                    yield "video", path
                else:
                    yield "image", path
            elif ptype == "audio_url":
                yield "audio", parse_url(part["audio_url"]["url"])


class ModelBackend(
    MiniCPMInferMixin,
    GenericInferMixin,
    MingFlashOmniInferMixin,
    LongCatNextInferMixin,
    Salmonn2InferMixin,
    VideoLLaMA3InferMixin,
    TempoInferMixin,
    OmniVinciInferMixin,
    BaichuanOmniInferMixin,
):
    """Wraps a HuggingFace model for inference, pinned to a specific GPU."""

    def __init__(self, args, device_ids: list[int] | None = None, force_single_device: bool = False):
        self.model_name = args.model
        _lower = args.model.lower().replace("_", "-")
        self.is_minicpm_omni = "minicpm-o" in _lower
        self.is_minicpm = "minicpm" in _lower
        self.is_ming_flash_omni = "ming-flash-omni" in _lower
        self.is_longcat_next = "longcat-next" in _lower
        self.is_salmonn2 = "salmonn2" in _lower.replace("-", "")
        self.is_videollama3 = "videollama3" in _lower.replace("-", "")
        self.is_tempo = "vision-cair/tempo" in _lower or _lower.endswith("/tempo-6b")
        self.is_omnivinci = "omnivinci" in _lower
        self.is_baichuan_omni = "baichuan-omni" in _lower
        self.omni = args.omni
        self.max_video_frames = args.max_video_frames
        self.device_ids = list(device_ids or [0])
        self.device_id = self.device_ids[0]
        self.tensor_parallel_size = len(self.device_ids)

        torch.cuda.set_device(self.device_id)

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map[args.dtype]

        if self.tensor_parallel_size > 1 and (
            self.is_minicpm or self.is_salmonn2 or self.is_videollama3 or self.is_tempo
        ):
            raise ValueError(
                f"{args.model} does not support tp={self.tensor_parallel_size} in transformers_serve; "
                "use replicas instead."
            )

        # Custom architectures — handle separately
        if self.is_salmonn2:
            self._init_salmonn2(args, dtype)
            return
        if self.is_videollama3:
            self._init_videollama3(args, dtype)
            return
        if self.is_tempo:
            self._init_tempo(args, dtype)
            return
        if self.is_ming_flash_omni:
            self._init_ming_flash_omni(args, dtype)
            return
        if self.is_longcat_next:
            self._init_longcat_next(args, dtype)
            return
        if self.is_omnivinci:
            self._init_omnivinci(args, dtype)
            return
        if self.is_baichuan_omni:
            self._init_baichuan_omni(args, dtype)
            return

        if self.tensor_parallel_size > 1:
            device_map = "auto"
        elif force_single_device or self.is_minicpm:
            device_map = "cuda"
        else:
            device_map = "cuda"

        # CUDA performance optimizations (generic, benefits all models)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        print(f"Loading model on devices {self.device_ids}: {args.model}")
        print(f"  dtype={args.dtype}, device_map={device_map}, omni={self.is_minicpm_omni}, tp={self.tensor_parallel_size}")

        # Some envs have a mismatched `flash_attn` wheel installed. We use
        # SDPA for this server, so force Transformers to treat FlashAttention
        # as unavailable before importing modeling code.
        import transformers.utils as hf_utils
        from transformers.utils import import_utils as hf_import_utils
        hf_utils.is_flash_attn_2_available = lambda: False
        hf_utils.is_flash_attn_greater_or_equal = lambda *args, **kwargs: False
        hf_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
        hf_utils.is_torchvision_available = lambda: False
        hf_import_utils.is_flash_attn_2_available = lambda: False
        hf_import_utils.is_flash_attn_greater_or_equal = lambda *args, **kwargs: False
        hf_import_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
        hf_import_utils.is_torchvision_available = lambda: False
        hf_import_utils._torchvision_available = False
        hf_import_utils._torchvision_version = "0.0"

        from transformers import AutoModel, AutoTokenizer

        if self.is_minicpm_omni:
            model_kwargs = dict(
                trust_remote_code=True,
                attn_implementation="sdpa",
                torch_dtype=dtype,
                init_vision=True,
                init_audio=True,
                init_tts=False,
            )
            if device_map != "cuda":
                model_kwargs["device_map"] = device_map
            self.model = AutoModel.from_pretrained(args.model, **model_kwargs)
            if device_map == "cuda":
                self.model = self.model.cuda()
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            self.processor = None
        elif self.is_minicpm:
            model_kwargs = dict(
                trust_remote_code=True,
                attn_implementation="sdpa",
                torch_dtype=dtype,
            )
            if device_map != "cuda":
                model_kwargs["device_map"] = device_map
            self.model = AutoModel.from_pretrained(args.model, **model_kwargs)
            if device_map == "cuda":
                self.model = self.model.cuda()
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            self.processor = None
        else:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            model_kwargs = dict(
                trust_remote_code=True,
                torch_dtype=dtype,
                attn_implementation="sdpa",
            )
            if device_map != "cuda":
                model_kwargs["device_map"] = device_map
            try:
                from transformers import AutoModelForImageTextToText
                self.model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
            except Exception:
                self.model = AutoModel.from_pretrained(args.model, **model_kwargs)
            if device_map == "cuda":
                self.model = self.model.cuda()
            self.model.eval()

        if args.compile and not self.is_minicpm:
            print("Applying torch.compile (reduce-overhead)...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("torch.compile applied — first inference will be slow (warmup)")
        elif args.compile and self.is_minicpm:
            print("Skipping torch.compile for MiniCPM (incompatible with model.chat())")

        torch.cuda.empty_cache()
        print(f"Model loaded on devices {self.device_ids}: {args.model}")

    @property
    def primary_device_id(self) -> int:
        return self.device_id

    def media_signature(self, request: ChatRequest) -> tuple[Any, ...]:
        """Stable signature for batching and media-cache affinity."""
        base_sig = extract_media_signature(request)
        if base_sig and base_sig[0] == "video":
            return base_sig + (self.effective_max_video_frames(), bool(self.omni))
        return base_sig

    def effective_max_video_frames(self) -> int:
        """Model-aware effective video sampling cap used for cache keys and preload."""
        if self.max_video_frames > 0:
            return self.max_video_frames
        return 0

    def batch_key(self, request: ChatRequest) -> tuple[Any, ...]:
        """Requests are batch-compatible only when media and generation params match."""
        return (
            self.media_signature(request),
            int(request.max_tokens),
            round(float(request.temperature), 4),
            round(float(request.top_p), 4),
        )

    def supports_batch_inference(self) -> bool:
        return self.is_salmonn2 or self.is_videollama3 or self.is_ming_flash_omni

    def warmup(self):
        """Run warmup passes to trigger JIT compilation, CUDA kernel caching,
        and tokenizer initialization.  Called once per replica at startup."""
        import time as _time
        torch.cuda.set_device(self.device_id)
        t0 = _time.monotonic()
        print(f"  [GPU {self.device_id}] Warming up...", flush=True)

        try:
            # 1. Tokenizer warmup — first call compiles templates, loads vocab
            if self.tokenizer is not None:
                self.tokenizer.encode("warmup", return_tensors="pt")
                if hasattr(self.tokenizer, "apply_chat_template"):
                    try:
                        self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": "warmup"}],
                            add_generation_prompt=True,
                        )
                    except Exception:
                        pass

            # 2. Generate warmup — triggers torch.compile, CUDA kernels,
            #    cuBLAS handle creation, KV cache allocation, sampling.
            if getattr(self, "is_videollama3", False):
                self._warmup_videollama3()
            elif getattr(self, "is_tempo", False):
                self._warmup_tempo()
            elif getattr(self, "is_ming_flash_omni", False):
                self._warmup_ming_flash_omni()
            elif getattr(self, "is_longcat_next", False):
                self._warmup_longcat_next()
            elif getattr(self, "is_minicpm", False):
                self._warmup_minicpm()
            elif getattr(self, "is_omnivinci", False):
                self._warmup_omnivinci()
            elif getattr(self, "is_baichuan_omni", False):
                self._warmup_baichuan_omni()
            else:
                dummy_ids = torch.tensor([[1, 2, 3]], device=f"cuda:{self.device_id}")
                with torch.inference_mode():
                    self.model.generate(dummy_ids, max_new_tokens=2, do_sample=False)

            # 3. CUDA sync to ensure all kernels are compiled
            torch.cuda.synchronize(self.device_id)

        except Exception as e:
            print(f"  [GPU {self.device_id}] Warmup error (non-fatal): {e}", flush=True)

        elapsed = _time.monotonic() - t0
        print(f"  [GPU {self.device_id}] Warmup complete ({elapsed:.1f}s)", flush=True)

    def infer(self, request: ChatRequest) -> str:
        """Run inference and return the assistant response text."""
        return self.generate_prepared(self.prepare(request))

    def preload_request(self, request: ChatRequest):
        """Pre-process media into cache from the media thread pool.

        Called by the engine's media pool so the GPU inference thread
        never blocks on cold video/audio decoding.  Model-specific
        mixins override _preload_* to cache processed tensors.
        """
        if self.is_salmonn2:
            self._salmonn2_preload(request)
        elif self.is_videollama3:
            self._videollama3_preload(request)
        elif self.is_tempo:
            self._tempo_preload(request)
        elif self.is_minicpm:
            self._minicpm_preload(request)
        elif self.is_ming_flash_omni:
            self._ming_preload(request)
        elif self.is_longcat_next:
            self._longcat_next_preload(request)
        elif self.is_omnivinci:
            self._omnivinci_preload(request)
        elif self.is_baichuan_omni:
            self._baichuan_omni_preload(request)
        else:
            self._preload_request_media(request)

    def _preload_request_media(self, request: ChatRequest):
        """Generic media warmup for models without a custom preload hook."""
        effective_max_frames = self.effective_max_video_frames()
        for media_type, path in iter_request_media(request):
            if media_type == "video":
                if self.is_minicpm_omni and self.omni:
                    load_video_omni_chunks(path, effective_max_frames)
                else:
                    load_video_frames(path, effective_max_frames)
            elif media_type == "audio":
                load_audio(path)
            elif media_type == "image":
                load_image(f"file://{path}")

    def preload_path(self, path: str, media_type: str):
        """Pre-load a single media path into the cache."""
        if self.is_salmonn2:
            return self._salmonn2_preload_path(path, media_type)
        if self.is_videollama3:
            return self._videollama3_preload_path(path, media_type)
        if self.is_tempo:
            return self._tempo_preload_path(path, media_type)
        if self.is_minicpm:
            return self._minicpm_preload_path(path, media_type)
        if self.is_ming_flash_omni:
            return self._ming_preload_path(path, media_type)
        if self.is_longcat_next:
            return self._longcat_next_preload_path(path, media_type)
        if self.is_omnivinci:
            return self._omnivinci_preload_path(path, media_type)
        if self.is_baichuan_omni:
            return self._baichuan_omni_preload_path(path, media_type)

        path = parse_url(path)
        if media_type == "auto":
            ext = os.path.splitext(path)[1].lower()
            if ext in _VIDEO_EXTS:
                media_type = "video"
            elif ext in _AUDIO_EXTS:
                media_type = "audio"
            elif ext in _IMAGE_EXTS:
                media_type = "image"

        effective_max_frames = self.effective_max_video_frames()
        if media_type == "video":
            if self.is_minicpm_omni and self.omni:
                load_video_omni_chunks(path, effective_max_frames)
            else:
                load_video_frames(path, effective_max_frames)
            if self.omni:
                audio_path = os.path.splitext(path)[0] + ".wav"
                if os.path.exists(audio_path):
                    load_audio(audio_path)
        elif media_type == "audio":
            load_audio(path)
        elif media_type == "image":
            load_image(f"file://{path}")

    def prepare(self, request: ChatRequest):
        """CPU-side request preparation before GPU generation."""
        if self.is_salmonn2:
            return self._prepare_salmonn2(request)
        if self.is_videollama3:
            return self._prepare_videollama3(request)
        if self.is_tempo:
            return self._prepare_tempo(request)
        if self.is_ming_flash_omni:
            return self._prepare_ming_flash_omni(request)
        if self.is_longcat_next:
            return self._prepare_longcat_next(request)
        if self.is_omnivinci:
            return self._prepare_omnivinci(request)
        if self.is_baichuan_omni:
            return self._prepare_baichuan_omni(request)
        if self.is_minicpm_omni or self.is_minicpm:
            return self._prepare_minicpm(request)
        return self._prepare_generic(request)

    def prepare_batch(self, requests: list[ChatRequest]):
        """CPU-side preparation for a keyed batch."""
        if self.is_salmonn2:
            return self._prepare_salmonn2_batch(requests)
        if self.is_videollama3:
            return self._prepare_videollama3_batch(requests)
        if self.is_ming_flash_omni:
            return self._prepare_ming_flash_omni_batch(requests)
        if self.is_omnivinci:
            return self._prepare_omnivinci_batch(requests)
        if self.is_baichuan_omni:
            return self._prepare_baichuan_omni_batch(requests)
        return [self.prepare(req) for req in requests]

    def generate_prepared(self, prepared) -> str:
        """GPU-side generation from a prepared request payload."""
        torch.cuda.set_device(self.device_id)
        if self.is_salmonn2:
            return self._generate_salmonn2_prepared(prepared)
        if self.is_videollama3:
            return self._generate_videollama3_prepared(prepared)
        if self.is_tempo:
            return self._generate_tempo_prepared(prepared)
        if self.is_ming_flash_omni:
            return self._generate_ming_flash_omni_prepared(prepared)
        if self.is_longcat_next:
            return self._generate_longcat_next_prepared(prepared)
        if self.is_omnivinci:
            return self._generate_omnivinci_prepared(prepared)
        if self.is_baichuan_omni:
            return self._generate_baichuan_omni_prepared(prepared)
        if self.is_minicpm_omni or self.is_minicpm:
            return self._generate_minicpm_prepared(prepared)
        return self._generate_generic_prepared(prepared)

    def generate_batch_prepared(self, prepared_batch) -> list[str | Exception]:
        """GPU-side generation from a prepared batch payload."""
        torch.cuda.set_device(self.device_id)
        if self.is_salmonn2:
            return self._generate_salmonn2_batch_prepared(prepared_batch)
        if self.is_videollama3:
            return self._generate_videollama3_batch_prepared(prepared_batch)
        if self.is_ming_flash_omni:
            return self._generate_ming_flash_omni_batch_prepared(prepared_batch)
        if self.is_omnivinci:
            return self._generate_omnivinci_batch_prepared(prepared_batch)
        if self.is_baichuan_omni:
            return self._generate_baichuan_omni_batch_prepared(prepared_batch)

        results = []
        for prepared in prepared_batch:
            try:
                results.append(self.generate_prepared(prepared))
            except Exception as e:
                results.append(e)
        return results

    def infer_batch(self, requests: list[ChatRequest]) -> list[str | Exception]:
        """Batch inference for multiple requests. Returns list of results.

        Models that support same-video batching use it when all requests
        share a video.  Default falls back to sequential inference.
        """
        prepared = self.prepare_batch(requests)
        return self.generate_batch_prepared(prepared)
