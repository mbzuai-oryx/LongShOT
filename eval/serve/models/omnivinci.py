"""NVIDIA OmniVinci (VILA-based omni model) inference via custom HF remote code."""

import os
from collections import OrderedDict
from typing import Any

import torch

from serve.media import _load_with_dedup, parse_url, load_audio, load_video_frames
from serve.schemas import ChatRequest


_VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".MP4")
_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


class OmniVinciInferMixin:
    """Mixin for nvidia/omnivinci — VILA-based omni model with video+audio+image support."""

    @staticmethod
    def _patch_omnivinci_flash_attn() -> str:
        """Detect best available attention and patch OmniVinci's hardcoded flash_attention_2.

        The SiglipVisionTowerDynamicS2 class hardcodes attn_implementation="flash_attention_2".
        If flash_attn isn't installed, we patch the cached module source to use sdpa or eager.
        Returns the attn_implementation string to use for the top-level model.
        """
        has_flash_attn = False
        try:
            import flash_attn  # noqa: F401
            has_flash_attn = True
            print("  flash_attn available — using flash_attention_2", flush=True)
            attn_impl = "flash_attention_2"
        except ImportError:
            for fallback in ("sdpa", "eager"):
                try:
                    if fallback == "sdpa":
                        import torch.nn.functional as F
                        q = torch.randn(1, 1, 4, 8, device="cpu")
                        F.scaled_dot_product_attention(q, q, q)
                    attn_impl = fallback
                    break
                except Exception:
                    continue
            else:
                attn_impl = "eager"
            print(f"  flash_attn not installed — patching OmniVinci to use {attn_impl}", flush=True)

        import glob
        search_roots = [
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "modules",
                         "transformers_modules", "nvidia", "omnivinci", "*"),
            os.path.join("/cache", "hf", "modules",
                         "transformers_modules", "nvidia", "omnivinci", "*"),
        ]
        for root in search_roots:
            for fpath in glob.glob(os.path.join(root, "*.py")):
                try:
                    content = open(fpath).read()
                    changed = False

                    # Patch hardcoded flash_attention_2 references when FA is unavailable
                    if not has_flash_attn and 'attn_implementation="flash_attention_2"' in content:
                        content = content.replace(
                            'attn_implementation="flash_attention_2"',
                            f'attn_implementation="{attn_impl}"',
                        )
                        changed = True

                    # Newer transformers checks `_supports_flash_attn` (not the old
                    # `_supports_flash_attn_2`).  Add the new attribute if missing.
                    if "_supports_flash_attn_2 = True" in content and "_supports_flash_attn = True" not in content:
                        content = content.replace(
                            "_supports_flash_attn_2 = True",
                            "_supports_flash_attn_2 = True\n    _supports_flash_attn = True",
                        )
                        changed = True

                    if changed:
                        with open(fpath, "w") as f:
                            f.write(content)
                        print(f"  Patched {os.path.basename(fpath)}", flush=True)
                except Exception as e:
                    print(f"  Warning: failed to patch {fpath}: {e}", flush=True)

        return attn_impl

    @staticmethod
    def _shim_omnivinci_imports():
        """Patch transformers.modeling_utils for OmniVinci compat.

        OmniVinci's modeling_vila.py imports `no_init_weights` which was
        removed in transformers >=5.x.  Inject a lightweight replacement
        before the dynamic module is loaded.

        Also patches _finalize_model_loading to tolerate missing
        `all_tied_weights_keys` on submodels (e.g. MultimodalProjector)
        written for older transformers.
        """
        import transformers.modeling_utils as mu

        if not hasattr(mu, "no_init_weights"):
            from contextlib import contextmanager

            @contextmanager
            def no_init_weights(_enable=True):
                if not _enable:
                    yield
                    return
                old = {}
                for cls in [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]:
                    if hasattr(cls, "reset_parameters"):
                        old[cls] = cls.reset_parameters
                        cls.reset_parameters = lambda self: None
                try:
                    yield
                finally:
                    for cls, fn in old.items():
                        cls.reset_parameters = fn

            mu.no_init_weights = no_init_weights

        if not getattr(mu.PreTrainedModel, "_omnivinci_tied_keys_patched", False):
            _orig_finalize = mu.PreTrainedModel._finalize_model_loading

            @staticmethod
            def _safe_finalize(model, *args, **kwargs):
                if not hasattr(model, "all_tied_weights_keys"):
                    model.all_tied_weights_keys = {}
                return _orig_finalize(model, *args, **kwargs)

            mu.PreTrainedModel._finalize_model_loading = _safe_finalize
            mu.PreTrainedModel._omnivinci_tied_keys_patched = True

    def _init_omnivinci(self, args, dtype):
        """Load OmniVinci with its custom VILAForCausalLM model class.

        Uses trust_remote_code=True so transformers pulls the custom VILA
        modeling code directly from the HF-cached repo — no local clone needed.
        """
        from transformers import AutoConfig, AutoProcessor, AutoModel
        self._shim_omnivinci_imports()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        print(f"Loading OmniVinci on devices {self.device_ids}: {args.model}", flush=True)

        self._omnivinci_dtype = dtype

        num_video_frames = self.max_video_frames if self.max_video_frames > 0 else 128
        self._omnivinci_num_video_frames = num_video_frames

        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

        # Patch source files after config download ensures files exist,
        # but before AutoModel.from_pretrained imports the modeling module.
        attn_impl = self._patch_omnivinci_flash_attn()

        # Source-file patch may miss already-imported modules, so also patch
        # any loaded VILA classes directly.
        import sys
        for mod_name, mod in list(sys.modules.items()):
            if "omnivinci" not in mod_name or mod is None:
                continue
            for attr_name in dir(mod):
                cls = getattr(mod, attr_name, None)
                if isinstance(cls, type) and getattr(cls, "_supports_flash_attn_2", False) and not getattr(cls, "_supports_flash_attn", False):
                    cls._supports_flash_attn = True

        config.load_audio_in_video = True
        config.num_video_frames = num_video_frames
        config.audio_chunk_length = "max_3600"

        # Pin to a single GPU when possible — device_map="auto" splits layers
        # across GPUs via accelerate hooks, adding cross-device transfer overhead
        # on every forward pass. Only use "auto" for models too large for one GPU.
        primary_device = f"cuda:{self.device_ids[0]}"
        if self.tensor_parallel_size > 1:
            free_mem = torch.cuda.mem_get_info(self.device_ids[0])[0]
            device_map = "auto" if free_mem < 30 * 1024**3 else primary_device
        else:
            device_map = primary_device

        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            config=config,
            device_map=device_map,
            attn_implementation=attn_impl,
        )

        print(f"  device_map={device_map}", flush=True)
        self.model = AutoModel.from_pretrained(args.model, **model_kwargs)
        self.model.eval()

        # VILA loads sub-models (mm_projector, vision tower, audio encoder)
        # internally — device_map="cuda:0" only handles the main state_dict.
        # Force-move everything to the target GPU.
        target_device = torch.device(primary_device)
        self.model = self.model.to(target_device)

        # Ensure generation_config exists (may be missing after .to() on some
        # transformers versions or when loading replicas)
        if not hasattr(self.model, "generation_config") or self.model.generation_config is None:
            from transformers import GenerationConfig
            self.model.generation_config = GenerationConfig()

        self.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

        self.processor.config.load_audio_in_video = True
        self.processor.config.num_video_frames = num_video_frames
        self.processor.config.audio_chunk_length = "max_3600"

        self.model.config.load_audio_in_video = True
        self.model.config.num_video_frames = num_video_frames
        self.model.config.audio_chunk_length = "max_3600"

        self.tokenizer = getattr(self.processor, "tokenizer", None)

        self._omnivinci_gen_config = getattr(self.model, "default_generation_config", None) or self.model.generation_config
        self._omnivinci_gen_config.update(max_new_tokens=1024, max_length=99999999)

        # torch.compile is incompatible with accelerate dispatch hooks injected
        # by device_map="auto" — causes constant recompilation then eager fallback.
        # Only compile when the model lives on a single device (no accelerate hooks).
        uses_accelerate = hasattr(self.model, "hf_device_map") and len(set(
            v for v in self.model.hf_device_map.values() if isinstance(v, int)
        )) > 1
        if args.compile and not uses_accelerate:
            print("Applying torch.compile to OmniVinci LLM backbone...", flush=True)
            self.model.llm = torch.compile(self.model.llm, mode="reduce-overhead")
        elif args.compile:
            print(
                "Skipping torch.compile — model uses accelerate device_map across "
                f"{len(set(self.model.hf_device_map.values()))} devices (incompatible with compile)",
                flush=True,
            )

        # Cache the output of __embed_media_tokens (vision tower + audio encoder)
        # keyed by video path. This is the GPU-heavy part (~10-20s per video)
        # that's identical across questions about the same video.
        self._omnivinci_embed_cache = OrderedDict()
        self._omnivinci_embed_cache_max = 10
        self._patch_embed_with_cache()

        torch.cuda.empty_cache()
        print(
            f"OmniVinci loaded on devices {self.device_ids}: {args.model} "
            f"(dtype={dtype}, tp={self.tensor_parallel_size}, "
            f"num_video_frames={num_video_frames})",
            flush=True,
        )

    def _patch_embed_with_cache(self):
        """Cache encode_video output (vision tower + mm_projector) by video path.

        encode_video is the most expensive call (~10-20s per video), running
        SigLIP on all frames then the mm_projector. Its output depends only on
        the video frames, not the text question. Caching it across questions
        about the same video gives a major speedup.
        """
        import copy
        model = self.model
        embed_cache = self._omnivinci_embed_cache
        cache_max = self._omnivinci_embed_cache_max
        mixin = self

        orig_encode_video = model.encode_video

        def cached_encode_video(inp, block_sizes=None, mm_info=None, num_frames=None):
            cache_key = getattr(mixin, "_omnivinci_current_media_key", None)

            if cache_key and cache_key in embed_cache:
                embed_cache.move_to_end(cache_key)
                return copy.deepcopy(embed_cache[cache_key])

            result = orig_encode_video(inp, block_sizes=block_sizes, mm_info=mm_info, num_frames=num_frames)

            if cache_key:
                if len(embed_cache) >= cache_max:
                    embed_cache.popitem(last=False)
                embed_cache[cache_key] = copy.deepcopy(result)

            return result

        model.encode_video = cached_encode_video
        self._omnivinci_current_media_key = None

    def _warmup_omnivinci(self):
        """Warmup: run a minimal text-only forward pass."""
        try:
            conversation = [{
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
            }]
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor([text])
            input_ids = inputs.input_ids
            target_device = f"cuda:{self.device_ids[0]}"
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, device=target_device)
            elif input_ids.device.type != "cuda":
                input_ids = input_ids.to(target_device)

            gen_config = self._omnivinci_gen_config
            gen_config.update(max_new_tokens=2, max_length=99999999)

            with torch.inference_mode():
                self.model.generate(
                    input_ids=input_ids,
                    media=getattr(inputs, "media", None),
                    media_config=getattr(inputs, "media_config", None),
                    generation_config=gen_config,
                )
        except Exception as e:
            print(f"  [OmniVinci] Warmup error (non-fatal): {e}", flush=True)

    def _omnivinci_messages_from_request(self, request: ChatRequest) -> list[dict]:
        """Convert OpenAI-format request to OmniVinci conversation format.

        The VILA processor handles video frame extraction and audio extraction
        internally, so we always pass the original video file path — never
        pre-extracted frames.
        """
        from serve.models.base import extract_video_path, extract_audio_path

        video_path = extract_video_path(request)
        audio_path = extract_audio_path(request)

        text_parts = []
        for msg in request.messages:
            raw = msg.get("content", "")
            if isinstance(raw, str):
                text_parts.append(raw)
            elif isinstance(raw, list):
                for part in raw:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))

        prompt = " ".join(text_parts).strip() or "Describe the input."

        content = []
        if video_path:
            content.append({"type": "video", "video": video_path})
        elif audio_path:
            content.append({"type": "audio", "audio": audio_path})
        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def _omnivinci_preload(self, request: ChatRequest):
        """No-op: media embedding cache is populated on first generate call."""
        pass

    def _omnivinci_preload_path(self, path: str, media_type: str):
        """No-op: media embedding cache is populated on first generate call."""
        pass

    def _sync_omnivinci_config(self):
        """Ensure audio/video config is set on both model and processor before each call."""
        nf = self._omnivinci_num_video_frames
        for cfg in (self.model.config, self.processor.config):
            cfg.load_audio_in_video = True
            cfg.num_video_frames = nf
            cfg.audio_chunk_length = "max_3600"

    def _prepare_omnivinci(self, request: ChatRequest) -> dict:
        """CPU-side preparation: build processor inputs for OmniVinci."""
        self._sync_omnivinci_config()
        from serve.models.base import extract_video_path, extract_audio_path

        video_path = extract_video_path(request)
        audio_path = extract_audio_path(request)

        messages = self._omnivinci_messages_from_request(request)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor([text])

        return {
            "input_ids": inputs.input_ids,
            "media": getattr(inputs, "media", None),
            "media_config": getattr(inputs, "media_config", None),
            "media_key": (video_path, audio_path),
            "request": request,
        }

    def _generate_omnivinci_prepared(self, prepared: dict) -> str:
        """GPU-side generation from prepared OmniVinci inputs."""
        request = prepared["request"]

        # Set cache key so __embed_media_tokens can cache/reuse vision encoder output
        self._omnivinci_current_media_key = prepared.get("media_key")

        input_ids = prepared["input_ids"]
        target_device = f"cuda:{self.device_ids[0]}"
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, device=target_device)
        elif input_ids.device.type != "cuda":
            input_ids = input_ids.to(target_device)

        gen_config = self._omnivinci_gen_config
        gen_kwargs = dict(max_new_tokens=request.max_tokens, max_length=99999999)
        if request.temperature > 0:
            gen_kwargs["temperature"] = request.temperature
            gen_kwargs["top_p"] = request.top_p
            gen_kwargs["do_sample"] = True
        gen_config.update(**gen_kwargs)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                media=prepared["media"],
                media_config=prepared["media_config"],
                generation_config=gen_config,
            )

        text = self.processor.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

        prompt_text = self.processor.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )[0]
        if text.startswith(prompt_text):
            text = text[len(prompt_text):]

        return text.strip()

    def _prepare_omnivinci_batch(self, requests: list[ChatRequest]):
        """Prepare a batch of OmniVinci requests — sequential since the model
        processes media per-item through its custom processor."""
        return [self._prepare_omnivinci(req) for req in requests]

    def _generate_omnivinci_batch_prepared(self, prepared_batch) -> list[str | Exception]:
        """Generate from a batch of prepared OmniVinci items — sequential with
        cache clearing between items to manage VRAM on large video inputs."""
        results = []
        for prepared in prepared_batch:
            try:
                results.append(self._generate_omnivinci_prepared(prepared))
            except Exception as e:
                results.append(e)
        return results
