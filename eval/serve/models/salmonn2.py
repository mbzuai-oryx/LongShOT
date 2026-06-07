"""SALMONN2+ support -- custom Qwen2.5-VL + Whisper audio encoder."""

import os
import sys
import math
import time
import tempfile
import copy

import numpy as np
import torch

from serve.schemas import ChatRequest
from serve.media import _load_with_dedup, parse_url


class Salmonn2InferMixin:
    _s2_shared: dict = {}

    @staticmethod
    def _ensure_salmonn2_importable():
        """Clone repo, mock liger_kernel, and shim missing transformers classes."""
        import sys, types, subprocess as sp

        # Mock liger_kernel (only used for training loss, not inference)
        if "liger_kernel" not in sys.modules:
            try:
                import liger_kernel  # noqa: F401
            except ImportError:
                mk = types.ModuleType("liger_kernel")
                mk_t = types.ModuleType("liger_kernel.transformers")
                mk_m = types.ModuleType("liger_kernel.transformers.model")
                mk_l = types.ModuleType("liger_kernel.transformers.model.loss_utils")
                mk_l.LigerForCausalLMLoss = lambda **kw: (_ for _ in ()).throw(
                    RuntimeError("liger_kernel not installed — training not supported"))
                mk_m.loss_utils = mk_l
                mk_t.model = mk_m
                mk.transformers = mk_t
                for n, m in [("liger_kernel", mk),
                             ("liger_kernel.transformers", mk_t),
                             ("liger_kernel.transformers.model", mk_m),
                             ("liger_kernel.transformers.model.loss_utils", mk_l)]:
                    sys.modules[n] = m

        # Shim classes/functions that SALMONN2+ imports but may have moved or
        # been removed across transformers versions (repo targets 4.51, env may
        # have 5.x where APIs were reorganised).
        from transformers import cache_utils
        for cls_name in ("SlidingWindowCache", "StaticCache", "EncoderDecoderCache"):
            if not hasattr(cache_utils, cls_name):
                setattr(cache_utils, cls_name, type(cls_name, (), {}))

        from transformers import modeling_rope_utils
        if not hasattr(modeling_rope_utils, "dynamic_rope_update"):
            modeling_rope_utils.dynamic_rope_update = lambda *a, **kw: None
        if not hasattr(modeling_rope_utils, "rope_config_validation"):
            modeling_rope_utils.rope_config_validation = lambda *a, **kw: None
        # 'default' rope type removed in 5.x — provide standard RoPE (no scaling)
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if "default" not in ROPE_INIT_FUNCTIONS:
            def _default_rope(config=None, device=None, seq_len=None, **kw):
                import torch as _t
                base = config.rope_theta
                prf = getattr(config, "partial_rotary_factor", 1.0)
                head_dim = config.hidden_size // config.num_attention_heads
                dim = int(head_dim * prf)
                inv_freq = 1.0 / (base ** (_t.arange(0, dim, 2, dtype=_t.int64,
                                                      device=device).float() / dim))
                return inv_freq, 1.0  # attention_factor = 1.0 (NOT base!)
            ROPE_INIT_FUNCTIONS["default"] = _default_rope

        from transformers import modeling_flash_attention_utils
        if not hasattr(modeling_flash_attention_utils, "flash_attn_supports_top_left_mask"):
            modeling_flash_attention_utils.flash_attn_supports_top_left_mask = lambda: False

        # BERT utilities: moved from modeling_utils → pytorch_utils in 5.x,
        # find_pruneable_heads_and_indices removed entirely (never used at inference).
        from transformers import modeling_utils as _mu
        if not hasattr(_mu, "apply_chunking_to_forward"):
            from transformers.pytorch_utils import apply_chunking_to_forward
            _mu.apply_chunking_to_forward = apply_chunking_to_forward
        if not hasattr(_mu, "prune_linear_layer"):
            try:
                from transformers.pytorch_utils import prune_linear_layer
                _mu.prune_linear_layer = prune_linear_layer
            except ImportError:
                _mu.prune_linear_layer = lambda *a, **kw: None
        if not hasattr(_mu, "find_pruneable_heads_and_indices"):
            def _find_pruneable(heads, n_heads, head_size, already_pruned):
                import torch as _t
                mask = _t.ones(n_heads, head_size)
                for h in already_pruned:
                    mask[h] = 0
                for h in heads:
                    mask[h] = 0
                mask = mask.view(-1).eq(1)
                return heads, mask.nonzero().squeeze()
            _mu.find_pruneable_heads_and_indices = _find_pruneable

        # transformers 5.x raises AttributeError for config attributes that
        # older versions defaulted.  SALMONN2+ (built for 4.51) accesses many
        # of these.  Provide sensible defaults for the full set.
        from transformers.configuration_utils import PretrainedConfig
        if not getattr(PretrainedConfig, "_salmonn2_patched", False):
            _CONFIG_DEFAULTS = {
                "pad_token_id": None, "bos_token_id": None, "eos_token_id": None,
                "decoder_start_token_id": None, "sep_token_id": None,
                "initializer_range": 0.02, "layer_norm_eps": 1e-6,
                "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0,
                "classifier_dropout": None, "chunk_size_feed_forward": 0,
                "add_cross_attention": False, "output_attentions": False,
                "output_hidden_states": False, "use_return_dict": True,
                "is_decoder": False, "tie_word_embeddings": True,
                "torchscript": False, "pruned_heads": {},
                "position_embedding_type": "absolute",
                "base_model_tp_plan": None, "base_model_pp_plan": None,
            }
            _orig_ga = PretrainedConfig.__getattribute__
            def _safe_ga(self, key):
                try:
                    return _orig_ga(self, key)
                except AttributeError:
                    if key in _CONFIG_DEFAULTS:
                        return _CONFIG_DEFAULTS[key]
                    raise
            PretrainedConfig.__getattribute__ = _safe_ga
            PretrainedConfig._salmonn2_patched = True

        # transformers 5.x sets all_tied_weights_keys in post_init(), but
        # SALMONN2+'s custom BertModel calls init_weights() → tie_weights()
        # during __init__ (before post_init runs).  Ensure the attr exists.
        from transformers.modeling_utils import PreTrainedModel
        if not getattr(PreTrainedModel, "_salmonn2_tie_patched", False):
            _orig_init_weights = PreTrainedModel.init_weights
            def _safe_init_weights(self_model):
                if not hasattr(self_model, "all_tied_weights_keys"):
                    self_model.all_tied_weights_keys = {}
                return _orig_init_weights(self_model)
            PreTrainedModel.init_weights = _safe_init_weights
            PreTrainedModel._salmonn2_tie_patched = True

        # get_head_mask removed in transformers 5.x; BERT Q-Former calls it
        if not hasattr(PreTrainedModel, "get_head_mask"):
            def _get_head_mask(self_model, head_mask, num_hidden_layers, is_attention_chunked=False):
                if head_mask is not None:
                    import torch as _t
                    head_mask = self_model._convert_head_mask_to_5d(head_mask, num_hidden_layers) \
                        if hasattr(self_model, "_convert_head_mask_to_5d") \
                        else head_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
                            num_hidden_layers, -1, -1, -1, -1)
                    if is_attention_chunked:
                        head_mask = head_mask.unsqueeze(-1)
                else:
                    head_mask = [None] * num_hidden_layers
                return head_mask
            PreTrainedModel.get_head_mask = _get_head_mask

        # Clone repo if not present
        repo_root = os.environ.get(
            "SALMONN2_REPO_PATH",
            "/tmp/video-salmonn2-repo/video_SALMONN2_plus",
        )
        if not os.path.exists(repo_root):
            parent = os.path.dirname(repo_root)
            print(f"Cloning video-SALMONN-2 repo to {parent} ...")
            sp.run(["git", "clone", "--depth", "1",
                    "https://github.com/bytedance/video-SALMONN-2.git", parent],
                   capture_output=True, timeout=300)

        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        return repo_root

    def _init_salmonn2(self, args, dtype):
        """Load video-SALMONN2+ model with its custom architecture."""
        import sys
        from transformers import AutoTokenizer, WhisperFeatureExtractor

        self._ensure_salmonn2_importable()
        from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus as SALMONN2Model

        # CUDA optimizations (idempotent)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        print(f"Loading SALMONN2+ model on GPU {self.device_id}: {args.model}", flush=True)
        t0 = time.monotonic()

        self.model = SALMONN2Model.from_pretrained(
            args.model, attn_implementation="sdpa", torch_dtype=dtype,
            device_map=f"cuda:{self.device_id}",
        )
        self.model.eval()
        # Sanity check: verify weights are real (not random)
        lm_norm = self.model.lm_head.weight.float().norm().item()
        emb_norm = self.model.model.embed_tokens.weight.float().norm().item()
        print(f"  from_pretrained: {time.monotonic()-t0:.1f}s, "
              f"lm_head.norm={lm_norm:.1f}, embed.norm={emb_norm:.1f}", flush=True)

        # Patch prepare_inputs_for_generation: transformers 5.x passes
        # cache_position=None on first call, but SALMONN2+ code does
        # cache_position[0] without a None guard.
        import types as _types
        _orig_prepare = self.model.prepare_inputs_for_generation.__func__
        def _safe_prepare(self_model, input_ids, cache_position=None, **kwargs):
            if cache_position is None:
                # Distinguish prefill (first call) from decode (subsequent calls)
                # by checking if KV cache already has content.
                past = kwargs.get("past_key_values", None)
                is_decode = past is not None and (
                    (hasattr(past, "get_seq_length") and past.get_seq_length() > 0)
                    or (isinstance(past, (list, tuple)) and len(past) > 0)
                )
                if is_decode:
                    seq_len = past.get_seq_length() if hasattr(past, "get_seq_length") else 1
                    cache_position = torch.tensor([seq_len], device=input_ids.device)
                else:
                    cache_position = torch.tensor([0], device=input_ids.device)
            return _orig_prepare(self_model, input_ids, cache_position=cache_position, **kwargs)
        self.model.prepare_inputs_for_generation = _types.MethodType(_safe_prepare, self.model)

        # Patch get_rope_index: SALMONN2+ mixes device=input_ids.device (GPU)
        # with bare torch.arange (CPU).  Force all computation on CPU, then
        # move results back to the model device.
        _orig_rope = self.model.get_rope_index.__func__
        def _safe_rope(self_model, input_ids=None, image_grid_thw=None,
                        video_grid_thw=None, audio_lengths=None,
                        second_per_grid_ts=None, attention_mask=None):
            dev = input_ids.device if input_ids is not None else torch.device("cpu")
            args_cpu = dict(
                input_ids=input_ids.cpu() if input_ids is not None else None,
                image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
                video_grid_thw=video_grid_thw.cpu() if video_grid_thw is not None else None,
                audio_lengths=audio_lengths,
                second_per_grid_ts=second_per_grid_ts.cpu() if isinstance(second_per_grid_ts, torch.Tensor) else second_per_grid_ts,
                attention_mask=attention_mask.cpu() if attention_mask is not None else None,
            )
            pos_ids, deltas = _orig_rope(self_model, **args_cpu)
            return pos_ids.to(dev), deltas.to(dev)
        self.model.get_rope_index = _types.MethodType(_safe_rope, self.model)

        # Reuse tokenizer / processors across replicas (read-only, thread-safe)
        shared = Salmonn2InferMixin._s2_shared
        if "tokenizer" not in shared:
            shared["tokenizer"] = AutoTokenizer.from_pretrained(
                args.model, model_max_length=131072, padding_side="right", use_fast=False,
            )
            # Video processor: use repo's custom processor on 4.x, standard on 5.x
            try:
                from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
                img_proc = Qwen2VLImageProcessorFast.from_pretrained(args.model)
                img_proc.max_pixels = 61250
                img_proc.min_pixels = 28 * 28 * 4
                if hasattr(img_proc, "size") and isinstance(img_proc.size, dict):
                    img_proc.size["longest_edge"] = 61250
                    img_proc.size["shortest_edge"] = 28 * 28 * 4
                shared["vid_processor"] = img_proc
                shared["vid_processor_type"] = "repo"
            except Exception:
                from transformers import Qwen2VLImageProcessor, Qwen2VLVideoProcessor, Qwen2VLProcessor
                img_proc = Qwen2VLImageProcessor(min_pixels=28 * 28 * 4, max_pixels=61250)
                vid_proc = Qwen2VLVideoProcessor(min_pixels=28 * 28 * 4, max_pixels=61250)
                shared["vid_processor"] = Qwen2VLProcessor(
                    image_processor=img_proc, tokenizer=shared["tokenizer"],
                    video_processor=vid_proc,
                )
                shared["vid_processor_type"] = "standard"
            shared["audio_proc"] = WhisperFeatureExtractor(
                feature_size=128, sampling_rate=16000, hop_length=160, chunk_length=30,
            )

        self.tokenizer = shared["tokenizer"]
        # Set chat template once (avoids deepcopy per request)
        self.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
        self.processor = None
        self._s2_vid_proc = shared["vid_processor"]
        self._s2_vid_proc_type = shared["vid_processor_type"]
        self._s2_audio_proc = shared["audio_proc"]

        vp = self._s2_vid_proc.video_processor if hasattr(self._s2_vid_proc, "video_processor") else self._s2_vid_proc
        self._s2_merge_size = getattr(vp, "merge_size", 2)
        self._s2_temporal_patch = getattr(vp, "temporal_patch_size", 2)
        # SALMONN2+ recommends 768 max frames but that OOMs on <100GB GPUs.
        # Default to 256 (safe for 80-95GB GPUs); override via --max-video-frames.
        self._s2_max_frames = args.max_video_frames or 256
        self._s2_min_frames = 16
        self._s2_base_interval = 0.1
        self._s2_max_frame_pixels = 61250

        if args.compile:
            print("Applying torch.compile (reduce-overhead) to SALMONN2+...", flush=True)
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("torch.compile applied — first inference will be slow (warmup)", flush=True)

        torch.cuda.empty_cache()
        print(f"SALMONN2+ loaded on GPU {self.device_id}: {args.model} "
              f"(total {time.monotonic()-t0:.1f}s)", flush=True)

    def _salmonn2_preload(self, request: ChatRequest):
        """Pre-process video+audio into cache from the media thread pool.

        Called by engine._preload_media so the GPU inference thread never
        blocks on cold video decoding or audio extraction.
        """
        from serve.models.base import extract_video_path
        video_path = extract_video_path(request)
        # Also extract audio_path if present
        audio_path = None
        for msg in request.messages:
            raw = msg.get("content", "")
            if not isinstance(raw, list):
                continue
            for part in raw:
                if isinstance(part, dict) and part.get("type") == "audio_url":
                    audio_path = parse_url(part["audio_url"]["url"])

        if video_path is None:
            return

        self._salmonn2_load_video_audio(video_path, audio_path)

    def _salmonn2_preload_path(self, path: str, media_type: str):
        """Preload a raw video path into the SALMONN2 media cache."""
        path = parse_url(path)
        ext = os.path.splitext(path)[1].lower()
        if media_type == "auto":
            media_type = "video" if ext in (".mp4", ".avi", ".mkv", ".mov", ".webm") else media_type
        if media_type == "video":
            self._salmonn2_load_video_audio(path, None)

    def _salmonn2_load_video_audio(self, video_path: str, audio_path: str | None):
        """Load and cache shared video/audio features for a video."""
        cache_key = f"salmonn2:{video_path}:{self._s2_max_frames}"

        def _load_video_audio():
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                vf = pool.submit(self._salmonn2_process_video, video_path)
                af = pool.submit(self._salmonn2_process_audio, video_path, audio_path)
                pv, gt, spgt = vf.result()
                audio_feat, al = af.result()
            return (pv, gt, spgt, audio_feat, al)

        return _load_with_dedup(cache_key, _load_video_audio)

    def _salmonn2_parse_request(self, request):
        """Extract video_path, audio_path, prompt_text from request."""
        from serve.models.base import extract_video_path
        video_path = extract_video_path(request)
        audio_path = None
        prompt_parts = []
        for msg in request.messages:
            raw = msg.get("content", "")
            if isinstance(raw, str):
                if msg.get("role") == "user":
                    prompt_parts.append(raw)
                continue
            if not isinstance(raw, list):
                continue
            for part in raw:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype == "text":
                    if msg.get("role") == "user":
                        prompt_parts.append(part.get("text", ""))
                elif ptype == "audio_url":
                    audio_path = parse_url(part["audio_url"]["url"])

        prompt_text = "\n".join(prompt_parts).strip() or "Describe the video."
        return video_path, audio_path, prompt_text, []

    def _prepare_salmonn2(self, request: ChatRequest):
        """Prepare SALMONN2 inputs on CPU."""
        video_path, audio_path, prompt_text, _ = self._salmonn2_parse_request(request)

        if video_path is None:
            return {"mode": "text", "request": request, "prompt_text": prompt_text}

        pixel_values, grid_thw, second_per_grid_ts, audio_feature, audio_lengths = \
            self._salmonn2_load_video_audio(video_path, audio_path)
        input_ids = self._salmonn2_tokenize(
            prompt_text, grid_thw, audio_lengths, second_per_grid_ts,
        )

        return {
            "mode": "single",
            "request": request,
            "video_path": video_path,
            "prompt_text": prompt_text,
            "input_ids": input_ids,
            "grid_thw": grid_thw,
            "pixel_values": pixel_values,
            "audio_feature": audio_feature,
            "audio_lengths": audio_lengths,
            "second_per_grid_ts": second_per_grid_ts,
        }

    def _prepare_salmonn2_batch(self, requests: list):
        """Prepare a same-video SALMONN2 batch on CPU."""
        if len(requests) <= 1:
            return [self._prepare_salmonn2(requests[0])] if requests else []

        parsed = [self._salmonn2_parse_request(req) for req in requests]
        video_paths = [p[0] for p in parsed]
        if len(set(v for v in video_paths if v)) != 1 or video_paths[0] is None:
            return [self._prepare_salmonn2(req) for req in requests]

        video_path = video_paths[0]
        audio_path = parsed[0][1]
        pixel_values, grid_thw, second_per_grid_ts, audio_feature, audio_lengths = \
            self._salmonn2_load_video_audio(video_path, audio_path)

        all_ids = []
        for _, _, prompt_text, _ in parsed:
            ids = self._salmonn2_tokenize(prompt_text, grid_thw, audio_lengths, second_per_grid_ts)
            all_ids.append(ids)

        batch_size = len(all_ids)
        max_len = max(ids.shape[0] for ids in all_ids)
        pad_id = self.tokenizer.pad_token_id or 0
        padded_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, ids in enumerate(all_ids):
            seq_len = ids.shape[0]
            offset = max_len - seq_len
            padded_ids[i, offset:] = ids
            attention_mask[i, offset:] = 1

        return {
            "mode": "batch",
            "requests": requests,
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "grid_thw_batch": (
                grid_thw.unsqueeze(0) if grid_thw.dim() == 1 else grid_thw
            ).expand(batch_size, -1).contiguous(),
            "pixel_values": pixel_values.repeat(batch_size, 1),
            "audio_feature": audio_feature.repeat(batch_size, 1) if audio_feature is not None else None,
            "audio_lengths": audio_lengths * batch_size if audio_lengths else None,
            "second_per_grid_ts": torch.tensor(second_per_grid_ts * batch_size),
            "pad_id": pad_id,
            "max_len": max_len,
        }

    def _generate_salmonn2_prepared(self, prepared) -> str:
        """Run SALMONN2 generation from prepared CPU tensors."""
        if prepared["mode"] == "text":
            return self._salmonn2_text_only(prepared["request"], prepared["prompt_text"])

        request = prepared["request"]
        device = self.model.device
        input_ids = prepared["input_ids"]
        input_ids_batch = input_ids.unsqueeze(0).to(device)
        grid_thw_batch = (
            prepared["grid_thw"].unsqueeze(0)
            if prepared["grid_thw"].dim() == 1 else prepared["grid_thw"]
        ).to(device)

        gen_kwargs = dict(max_new_tokens=request.max_tokens, do_sample=False)
        if request.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = request.temperature
            gen_kwargs["top_p"] = request.top_p

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids_batch,
                attention_mask=torch.ones_like(input_ids_batch),
                pixel_values_videos=prepared["pixel_values"].to(device, self.model.dtype),
                video_grid_thw=grid_thw_batch,
                audio_feature=prepared["audio_feature"].to(device, self.model.dtype)
                if prepared["audio_feature"] is not None else None,
                audio_lengths=prepared["audio_lengths"],
                second_per_grid_ts=torch.tensor(prepared["second_per_grid_ts"]),
                **gen_kwargs,
            )

        prompt_len = input_ids.shape[0]
        generated = outputs[0][prompt_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text.strip()

    def _generate_salmonn2_batch_prepared(self, prepared_batch) -> list[str | Exception]:
        """Run batched SALMONN2 generation from prepared CPU tensors."""
        if isinstance(prepared_batch, list):
            results = []
            for prepared in prepared_batch:
                try:
                    results.append(self._generate_salmonn2_prepared(prepared))
                except Exception as e:
                    results.append(e)
            return results

        prepared = prepared_batch
        requests = prepared["requests"]
        device = self.model.device
        batch_size = len(requests)
        gen_kwargs = dict(max_new_tokens=max(r.max_tokens for r in requests), do_sample=False)
        if requests and requests[0].temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = requests[0].temperature
            gen_kwargs["top_p"] = requests[0].top_p

        print(f"  [GPU {self.device_id}] [SALMONN2] BATCH {batch_size} requests, "
              f"max_len={prepared['max_len']}", flush=True)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=prepared["input_ids"].to(device),
                attention_mask=prepared["attention_mask"].to(device),
                pixel_values_videos=prepared["pixel_values"].to(device, self.model.dtype),
                video_grid_thw=prepared["grid_thw_batch"].to(device),
                audio_feature=prepared["audio_feature"].to(device, self.model.dtype)
                if prepared["audio_feature"] is not None else None,
                audio_lengths=prepared["audio_lengths"],
                second_per_grid_ts=prepared["second_per_grid_ts"],
                **gen_kwargs,
            )

        results = []
        pad_id = prepared["pad_id"]
        max_len = prepared["max_len"]
        for i in range(batch_size):
            generated = outputs[i][max_len:]
            if pad_id is not None:
                mask = generated != pad_id
                if mask.any():
                    generated = generated[:mask.nonzero()[-1].item() + 1]
                else:
                    generated = generated[:0]
            text = self.tokenizer.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            results.append(text.strip())
        return results

    def _infer_salmonn2(self, request: ChatRequest) -> str:
        """Compatibility wrapper for direct inference."""
        return self._generate_salmonn2_prepared(self._prepare_salmonn2(request))

    def _infer_salmonn2_batch(self, requests: list) -> list:
        """Compatibility wrapper for direct batch inference."""
        return self._generate_salmonn2_batch_prepared(self._prepare_salmonn2_batch(requests))

    def _salmonn2_text_only(self, request: ChatRequest, prompt_text: str) -> str:
        """Fallback for text-only requests."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ]
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )
        # apply_chat_template may return BatchEncoding (transformers 5.x) or tensor
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        input_ids = ids.to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(input_ids, max_new_tokens=request.max_tokens, do_sample=False)
        return self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

    def _salmonn2_process_video(self, video_path: str):
        """Extract video frames and process through Qwen2.5-VL video processor."""
        from decord import VideoReader, cpu as decord_cpu
        from PIL import Image

        vr = VideoReader(video_path, ctx=decord_cpu(0), num_threads=1)
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        video_length = total_frames / fps

        # Frame sampling — follow SALMONN2+ recommended strategy
        avg_fps = max(round(fps * self._s2_base_interval), 1)
        frame_idx = list(range(0, total_frames, avg_fps))

        if len(frame_idx) > self._s2_max_frames:
            frame_idx = np.linspace(0, total_frames - 1, self._s2_max_frames, dtype=int).tolist()
        elif len(frame_idx) < self._s2_min_frames:
            n = min(self._s2_min_frames, total_frames)
            frame_idx = np.linspace(0, total_frames - 1, n, dtype=int).tolist()

        # Read frames as NCHW numpy (repo processor) or PIL (standard processor)
        frames_np = vr.get_batch(frame_idx).asnumpy()  # NHWC
        actual_fps = len(frame_idx) / max(video_length, 0.01)

        if self._s2_vid_proc_type == "repo":
            # Repo's custom processor: takes NCHW numpy, handles temporal patching.
            # Save/restore pixel bounds instead of deepcopy (single-threaded executor).
            video_nchw = frames_np.transpose(0, 3, 1, 2)  # NHWC → NCHW
            proc = self._s2_vid_proc
            orig_max, orig_min = proc.max_pixels, proc.min_pixels
            orig_size = dict(proc.size) if hasattr(proc, "size") and isinstance(proc.size, dict) else None
            new_pixel = self._s2_max_frame_pixels
            if len(frame_idx) < self._s2_max_frames:
                new_pixel = int(0.95 * self._s2_max_frames / len(frame_idx) * new_pixel)
            proc.max_pixels = new_pixel
            proc.min_pixels = 28 * 28
            if orig_size is not None:
                proc.size["longest_edge"] = new_pixel
                proc.size["shortest_edge"] = 28 * 28
            try:
                result = proc.preprocess(images=None, videos=video_nchw, return_tensors="pt")
            finally:
                proc.max_pixels = orig_max
                proc.min_pixels = orig_min
                if orig_size is not None:
                    proc.size.update(orig_size)
            pixel_values = result["pixel_values_videos"]
            grid_thw = result["video_grid_thw"][0]
        else:
            # Standard transformers 5.x processor
            frames_pil = [Image.fromarray(f) for f in frames_np]
            result = self._s2_vid_proc(
                images=None, text="<video>", videos=[frames_pil], return_tensors="pt",
            )
            pixel_values = result["pixel_values_videos"]
            grid_thw = result["video_grid_thw"][0]

        second_per_grid_ts = [self._s2_temporal_patch / actual_fps]

        return pixel_values, grid_thw, second_per_grid_ts

    def _salmonn2_process_audio(self, video_path: str, audio_path: str | None = None):
        """Load or extract audio and create Whisper mel spectrograms.

        Priority: explicit audio_path (from request) -> pre-extracted .wav
        alongside video -> torchcodec -> ffmpeg+librosa extraction.
        """
        import copy

        audio_np = None

        # 1. Explicit audio path from request (e.g. :omni flag sends audio_url)
        if audio_path and os.path.exists(audio_path):
            try:
                import librosa
                audio_np, _ = librosa.load(audio_path, sr=16000, mono=True)
            except Exception as e:
                print(f"  [GPU {self.device_id}] [SALMONN2] audio_url load failed: {e}", flush=True)

        # 2. Pre-extracted .wav alongside the video file
        if audio_np is None:
            wav_path = os.path.splitext(video_path)[0] + ".wav"
            if os.path.exists(wav_path):
                try:
                    import librosa
                    audio_np, _ = librosa.load(wav_path, sr=16000, mono=True)
                except Exception as e:
                    print(f"  [GPU {self.device_id}] [SALMONN2] wav load failed ({wav_path}): {e}", flush=True)
            else:
                print(f"  [GPU {self.device_id}] [SALMONN2] no wav at {wav_path}", flush=True)

        # 3. torchcodec
        if audio_np is None:
            try:
                from torchcodec.decoders import AudioDecoder
                dec = AudioDecoder(video_path, sample_rate=16000, num_channels=1)
                audio_np = dec.get_all_samples().data.numpy().squeeze(0)
            except Exception as e:
                print(f"  [GPU {self.device_id}] [SALMONN2] torchcodec failed: {e}", flush=True)

        if audio_np is None:
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                import subprocess
                subprocess.run(
                    ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1",
                     "-acodec", "pcm_s16le", tmp.name],
                    capture_output=True, timeout=120,
                )
                import librosa
                audio_np, _ = librosa.load(tmp.name, sr=16000, mono=True)
                os.unlink(tmp.name)
            except Exception:
                pass

        if audio_np is None:
            print(f"  [GPU {self.device_id}] [SALMONN2] No audio extracted from {video_path}", flush=True)
            return None, None

        # Pad if < 1 second
        if audio_np.shape[0] < 16000:
            audio_np = np.pad(audio_np, (0, 16000 - audio_np.shape[0]))

        # Split into 30-second chunks → mel spectrograms
        sr = 16000
        chunk_samples = 30 * sr
        chunks = [audio_np[k:k + chunk_samples] for k in range(0, len(audio_np), chunk_samples)]

        # WhisperFeatureExtractor is stateless — no deepcopy needed
        spectrograms = [
            self._s2_audio_proc(c, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
            for c in chunks
        ]
        audio_feature = torch.stack(spectrograms, dim=0)

        # 60 audio tokens per 30s chunk (Q-Former with query_length=1, 0.5s window)
        audio_lengths = [math.ceil(len(audio_np) / chunk_samples) * 60]

        return audio_feature, audio_lengths

    def _salmonn2_tokenize(self, prompt_text: str, grid_thw, audio_lengths, second_per_grid_ts):
        """Build input_ids with interleaved <|video_pad|> and <|audio_pad|> tokens."""
        import copy

        merge = self._s2_merge_size

        # Ensure <video> placeholder exists
        if "<video>" not in prompt_text:
            prompt_text = "<video>\n" + prompt_text

        parts = prompt_text.split("<video>")
        t_dim = grid_thw[0].item()
        spatial = grid_thw[1].item() * grid_thw[2].item() // (merge ** 2)

        if audio_lengths is not None and audio_lengths[0] > 0:
            # Distribute audio tokens across temporal steps
            per_ts = self._split_into_groups(
                audio_lengths, [t_dim],
                [second_per_grid_ts[0]] if second_per_grid_ts else None,
            )
            new_parts = []
            for i in range(len(parts) - 1):
                new_parts.append(parts[i])
                repl = "<|vision_start|>"
                for t in range(t_dim):
                    repl += "<|video_pad|>" * spatial
                    repl += "<|audio_pad|>" * per_ts[0][t]
                repl += "<|vision_end|>"
                new_parts.append(repl)
            new_parts.append(parts[-1])
            content = "".join(new_parts)
        else:
            # Video only — no audio tokens
            new_parts = []
            for i in range(len(parts) - 1):
                new_parts.append(parts[i])
                repl = "<|vision_start|>" + "<|video_pad|>" * (t_dim * spatial) + "<|vision_end|>"
                new_parts.append(repl)
            new_parts.append(parts[-1])
            content = "".join(new_parts)

        # Apply chat template (tokenizer.chat_template set once in _init_salmonn2)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                  return_tensors="pt")
        # transformers 5.x returns BatchEncoding; extract input_ids tensor
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        if isinstance(ids, torch.Tensor):
            return ids.squeeze(0).long()
        return torch.tensor(ids, dtype=torch.long)

    @staticmethod
    def _split_into_groups(counts, groups, second_per_grid_ts=None):
        """Distribute audio token counts across temporal groups."""
        result = []
        if second_per_grid_ts is None:
            for count, g in zip(counts, groups):
                g = g.item() if hasattr(g, "item") else int(g)
                base, rem = divmod(count, g)
                gl = [base] * g
                if rem > 0:
                    step = g / rem
                    for i in range(1, rem + 1):
                        idx = min(math.floor(i * step) - 1, g - 1)
                        gl[idx] += 1
                result.append(gl)
        else:
            for count, g, sec in zip(counts, groups, second_per_grid_ts):
                g = g.item() if hasattr(g, "item") else int(g)
                frame_idx = (torch.arange(g) * sec * 2).long()
                per = torch.diff(frame_idx).tolist()
                remainder = count - sum(per)
                if remainder < 0:
                    print(f"  [SALMONN2] Warning: negative audio remainder ({remainder}), clamping to 0", flush=True)
                per.append(max(0, remainder))
                result.append(per)
        return result
