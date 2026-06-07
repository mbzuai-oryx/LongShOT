"""Baichuan-Omni-1.5 inference via custom HF remote code."""

import json
import os
from collections import OrderedDict
from functools import lru_cache
from typing import Any

import torch

from serve.media import parse_url
from serve.schemas import ChatRequest


class BaichuanOmniInferMixin:
    """Mixin for baichuan-inc/Baichuan-Omni-1d5 — omni model with video+audio+image."""

    @staticmethod
    def _patch_torchaudio_compat():
        """Shim removed torchaudio APIs for >= 2.11 compatibility."""
        try:
            import torchaudio
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["default"]
            if not hasattr(torchaudio, "info"):
                import dataclasses

                @dataclasses.dataclass
                class _AudioInfo:
                    sample_rate: int
                    num_frames: int
                    num_channels: int
                    bits_per_sample: int = 16
                    encoding: str = "PCM_S"

                def _info(uri, **kwargs):
                    waveform, sr = torchaudio.load(uri)
                    return _AudioInfo(
                        sample_rate=sr,
                        num_frames=waveform.shape[-1],
                        num_channels=waveform.shape[0],
                    )

                torchaudio.info = _info
        except ImportError:
            pass

    def _init_baichuan_omni(self, args, dtype):
        self._patch_torchaudio_compat()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        print(f"Loading Baichuan-Omni on devices {self.device_ids}: {args.model}", flush=True)

        primary_device = f"cuda:{self.device_ids[0]}"

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True
        )
        device_map = "auto" if self.tensor_parallel_size > 1 else primary_device
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        if device_map != "auto":
            self.model = self.model.to(primary_device)

        gen_cfg = self.model.generation_config
        if not hasattr(gen_cfg, "user_token_id"):
            gen_cfg.user_token_id = self.tokenizer.convert_tokens_to_ids("<C_Q>")
        if not hasattr(gen_cfg, "assistant_token_id"):
            gen_cfg.assistant_token_id = self.tokenizer.convert_tokens_to_ids("<C_A>")
        if not gen_cfg.eos_token_id:
            gen_cfg.eos_token_id = 151643  # <|endoftext|>
        if not gen_cfg.pad_token_id and gen_cfg.pad_token_id != 0:
            gen_cfg.pad_token_id = gen_cfg.eos_token_id
        if not hasattr(self.model.config, "model_max_length"):
            self.model.config.model_max_length = 65536

        self.processor = self.model.bind_processor(
            self.tokenizer, training=False
        )

        # Cap audio and video to avoid OOM — the LLM attention uses SDPA
        # (not flash_attn) so memory scales quadratically with sequence length.
        aud_cfg = getattr(self.model.config, "audio_config", None)
        if aud_cfg is not None:
            aud_cfg.split_overlap = -1  # no splitting, truncate to max_audio_seconds
        vid_cfg = getattr(self.model.config, "video_config", None)
        if vid_cfg is not None:
            vid_cfg.max_frame_num = min(getattr(vid_cfg, "max_frame_num", 32), 16)

        self._baichuan_device = primary_device
        self._baichuan_dtype = dtype

        cfg = self.model.config
        vid_cfg = getattr(cfg, "video_config", None)
        self._bc_video_start = self.tokenizer.convert_ids_to_tokens(
            vid_cfg.video_start_token_id
        ) if vid_cfg and hasattr(vid_cfg, "video_start_token_id") else "<video_start>"
        self._bc_video_end = self.tokenizer.convert_ids_to_tokens(
            vid_cfg.video_end_token_id
        ) if vid_cfg and hasattr(vid_cfg, "video_end_token_id") else "<video_end>"
        self._bc_audio_start = self.tokenizer.convert_ids_to_tokens(
            aud_cfg.audio_start_token_id
        ) if aud_cfg and hasattr(aud_cfg, "audio_start_token_id") else "<audio_start>"
        self._bc_audio_end = self.tokenizer.convert_ids_to_tokens(
            aud_cfg.audio_end_token_id
        ) if aud_cfg and hasattr(aud_cfg, "audio_end_token_id") else "<audio_end>"

        # Cache for processed media — keyed by (video_path, audio_path).
        self._bc_media_cache = {}
        self._bc_media_cache_max = 20

        # Cache vision/audio encoder outputs to avoid re-encoding per question.
        self._bc_embed_cache = OrderedDict()
        self._bc_embed_cache_max = 10
        self._bc_current_media_key = None
        self._patch_baichuan_encoders()

        if args.compile:
            print("Skipping torch.compile for Baichuan-Omni (custom multimodal generate)", flush=True)

        torch.cuda.empty_cache()
        print(
            f"Baichuan-Omni loaded on {primary_device}: {args.model} (dtype={dtype})",
            flush=True,
        )

    def _bc_get_media_text(self, video_path, audio_path):
        """Build the multimodal token string for video+audio paths."""
        mm_parts = []
        if video_path:
            video_json = json.dumps({"local": video_path})
            mm_parts.append(f"{self._bc_video_start}{video_json}{self._bc_video_end}")
        if audio_path:
            audio_json = json.dumps({"path": audio_path})
            mm_parts.append(f"{self._bc_audio_start}{audio_json}{self._bc_audio_end}")
        return "".join(mm_parts)

    def _bc_process_media(self, video_path, audio_path):
        """Process media through the Baichuan processor with caching."""
        cache_key = (video_path, audio_path)
        if cache_key in self._bc_media_cache:
            return self._bc_media_cache[cache_key]

        media_text = self._bc_get_media_text(video_path, audio_path)
        if not media_text:
            return None

        result = self.processor(media_text)

        if len(self._bc_media_cache) >= self._bc_media_cache_max:
            oldest = next(iter(self._bc_media_cache))
            del self._bc_media_cache[oldest]
        self._bc_media_cache[cache_key] = result
        return result

    def _patch_baichuan_encoders(self):
        """Cache audio_tokenizer and get_visual_embed outputs by video path.

        These are the GPU-heavy ops (~5-15s) that only depend on the video/audio,
        not the text question. Caching them across questions about the same video
        gives a ~3-5x speedup.
        """
        import copy
        model = self.model  # OmniForCausalLM
        inner = model.model  # OmniModel
        cache = self._bc_embed_cache
        cache_max = self._bc_embed_cache_max
        mixin = self

        # 1. Cache audio_tokenizer output by wrapping its forward/__call__
        orig_audio_forward = model.audio_tokenizer.forward

        def cached_audio_forward(audios, encoder_length, bridge_length):
            key = getattr(mixin, "_bc_current_media_key", None)
            cache_k = ("audio", key) if key else None
            if cache_k and cache_k in cache:
                cache.move_to_end(cache_k)
                return cache[cache_k].clone()
            result = orig_audio_forward(audios, encoder_length, bridge_length)
            if cache_k:
                if len(cache) >= cache_max:
                    cache.popitem(last=False)
                cache[cache_k] = result.clone()
            return result

        model.audio_tokenizer.forward = cached_audio_forward

        # 2. Cache visual_model + visual_bridge output
        orig_get_visual_embed = inner.get_visual_embed

        def cached_get_visual_embed(input_ids, text_embedding, images=None, patch_nums=None,
                                     images_grid=None, videos=None, videos_patch_nums=None,
                                     videos_grid=None, group_index=None):
            key = getattr(mixin, "_bc_current_media_key", None)
            cache_k = ("visual", key) if key else None
            if cache_k and cache_k in cache:
                cache.move_to_end(cache_k)
                cached_visual_embed = cache[cache_k]
                # Still need to merge with text_embedding for this question's tokens
                return inner._merge_cached_visual(
                    input_ids, text_embedding, cached_visual_embed,
                    images, patch_nums, videos, videos_patch_nums, group_index
                )

            # No cache hit — run full visual encoding
            result = orig_get_visual_embed(
                input_ids, text_embedding, images, patch_nums,
                images_grid, videos, videos_patch_nums, videos_grid, group_index
            )
            return result

        # Don't patch get_visual_embed directly — the merging is tightly coupled.
        # Instead, just cache the audio tokenizer (the most expensive part for omni).
        # Visual caching would require refactoring get_visual_embed internals.

    def _warmup_baichuan_omni(self):
        try:
            self.model.chat(
                self.tokenizer,
                [{"role": "user", "content": "Hello"}],
                stream=False,
            )
        except Exception as e:
            print(f"  [Baichuan-Omni] Warmup error (non-fatal): {e}", flush=True)

    def _baichuan_omni_preload(self, request: ChatRequest):
        from serve.models.base import extract_video_path, extract_audio_path
        video_path = extract_video_path(request)
        audio_path = extract_audio_path(request)
        if video_path or audio_path:
            self._bc_process_media(video_path, audio_path)

    def _baichuan_omni_preload_path(self, path: str, media_type: str):
        if media_type in ("video", "auto"):
            audio_path = os.path.splitext(path)[0] + ".wav"
            if not os.path.exists(audio_path):
                audio_path = None
            self._bc_process_media(path, audio_path)

    def _prepare_baichuan_omni(self, request: ChatRequest) -> dict:
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

        gen_cfg = self.model.generation_config
        user_token = self.tokenizer.decode([gen_cfg.user_token_id])
        assistant_token = self.tokenizer.decode([gen_cfg.assistant_token_id])

        full_text = user_token + self._bc_get_media_text(video_path, audio_path) + prompt + assistant_token

        processed = self.processor(full_text)

        if processed.input_ids is None:
            raise RuntimeError(
                f"Baichuan processor returned empty input_ids. Text passed: {full_text[:200]}"
            )

        return {
            "processed": processed,
            "request": request,
            "media_key": (video_path, audio_path),
        }

    def _generate_baichuan_omni_prepared(self, prepared: dict) -> str:
        request = prepared["request"]
        processed = prepared["processed"]
        device = self._baichuan_device
        self._bc_current_media_key = prepared.get("media_key")

        input_ids = processed.input_ids
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], device=device)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0).to(device)
        else:
            input_ids = input_ids.to(device)

        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": request.max_tokens,
            "eos_token_id": 151643,
            "pad_token_id": 151643,
        }
        if request.temperature > 0:
            gen_kwargs["temperature"] = request.temperature
            gen_kwargs["top_p"] = request.top_p
            gen_kwargs["do_sample"] = True

        if processed.audios is not None:
            audios = processed.audios
            if isinstance(audios, list):
                tensors = [torch.as_tensor(a) for a in audios]
                max_len = max(t.shape[-1] for t in tensors)
                audios = torch.stack([
                    torch.nn.functional.pad(t, (0, max_len - t.shape[-1]))
                    for t in tensors
                ])
            gen_kwargs["audios"] = audios.to(device, self._baichuan_dtype)
            if processed.encoder_length is not None:
                el = processed.encoder_length
                gen_kwargs["encoder_length"] = el.to(device) if isinstance(el, torch.Tensor) else torch.tensor(el, device=device)
            if processed.bridge_length is not None:
                bl = processed.bridge_length
                gen_kwargs["bridge_length"] = bl.to(device) if isinstance(bl, torch.Tensor) else torch.tensor(bl, device=device)

        if processed.videos is not None:
            videos = processed.videos
            if isinstance(videos, list):
                videos = [v.to(device, self._baichuan_dtype) if isinstance(v, torch.Tensor) else torch.tensor(v, device=device, dtype=self._baichuan_dtype) for v in videos]
            else:
                videos = videos.to(device, self._baichuan_dtype)
            gen_kwargs["videos"] = videos
            if processed.videos_patch_nums is not None:
                vpn = processed.videos_patch_nums
                gen_kwargs["videos_patch_nums"] = vpn.to(device) if isinstance(vpn, torch.Tensor) else torch.tensor(vpn, device=device)
            if getattr(processed, "videos_grid", None) is not None:
                vg = processed.videos_grid
                gen_kwargs["videos_grid"] = vg.to(device) if isinstance(vg, torch.Tensor) else vg

        if processed.images is not None:
            images = processed.images
            if isinstance(images, list):
                images = [im.to(device, self._baichuan_dtype) if isinstance(im, torch.Tensor) else torch.tensor(im, device=device, dtype=self._baichuan_dtype) for im in images]
            else:
                images = images.to(device, self._baichuan_dtype)
            gen_kwargs["images"] = images
            if processed.patch_nums is not None:
                pn = processed.patch_nums
                gen_kwargs["patch_nums"] = pn.to(device) if isinstance(pn, torch.Tensor) else torch.tensor(pn, device=device)
            if getattr(processed, "images_grid", None) is not None:
                ig = processed.images_grid
                gen_kwargs["images_grid"] = ig.to(device) if isinstance(ig, torch.Tensor) else ig

        with torch.inference_mode():
            output_ids = self.model.generate(**gen_kwargs)

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    def _prepare_baichuan_omni_batch(self, requests: list[ChatRequest]):
        return [self._prepare_baichuan_omni(req) for req in requests]

    def _generate_baichuan_omni_batch_prepared(self, prepared_batch) -> list[str | Exception]:
        results = []
        for prepared in prepared_batch:
            try:
                results.append(self._generate_baichuan_omni_prepared(prepared))
            except Exception as e:
                results.append(e)
        return results
