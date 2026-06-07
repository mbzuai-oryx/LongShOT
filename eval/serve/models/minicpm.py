"""MiniCPM-o / MiniCPM-V inference via model.chat()."""

import os

import numpy as np
import torch

from serve.schemas import ChatRequest
from serve.media import parse_url, load_audio, load_image, load_minicpm_video_content


class MiniCPMInferMixin:
    def _minicpm_max_inp_length(self) -> int:
        override = os.getenv("MINICPM_MAX_INP_LENGTH", "").strip()
        if override:
            try:
                value = int(override)
                if value > 0:
                    return value
            except ValueError:
                pass

        candidates = []
        config = getattr(self.model, "config", None)
        if config is not None:
            for attr in ("max_inp_length", "max_position_embeddings"):
                value = getattr(config, attr, None)
                if isinstance(value, int) and value > 0:
                    candidates.append(value)
            llm_config = getattr(config, "llm_config", None)
            if llm_config is not None:
                for attr in ("max_inp_length", "max_position_embeddings"):
                    value = getattr(llm_config, attr, None)
                    if isinstance(value, int) and value > 0:
                        candidates.append(value)

        return max(candidates) if candidates else 8192

    def _minicpm_find_audio_path(self, raw_content) -> str | None:
        if not isinstance(raw_content, list):
            return None
        for part in raw_content:
            if isinstance(part, dict) and part.get("type") == "audio_url":
                return parse_url(part["audio_url"]["url"])
        return None

    def _minicpm_video_content(self, video_path: str, audio_path: str | None = None):
        return load_minicpm_video_content(
            video_path,
            audio_path=audio_path,
            use_audio=bool(self.is_minicpm_omni and self.omni),
        )

    def _warmup_minicpm(self):
        """MiniCPM uses model.chat(), so generate()-based warmup is invalid."""
        chat_kwargs = dict(
            msgs=[{"role": "user", "content": ["warmup"]}],
            tokenizer=self.tokenizer,
            do_sample=False,
            max_new_tokens=2,
            max_slice_nums=1,
            use_image_id=False,
        )
        with torch.inference_mode(), torch.cuda.device(self.device_id):
            self.model.chat(**chat_kwargs)

    def _minicpm_preload(self, request: ChatRequest):
        for msg in request.messages:
            raw_content = msg.get("content", "")
            if not isinstance(raw_content, list):
                continue
            audio_path = self._minicpm_find_audio_path(raw_content)
            video_seen = False
            for part in raw_content:
                ptype = part.get("type", "")
                if ptype == "video_url":
                    video_seen = True
                    path = parse_url(part["video_url"]["url"])
                    self._minicpm_video_content(path, audio_path=audio_path)
                elif ptype == "image_url":
                    load_image(part["image_url"]["url"])
                elif ptype == "audio_url" and not video_seen:
                    load_audio(parse_url(part["audio_url"]["url"]))

    def _minicpm_preload_path(self, path: str, media_type: str):
        path = parse_url(path)
        if media_type == "video":
            audio_path = None
            if self.is_minicpm_omni and self.omni:
                candidate = path.rsplit(".", 1)[0] + ".wav"
                if candidate != path and os.path.exists(candidate):
                    audio_path = candidate
            self._minicpm_video_content(path, audio_path=audio_path)
        elif media_type == "audio":
            load_audio(path)
        elif media_type == "image":
            load_image(f"file://{path}")

    def _prepare_minicpm(self, request: ChatRequest) -> dict:
        """Prepare MiniCPM inputs on CPU for model.chat()."""
        msgs = []
        has_audio = False
        has_video = False
        for msg in request.messages:
            role = msg["role"]
            raw_content = msg.get("content", "")

            if isinstance(raw_content, str):
                msgs.append({"role": role, "content": [raw_content]})
                continue

            audio_path = self._minicpm_find_audio_path(raw_content)
            content_list = []
            video_has_audio = False
            for part in raw_content:
                ptype = part.get("type", "")
                if ptype == "text":
                    content_list.append(part.get("text", ""))
                elif ptype == "image_url":
                    url = part["image_url"]["url"]
                    content_list.append(load_image(url))
                elif ptype == "video_url":
                    has_video = True
                    path = parse_url(part["video_url"]["url"])
                    video_content = self._minicpm_video_content(path, audio_path=audio_path)
                    content_list.extend(video_content)
                    if any(isinstance(c, np.ndarray) for c in video_content):
                        has_audio = True
                        video_has_audio = True
                elif ptype == "audio_url":
                    if video_has_audio:
                        continue
                    path = parse_url(part["audio_url"]["url"])
                    audio_np = load_audio(path)
                    content_list.append(audio_np)
                    has_audio = True

            msgs.append({"role": role, "content": content_list})

        do_sample = request.temperature > 0
        chat_kwargs = dict(
            msgs=msgs,
            tokenizer=self.tokenizer,
            do_sample=do_sample,
            max_new_tokens=request.max_tokens,
            max_inp_length=self._minicpm_max_inp_length(),
            max_slice_nums=1 if has_video else 2,
            use_image_id=False,
            num_beams=1,
            use_cache=True,
        )
        if do_sample:
            chat_kwargs["temperature"] = request.temperature
            if request.top_p is not None:
                chat_kwargs["top_p"] = request.top_p

        if self.is_minicpm_omni and self.omni and has_audio:
            if "4_5" in self.model_name or "4.5" in self.model_name:
                chat_kwargs["omni_mode"] = True
            else:
                chat_kwargs["omni_input"] = True
            chat_kwargs["max_slice_nums"] = 1
            chat_kwargs["return_dict"] = True

        return {
            "chat_kwargs": chat_kwargs,
            "has_audio": has_audio,
            "has_video": has_video,
        }

    def _generate_minicpm_prepared(self, prepared: dict) -> str:
        """Run MiniCPM generation from prepared chat kwargs."""
        chat_kwargs = prepared["chat_kwargs"]
        has_audio = prepared["has_audio"]
        has_video = prepared.get("has_video", False)
        msgs = chat_kwargs["msgs"]
        original_max_slice_nums = chat_kwargs.get("max_slice_nums")

        print(
            f"  [GPU {self.device_id}] has_audio={has_audio}, has_video={has_video}, "
            f"omni={self.omni}, msgs={len(msgs)}, max_inp_length={chat_kwargs.get('max_inp_length')}",
            flush=True,
        )

        with torch.inference_mode(), torch.cuda.device(self.device_id):
            try:
                result = self.model.chat(**chat_kwargs)
            except (AssertionError, RuntimeError) as e:
                emsg = str(e).lower()
                if has_audio and (
                    isinstance(e, AssertionError)
                    or "audio" in emsg
                    or "sequence length" in emsg
                    or "indexing error" in emsg
                    or "sizes of tensors must match" in emsg
                    or "illegal memory" in emsg
                    or "cuda error" in emsg
                ):
                    print(f"  [GPU {self.device_id}] [FALLBACK] Omni failed ({e}), retrying vision-only", flush=True)
                    torch.cuda.empty_cache()
                    for m in msgs:
                        m["content"] = [c for c in m["content"] if not isinstance(c, np.ndarray)]
                    chat_kwargs["msgs"] = msgs
                    chat_kwargs.pop("omni_mode", None)
                    chat_kwargs.pop("omni_input", None)
                    chat_kwargs.pop("return_dict", None)
                    if has_video:
                        chat_kwargs["max_slice_nums"] = 1
                    elif original_max_slice_nums is not None:
                        chat_kwargs["max_slice_nums"] = original_max_slice_nums
                    result = self.model.chat(**chat_kwargs)
                else:
                    raise

        if isinstance(result, dict):
            return result.get("text", str(result))
        return str(result)

    def _infer_minicpm(self, request: ChatRequest) -> str:
        """Compatibility wrapper for direct inference."""
        return self._generate_minicpm_prepared(self._prepare_minicpm(request))
