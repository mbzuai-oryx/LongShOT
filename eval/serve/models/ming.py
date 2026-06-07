"""Ming Flash Omni 2.0 inference via its custom HF remote code."""

import os
from typing import Any

import torch

from serve.media import _load_with_dedup, parse_url
from serve.schemas import ChatRequest


class MingFlashOmniInferMixin:
    """Mixin for inclusionAI/Ming-flash-omni-2.0."""

    _MING_GITHUB_REPO = "https://github.com/inclusionAI/Ming.git"
    _MING_GITHUB_BRANCH = "Ming_Flash_2.0_update_v0318"
    _MING_CLONE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ming-flash-omni-2.0")

    @staticmethod
    def _ensure_ming_repo() -> str:
        """Clone the GitHub fix branch (once) and return the local path."""
        import subprocess
        clone_dir = MingFlashOmniInferMixin._MING_CLONE_DIR
        if os.path.isdir(os.path.join(clone_dir, ".git")):
            print(f"  Ming repo already cloned: {clone_dir}", flush=True)
            return clone_dir
        print(f"  Cloning Ming fix branch into {clone_dir} ...", flush=True)
        subprocess.check_call([
            "git", "clone", "--single-branch",
            "-b", MingFlashOmniInferMixin._MING_GITHUB_BRANCH,
            "--depth", "1",
            MingFlashOmniInferMixin._MING_GITHUB_REPO,
            clone_dir,
        ])
        return clone_dir

    def _init_ming_flash_omni(self, args, dtype):
        """Load Ming Flash Omni with its custom model class."""
        from transformers import AutoConfig, AutoProcessor

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        print(f"Loading Ming Flash Omni on devices {self.device_ids}: {args.model}", flush=True)

        code_dir = self._ensure_ming_repo()

        import sys
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)

        self._ming_dtype = dtype
        self._ming_text_only = True
        self._ming_config = AutoConfig.from_pretrained(code_dir, trust_remote_code=True)
        self._ming_config._attn_implementation_internal = "sdpa"
        if hasattr(self._ming_config, "llm_config"):
            self._ming_config.llm_config._attn_implementation_internal = "sdpa"

        self.processor = AutoProcessor.from_pretrained(code_dir, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, "tokenizer", None)

        from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration as model_cls

        model_kwargs = dict(
            torch_dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            load_image_gen=False,
            load_talker=False,
        )

        if self.tensor_parallel_size > 1:
            max_mem = {dev: "90GiB" for dev in self.device_ids}
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = max_mem
        else:
            model_kwargs["device_map"] = {"": f"cuda:{self.device_id}"}

        self.model = model_cls.from_pretrained(args.model, **model_kwargs)
        self.model.eval()

        if args.compile:
            print("Skipping torch.compile for Ming Flash Omni (custom multimodal generate)", flush=True)

        torch.cuda.empty_cache()
        print(
            f"Ming Flash Omni loaded on devices {self.device_ids}: {args.model} "
            f"(attn=sdpa, tp={self.tensor_parallel_size}, text_only={self._ming_text_only})",
            flush=True,
        )

    def _warmup_ming_flash_omni(self):
        """Warmup: tokenizer/template only. Full multimodal warmup is expensive."""
        pass

    def _ming_role(self, role: str) -> str:
        role = (role or "user").lower()
        if role == "user":
            return "HUMAN"
        if role == "assistant":
            return "ASSISTANT"
        if role == "system":
            return "SYSTEM"
        return role.upper()

    def _ming_system_text(self, request: ChatRequest) -> str | None:
        """Extract system message text from the request."""
        for msg in request.messages:
            if (msg.get("role", "") or "").lower() == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
        return None

    def _ming_template_kwargs(self, request: ChatRequest) -> dict[str, Any]:
        extra = request.extra_body or {}
        template_kwargs = {}

        sys_text = self._ming_system_text(request)
        if sys_text:
            template_kwargs["sys_prompt_exp"] = sys_text
        if "sys_prompt_exp" in extra:
            template_kwargs["sys_prompt_exp"] = extra["sys_prompt_exp"]
        if "use_cot_system_prompt" in extra:
            template_kwargs["use_cot_system_prompt"] = bool(extra["use_cot_system_prompt"])
        return template_kwargs

    def _ming_messages_from_request(self, request: ChatRequest) -> list[dict[str, Any]]:
        """Convert an OpenAI-format request into Ming's chat format.

        The upstream processor's apply_chat_template only accepts HUMAN and
        ASSISTANT roles.  System messages are extracted separately and passed
        via the sys_prompt_exp kwarg instead.
        """
        messages = []
        for msg in request.messages:
            role = self._ming_role(msg.get("role", "user"))
            raw_content = msg.get("content", "")

            # Skip system messages — handled via _ming_template_kwargs
            if role == "SYSTEM":
                continue

            if isinstance(raw_content, str):
                messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": raw_content}],
                })
                continue

            content_parts = []
            for part in raw_content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype == "text":
                    content_parts.append({"type": "text", "text": part.get("text", "")})
                elif ptype == "image_url":
                    path = parse_url(part["image_url"]["url"])
                    ext = os.path.splitext(path)[1].lower()
                    if ext in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
                        content_parts.append({"type": "video", "video": path})
                    else:
                        content_parts.append({"type": "image", "image": path})
                elif ptype == "video_url":
                    content_parts.append({"type": "video", "video": parse_url(part["video_url"]["url"])})
                elif ptype == "audio_url":
                    content_parts.append({"type": "audio", "audio": parse_url(part["audio_url"]["url"])})

            if content_parts:
                messages.append({"role": role, "content": content_parts})

        if not messages:
            messages = [{
                "role": "HUMAN",
                "content": [{"type": "text", "text": "Describe the input."}],
            }]
        return messages

    def _ming_media_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract only media items for processor-side media loading."""
        media_messages = []
        for msg in messages:
            content = msg.get("content", [])
            media_parts = [part for part in content if part.get("type") != "text"]
            if media_parts:
                media_messages.append({"role": msg["role"], "content": media_parts})
        return media_messages

    def _normalize_ming_media_inputs(self, media_inputs):
        if isinstance(media_inputs, tuple) and len(media_inputs) == 3:
            return media_inputs
        if isinstance(media_inputs, tuple) and len(media_inputs) == 2:
            image_inputs, video_inputs = media_inputs
            return image_inputs, video_inputs, None
        raise ValueError(
            f"Unexpected process_vision_info return shape for {self.model_name}: {type(media_inputs)}"
        )

    def _ming_media_cache_key(self, request: ChatRequest) -> str | None:
        media_sig = self.media_signature(request)
        if media_sig == ("text",):
            return None
        return f"ming_media:{repr(media_sig)}"

    def _ming_get_media_inputs(
        self,
        request: ChatRequest,
        messages: list[dict[str, Any]] | None = None,
    ):
        """Load or reuse Ming processor media inputs for a request."""
        cache_key = self._ming_media_cache_key(request)
        if cache_key is None:
            return None, None, None

        messages = messages or self._ming_messages_from_request(request)
        media_messages = self._ming_media_messages(messages)
        if not media_messages:
            return None, None, None

        def _process():
            return self._normalize_ming_media_inputs(
                self.processor.process_vision_info(media_messages)
            )

        return _load_with_dedup(cache_key, _process)

    def _ming_repeat_media_inputs(self, media_inputs, batch_size: int):
        """Repeat shared media references so processor batching can align them."""
        if media_inputs is None:
            return None
        if isinstance(media_inputs, list):
            return media_inputs * batch_size
        if isinstance(media_inputs, tuple):
            return list(media_inputs) * batch_size
        return [media_inputs for _ in range(batch_size)]

    def _ming_has_media_inputs(self, media_inputs) -> bool:
        if media_inputs is None:
            return False
        try:
            return len(media_inputs) > 0
        except TypeError:
            return True

    def _ming_preload(self, request: ChatRequest):
        """Warm the cached processor media inputs for Ming."""
        self._ming_get_media_inputs(request)

    def _ming_preload_path(self, path: str, media_type: str):
        """Preload a raw media path into Ming's processor media cache."""
        path = parse_url(path)
        if media_type == "auto":
            ext = os.path.splitext(path)[1].lower()
            if ext in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
                media_type = "video"
            elif ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
                media_type = "audio"
            else:
                media_type = "image"

        part_key = {
            "video": {"type": "video", "video": path},
            "audio": {"type": "audio", "audio": path},
            "image": {"type": "image", "image": path},
        }.get(media_type)
        if part_key is None:
            return

        content = [part_key]
        audio_sig = ()
        if media_type == "video" and self.omni:
            audio_path = os.path.splitext(path)[0] + ".wav"
            if os.path.exists(audio_path):
                content.append({"type": "audio", "audio": audio_path})
                audio_sig = (audio_path,)

        messages = [{"role": "HUMAN", "content": content}]
        if media_type == "video":
            sig = ("video", path, audio_sig, self.max_video_frames or 0, bool(self.omni))
        elif media_type == "audio":
            sig = ("media", (), (path,))
        else:
            sig = ("media", (path,), ())
        cache_key = f"ming_media:{repr(sig)}"
        _load_with_dedup(
            cache_key,
            lambda: self._normalize_ming_media_inputs(
                self.processor.process_vision_info(messages)
            ),
        )

    def _prepare_ming_flash_omni(self, request: ChatRequest) -> dict:
        """Prepare Ming Flash Omni inputs on CPU."""
        messages = self._ming_messages_from_request(request)
        try:
            text = self.processor.apply_chat_template(messages, **self._ming_template_kwargs(request))
        except TypeError:
            text = self.processor.apply_chat_template(messages)

        image_inputs, video_inputs, audio_inputs = self._ming_get_media_inputs(request, messages)

        processor_kwargs = dict(
            text=[text],
            return_tensors="pt",
        )
        if self._ming_has_media_inputs(image_inputs):
            processor_kwargs["images"] = image_inputs
        if self._ming_has_media_inputs(video_inputs):
            processor_kwargs["videos"] = video_inputs
        if self._ming_has_media_inputs(audio_inputs):
            processor_kwargs["audios"] = audio_inputs

        try:
            inputs = self.processor(
                **processor_kwargs,
                audio_kwargs={"use_whisper_encoder": True},
            )
        except TypeError:
            inputs = self.processor(**processor_kwargs)

        for key, value in list(inputs.items()):
            if not isinstance(value, torch.Tensor):
                continue
            if key in {"pixel_values", "pixel_values_videos", "audio_feats"}:
                inputs[key] = value.to(dtype=dtype_for_tensor(value, self._ming_dtype))

        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        return {
            "inputs": inputs,
            "prompt_len": prompt_len,
            "request": request,
        }

    def _prepare_ming_flash_omni_batch(self, requests: list[ChatRequest]):
        """Prepare a same-media Ming batch on CPU."""
        if len(requests) <= 1:
            return [self._prepare_ming_flash_omni(requests[0])] if requests else []

        base_sig = self.media_signature(requests[0])
        if any(self.media_signature(req) != base_sig for req in requests[1:]):
            return [self._prepare_ming_flash_omni(req) for req in requests]

        messages_batch = [self._ming_messages_from_request(req) for req in requests]
        texts = []
        for req, messages in zip(requests, messages_batch):
            try:
                texts.append(self.processor.apply_chat_template(messages, **self._ming_template_kwargs(req)))
            except TypeError:
                texts.append(self.processor.apply_chat_template(messages))

        image_inputs, video_inputs, audio_inputs = self._ming_get_media_inputs(requests[0], messages_batch[0])
        processor_kwargs = dict(
            text=texts,
            return_tensors="pt",
        )
        if self._ming_has_media_inputs(image_inputs):
            processor_kwargs["images"] = self._ming_repeat_media_inputs(image_inputs, len(requests))
        if self._ming_has_media_inputs(video_inputs):
            processor_kwargs["videos"] = self._ming_repeat_media_inputs(video_inputs, len(requests))
        if self._ming_has_media_inputs(audio_inputs):
            processor_kwargs["audios"] = self._ming_repeat_media_inputs(audio_inputs, len(requests))

        try:
            try:
                inputs = self.processor(
                    **processor_kwargs,
                    audio_kwargs={"use_whisper_encoder": True},
                )
            except TypeError:
                inputs = self.processor(**processor_kwargs)
        except Exception:
            return [self._prepare_ming_flash_omni(req) for req in requests]

        for key, value in list(inputs.items()):
            if not isinstance(value, torch.Tensor):
                continue
            if key in {"pixel_values", "pixel_values_videos", "audio_feats"}:
                inputs[key] = value.to(dtype=dtype_for_tensor(value, self._ming_dtype))

        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        return {
            "mode": "batch",
            "request": requests[0],
            "requests": requests,
            "inputs": inputs,
            "prompt_len": prompt_len,
        }

    def _generate_ming_flash_omni_prepared(self, prepared: dict) -> str:
        """Run Ming Flash Omni generation from prepared CPU inputs."""
        request = prepared["request"]
        device = torch.device(f"cuda:{self.device_id}")
        inputs = {}
        for key, value in prepared["inputs"].items():
            if isinstance(value, torch.Tensor):
                if key in {"pixel_values", "pixel_values_videos", "audio_feats"}:
                    inputs[key] = value.to(device=device, dtype=dtype_for_tensor(value, self._ming_dtype))
                else:
                    inputs[key] = value.to(device=device)
            else:
                inputs[key] = value

        gen_kwargs = dict(
            max_new_tokens=request.max_tokens,
            use_cache=True,
            num_logits_to_keep=1,
            do_sample=request.temperature > 0,
        )
        gen_terminator = getattr(self.processor, "gen_terminator", None)
        if gen_terminator is not None:
            gen_kwargs["eos_token_id"] = gen_terminator
        if request.temperature > 0:
            gen_kwargs["temperature"] = request.temperature
            gen_kwargs["top_p"] = request.top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = prepared["prompt_len"]
        generated = output_ids[0][prompt_len:]
        text = self.processor.batch_decode(
            [generated],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text.strip()

    def _infer_ming_flash_omni(self, request: ChatRequest) -> str:
        """Compatibility wrapper for direct inference."""
        return self._generate_ming_flash_omni_prepared(self._prepare_ming_flash_omni(request))

    def _generate_ming_flash_omni_batch_prepared(self, prepared_batch) -> list[str | Exception]:
        """Run Ming Flash Omni batch generation from prepared CPU inputs."""
        if isinstance(prepared_batch, list):
            results = []
            for prepared in prepared_batch:
                try:
                    results.append(self._generate_ming_flash_omni_prepared(prepared))
                except Exception as e:
                    results.append(e)
            return results

        prepared = prepared_batch
        requests = prepared["requests"]
        device = torch.device(f"cuda:{self.device_id}")
        inputs = {}
        for key, value in prepared["inputs"].items():
            if isinstance(value, torch.Tensor):
                if key in {"pixel_values", "pixel_values_videos", "audio_feats"}:
                    inputs[key] = value.to(device=device, dtype=dtype_for_tensor(value, self._ming_dtype))
                else:
                    inputs[key] = value.to(device=device)
            else:
                inputs[key] = value

        gen_kwargs = dict(
            max_new_tokens=max(r.max_tokens for r in requests),
            use_cache=True,
            num_logits_to_keep=1,
            do_sample=requests[0].temperature > 0,
        )
        gen_terminator = getattr(self.processor, "gen_terminator", None)
        if gen_terminator is not None:
            gen_kwargs["eos_token_id"] = gen_terminator
        if requests[0].temperature > 0:
            gen_kwargs["temperature"] = requests[0].temperature
            gen_kwargs["top_p"] = requests[0].top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = prepared["prompt_len"]
        results = []
        for row in output_ids:
            text = self.processor.batch_decode(
                [row[prompt_len:]],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            results.append(text.strip())
        return results


def dtype_for_tensor(value: torch.Tensor, target_dtype: torch.dtype) -> torch.dtype:
    """Keep integer tensors intact while casting floating tensors."""
    return target_dtype if value.is_floating_point() else value.dtype
