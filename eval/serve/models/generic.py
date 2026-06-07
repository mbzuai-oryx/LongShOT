"""Generic VLM inference via processor + model.generate()."""

import torch

from serve.schemas import ChatRequest
from serve.media import parse_url, load_image, load_video_frames, load_audio


class GenericInferMixin:
    def _prepare_generic(self, request: ChatRequest) -> dict:
        """Prepare generic VLM inputs on CPU.

        Uses structured content format for apply_chat_template (required by
        Qwen2.5-VL family and other modern VLMs to insert correct vision tokens).
        Falls back to flat text with <image>/<video> placeholders for older models.
        """
        images = []
        videos = []
        audios = []

        structured_messages = []
        flat_messages = []

        for msg in request.messages:
            role = msg["role"]
            raw_content = msg.get("content", "")

            if isinstance(raw_content, str):
                structured_messages.append({"role": role, "content": raw_content})
                flat_messages.append({"role": role, "content": raw_content})
                continue

            structured_parts = []
            flat_text_parts = []
            for part in raw_content:
                ptype = part.get("type", "")
                if ptype == "text":
                    text_val = part.get("text", "")
                    structured_parts.append({"type": "text", "text": text_val})
                    flat_text_parts.append(text_val)
                elif ptype == "image_url":
                    url = part["image_url"]["url"]
                    images.append(load_image(url))
                    structured_parts.append({"type": "image", "image": parse_url(url)})
                    flat_text_parts.append("<image>")
                elif ptype == "video_url":
                    path = parse_url(part["video_url"]["url"])
                    frames = load_video_frames(path, self.max_video_frames)
                    videos.append(frames)
                    structured_parts.append({"type": "video", "video": path})
                    flat_text_parts.append("<video>")
                elif ptype == "audio_url":
                    path = parse_url(part["audio_url"]["url"])
                    audios.append(load_audio(path))
                    structured_parts.append({"type": "text", "text": "<audio>"})
                    flat_text_parts.append("<audio>")

            structured_messages.append({"role": role, "content": structured_parts})
            flat_messages.append({"role": role, "content": " ".join(flat_text_parts)})

        try:
            text = self.processor.apply_chat_template(
                structured_messages, add_generation_prompt=True, tokenize=False
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            text = self.processor.apply_chat_template(
                flat_messages, add_generation_prompt=True, tokenize=False
            )

        proc_kwargs = dict(text=text, return_tensors="pt")
        if images:
            proc_kwargs["images"] = images
        if videos:
            proc_kwargs["videos"] = videos
        if audios:
            proc_kwargs["audios"] = audios

        inputs = self.processor(**proc_kwargs)
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        return {
            "inputs": inputs,
            "prompt_len": prompt_len,
            "request": request,
        }

    def _generate_generic_prepared(self, prepared: dict) -> str:
        """Run generic VLM generation from prepared CPU inputs."""
        request = prepared["request"]
        inputs = {
            k: v.to(self.model.device) if hasattr(v, "to") else v
            for k, v in prepared["inputs"].items()
        }

        gen_kwargs = dict(
            max_new_tokens=request.max_tokens,
            do_sample=request.temperature > 0,
        )
        if request.temperature > 0:
            gen_kwargs["temperature"] = request.temperature
            gen_kwargs["top_p"] = request.top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = prepared["prompt_len"]
        generated = output_ids[0][prompt_len:]
        text_out = self.processor.decode(generated, skip_special_tokens=True)
        return text_out.strip()

    def _infer_generic(self, request: ChatRequest) -> str:
        """Compatibility wrapper for direct inference."""
        return self._generate_generic_prepared(self._prepare_generic(request))
