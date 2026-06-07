"""Video-RAG pipeline (open-ended variant).

Refactored from the upstream `vidrag_pipeline.py` script in
https://github.com/Leon1207/Video-RAG-master into a thread-safe class with
pooled VLMs across multiple GPUs.

Models match upstream defaults: LLaVA-Video-7B-Qwen2, Whisper-large,
CLIP-ViT-Large-336, Contriever. Only behavioural change: open-ended final
prompt (no MCQ-letter coercion).
"""

import base64
import copy
import io
import json
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ffmpeg
import numpy as np
import torch
import torchaudio
from decord import VideoReader, cpu
from PIL import Image


def _device_str(gpu_id: int) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{gpu_id}"


def process_video(video_path: str, max_frames_num: int, fps: int = 1, force_sample: bool = True):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "", 0.0
    vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    sample_fps = round(vr.get_avg_fps() / fps)
    frame_idx = list(range(0, len(vr), max(sample_fps, 1)))
    if len(frame_idx) > max_frames_num or force_sample:
        uniform = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform.tolist()
    frame_time = ",".join(f"{i / vr.get_avg_fps():.2f}s" for i in frame_idx)
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames, frame_time, video_time


RETRIEVE_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "To answer the question step by step, you can provide your retrieve request "
    "to assist you by the following json format:\n"
    "{{\n"
    '    "ASR": Optional[str]. Subtitles of the video relevant to the question, '
    "in two sentences. null if not needed.\n"
    '    "DET": Optional[list]. Up to five physical entities related to the '
    "question (no abstract concepts). null if not needed.\n"
    '    "TYPE": Optional[list]. Subset of [\'location\', \'number\', \'relation\']. '
    "null if not needed.\n"
    "}}\n"
    "Return only the JSON object."
)


FINAL_PROMPT_HEADER = (
    "You are given a video. Use the frames together with any auxiliary "
    "information below to answer the question with a concise, free-form "
    "answer. Do not output a multiple-choice letter unless the question "
    "itself requires one.\n"
)


@dataclass
class _VLMInstance:
    """One LLaVA-Video copy bound to a specific GPU."""
    tokenizer: Any
    model: Any
    image_processor: Any
    device: str


class VideoRAGPipeline:
    """Per-video Video-RAG inference pipeline with a multi-GPU VLM pool.

    `gpus.vlm` may be an int (single VLM) or list (one VLM per id, allowing
    duplicates to fit multiple copies on one GPU). Aux models are singletons
    on `gpus.aux`; concurrent access is serialized with a lock.
    """

    def __init__(self, config: Dict[str, Any], repo_path: Path):
        self.config = config
        self.repo_path = repo_path

        infer = config.get("inference", {})
        self.max_frames = int(infer.get("max_frames", 32))
        self.fps = int(infer.get("fps", 1))
        self.rag_threshold = float(infer.get("rag_threshold", 0.3))
        self.clip_threshold = float(infer.get("clip_threshold", 0.3))
        self.beta = float(infer.get("beta", 3.0))
        self.use_ocr = bool(infer.get("use_ocr", True))
        self.use_asr = bool(infer.get("use_asr", True))
        self.use_det = bool(infer.get("use_det", False))
        self.max_new_tokens = int(infer.get("max_new_tokens", 1024))

        gpus = config.get("gpus", {})
        vlm_spec: Union[int, List[int]] = gpus.get("vlm", 0)
        self.vlm_gpus: List[int] = (
            list(vlm_spec) if isinstance(vlm_spec, (list, tuple)) else [int(vlm_spec)]
        )
        self.aux_gpu = int(gpus.get("aux", max(self.vlm_gpus) + 1 if self.vlm_gpus else 1))
        self._aux_device = _device_str(self.aux_gpu)

        cache_dir = config.get("paths", {}).get("cache_dir", "./cache")
        self.audio_cache_dir = Path(cache_dir) / "audio"
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)

        # VLM pool — populated in setup_models().
        self._vlm_pool: "queue.Queue[_VLMInstance]" = queue.Queue()
        self._vlm_instances: List[_VLMInstance] = []

        # Aux singletons + their lock.
        self._aux_lock = threading.Lock()
        self._asr_model = None
        self._asr_processor = None
        self._clip_model = None
        self._clip_processor = None
        self._embed_tokenizer = None
        self._embed_model = None
        self._ocr_reader = None
        self._spacy_nlp = None

    @property
    def num_workers(self) -> int:
        return len(self.vlm_gpus)

    # ------------------------------------------------------------------ setup

    def setup_models(self) -> None:
        models_cfg = self.config.get("models", {})

        # ----- VLM pool: one LLaVA-Video copy per gpus.vlm entry.
        from llava.model.builder import load_pretrained_model

        vlm_id = models_cfg.get("vlm", "lmms-lab/LLaVA-Video-7B-Qwen2")
        for gpu_id in self.vlm_gpus:
            device = _device_str(gpu_id)
            print(f"[video_rag] loading VLM {vlm_id} on {device}")
            tokenizer, model, image_processor, _ = load_pretrained_model(
                vlm_id,
                None,
                "llava_qwen",
                torch_dtype="bfloat16",
                device_map=device,
                overwrite_config={},
            )
            # LLaVA-NeXT's builder hard-codes the vision tower to "cuda"
            # (i.e. cuda:0) when device_map != "auto" — see builder.py:293.
            # Force everything onto the target device.
            model.to(device)
            vision_tower = model.get_vision_tower()
            if vision_tower is not None:
                vision_tower.to(device=device, dtype=torch.bfloat16)
            model.eval()
            inst = _VLMInstance(tokenizer, model, image_processor, device)
            self._vlm_instances.append(inst)
            self._vlm_pool.put(inst)

        # ----- Whisper ASR.
        if self.use_asr:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            asr_id = models_cfg.get("asr", "openai/whisper-large")
            print(f"[video_rag] loading ASR {asr_id} on {self._aux_device}")
            self._asr_processor = WhisperProcessor.from_pretrained(asr_id)
            self._asr_model = WhisperForConditionalGeneration.from_pretrained(
                asr_id, torch_dtype=torch.float16
            ).to(self._aux_device)
            self._asr_model.eval()

        # ----- CLIP (DET branch frame filter).
        if self.use_det:
            from transformers import CLIPModel, CLIPProcessor

            clip_id = models_cfg.get("clip", "openai/clip-vit-large-patch14-336")
            print(f"[video_rag] loading CLIP {clip_id} on {self._aux_device}")
            self._clip_model = CLIPModel.from_pretrained(
                clip_id, torch_dtype=torch.float16
            ).to(self._aux_device)
            self._clip_processor = CLIPProcessor.from_pretrained(clip_id)
            self._clip_model.eval()

        # ----- Contriever for OCR/ASR retrieval.
        from transformers import AutoModel, AutoTokenizer

        embed_id = models_cfg.get("text_embed", "facebook/contriever")
        print(f"[video_rag] loading text embedder {embed_id} on {self._aux_device}")
        self._embed_tokenizer = AutoTokenizer.from_pretrained(embed_id)
        self._embed_model = AutoModel.from_pretrained(embed_id).to(self._aux_device)
        self._embed_model.eval()

        # ----- EasyOCR.
        if self.use_ocr:
            import easyocr

            print("[video_rag] loading EasyOCR")
            self._ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

        # ----- spacy keyword filter.
        try:
            import spacy

            self._spacy_nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"[video_rag] spacy unavailable ({e}); skipping keyword filter")
            self._spacy_nlp = None

    # ----------------------------------------------------------------- VLM I/O

    def _vlm_inference(self, vlm: _VLMInstance, prompt: str, video_tensor) -> str:
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        question = (DEFAULT_IMAGE_TOKEN + prompt) if video_tensor is not None else prompt
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt_question, vlm.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(vlm.device)
        )

        with torch.no_grad():
            cont = vlm.model.generate(
                input_ids,
                images=video_tensor,
                modalities=["video"] if video_tensor is not None else None,
                do_sample=False,
                temperature=0,
                max_new_tokens=self.max_new_tokens,
            )
        return vlm.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

    def _prepare_video_tensor(self, vlm: _VLMInstance, frames: np.ndarray):
        v = vlm.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        v = v.to(vlm.device).bfloat16()
        return [v]

    # --------------------------------------------------------------- helpers

    def _embed(self, texts: List[str]) -> np.ndarray:
        with self._aux_lock, torch.no_grad():
            inputs = self._embed_tokenizer(
                texts, return_tensors="pt", truncation=True, padding=True, max_length=512
            ).to(self._aux_device)
            out = self._embed_model(**inputs).last_hidden_state.mean(dim=1)
            out = torch.nn.functional.normalize(out, dim=1)
            return out.cpu().numpy()

    def _retrieve(self, docs: List[str], queries: List[str]) -> List[str]:
        if not docs:
            return []
        q_vecs = self._embed(queries)
        q = q_vecs.mean(axis=0)
        q = q / (np.linalg.norm(q) + 1e-9)
        d_vecs = self._embed(docs)
        sims = d_vecs @ q
        idx = [i for i, s in enumerate(sims) if s >= self.rag_threshold]
        return [docs[i] for i in idx]

    def _filter_keywords(self, kws: Optional[List[str]]) -> List[str]:
        if not kws:
            return []
        if self._spacy_nlp is None:
            return [k for k in kws if k and k != "video"]
        out = []
        for phrase in kws:
            doc = self._spacy_nlp(phrase)
            if len(doc) == 1:
                if doc[0].pos_ in ("NOUN", "ADJ", "VERB") and phrase != "video":
                    out.append(phrase)
            elif len(doc) == 2 and (
                (doc[0].pos_ == "ADJ" and doc[1].pos_ in ("NOUN", "PROPN"))
                or (doc[0].pos_ in ("NOUN", "PROPN") and doc[1].pos_ in ("NOUN", "PROPN"))
                or (doc[0].pos_ == "VERB" and doc[1].pos_ in ("NOUN", "PROPN"))
            ):
                if phrase != "video":
                    out.append(phrase)
            elif (
                len(doc) == 3
                and doc[0].pos_ == "ADJ"
                and doc[1].pos_ in ("NOUN", "PROPN")
                and doc[2].pos_ in ("NOUN", "PROPN")
            ):
                if phrase != "video":
                    out.append(phrase)
        return out

    # --------------------------------------------------------------------- ASR

    def _get_asr_docs(self, video_path: str) -> List[str]:
        stem = Path(video_path).stem
        cache_txt = self.audio_cache_dir / f"{stem}.txt"
        if cache_txt.exists():
            return [ln.strip() for ln in cache_txt.read_text().splitlines() if ln.strip()]

        # Cache miss: run Whisper. Lock since we share one ASR model across workers.
        with self._aux_lock:
            # Re-check inside the lock — another worker may have just filled it.
            if cache_txt.exists():
                return [ln.strip() for ln in cache_txt.read_text().splitlines() if ln.strip()]

            audio_path = self.audio_cache_dir / f"{stem}.wav"
            try:
                if not audio_path.exists():
                    (
                        ffmpeg.input(video_path)
                        .output(str(audio_path), acodec="pcm_s16le", ac=1, ar="16k")
                        .overwrite_output()
                        .run(quiet=True)
                    )
            except Exception as e:
                print(f"[video_rag] ffmpeg failed for {video_path}: {e}")
                return []

            try:
                speech, sr = torchaudio.load(str(audio_path))
                speech = speech.mean(dim=0)
                if sr != 16000:
                    speech = torchaudio.transforms.Resample(sr, 16000)(speech)
            except Exception as e:
                print(f"[video_rag] torchaudio.load failed: {e}")
                return []

            chunk_len = 30 * 16000
            transcripts: List[str] = []
            for i in range(0, len(speech), chunk_len):
                chunk = speech[i : i + chunk_len]
                try:
                    inputs = self._asr_processor(
                        chunk.numpy(), sampling_rate=16000, return_tensors="pt"
                    )
                    input_features = inputs["input_features"].to(self._aux_device, torch.float16)
                    with torch.no_grad():
                        pred_ids = self._asr_model.generate(
                            input_features, no_repeat_ngram_size=2, early_stopping=True
                        )
                    text = self._asr_processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
                    transcripts.append(text.strip())
                except Exception as e:
                    print(f"[video_rag] whisper chunk failed: {e}")

            cache_txt.write_text("\n".join(transcripts))
            return transcripts

    # --------------------------------------------------------------------- OCR

    def _get_ocr_docs(self, frames: np.ndarray) -> List[str]:
        seen: set = set()
        docs: List[str] = []
        with self._aux_lock:
            for img in frames:
                try:
                    results = self._ocr_reader.readtext(img)
                except Exception:
                    continue
                line = ""
                for _, text, conf in results:
                    if conf > 0.5 and text not in seen:
                        line += f"{text}; "
                        seen.add(text)
                if line:
                    docs.append(line)
        return docs

    # --------------------------------------------------------------------- DET

    def _get_det_top_idx(self, raw_video: List[np.ndarray], request_det: List[str]):
        clip_text = ["A picture of " + t for t in request_det] or ["A picture of object"]
        with self._aux_lock:
            feats = []
            for frame in raw_video:
                t = self._clip_processor(images=frame, return_tensors="pt")["pixel_values"]
                t = t.to(self._aux_device, dtype=torch.float16).squeeze(0)
                feats.append(t)
            video_tensor = torch.stack(feats, dim=0)

            clip_inputs = self._clip_processor(
                text=clip_text, return_tensors="pt", padding=True, truncation=True
            ).to(self._aux_device)
            with torch.no_grad():
                img_feats = self._clip_model.get_image_features(video_tensor)
                txt_feats = self._clip_model.get_text_features(**clip_inputs)
                sims = (img_feats @ txt_feats.T).squeeze(0).mean(1).cpu().numpy().astype(np.float64)
        alpha = self.beta * (len(sims) / 16.0)
        sims = sims * alpha / (np.sum(sims) + 1e-9)
        return [i for i, s in enumerate(sims) if s > self.clip_threshold]

    # --------------------------------------------------------------------- run

    def run(self, video_path: str, question: str) -> Dict[str, Any]:
        # Acquire one VLM for this sample's lifetime so both VLM calls hit the
        # same GPU (avoids cross-device tensor moves and keeps GPU caches warm).
        vlm = self._vlm_pool.get()
        try:
            frames, frame_time, video_time = process_video(
                video_path, self.max_frames, self.fps, force_sample=True
            )
            raw_video = [f for f in frames]
            video_tensor = self._prepare_video_tensor(vlm, frames)

            ocr_docs_total = self._get_ocr_docs(frames) if self.use_ocr else []
            asr_docs_total = self._get_asr_docs(video_path) if self.use_asr else []

            # Step 0: VLM (text-only) plans what to retrieve.
            retrieve_prompt = RETRIEVE_PROMPT_TEMPLATE.format(question=question)
            try:
                json_request_raw = self._vlm_inference(vlm, retrieve_prompt, None)
                request = self._extract_json(json_request_raw)
            except Exception as e:
                print(f"[video_rag] retrieve-plan call failed: {e}")
                request = {}

            det_kws_raw = request.get("DET") if isinstance(request.get("DET"), list) else []
            det_kws = self._filter_keywords(det_kws_raw)
            asr_query = request.get("ASR") if isinstance(request.get("ASR"), str) else None

            # Step 1a: DET branch (off unless USE_DET + APE service).
            det_blocks: List[str] = []
            if self.use_det and det_kws:
                try:
                    det_top_idx = self._get_det_top_idx(raw_video, det_kws)
                    if det_top_idx:
                        # APE service hookup intentionally omitted; keep wiring
                        # in place for when a det helper is added.
                        det_blocks = []
                except Exception as e:
                    print(f"[video_rag] DET branch failed: {e}")

            # Step 1b: OCR / ASR retrieval.
            ocr_hits: List[str] = []
            if self.use_ocr and ocr_docs_total:
                ocr_hits = self._retrieve(ocr_docs_total, [question] + det_kws)

            asr_hits: List[str] = []
            if self.use_asr and asr_docs_total:
                queries = [question] + ([asr_query] if asr_query else [])
                asr_hits = self._retrieve(asr_docs_total, queries)

            # Step 2: open-ended final prompt.
            aux_blocks: List[str] = []
            if det_blocks:
                aux_blocks.append(
                    f"Video has {self.max_frames} frames in total; detected objects per frame:\n"
                    + "\n".join(det_blocks)
                )
            if asr_hits:
                aux_blocks.append(
                    "Video Automatic Speech Recognition (chronological): " + " ".join(asr_hits)
                )
            if ocr_hits:
                aux_blocks.append("Video OCR (chronological): " + "; ".join(ocr_hits))

            final_prompt = FINAL_PROMPT_HEADER
            if aux_blocks:
                final_prompt += "\n".join(aux_blocks) + "\n"
            final_prompt += f"\nQuestion: {question}\nAnswer:"

            answer = self._vlm_inference(vlm, final_prompt, video_tensor)

            return {
                "answer": answer,
                "retrieve_request": request,
                "num_ocr_hits": len(ocr_hits),
                "num_asr_hits": len(asr_hits),
                "num_det_hits": len(det_blocks),
                "num_frames": len(frames),
                "video_time": video_time,
            }
        finally:
            self._vlm_pool.put(vlm)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return {}
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
