"""VideoMind agent adapter — Chain-of-LoRA video reasoning.

Wraps the VideoMind pipeline (yeliudev/VideoMind) behind the BaseAgent
interface.  A single Qwen2-VL model is loaded once with multiple LoRA
adapters (planner, grounder, verifier, answerer) that are switched at
inference time.

Multi-GPU: loads one independent model replica per GPU and pools them
via a thread-safe queue.  Each worker thread acquires a replica for
the duration of one sample, then releases it.  Video I/O runs outside
the pool acquire for concurrent prefetch.

Requires: transformers==4.45.2 (pinned for VideoMind compat).
"""

import json
import os
import queue
import sys
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from agents.base import AgentResponse, BaseAgent
from agents.config import AGENTS_DIR
from agents.registry import register_agent


def _set_torch_perf_flags():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class _Replica:
    """One model instance pinned to a specific GPU."""
    model: Any
    processor: Any
    device: torch.device
    adapter_state: Dict[str, bool]


@register_agent("videomind")
class VideoMindAgent(BaseAgent):
    """VideoMind: Chain-of-LoRA planner/grounder/verifier/answerer."""

    name = "videomind"
    repo_url = "https://github.com/yeliudev/VideoMind"

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self._repo_path: Optional[Path] = None
        self._pool: Optional[queue.Queue] = None
        self._num_replicas: int = 0

    # ------------------------------------------------------------------
    # BaseAgent lifecycle
    # ------------------------------------------------------------------

    def setup(self, config: Dict[str, Any]) -> None:
        self.config = config
        repo_rel = config.get("repo", {}).get("path", "repos/VideoMind")
        self._repo_path = AGENTS_DIR / repo_rel
        if not self._repo_path.exists():
            raise FileNotFoundError(
                f"VideoMind repo not found at {self._repo_path}. "
                f"Clone with: git clone https://github.com/yeliudev/VideoMind {self._repo_path}"
            )

        repo_str = str(self._repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def _resolve_model_paths(self):
        """Download HF models and return (videomind_path, vm_config)."""
        from huggingface_hub import snapshot_download
        # Import model module first to register the agent_qwen2_vl config type
        import videomind.model.model  # noqa: F401
        from transformers import AutoConfig

        models_cfg = self.config.get("models", {})
        videomind_path = models_cfg.get("videomind", "yeliudev/VideoMind-7B")

        if not os.path.isdir(videomind_path):
            print(f"[VideoMind] Downloading {videomind_path} from HuggingFace Hub...")
            videomind_path = snapshot_download(videomind_path)

        _BASE_MODEL_MAP = {
            "Qwen2-VL-7B-Instruct": "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen2-VL-2B-Instruct": "Qwen/Qwen2-VL-2B-Instruct",
        }
        vm_config = AutoConfig.from_pretrained(videomind_path)
        base_path = getattr(vm_config, "base_model_path", None)
        if base_path and not os.path.isdir(base_path):
            base_name = os.path.basename(base_path)
            hf_id = _BASE_MODEL_MAP.get(base_name, f"Qwen/{base_name}")
            print(f"[VideoMind] Downloading base model {hf_id}...")
            local_base = snapshot_download(hf_id)
            vm_config.base_model_path = local_base

        return videomind_path, vm_config

    def _load_replica(self, videomind_path: str, vm_config, gpu_id: int) -> _Replica:
        """Load one full model replica on a specific GPU."""
        import nncore
        from videomind.model.builder import build_model

        dtype_str = self.config.get("dtype", "float16")
        dtype = getattr(torch, dtype_str, torch.float16)

        device = f"cuda:{gpu_id}"
        print(f"[VideoMind] Loading replica on {device}...")
        model, processor = build_model(
            videomind_path, config=vm_config, device=device, dtype=dtype
        )

        adapter_state = {"planner": False, "verifier": False, "answerer": False}
        for role in ("planner", "verifier", "answerer"):
            adapter_path = nncore.join(videomind_path, role)
            if nncore.is_dir(adapter_path):
                model.load_adapter(adapter_path, adapter_name=role)
                adapter_state[role] = True

        return _Replica(
            model=model,
            processor=processor,
            device=torch.device(device),
            adapter_state=adapter_state,
        )

    def start_servers(self) -> None:
        """Load model replicas across available GPUs."""
        _set_torch_perf_flags()

        videomind_path, vm_config = self._resolve_model_paths()

        gpu_ids = self.config.get("gpus", None)
        if gpu_ids is None:
            n_visible = torch.cuda.device_count()
            gpu_ids = list(range(n_visible))
        elif isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]

        self._num_replicas = len(gpu_ids)
        self._pool = queue.Queue()

        for gid in gpu_ids:
            replica = self._load_replica(videomind_path, vm_config, gid)
            self._pool.put(replica)

        print(f"[VideoMind] Ready. {self._num_replicas} replicas on GPUs {gpu_ids}. "
              f"Adapters: {replica.adapter_state}")

    def stop_servers(self) -> None:
        if self._pool is not None:
            while not self._pool.empty():
                try:
                    replica = self._pool.get_nowait()
                    del replica.model
                    del replica.processor
                except queue.Empty:
                    break
            self._pool = None
        self._num_replicas = 0
        torch.cuda.empty_cache()

    @property
    def recommended_concurrency(self) -> Optional[int]:
        return max(self._num_replicas * 3, 4)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _prefetch_video(self, video_path: str) -> float:
        """Decode video frames OUTSIDE the GPU pool so ffmpeg overlaps
        with another sample's model inference."""
        from videomind.dataset.utils import process_vision_info
        from videomind.utils.io import get_duration

        duration = get_duration(video_path)

        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path,
                 "min_pixels": 36 * 28 * 28, "max_pixels": 64 * 28 * 28,
                 "max_frames": 150, "fps": 1.0},
                {"type": "text", "text": "prefetch"},
            ],
        }]
        try:
            process_vision_info(messages)
        except Exception:
            pass

        return duration

    def process_sample(
        self,
        video_path: str,
        question: str,
        subtitles: Optional[str] = None,
    ) -> AgentResponse:
        duration = self._prefetch_video(video_path)

        replica = self._pool.get()
        try:
            return self._run_pipeline(replica, video_path, question, subtitles, duration)
        finally:
            self._pool.put(replica)

    @torch.inference_mode()
    def _run_pipeline(
        self,
        r: _Replica,
        video_path: str,
        question: str,
        subtitles: Optional[str] = None,
        duration: float = 0,
    ) -> AgentResponse:
        from videomind.utils.io import get_duration
        from videomind.utils.parser import parse_query, parse_span

        inf_cfg = self.config.get("inference", {})
        if duration <= 0:
            duration = get_duration(video_path)
        reasoning_steps: List[Dict[str, Any]] = []
        metadata: Dict[str, Any] = {"duration": duration, "agents": []}

        use_planner = (
            r.adapter_state.get("planner")
            and (inf_cfg.get("auto_planning") or inf_cfg.get("auto_rephrasing"))
        )

        do_grounding = True
        query = question

        # ---- Planner ----
        if use_planner:
            planner_resp = self._run_planner(r, video_path, question)
            reasoning_steps.append({"role": "planner", "response": planner_resp})
            metadata["agents"].append("planner")

            try:
                parsed = json.loads(planner_resp)
                action = parsed[0] if isinstance(parsed, list) else parsed
                if (
                    inf_cfg.get("auto_rephrasing")
                    and action["type"].lower() == "grounder"
                    and action.get("value")
                ):
                    query = action["value"]
                elif inf_cfg.get("auto_planning") and action["type"].lower() == "answerer":
                    do_grounding = False
            except Exception:
                pass

        pred = [[0, duration]]
        conf = [0.0]

        # ---- Grounder ----
        if do_grounding:
            query = parse_query(query)
            grounder_resp, pred, conf = self._run_grounder(r, video_path, query, duration)
            reasoning_steps.append({"role": "grounder", "pred": pred[:3]})
            metadata["agents"].append("grounder")

            # ---- Verifier ----
            if (
                r.adapter_state.get("verifier")
                and inf_cfg.get("use_verifier", True)
                and len(pred) > 1
            ):
                pred, conf, probs, ranks = self._run_verifier(
                    r, video_path, question, pred, conf, duration
                )
                metadata["agents"].append("verifier")

        # ---- Answerer ----
        selected = pred[0] if pred else [0, duration]
        min_len = inf_cfg.get("min_answer_segment_len", 32)
        s, e = parse_span(selected, duration, min_len)

        answer = self._run_answerer(r, video_path, question, s, e, inf_cfg)
        metadata["agents"].append("answerer")
        metadata["selected_segment"] = [s, e]

        return AgentResponse(
            answer=answer,
            reasoning_steps=reasoning_steps,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Individual agent roles
    # ------------------------------------------------------------------

    @staticmethod
    def _switch_adapter(r: _Replica, name: str) -> None:
        r.model.base_model.disable_adapter_layers()
        r.model.base_model.enable_adapter_layers()
        r.model.set_adapter(name)

    @staticmethod
    def _generate_text(r: _Replica, data: dict, max_new_tokens: int = 256) -> str:
        output_ids = r.model.generate(
            **data,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            max_new_tokens=max_new_tokens,
        )
        output_ids = output_ids[0, data["input_ids"].size(1):]
        if len(output_ids) and output_ids[-1] == r.processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]
        return r.processor.decode(output_ids, skip_special_tokens=True).strip()

    @staticmethod
    def _prepare_inputs(r: _Replica, messages: list) -> dict:
        from videomind.dataset.utils import process_vision_info

        text = r.processor.apply_chat_template(messages, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        data = r.processor(text=[text], images=images, videos=videos, return_tensors="pt")
        return data.to(r.device)

    def _run_planner(self, r: _Replica, video_path: str, question: str) -> str:
        from videomind.constants import PLANNER_PROMPT

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video", "video": video_path,
                    "min_pixels": 36 * 28 * 28, "max_pixels": 64 * 28 * 28,
                    "max_frames": 100, "fps": 1.0,
                },
                {"type": "text", "text": PLANNER_PROMPT.format(question)},
            ],
        }]

        data = self._prepare_inputs(r, messages)
        self._switch_adapter(r, "planner")
        return self._generate_text(r, data)

    def _run_grounder(
        self, r: _Replica, video_path: str, query: str, duration: float
    ) -> tuple:
        from videomind.constants import GROUNDER_PROMPT

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video", "video": video_path,
                    "min_pixels": 36 * 28 * 28, "max_pixels": 64 * 28 * 28,
                    "max_frames": 150, "fps": 1.0,
                },
                {"type": "text", "text": GROUNDER_PROMPT.format(query)},
            ],
        }]

        data = self._prepare_inputs(r, messages)
        self._switch_adapter(r, "grounder")
        response = self._generate_text(r, data)

        if len(r.model.reg) > 0:
            blob = r.model.reg[0].cpu().float()
            pred = blob[:, :2] * duration
            conf = blob[:, -1].tolist()
            pred = pred.clamp(min=0, max=duration)
            unit = 0.001
            pred = (torch.round(pred / unit).long() * unit).tolist()
            pred = [[min(s, e), max(s, e)] for s, e in pred]
        else:
            pred = [[0, duration]]
            conf = [0.0]

        return response, pred, conf

    def _run_verifier(
        self,
        r: _Replica,
        video_path: str,
        question: str,
        pred: list,
        conf: list,
        duration: float,
    ) -> tuple:
        from videomind.constants import VERIFIER_PROMPT
        from videomind.dataset.utils import process_vision_info
        from videomind.utils.parser import parse_span

        probs = []
        for cand in pred[:5]:
            s0, e0 = parse_span(cand, duration, 2)
            offset = (e0 - s0) / 2
            s1, e1 = parse_span([s0 - offset, e0 + offset], duration)
            s_pct = (s0 - s1) / (e1 - s1) if e1 > s1 else 0.0
            e_pct = (e0 - s1) / (e1 - s1) if e1 > s1 else 1.0

            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video", "video": video_path,
                        "video_start": s1, "video_end": e1,
                        "min_pixels": 36 * 28 * 28, "max_pixels": 64 * 28 * 28,
                        "max_frames": 64, "fps": 2.0,
                    },
                    {"type": "text", "text": VERIFIER_PROMPT.format(question)},
                ],
            }]

            text = r.processor.apply_chat_template(messages, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            data = r.processor(text=[text], images=images, videos=videos, return_tensors="pt")

            video_grid_thw = data["video_grid_thw"][0]
            num_frames = int(video_grid_thw[0])
            window = int(video_grid_thw[1] * video_grid_thw[2] / 4)

            pos_s = min(max(0, round(s_pct * num_frames)), num_frames)
            pos_e = min(max(0, round(e_pct * num_frames)), num_frames)

            base_idx = torch.nonzero(
                data["input_ids"][0] == r.model.config.vision_start_token_id
            ).item()
            pos_s_tok = pos_s * window + base_idx + 1
            pos_e_tok = pos_e * window + base_idx + 2

            input_ids = data["input_ids"][0].tolist()
            input_ids.insert(pos_s_tok, r.model.config.seg_s_token_id)
            input_ids.insert(pos_e_tok, r.model.config.seg_e_token_id)
            data["input_ids"] = torch.LongTensor([input_ids])
            data["attention_mask"] = torch.ones_like(data["input_ids"])

            data = data.to(r.device)
            self._switch_adapter(r, "verifier")

            with torch.inference_mode():
                logits = r.model(**data).logits[0, -1].softmax(dim=-1)
            score = (logits[9454] - logits[2753]).sigmoid().item()
            probs.append(score)

        ranks = torch.Tensor(probs).argsort(descending=True).tolist()
        pred_reranked = [pred[i] for i in ranks]
        conf_reranked = [conf[i] for i in ranks]
        return pred_reranked, conf_reranked, probs, ranks

    def _run_answerer(
        self,
        r: _Replica,
        video_path: str,
        question: str,
        s: float,
        e: float,
        inf_cfg: dict,
    ) -> str:
        min_pix = inf_cfg.get("answerer_min_pixels", 128) * 28 * 28
        max_pix = inf_cfg.get("answerer_max_pixels", 256) * 28 * 28
        max_frames = inf_cfg.get("answerer_max_frames", 32)
        fps = inf_cfg.get("answerer_fps", 2.0)
        max_new_tokens = inf_cfg.get("max_new_tokens", 256)

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video", "video": video_path,
                    "video_start": s, "video_end": e,
                    "min_pixels": min_pix, "max_pixels": max_pix,
                    "max_frames": max_frames, "fps": fps,
                },
                {"type": "text", "text": question},
            ],
        }]

        data = self._prepare_inputs(r, messages)

        if r.adapter_state.get("answerer"):
            self._switch_adapter(r, "answerer")
            context = nullcontext
        else:
            context = r.model.disable_adapter

        with context():
            return self._generate_text(r, data, max_new_tokens=max_new_tokens)
