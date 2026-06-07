"""Vgent agent adapter — graph-based retrieval-augmented video QA.

Wraps the Vgent pipeline (xiaoqian-shen/Vgent) behind the BaseAgent
interface.  Removes the upstream torchrun/distributed requirement and
runs entirely in-process with multi-GPU replica pooling (same pattern
as VideoMind).

Key changes from upstream:
  - No torch.distributed — single-process, multi-GPU via thread pool
  - Open-ended answer generation (upstream was MCQ-locked)
  - Graph construction as preprocessing step
  - VLM pinned to specific GPU per replica (upstream used .to("cuda"))
  - Bypasses HF processor — uses generate_fast for speed
  - Per-GPU LRU video cache to avoid reloading same video
"""

import json
import os
import pickle
import queue
import re
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from agents.base import AgentResponse, BaseAgent
from agents.config import AGENTS_DIR
from agents.registry import register_agent


@dataclass
class _Replica:
    """One VLM + embedding model pinned to a specific GPU."""
    vlm: Any
    processor: Any
    image_processor: Any
    embedding_model: Any
    embedding_tokenizer: Any
    device: torch.device
    gpu_id: int
    video_cache: OrderedDict = field(default_factory=OrderedDict)
    cache_lock: threading.Lock = field(default_factory=threading.Lock)


OPEN_ENDED_PROMPT = (
    "Based on the video and the information gathered, "
    "answer the following question in detail.\n"
)

VIDEO_CACHE_SIZE = 0


@register_agent("vgent")
class VgentAgent(BaseAgent):
    """Vgent: graph-based retrieval-reasoning-augmented video QA."""

    name = "vgent"
    repo_url = "https://github.com/xiaoqian-shen/Vgent"

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self._repo_path: Optional[Path] = None
        self._replicas: List[_Replica] = []
        self._replica_queue: Optional[queue.Queue] = None
        self._thread_local = threading.local()
        self._num_replicas: int = 0
        self._graph_cfg: Dict[str, Any] = {}
        self._inf_cfg: Dict[str, Any] = {}

    def setup(self, config: Dict[str, Any]) -> None:
        self.config = config
        repo_rel = config.get("repo", {}).get("path", "repos/Vgent")
        self._repo_path = AGENTS_DIR / repo_rel
        if not self._repo_path.exists():
            raise FileNotFoundError(
                f"Vgent repo not found at {self._repo_path}. "
                f"Clone with: git clone https://github.com/xiaoqian-shen/Vgent {self._repo_path}"
            )

        repo_str = str(self._repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        self._graph_cfg = config.get("graph", {})
        self._inf_cfg = config.get("inference", {})

    def _load_replica(self, gpu_id: int) -> _Replica:
        from vgent_models.qwenvl import load_model
        from transformers import AutoModel, AutoTokenizer

        models_cfg = self.config.get("models", {})
        vlm_name = models_cfg.get("vlm", "Qwen/Qwen2.5-VL-7B-Instruct")
        embed_name = models_cfg.get("embedding", "BAAI/bge-large-en-v1.5")

        device = f"cuda:{gpu_id}"
        print(f"[vgent] Loading VLM replica on {device}...")

        _, vlm, processor, _ = load_model(vlm_name, device=device)
        image_processor = processor

        embedding_tokenizer = AutoTokenizer.from_pretrained(embed_name)
        embedding_model = AutoModel.from_pretrained(embed_name).to(device)

        return _Replica(
            vlm=vlm,
            processor=processor,
            image_processor=image_processor,
            embedding_model=embedding_model,
            embedding_tokenizer=embedding_tokenizer,
            device=torch.device(device),
            gpu_id=gpu_id,
        )

    def start_servers(self) -> None:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True

        gpu_ids = self.config.get("gpus", None)
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        elif isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]

        self._num_replicas = len(gpu_ids)
        self._replicas = []
        self._replica_queue = queue.Queue()

        for gid in gpu_ids:
            replica = self._load_replica(gid)
            self._replicas.append(replica)
            self._replica_queue.put(replica)

        print(f"[vgent] Ready. {self._num_replicas} replicas on GPUs {gpu_ids}.")

    def stop_servers(self) -> None:
        for replica in self._replicas:
            del replica.vlm
            del replica.processor
            del replica.embedding_model
        self._replicas = []
        self._replica_queue = None
        self._num_replicas = 0
        torch.cuda.empty_cache()

    @property
    def recommended_concurrency(self) -> Optional[int]:
        return max(self._num_replicas, 1)

    def needs_preprocessing(self) -> bool:
        return True

    def preprocess(self, video_dir: str, output_dir: str, **kwargs) -> None:
        graph_dir = self._graph_cfg.get("cache_dir", output_dir)
        os.makedirs(graph_dir, exist_ok=True)
        print(f"[vgent] Graphs should be pre-built. Run preprocess_graphs.py if needed.")

    def _get_thread_replica(self) -> _Replica:
        """Assign a replica to this thread permanently on first call."""
        if not hasattr(self._thread_local, 'replica'):
            replica = self._replica_queue.get()
            self._thread_local.replica = replica
            torch.cuda.set_device(replica.gpu_id)
        return self._thread_local.replica

    def process_sample(
        self,
        video_path: str,
        question: str,
        subtitles: Optional[str] = None,
    ) -> AgentResponse:
        replica = self._get_thread_replica()
        try:
            result = self._run_pipeline(replica, video_path, question, subtitles)
            return result
        except torch.cuda.OutOfMemoryError:
            replica.video_cache.clear()
            torch.cuda.empty_cache()
            raise

    @torch.inference_mode()
    def _run_pipeline(
        self,
        r: _Replica,
        video_path: str,
        question: str,
        subtitles_text: Optional[str] = None,
    ) -> AgentResponse:
        from vgent_utils.prompts import GRAPH_PROMPT, REASONING_PROMPT, AGGREGATE_PROMPT
        from vgent_utils.retrieval import (
            compute_text_similarity,
            allocate_node,
            node2indices,
            extract_choices,
            count_and_sort_filtered,
        )

        chunk_size = self._graph_cfg.get("chunk_size", 64)
        fps_cfg = self._graph_cfg.get("fps", 1.0)
        n_retrieval = self._graph_cfg.get("n_retrieval", 20)
        n_refine = self._graph_cfg.get("n_refine", 5)
        total_pixels = self._graph_cfg.get("total_pixels", 16384)
        uniform_frame = self._graph_cfg.get("uniform_frame", 450)
        graph_dir = self._graph_cfg.get("cache_dir", "")

        reasoning_steps: List[Dict[str, Any]] = []
        metadata: Dict[str, Any] = {}

        video_inputs, raw_video, frame_idx, vid_fps = self._load_video_cached(video_path, r)
        num_frames = len(video_inputs[0])
        metadata["num_frames"] = num_frames

        subtitle_list = self._parse_subtitles(subtitles_text)

        # --- Load graph ---
        vname = Path(video_path).stem
        video_graph, entity_graph = None, None
        if num_frames >= chunk_size * n_retrieval:
            graph_file = os.path.join(graph_dir, f"{vname}.pkl") if graph_dir else ""
            if graph_file and os.path.exists(graph_file):
                saved = pickle.load(open(graph_file, "rb"))
                video_graph = saved["video_graph"]
                entity_graph = saved["entity_graph"]

        # --- Keyword extraction (text-only, fast) ---
        query_list, llm_info = self._extract_keywords(r, question)
        reasoning_steps.append({"role": "keyword_extraction", "info": llm_info})

        # --- Node retrieval ---
        node_result = self._retrieve_nodes(
            r, question, query_list, video_inputs, video_graph, entity_graph,
            subtitle_list, llm_info, chunk_size, fps_cfg, n_retrieval,
        )
        reasoning_steps.append({"role": "retrieval", "nodes": node_result["nodes"][:10]})

        # --- Node refinement ---
        node_result, sql_check, check_result = self._refine_nodes(
            r, node_result, question, llm_info, video_inputs,
            subtitle_list, chunk_size, fps_cfg, None,
        )
        reasoning_steps.append({"role": "refinement", "nodes": node_result["nodes"][:n_refine]})

        # --- Answer aggregation (open-ended) ---
        answer = self._aggregate_open_ended(
            r, node_result, llm_info, video_inputs, raw_video,
            None, subtitle_list, question, video_graph, sql_check,
            check_result, vid_fps, chunk_size, n_refine, total_pixels,
            uniform_frame,
        )

        metadata["node_list"] = node_result["nodes"][:n_refine]
        metadata["llm_info"] = llm_info

        del video_inputs, raw_video
        torch.cuda.empty_cache()

        return AgentResponse(
            answer=answer,
            reasoning_steps=reasoning_steps,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_video_cached(self, video_path: str, r: _Replica):
        """Load video with per-replica LRU cache."""
        cache_key = video_path
        with r.cache_lock:
            if cache_key in r.video_cache:
                r.video_cache.move_to_end(cache_key)
                return r.video_cache[cache_key]

        result = self._load_video(video_path, r)

        with r.cache_lock:
            r.video_cache[cache_key] = result
            while len(r.video_cache) > VIDEO_CACHE_SIZE:
                r.video_cache.popitem(last=False)

        return result

    def _load_video(self, video_path: str, r: _Replica):
        from vgent_models.utils import _ffprobe_meta, smart_nframes, smart_resize
        from vgent_models.utils import VIDEO_MAX_PIXELS
        import subprocess

        fps_cfg = self._graph_cfg.get("fps", 1.0)
        chunk_size = self._graph_cfg.get("chunk_size", 64)
        total_pixels = self._graph_cfg.get("total_pixels", 16384)
        min_pixels = 16 * 28 * 28

        total_frames, video_fps, width, height = _ffprobe_meta(video_path)
        ele = {"video": video_path, "fps": fps_cfg}
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        target_fps = nframes / max(total_frames, 1) * video_fps

        num_chunks = max(1, int(np.ceil(nframes / chunk_size)))
        max_frames = min(nframes, num_chunks * chunk_size)
        if nframes > max_frames:
            nframes = max_frames
            target_fps = nframes / max(total_frames, 1) * video_fps

        # Compute resized dimensions for graph processing
        tp = total_pixels * num_chunks * 28 * 28
        max_px = max(min(VIDEO_MAX_PIXELS, tp / nframes * 2.0), int(min_pixels * 1.05))
        max_px = min(tp, max_px)
        rh, rw = smart_resize(height, width, factor=28, min_pixels=min_pixels, max_pixels=max_px)

        # ONE ffmpeg call: output raw frames (original resolution, target fps)
        cmd = [
            "ffmpeg", "-v", "quiet", "-i", video_path,
            "-vf", f"fps={target_fps:.4f}",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        raw = np.frombuffer(result.stdout, dtype=np.uint8)
        actual_frames = len(raw) // (height * width * 3)
        if actual_frames == 0:
            raise RuntimeError(f"ffmpeg produced 0 frames for {video_path}")
        raw = raw[:actual_frames * height * width * 3].reshape(actual_frames, height, width, 3)
        if actual_frames != nframes:
            idx = np.linspace(0, actual_frames - 1, nframes, dtype=int)
            raw = raw[idx]

        raw_video = torch.from_numpy(raw.copy()).permute(0, 3, 1, 2)  # TCHW uint8

        # Resize on CPU — only small chunks are moved to GPU for VLM calls
        video = torch.nn.functional.interpolate(
            raw_video.float(), size=(rh, rw),
            mode='bicubic', align_corners=False,
        )

        frame_idx = list(range(0, total_frames, max(1, round(video_fps))))
        sample_fps = round(nframes / max(total_frames, 1e-6) * video_fps, 2)

        return [video], [raw_video], frame_idx, sample_fps

    def _vlm_call(self, r: _Replica, text: str, video=None, max_new_tokens=512, size_list=None, fps=None):
        """VLM call — uses generate_fast for video, mllm_response for text-only."""
        if video is not None:
            from vgent_models.qwenvl import generate_fast
            if isinstance(video, torch.Tensor) and video.device != r.device:
                video = video.to(r.device)
            return generate_fast(r.vlm, r.processor, video, text, r.device,
                                  max_new_tokens=max_new_tokens)
        else:
            from vgent_models.qwenvl import mllm_response
            return mllm_response(
                r.vlm, r.processor, r.image_processor,
                text, None, None, max_new_tokens=max_new_tokens,
            )

    def _vlm_call_batch(self, r: _Replica, text: str, video_list: list,
                         max_new_tokens=256):
        """Batched VLM call for multiple video chunks."""
        from vgent_models.qwenvl import generate_fast_batch
        videos_on_device = []
        for v in video_list:
            if isinstance(v, torch.Tensor) and v.device != r.device:
                videos_on_device.append(v.to(r.device))
            else:
                videos_on_device.append(v)
        return generate_fast_batch(r.vlm, r.processor, videos_on_device, text,
                                    r.device, max_new_tokens=max_new_tokens)

    def _generate_entities(self, r, prompt, video_input, max_new_tokens=512):
        if isinstance(video_input, torch.Tensor) and len(video_input) > 8:
            sub_idx = torch.linspace(0, len(video_input) - 1, 8).round().long()
            video_input = video_input[sub_idx]
        for _ in range(2):
            try:
                response = self._vlm_call(r, prompt, video_input, max_new_tokens)
                info = json.loads(response.replace("```json", "").replace("```", "").strip())
                entities = [
                    f"{e['entity name']}, {e['description']}"
                    for e in info.get("entities", [])
                    if "entity name" in e and "description" in e
                ]
                actions = [
                    f"{e['entity name']}, {e['action description']}"
                    for e in info.get("actions", [])
                    if "entity name" in e and "action description" in e
                ]
                scenes = [s["location"] for s in info.get("scenes", []) if "location" in s]
                return entities, actions, scenes
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return [], [], []

    def _extract_keywords(self, r, question):
        from vgent_utils.prompts import REASONING_PROMPT
        prompt = REASONING_PROMPT.format(query=question, candidates="")
        max_tokens = self._inf_cfg.get("keyword_max_new_tokens", 256)
        llm_info = None
        for _ in range(2):
            try:
                response = self._vlm_call(r, prompt, None, max_tokens)
                llm_info = json.loads(response.replace("```json", "").replace("```", "").strip())
                break
            except Exception:
                continue

        query_list = llm_info.get("keywords", []) if llm_info else []
        query_list = list(set(query_list))
        return query_list, llm_info

    def _retrieve_nodes(
        self, r, question, query_list, video_inputs, video_graph, entity_graph,
        subtitles, llm_info, chunk_size, fps, n_retrieval,
    ):
        from vgent_utils.retrieval import compute_text_similarity, allocate_node

        indices = None
        if "subtitle" in question.lower() and subtitles is not None:
            matches = re.findall(r"'((?:[^']|(?<=\w)'(?=\w))*)'", question)
            if matches:
                indices = []
                for time, text in subtitles:
                    if text in matches:
                        indices.append(time)
                return {"nodes": [], "indices": indices}

        total_chunks = round(np.ceil(len(video_inputs[0]) / chunk_size))

        if "beginning" in question.lower() or "at the start of" in question.lower():
            return {"nodes": list(range(min(3, total_chunks))), "indices": None}
        if "at the end of the video" in question.lower():
            return {"nodes": list(range(max(total_chunks - 3, 0), total_chunks)), "indices": None}

        if video_graph is None:
            if len(video_inputs[0]) <= 128:
                return {"nodes": list(range(total_chunks)), "indices": None}
            return {"nodes": [], "indices": None}

        query_with_q = query_list + [question]
        args_ns = type("A", (), {"n_retrieval": n_retrieval, "chunk_size": chunk_size})()
        node_list = allocate_node(
            args_ns, video_graph, entity_graph,
            query_with_q, r.embedding_model, r.embedding_tokenizer,
        )

        key_list = []
        for nid in node_list:
            nd = video_graph.nodes[nid]
            parts = (
                nd.get("entities", []) + nd.get("actions", []) + nd.get("scenes", [])
            )
            if nd.get("subtitles"):
                parts += nd["subtitles"]
            key_list.append("; ".join(parts))

        if key_list:
            sims = compute_text_similarity(
                query_with_q, key_list, r.embedding_model, r.embedding_tokenizer,
                return_all=True,
            )
            sorted_idx = torch.argsort(torch.mean(sims, dim=0), descending=True)
            node_list = [node_list[i] for i in sorted_idx]

        return {"nodes": node_list[:n_retrieval], "indices": None}

    def _refine_nodes(
        self, r, retrieved, question, llm_info, video_inputs,
        subtitles, chunk_size, fps, size_list,
    ):
        from vgent_utils.prompts import SQL_PROMPT, SQL_ANSWER_PROMPT, SQL_ANSWER_COUNT_PROMPT
        from vgent_utils.retrieval import count_and_sort_filtered

        if not retrieved["nodes"]:
            return retrieved, None, None

        question_type = llm_info.get("tool") if llm_info else None
        info = None
        obj_count = False

        if question_type == "order" or "order" in question.lower():
            from vgent_utils.retrieval import extract_choices
            try:
                choices = extract_choices(question, [""])
                info = {f"Q{i+1}": f"Is '{c.lower()}' shown in video?" for i, c in enumerate(choices)}
            except (IndexError, ValueError):
                pass
        elif question_type == "action counting" and "action" in question.lower():
            match = re.search(r"'(.*?)'", question)
            if match:
                info = {"Q1": f"Is there a scene featuring the '{match.group(1)}' action in the video?"}
        elif question_type == "object counting" or "how many" in question.lower():
            obj_count = True
            info = {"Q1": question}

        if info is None:
            prompt = SQL_PROMPT.format(query=question, candidates="")
            max_tokens = self._inf_cfg.get("refine_max_new_tokens", 256)
            for _ in range(2):
                try:
                    response = self._vlm_call(r, prompt, None, max_tokens)
                    info = json.loads(response.replace("```json", "").replace("```", "").strip())
                    break
                except Exception:
                    continue

        if info is None:
            return retrieved, None, None

        split_video = torch.split(video_inputs[0], chunk_size, dim=0)
        check_result = {}

        # Batch refinement: process all nodes in batches
        valid_nodes = [n for n in retrieved["nodes"] if n < len(split_video)]

        if obj_count:
            instruct_tpl = SQL_ANSWER_COUNT_PROMPT
        else:
            instruct_tpl = SQL_ANSWER_PROMPT

        # Prepare all chunks (subsampled to 8 frames)
        batch_chunks = []
        batch_nodes = []
        for node in valid_nodes:
            chunk = split_video[node]
            if len(chunk) > 8:
                sub_idx = torch.linspace(0, len(chunk) - 1, 8).round().long()
                chunk = chunk[sub_idx]

            subtitle_prompt = ""
            if subtitles is not None:
                start_t = node * chunk_size // fps
                end_t = (node + 1) * chunk_size // fps
                subs = [text for time, text in subtitles if start_t <= time < end_t]
                if subs:
                    subtitle_prompt = " This video's subtitles are listed below:\n" + " ".join(subs) + "\n"

            batch_chunks.append(chunk)
            batch_nodes.append(node)

        refine_batch = 8
        instruct = instruct_tpl.format(questions=info)
        for bi in range(0, len(batch_chunks), refine_batch):
            be = min(bi + refine_batch, len(batch_chunks))
            chunk_batch = batch_chunks[bi:be]
            node_batch = batch_nodes[bi:be]

            try:
                if len(chunk_batch) > 1:
                    outputs = self._vlm_call_batch(r, instruct, chunk_batch, max_new_tokens=256)
                else:
                    outputs = [self._vlm_call(r, instruct, chunk_batch[0], 256)]
            except Exception:
                outputs = [None] * len(chunk_batch)

            for j, output in enumerate(outputs):
                pred = None
                if output:
                    try:
                        pred = json.loads(output.replace("```json", "").replace("```", "").strip())
                    except Exception:
                        pass
                check_result[node_batch[j]] = pred

        _, sorted_nodes = count_and_sort_filtered(check_result)
        retrieved["nodes"] = sorted_nodes
        return retrieved, info, check_result

    def _aggregate_open_ended(
        self, r, refined, llm_info, video_inputs, raw_video, size_list,
        subtitles, question, video_graph, sql_check, check_result,
        vid_fps, chunk_size, n_refine, total_pixels, uniform_frame,
    ):
        from vgent_utils.prompts import AGGREGATE_PROMPT
        from vgent_utils.retrieval import node2indices
        from vgent_models.utils import resize_video
        from types import SimpleNamespace

        node_list = refined["nodes"]
        select_subtitles = None
        args_ns = SimpleNamespace(
            chunk_size=chunk_size, n_refine=n_refine,
            uniform_frame=uniform_frame,
        )

        if node_list and len(node_list) > 0:
            question_type = llm_info.get("tool") if llm_info else None
            indices, sorted_node_list = node2indices(node_list, question_type, video_inputs, args_ns)
            video_segments = video_inputs[0][indices]
            if subtitles is not None:
                if video_graph is None:
                    select_subtitles = []
                    for nid in sorted_node_list:
                        select_subtitles.extend(
                            [text for time, text in subtitles
                             if nid * chunk_size <= time < (nid + 1) * chunk_size]
                        )
                else:
                    select_subtitles = [text for _, text in subtitles]
        elif refined.get("indices") is not None:
            indices = refined["indices"]
            extend_indices = []
            if subtitles is not None:
                select_subtitles = []
                for idx in indices:
                    select_subtitles.extend([text for time, text in subtitles if time == idx])
                    extend_indices.extend(range(max(0, idx - 10), min(len(video_inputs[0]) - 1, idx + 10)))
            indices = sorted(set(extend_indices))
            video_segments = video_inputs[0][indices]
        else:
            n_uniform = min(uniform_frame, len(video_inputs[0]))
            indices = np.linspace(0, len(video_inputs[0]) - 1, n_uniform, dtype=int)
            video_segments = video_inputs[0][indices]
            if subtitles is not None:
                select_subtitles = [text for _, text in subtitles]

        # Build prompt
        input_prompt = f"Question: {question}\n"
        if select_subtitles:
            input_prompt += "This video's subtitles are listed below:\n"
            input_prompt += " ".join(select_subtitles) + "\n"

        multiple = llm_info.get("multiple", "no") if llm_info else "no"
        question_type = llm_info.get("tool") if llm_info else None
        if (
            sql_check is not None
            and check_result is not None
            and node_list
            and len(node_list) > 1
            and multiple == "yes"
            and question_type != "object counting"
        ):
            agg_text = ""
            for key, value in check_result.items():
                agg_text += f"video [{key}]:\n"
                if value and isinstance(value, dict):
                    for qid, q in sql_check.items():
                        if key in check_result and check_result[key] is not None:
                            ans = check_result[key].get(qid)
                            if ans and ans != "no":
                                agg_text += f"{q}: {ans}\n"
            if agg_text.strip():
                agg_prompt = AGGREGATE_PROMPT.format(
                    query=question, candidates="", input=agg_text
                )
                try:
                    agg_info = self._vlm_call(r, agg_prompt, None, 128)
                    input_prompt += f"Relevant information: {agg_info}\n"
                except Exception:
                    pass

        input_prompt += OPEN_ENDED_PROMPT

        # Map indices from resized video to raw video frame space
        raw_len = len(raw_video[0])
        resized_len = len(video_inputs[0])
        if resized_len > 0 and raw_len > 0:
            scale = raw_len / resized_len
            raw_indices = [min(int(i * scale), raw_len - 1) for i in indices]
            raw_indices = sorted(set(raw_indices))
        else:
            raw_indices = list(range(raw_len))

        max_answer_frames = 256
        video_segments_for_answer = raw_video[0][raw_indices] if raw_indices else raw_video[0]
        if len(video_segments_for_answer) > max_answer_frames:
            sub_idx = torch.linspace(0, len(video_segments_for_answer) - 1, max_answer_frames).round().long()
            video_segments_for_answer = video_segments_for_answer[sub_idx]
        # Move to GPU for fast resize + generate
        video_segments_for_answer = video_segments_for_answer.to(r.device).float()
        video_segments_for_answer, answer_fps = resize_video(
            video_segments_for_answer, vid_fps,
            total_pixels=total_pixels * 28 * 28, maximum_frames=max_answer_frames,
        )

        max_tokens = self._inf_cfg.get("answer_max_new_tokens", 512)
        answer = self._vlm_call(
            r, input_prompt, video_segments_for_answer,
            max_new_tokens=max_tokens,
        )
        return answer.strip()

    def _parse_subtitles(self, subtitles_text: Optional[str]):
        if not subtitles_text:
            return None
        lines = subtitles_text.strip().split("\n")
        result = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            result.append((i * 5, line))
        return result if result else None
