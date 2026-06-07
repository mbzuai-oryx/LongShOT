"""Video-RAG agent adapter.

Wraps the open-ended Video-RAG pipeline (see `pipeline.py`) behind the
`BaseAgent` interface so it slots into the LongShOT eval harness.

Unlike VideoExplorer this agent runs entirely in-process (no vLLM servers) —
upstream LLaVA-Video uses bespoke conversation templates and bundled mm
preprocessing, so we keep the original `llava` package's load path.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from agents.base import AgentResponse, BaseAgent
from agents.config import AGENTS_DIR
from agents.registry import register_agent


@register_agent("video_rag")
class VideoRAGAgent(BaseAgent):
    """Training-free RAG video QA agent (Leon1207/Video-RAG-master)."""

    name = "video_rag"

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self._repo_path: Optional[Path] = None
        self._pipeline = None
        self._shutting_down = False

    def setup(self, config: Dict[str, Any]) -> None:
        self.config = config
        repo_rel = config.get("repo", {}).get("path", "repos/Video-RAG-master")
        self._repo_path = AGENTS_DIR / repo_rel
        if not self._repo_path.exists():
            raise FileNotFoundError(
                f"Video-RAG repo not found at {self._repo_path}. "
                f"Clone with: git clone https://github.com/Leon1207/Video-RAG-master {self._repo_path}"
            )

        # Add repo to path so any optional helper imports resolve.
        repo_str = str(self._repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def start_servers(self) -> None:
        """No external servers — load all models in-process."""
        from agents.video_rag.pipeline import VideoRAGPipeline

        self._pipeline = VideoRAGPipeline(config=self.config, repo_path=self._repo_path)
        self._pipeline.setup_models()

    def stop_servers(self) -> None:
        self._shutting_down = True
        print("\n[video_rag] Cleaning up...")
        self._pipeline = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def process_sample(
        self,
        video_path: str,
        question: str,
        subtitles: Optional[str] = None,
    ) -> AgentResponse:
        if self._shutting_down:
            raise RuntimeError("shutdown in progress")
        if self._pipeline is None:
            raise RuntimeError("Agent not initialized. Call start_servers() first.")

        result = self._pipeline.run(video_path, question)

        answer = (result.get("answer") or "").strip()
        metadata = {
            "retrieve_request": result.get("retrieve_request"),
            "num_ocr_hits": result.get("num_ocr_hits"),
            "num_asr_hits": result.get("num_asr_hits"),
            "num_det_hits": result.get("num_det_hits"),
            "num_frames": result.get("num_frames"),
            "video_time": result.get("video_time"),
        }
        return AgentResponse(answer=answer, reasoning_steps=[], metadata=metadata)

    def get_server_endpoints(self) -> Dict[str, str]:
        return {}

    @property
    def recommended_concurrency(self):
        """Match worker count to the size of the VLM pool."""
        if self._pipeline is not None:
            return self._pipeline.num_workers
        # Pre-start estimate from config — len(gpus.vlm) if it's a list.
        vlm = self.config.get("gpus", {}).get("vlm", 0)
        if isinstance(vlm, (list, tuple)):
            return len(vlm)
        return 1
