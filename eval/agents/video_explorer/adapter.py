"""VideoExplorer agent adapter - wraps the original VideoDeepResearch code."""

import atexit
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from agents.base import AgentResponse, BaseAgent
from agents.config import AGENTS_DIR
from agents.registry import register_agent
from agents.server_manager import AgentServerManager, ServerSpec

_active_agent: Optional["VideoExplorerAgent"] = None


def _cleanup_handler(signum=None, frame=None):
    """Cleanup handler for signals and atexit.

    On SIGINT/SIGTERM we only flip the shutdown flag so in-flight workers raise
    a quiet RuntimeError instead of seeing state get nulled out from under them.
    Full teardown happens in the main thread's `finally` block in eval.py.
    """
    global _active_agent
    if _active_agent is not None:
        _active_agent._shutting_down = True
    if signum == signal.SIGINT:
        sys.exit(130)
    elif signum == signal.SIGTERM:
        sys.exit(143)


@register_agent("video_explorer")
class VideoExplorerAgent(BaseAgent):
    """Wraps VideoExplorer from RUC-NLPIR/VideoDeepResearch.

    Expects the repo to be pre-cloned at agents/repos/VideoDeepResearch.
    Uses official fine-tuned models for reproducibility.
    """

    name = "video_explorer"

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.server_manager: Optional[AgentServerManager] = None
        self.demo_class = None  # Store class, not instance
        self._repo_path: Optional[Path] = None
        # Cached components (initialized once, reused across samples)
        self._retriever = None
        self._vlm_client = None
        self._processor = None
        self._shutting_down = False

    def setup(self, config: Dict[str, Any]) -> None:
        """Initialize VideoExplorer from pre-cloned repo."""
        self.config = config

        repo_config = config.get("repo", {})
        repo_rel_path = repo_config.get("path", "repos/VideoDeepResearch")
        self._repo_path = AGENTS_DIR / repo_rel_path

        if not self._repo_path.exists():
            raise FileNotFoundError(
                f"VideoDeepResearch repo not found at {self._repo_path}. "
                f"Please clone it: git clone https://github.com/RUC-NLPIR/VideoDeepResearch {self._repo_path}"
            )

        self._add_repo_to_path()

    def _add_repo_to_path(self) -> None:
        """Add the repo to Python path for imports."""
        if self._repo_path and str(self._repo_path) not in sys.path:
            sys.path.insert(0, str(self._repo_path))

    def start_servers(self) -> None:
        """Start vLLM servers for VideoExplorer components."""
        global _active_agent
        _active_agent = self
        signal.signal(signal.SIGINT, _cleanup_handler)
        signal.signal(signal.SIGTERM, _cleanup_handler)
        atexit.register(_cleanup_handler)

        servers_config = self.config.get("servers", {})
        models_config = self.config.get("models", {})

        specs = []
        for server_name, server_cfg in servers_config.items():
            model_path = models_config.get(server_name)
            if not model_path:
                raise ValueError(f"No model specified for server '{server_name}'")

            spec = ServerSpec(
                name=server_name,
                model_path=model_path,
                gpu_id=server_cfg.get("gpu", 0),
                port=server_cfg.get("port", 8000),
                tensor_parallel=server_cfg.get("tensor_parallel", 1),
                max_model_len=server_cfg.get("max_model_len"),
                extra_args={
                    k: v for k, v in server_cfg.items()
                    if k not in ("gpu", "port", "tensor_parallel", "max_model_len")
                },
            )
            specs.append(spec)

        self.server_manager = AgentServerManager(specs)
        endpoints = self.server_manager.start_all()

        self._set_environment(endpoints)
        self._initialize_demo()

    def _set_environment(self, endpoints: Dict[str, str]) -> None:
        """Set environment variables expected by VideoExplorer."""
        # Map our server names to env vars expected by demo.py
        endpoint_mapping = {
            "temporal_grounder": "API_BASE_URL_TEMPORAL_GROUNDING",
            "planner": "API_BASE_URL",
            "vlm": "VLM_API_BASE",  # Custom env var we'll add support for
        }

        for name, endpoint in endpoints.items():
            env_var = endpoint_mapping.get(name)
            if env_var:
                os.environ[env_var] = f"{endpoint}/v1"

        # Set API keys (vLLM doesn't require real keys)
        os.environ["API_KEY"] = "EMPTY"
        os.environ["API_KEY_TEMPORAL_GROUNDING"] = "EMPTY"

        paths_config = self.config.get("paths", {})
        if "cache_dir" in paths_config:
            os.makedirs(paths_config["cache_dir"], exist_ok=True)
        if "clips_dir" in paths_config:
            os.makedirs(paths_config["clips_dir"], exist_ok=True)

    def _patch_torchvision_compat(self) -> None:
        """Patch torchvision/pytorchvideo/transformers for compatibility with older dependencies."""
        # Patch torchvision.transforms.functional_tensor
        try:
            import torchvision.transforms
            if not hasattr(torchvision.transforms, 'functional_tensor'):
                from torchvision.transforms import _functional_tensor
                torchvision.transforms.functional_tensor = _functional_tensor
                sys.modules['torchvision.transforms.functional_tensor'] = _functional_tensor
        except Exception:
            pass

        # Patch torchaudio.set_audio_backend
        try:
            import torchaudio
            if not hasattr(torchaudio, 'set_audio_backend'):
                torchaudio.set_audio_backend = lambda x: None
        except Exception:
            pass

        # Patch transformers.utils.import_utils.is_torch_fx_available (removed in newer versions)
        try:
            from transformers.utils import import_utils
            if not hasattr(import_utils, 'is_torch_fx_available'):
                import_utils.is_torch_fx_available = lambda: False
        except Exception:
            pass

    def _initialize_demo(self) -> None:
        """Initialize VideoQADemo from the repo."""
        import importlib.util

        self._patch_torchvision_compat()

        demo_path = self._repo_path / "eval" / "demo.py"
        if not demo_path.exists():
            raise FileNotFoundError(f"demo.py not found at {demo_path}")

        # Add eval directory to path for relative imports (prompt, etc.)
        eval_dir = str(self._repo_path / "eval")
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)

        try:
            spec = importlib.util.spec_from_file_location("video_explorer_demo", demo_path)
            demo_module = importlib.util.module_from_spec(spec)
            sys.modules["video_explorer_demo"] = demo_module
            spec.loader.exec_module(demo_module)

            self.demo_class = demo_module.VideoQADemo

            # Initialize shared components once
            self._initialize_shared_components()
        except Exception as e:
            raise ImportError(f"Failed to import VideoQADemo: {e}")

    def _initialize_shared_components(self) -> None:
        """Initialize retriever, VLM client, and processor once for reuse."""
        import torch
        from openai import OpenAI
        from transformers import AutoProcessor

        paths_cfg = self.config.get("paths", {})
        inference_cfg = self.config.get("inference", {})
        models_cfg = self.config.get("models", {})

        # Initialize VLM client
        vlm_api_base = os.environ.get("VLM_API_BASE")
        if vlm_api_base:
            self._vlm_client = OpenAI(base_url=vlm_api_base, api_key="EMPTY")
            vlm_model = models_cfg.get("vlm", "Qwen/Qwen2-VL-7B-Instruct")
            self._processor = AutoProcessor.from_pretrained(vlm_model, use_fast=True)
            self._processor.tokenizer.padding_side = 'left'

        from retriever_languagebind import Retrieval_Manager

        class Args:
            dataset_folder = paths_cfg.get("data_dir", "./data")
            dataset = "demo"
            clip_duration = inference_cfg.get("clip_duration", 10)
            retriever_type = inference_cfg.get("retriever_type", "languagebind")

        args = Args()
        clip_save_folder = f'{args.dataset_folder}/clips/{args.clip_duration}/'

        # Retrieval_Manager handles its own GPU-op serialization and fatal-CUDA
        # circuit breaking internally; see retriever_languagebind.py.
        self._retriever = Retrieval_Manager(args, clip_save_folder=clip_save_folder)

        if torch.cuda.is_available():
            retriever_gpu = int(inference_cfg.get("retriever_gpu", 0))
            print(f"[agent] loading retriever on logical GPU {retriever_gpu}")
            self._retriever.load_model_to_gpu(retriever_gpu)
            print("[agent] retriever ready")

    def stop_servers(self) -> None:
        """Stop all vLLM servers."""
        global _active_agent
        self._shutting_down = True
        print("\n[VideoExplorer] Cleaning up servers...")
        if self.server_manager:
            self.server_manager.stop_all()
            self.server_manager = None
        self.demo_class = None
        self._retriever = None
        self._vlm_client = None
        self._processor = None
        _active_agent = None

    def process_sample(
        self,
        video_path: str,
        question: str,
        subtitles: Optional[str] = None,
    ) -> AgentResponse:
        """Process a video QA sample using VideoExplorer."""
        if self._shutting_down:
            raise RuntimeError("shutdown in progress")
        if self.demo_class is None:
            raise RuntimeError("Agent not initialized. Call start_servers() first.")
        if self._retriever is not None and self._retriever.is_dead():
            # CUDA context is poisoned — no point processing more samples.
            self._shutting_down = True
            raise RuntimeError("retriever dead — aborting run")

        inference_cfg = self.config.get("inference", {})
        paths_cfg = self.config.get("paths", {})
        models_cfg = self.config.get("models", {})
        use_subtitles = inference_cfg.get("use_subtitles", True) and subtitles is not None

        demo_instance = self.demo_class(
            video_path=video_path,
            question=question,
            dataset_folder=paths_cfg.get("data_dir", "./data"),
            clip_duration=inference_cfg.get("clip_duration", 10),
            use_subtitle=use_subtitles,
            vlm_model_name=models_cfg.get("vlm"),
            planner_model_name=models_cfg.get("planner"),
            temporal_model_name=models_cfg.get("temporal_grounder"),
            retriever=self._retriever,
            vlm_client=self._vlm_client,
            processor=self._processor,
        )

        result = demo_instance.run()

        return self._parse_result(result)

    def _parse_result(self, result: Dict[str, Any]) -> AgentResponse:
        """Parse VideoExplorer's result into AgentResponse.

        `VideoQADemo.run()` returns `pred_answer`, which upstream passes
        through `_extract_final_answer` — that helper strips the body down
        to the first uppercase letter (designed for MCQ tasks). For
        open-ended LongSHOT QA we want the full <answer>...</answer> body,
        so re-extract it from the conversation history when available.
        """
        import re

        messages = result.get("messages") or []

        full_answer = ""
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            m = re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if m:
                full_answer = m[-1].strip()
                break

        # Fall back to the stripped upstream answer if the full body wasn't
        # captured (e.g. run hit MAX_DS_ROUND with no final <answer> tag).
        if not full_answer:
            full_answer = (
                result.get("pred_answer")
                or result.get("answer")
                or result.get("prediction")
                or ""
            )
        if full_answer == "-":
            full_answer = ""

        reasoning_steps = []
        for i, turn in enumerate(messages):
            reasoning_steps.append({
                "round": i,
                "role": turn.get("role"),
                "content": turn.get("content", ""),
            })

        metadata = {
            "pred_answer_raw": result.get("pred_answer"),
            "total_rounds": result.get("total_rounds"),
            "is_correct": result.get("is_correct"),
            "num_rounds": len(reasoning_steps) // 2 if reasoning_steps else 0,
        }

        return AgentResponse(
            answer=full_answer,
            reasoning_steps=reasoning_steps,
            metadata=metadata,
        )

    def get_server_endpoints(self) -> Dict[str, str]:
        """Return current server endpoints."""
        if self.server_manager:
            return self.server_manager.endpoints
        return {}
