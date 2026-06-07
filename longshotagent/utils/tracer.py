"""Per-request tracing for the video agent.

Each server startup creates a session folder under trace/.
Each request gets a JSONL file with the full reasoning chain:
system prompt, user query, LLM completions, tool calls, tool results, errors, final answer.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_session_dir: Optional[Path] = None


def init_session() -> Path:
    """Create a new session folder for this server startup. Call once at server start."""
    global _session_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _session_dir = Path("trace") / timestamp
    _session_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Trace session: %s", _session_dir)
    return _session_dir


class RequestTracer:
    """Traces a single request's full chain to a JSONL file."""

    def __init__(self, video_id: str, question: str):
        self._events = []
        self._start = time.time()
        self.video_id = video_id
        self.request_id = f"{video_id}_{int(self._start * 1000) % 100000000}"

        if _session_dir is None:
            self._path = None
            return
        self._path = _session_dir / f"{self.request_id}.jsonl"
        self._log("request", {"video_id": video_id, "question": question})

    def _log(self, event_type: str, data: Dict[str, Any]):
        entry = {
            "t": round(time.time() - self._start, 3),
            "event": event_type,
            **data,
        }
        self._events.append(entry)
        if self._path:
            with open(self._path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_system_prompt(self, prompt: str):
        self._log("system_prompt", {"chars": len(prompt)})

    def log_completion_request(self, iteration: int, tool_choice: str, message_count: int, total_chars: int):
        self._log("completion_request", {
            "iteration": iteration,
            "tool_choice": tool_choice,
            "message_count": message_count,
            "total_chars": total_chars,
        })

    def log_completion_response(self, iteration: int, content: str, tool_calls: list, reasoning: str = ""):
        tc_summary = [
            {"name": tc.function.name, "args": tc.function.arguments[:200]}
            for tc in (tool_calls or [])
        ]
        self._log("completion_response", {
            "iteration": iteration,
            "content": (content or "")[:500],
            "tool_calls": tc_summary,
            "reasoning_chars": len(reasoning) if reasoning else 0,
        })

    def log_tool_call(self, iteration: int, tool_name: str, arguments: dict):
        args_short = {k: (v[:100] if isinstance(v, str) and len(v) > 100 else v)
                      for k, v in arguments.items()}
        self._log("tool_call", {
            "iteration": iteration,
            "tool": tool_name,
            "arguments": args_short,
        })

    def log_tool_result(self, iteration: int, tool_name: str, result: dict, elapsed: float):
        result_summary = {}
        if isinstance(result, dict):
            result_summary["success"] = result.get("success", True)
            inner = result.get("result", {})
            if isinstance(inner, dict):
                if "results" in inner:
                    for mod, items in inner.get("results", {}).items():
                        if isinstance(items, list):
                            result_summary[f"{mod}_count"] = len(items)
                desc = inner.get("description")
                if isinstance(desc, str):
                    result_summary["description"] = desc[:300]
                trans = inner.get("transcription")
                if isinstance(trans, str):
                    result_summary["transcription"] = trans[:300]
                if "total_results" in inner:
                    result_summary["total_results"] = inner["total_results"]
            if "error" in result:
                result_summary["error"] = str(result["error"])[:200]
        self._log("tool_result", {
            "iteration": iteration,
            "tool": tool_name,
            "elapsed": round(elapsed, 2),
            "result": result_summary,
        })

    def log_tool_error(self, iteration: int, tool_name: str, error: str):
        self._log("tool_error", {
            "iteration": iteration,
            "tool": tool_name,
            "error": error[:500],
        })

    def log_answer(self, answer: str, metrics: dict):
        self._log("answer", {
            "answer": answer[:1000],
            "total_elapsed": round(time.time() - self._start, 2),
            **metrics,
        })

    def log_error(self, error: str):
        self._log("error", {"error": error[:500]})
