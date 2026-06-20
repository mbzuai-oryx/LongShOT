"""
Video understanding agent with LLM integration and tool calling capabilities.

This module provides the VideoAgent class that enables interactive querying
of processed video content using an LLM with vector search tools.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import openai

from preprocessing.vector_store import VectorStore
from utils import (
    REFINE_VIDEO_TOOL_DEFINITION,
    VERIFY_CLAIM_TOOL_DEFINITION,
    VIDEO_SEARCH_TOOL_DEFINITION,
    get_video_refiner,
)

logger = logging.getLogger(__name__)

TOOLS = [VIDEO_SEARCH_TOOL_DEFINITION, REFINE_VIDEO_TOOL_DEFINITION, VERIFY_CLAIM_TOOL_DEFINITION]

ENABLE_THINKING = True
MAX_TOOL_ITERATIONS = 10
MAX_SEARCH_ROUNDS = 10
DEFAULT_REFINE_SEGMENT_SECONDS = 5.0
MODEL_SERVER_LIMITS = httpx.Limits(max_connections=256, max_keepalive_connections=256)
MODEL_SERVER_CONCURRENCY_LIMIT = int(
    os.getenv("VIDEO_AGENT_LLM_CONCURRENCY_LIMIT", "32")
)
LLM_REQUEST_MAX_RETRIES = int(os.getenv("VIDEO_AGENT_LLM_REQUEST_MAX_RETRIES", "2"))
LLM_REQUEST_RETRY_BASE_SECONDS = float(
    os.getenv("VIDEO_AGENT_LLM_REQUEST_RETRY_BASE_SECONDS", "0.5")
)
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
MODEL_SERVER_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=900.0,
    write=900.0,
    pool=None,
)
FINALIZE_TOOL_LOOP_MESSAGE = {
    "role": "user",
    "content": "Use the tool results already provided and answer the user's question directly. Do not call any additional tools. Provide a thorough, detailed answer with your reasoning and evidence. Explain step by step how you arrived at your conclusion.",
}

# Pattern to strip leaked tool call XML from model output
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def _sanitize_response(content: str) -> str:
    """Strip any leaked tool call XML tags from final response."""
    if not content:
        return content
    sanitized = _TOOL_CALL_PATTERN.sub("", content).strip()
    return sanitized


class VideoAgent:
    """
    Interactive agent for querying video content using LLM with tool calling.

    The VideoAgent connects to a vLLM server and uses tool calling to search
    through preprocessed video data, enabling natural language queries about
    video content including both audio transcriptions and visual information.
    """

    def __init__(
        self,
        vllm_base_url: str = "http://localhost:8010/v1",
        model_name: str = "google/gemma-4-31B-it",        db_path: str = "./chroma_db",
        vlm_base_url: str = "http://localhost:8011/v1",
        vlm_model_name: str = "google/gemma-4-31B-it",        alm_base_url: str = "http://localhost:8013/v1",
        alm_model_name: str = "nvidia/audio-flamingo-3-hf",
        text_embedding_url: str = "http://localhost:8014/v1",
        visual_embedding_url: str = "http://localhost:8018/v1",
        videos_dir: str = "./videos",
        video_search_paths: list = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.client = openai.OpenAI(
            base_url=vllm_base_url,
            api_key="dummy",
            http_client=httpx.Client(
                limits=MODEL_SERVER_LIMITS,
                timeout=MODEL_SERVER_TIMEOUT,
            ),
        )
        self.async_client = openai.AsyncOpenAI(
            base_url=vllm_base_url,
            api_key="dummy",
            http_client=httpx.AsyncClient(
                limits=MODEL_SERVER_LIMITS,
                timeout=MODEL_SERVER_TIMEOUT,
            ),
        )
        self.model_name = model_name
        self.db_path = db_path
        self.vlm_base_url = vlm_base_url
        self.vlm_model_name = vlm_model_name
        self.alm_base_url = alm_base_url
        self.alm_model_name = alm_model_name
        self.text_embedding_url = text_embedding_url
        self.visual_embedding_url = visual_embedding_url
        self.videos_dir = videos_dir
        self.video_search_paths = video_search_paths or [videos_dir]

        self.vector_store = vector_store or VectorStore(db_path=db_path)

        self.conversation_histories: Dict[str, List[Dict[str, str]]] = {}
        self.MAX_CONVERSATIONS = 50
        self._conversation_lock = threading.RLock()
        self._component_lock = threading.Lock()
        self._sync_completion_slots = threading.BoundedSemaphore(
            MODEL_SERVER_CONCURRENCY_LIMIT
        )
        self._async_completion_slots = asyncio.Semaphore(MODEL_SERVER_CONCURRENCY_LIMIT)

        self.search_executor = None
        self.video_refiner = None

        try:
            with open("prompts/system_prompt.txt", "r") as f:
                self.system_prompt = f.read().strip()
        except FileNotFoundError:
            logger.warning("System prompt file not found, using default prompt")
            self.system_prompt = "You are a helpful video understanding assistant."

        logger.info("VideoAgent initialized with model: %s", model_name)

    def close(self):
        """Close sync HTTP clients owned by the agent."""
        self.client.close()
        if self.search_executor is not None:
            self.search_executor.close()
        if self.video_refiner is not None:
            self.video_refiner.close()

    async def aclose(self):
        """Close async HTTP clients owned by the agent."""
        self.client.close()
        await self.async_client.close()
        if self.search_executor is not None:
            self.search_executor.close()
            await self.search_executor.aclose()
        if self.video_refiner is not None:
            self.video_refiner.close()
            await self.video_refiner.aclose()

    def _get_search_executor(self):
        """Get or create the search executor (uses embedding servers)."""
        if self.search_executor is None:
            with self._component_lock:
                if self.search_executor is None:
                    from utils.tools import get_search_executor

                    self.search_executor = get_search_executor(
                        db_path=self.db_path,
                        text_embedding_url=self.text_embedding_url,
                        visual_embedding_url=self.visual_embedding_url,
                        vector_store=self.vector_store,
                    )
                    logger.info("Search executor initialized")
        return self.search_executor

    def _get_video_refiner(self):
        """Get or create the video refiner."""
        if self.video_refiner is None:
            with self._component_lock:
                if self.video_refiner is None:
                    self.video_refiner = get_video_refiner(
                        vlm_base_url=self.vlm_base_url,
                        vlm_model_name=self.vlm_model_name,
                        alm_base_url=self.alm_base_url,
                        alm_model_name=self.alm_model_name,
                        video_search_paths=self.video_search_paths,
                    )
                    logger.info("Video refiner initialized")
        return self.video_refiner

    def _get_video_duration_seconds(self, video_id: str) -> Optional[float]:
        """Look up video duration via the refiner's ffprobe cache."""
        try:
            original_path = None
            if self.vector_store:
                original_path = self.vector_store.get_video_path(video_id)
            refiner = self._get_video_refiner()
            path = refiner._find_video_file(video_id, original_path)
            if path:
                return refiner._get_video_duration(path)
        except Exception as e:
            logger.warning("Could not get duration for video_id=%s: %s", video_id, e)
        return None

    def _build_system_prompt_for_video(self, video_id: str) -> str:
        """Build system prompt with video duration injected."""
        duration = self._get_video_duration_seconds(video_id)
        if duration and duration > 0:
            mins, secs = divmod(int(duration), 60)
            return self.system_prompt + f"\n\nThis video is {duration:.1f} seconds long ({mins}m {secs}s)."
        return self.system_prompt

    @staticmethod
    def _normalize_message(message: Any) -> Dict[str, Any]:
        """Convert request messages into plain dicts for request-local chat."""
        if isinstance(message, dict):
            return {"role": message["role"], "content": message["content"]}
        return {"role": message.role, "content": message.content}

    def _build_stateless_messages(
        self, video_id: str, request_messages: List[Any]
    ) -> List[Dict[str, Any]]:
        """Build a request-local conversation for API calls."""
        messages = [
            {"role": "system", "content": self._build_system_prompt_for_video(video_id)}
        ]
        messages.extend(
            self._normalize_message(message) for message in request_messages
        )
        return messages

    def _create_completion(self, **kwargs):
        """Issue one completion request behind a bounded local bulkhead."""
        for attempt in range(LLM_REQUEST_MAX_RETRIES + 1):
            try:
                with self._sync_completion_slots:
                    return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                if (
                    attempt >= LLM_REQUEST_MAX_RETRIES
                    or not self._is_retryable_llm_error(e)
                ):
                    raise
                delay = self._retry_delay_seconds(
                    attempt, LLM_REQUEST_RETRY_BASE_SECONDS
                )
                logger.warning(
                    "Retrying LLM completion after attempt %d/%d failed: %s",
                    attempt + 1,
                    LLM_REQUEST_MAX_RETRIES + 1,
                    e,
                )
                time.sleep(delay)

    async def _create_completion_async(self, **kwargs):
        """Issue one async completion request behind a bounded local bulkhead."""
        for attempt in range(LLM_REQUEST_MAX_RETRIES + 1):
            try:
                async with self._async_completion_slots:
                    return await self.async_client.chat.completions.create(**kwargs)
            except Exception as e:
                if (
                    attempt >= LLM_REQUEST_MAX_RETRIES
                    or not self._is_retryable_llm_error(e)
                ):
                    raise
                delay = self._retry_delay_seconds(
                    attempt, LLM_REQUEST_RETRY_BASE_SECONDS
                )
                logger.warning(
                    "Retrying async LLM completion after attempt %d/%d failed: %s",
                    attempt + 1,
                    LLM_REQUEST_MAX_RETRIES + 1,
                    e,
                )
                await asyncio.sleep(delay)

    @staticmethod
    def _retry_delay_seconds(attempt: int, base_delay: float) -> float:
        """Compute capped exponential backoff delay."""
        return min(base_delay * (2**attempt), 4.0)

    @staticmethod
    def _is_retryable_llm_error(error: Exception) -> bool:
        """Retry only transport and transient server failures."""
        if isinstance(error, (openai.APIConnectionError, openai.APITimeoutError)):
            return True
        if isinstance(error, openai.RateLimitError):
            return True
        if isinstance(error, openai.InternalServerError):
            return True
        if isinstance(error, openai.APIStatusError):
            return error.status_code in RETRYABLE_STATUS_CODES
        if isinstance(error, httpx.TimeoutException):
            return True
        if isinstance(error, httpx.TransportError):
            return True
        return False

    def _get_conversation_history(self, video_id: str) -> list:
        """Get conversation history for a video, evicting oldest if at capacity."""
        with self._conversation_lock:
            if video_id not in self.conversation_histories:
                if len(self.conversation_histories) >= self.MAX_CONVERSATIONS:
                    oldest_key = next(iter(self.conversation_histories))
                    del self.conversation_histories[oldest_key]
                    logger.info(
                        "Evicted oldest conversation (%s) to stay under %d limit",
                        oldest_key,
                        self.MAX_CONVERSATIONS,
                    )
                self.conversation_histories[video_id] = [
                    {
                        "role": "system",
                        "content": self._build_system_prompt_for_video(video_id),
                    }
                ]
            return self.conversation_histories[video_id]

    def clear_conversation_history(self, video_id: str = None):
        """Clear conversation history for a video or all videos."""
        with self._conversation_lock:
            if video_id:
                if video_id in self.conversation_histories:
                    self.conversation_histories[video_id] = [
                        {
                            "role": "system",
                            "content": self._build_system_prompt_for_video(video_id),
                        }
                    ]
                    logger.info("Cleared conversation history for video: %s", video_id)
            else:
                self.conversation_histories.clear()
                logger.info("Cleared all conversation histories")

    def _completion_kwargs(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tool_choice: str = "auto",
    ) -> Dict[str, Any]:
        """Build completion kwargs while honoring request-level generation caps."""
        if ENABLE_THINKING:
            default_temp = 1.0
            default_top_p = 0.95
            default_top_k = 20
            default_presence_penalty = 1.5
        else:
            default_temp = 0.7
            default_top_p = 0.8
            default_top_k = 20
            default_presence_penalty = 0.0
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": tool_choice,
            "temperature": temperature if temperature is not None else default_temp,
            "top_p": default_top_p,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
                "top_k": default_top_k,
                "presence_penalty": default_presence_penalty,
            },
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return kwargs

    @staticmethod
    def _normalize_refine_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Repair common model mistakes before constructing RefineVideoTool."""
        normalized = dict(arguments)
        start_time = float(normalized["start_time"])
        end_time = float(normalized["end_time"])

        if end_time < start_time:
            logger.info(
                "Swapping refine_video bounds from [%.2fs-%.2fs]",
                start_time,
                end_time,
            )
            start_time, end_time = end_time, start_time

        if end_time == start_time:
            original_time = start_time
            start_time = max(0.0, start_time - (DEFAULT_REFINE_SEGMENT_SECONDS / 2.0))
            end_time = start_time + DEFAULT_REFINE_SEGMENT_SECONDS
            logger.info(
                "Expanding zero-length refine_video timestamp at %.2fs to [%.2fs-%.2fs]",
                original_time,
                start_time,
                end_time,
            )

        normalized["start_time"] = start_time
        normalized["end_time"] = end_time
        return normalized

    @staticmethod
    def _tool_policy(
        tool_names: List[str], iteration: int, search_rounds: int
    ) -> Tuple[str, bool]:
        """Decide whether to allow another tool round."""
        force_finalize = iteration >= MAX_TOOL_ITERATIONS
        if "search_video" in tool_names and search_rounds >= MAX_SEARCH_ROUNDS:
            force_finalize = True
        return ("none" if force_finalize else "auto"), force_finalize

    @staticmethod
    def _log_chat_metrics(
        video_id: str,
        started_at: float,
        metrics: Dict[str, Any],
    ) -> None:
        """Emit one compact metrics line per completed request."""
        logger.info(
            "Chat completed for video_id='%s' in %.2fs (%d tool rounds, %d search rounds%s)",
            video_id,
            time.perf_counter() - started_at,
            metrics["tool_iterations"],
            metrics["search_rounds"],
            ", forced finalize" if metrics["forced_finalize"] else "",
        )

    def execute_tool_call(self, tool_call, video_id: str = None) -> Dict[str, Any]:
        """Execute a tool call from the LLM."""
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        if video_id and not arguments.get("video_id"):
            arguments["video_id"] = video_id

        if function_name == "search_video":
            search_executor = self._get_search_executor()
            from utils.tools import VideoSearchTool

            modality = arguments.get("modality", ["audio", "visual"])
            if isinstance(modality, str):
                modality = [modality]
            tool_params = VideoSearchTool(
                video_id=arguments["video_id"],
                query=arguments["query"],
                modality=modality,
                max_results=arguments.get("max_results", 5),
            )
            result = search_executor.execute_search(tool_params)
            return {"success": True, "result": result}

        if function_name == "refine_video":
            missing = [
                k
                for k in ("start_time", "end_time", "modality")
                if arguments.get(k) is None
            ]
            if missing:
                raise ValueError(
                    f"Missing required argument(s) {missing} for refine_video."
                )

            video_refiner = self._get_video_refiner()
            from utils.tools import RefineVideoTool

            arguments = self._normalize_refine_arguments(arguments)
            tool_params = RefineVideoTool(
                video_id=arguments["video_id"],
                start_time=arguments["start_time"],
                end_time=arguments["end_time"],
                modality=arguments["modality"],
                query=arguments.get("query"),
            )
            result = video_refiner.execute_refinement(
                tool_params, vector_store=self.vector_store
            )
            return {"success": True, "result": result}

        if function_name == "verify_claim":
            video_refiner = self._get_video_refiner()
            from utils.tools import VerifyClaimTool

            tool_params = VerifyClaimTool(
                video_id=arguments["video_id"],
                start_time=float(arguments["start_time"]),
                end_time=float(arguments["end_time"]),
                claim=arguments["claim"],
            )
            result = video_refiner.execute_verify_claim(
                tool_params, vector_store=self.vector_store
            )
            return {"success": True, "result": result}

        raise ValueError(f"Unknown function: {function_name}")

    async def execute_tool_call_async(self, tool_call, video_id: str = None) -> Dict[str, Any]:
        """Execute a tool call from the LLM without blocking the event loop."""
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        if video_id and not arguments.get("video_id"):
            arguments["video_id"] = video_id

        if function_name == "search_video":
            if not arguments.get("query"):
                raise ValueError("Missing required argument 'query' for search_video.")

            search_executor = self._get_search_executor()
            from utils.tools import VideoSearchTool

            modality = arguments.get("modality", ["audio", "visual"])
            if isinstance(modality, str):
                modality = [modality]
            tool_params = VideoSearchTool(
                video_id=arguments["video_id"],
                query=arguments["query"],
                modality=modality,
                max_results=arguments.get("max_results", 5),
            )
            result = await search_executor.execute_search_async(tool_params)
            return {"success": True, "result": result}

        if function_name == "refine_video":
            missing = [
                k
                for k in ("start_time", "end_time", "modality")
                if arguments.get(k) is None
            ]
            if missing:
                raise ValueError(
                    f"Missing required argument(s) {missing} for refine_video."
                )

            video_refiner = self._get_video_refiner()
            from utils.tools import RefineVideoTool

            arguments = self._normalize_refine_arguments(arguments)
            tool_params = RefineVideoTool(
                video_id=arguments["video_id"],
                start_time=arguments["start_time"],
                end_time=arguments["end_time"],
                modality=arguments["modality"],
                query=arguments.get("query"),
            )
            result = await video_refiner.execute_refinement_async(
                tool_params, vector_store=self.vector_store
            )
            return {"success": True, "result": result}

        if function_name == "verify_claim":
            video_refiner = self._get_video_refiner()
            from utils.tools import VerifyClaimTool

            tool_params = VerifyClaimTool(
                video_id=arguments["video_id"],
                start_time=float(arguments["start_time"]),
                end_time=float(arguments["end_time"]),
                claim=arguments["claim"],
            )
            result = await video_refiner.execute_verify_claim_async(
                tool_params, vector_store=self.vector_store
            )
            return {"success": True, "result": result}

        raise ValueError(f"Unknown function: {function_name}")

    def _run_tool_loop(
        self,
        messages: list,
        message,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        video_id: str = None,
    ) -> Tuple[object, Dict[str, Any]]:
        """Execute tool calls until the LLM returns a final answer."""
        iteration = 0
        search_rounds = 0
        forced_finalize = False
        seen_search_hits: set = set()

        while message.tool_calls:
            iteration += 1
            tool_calls = message.tool_calls
            tool_names = [tool_call.function.name for tool_call in tool_calls]
            if "search_video" in tool_names:
                search_rounds += 1

            def _exec_tool(tc):
                try:
                    result = self.execute_tool_call(tc, video_id=video_id)
                except Exception as e:
                    logger.exception("Tool %s failed", tc.function.name)
                    result = {"success": False, "error": str(e)}
                if tc.function.name == "search_video":
                    result = self._dedup_search_result(result, seen_search_hits)
                return result

            if len(tool_calls) == 1:
                tool_call = tool_calls[0]
                logger.debug("LLM is calling tool: %s", tool_call.function.name)
                tool_result = _exec_tool(tool_call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result),
                    }
                )
                round_results = [tool_result]
            else:
                logger.debug("LLM is calling %d tools in parallel", len(tool_calls))
                round_results = []
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(tool_calls)
                ) as executor:
                    future_to_tool_call = {
                        executor.submit(_exec_tool, tool_call): tool_call
                        for tool_call in tool_calls
                    }
                    for future in concurrent.futures.as_completed(future_to_tool_call):
                        tool_call = future_to_tool_call[future]
                        tool_result = future.result()
                        round_results.append(tool_result)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(tool_result),
                            }
                        )

            all_failed = all(
                isinstance(r, dict) and not r.get("success", True)
                for r in round_results
            )
            if all_failed:
                iteration -= 1
                if "search_video" in tool_names:
                    search_rounds -= 1

            tool_choice, should_finalize = self._tool_policy(
                tool_names, iteration, search_rounds
            )
            completion_messages = messages
            if should_finalize:
                forced_finalize = True
                completion_messages = messages + [FINALIZE_TOOL_LOOP_MESSAGE.copy()]

            response = self._create_completion(
                **self._completion_kwargs(
                    completion_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tool_choice=tool_choice,
                )
            )
            message = response.choices[0].message
            messages.append(message)

            if tool_choice == "none":
                if message.tool_calls:
                    logger.warning(
                        "Model returned tool calls despite tool_choice='none'"
                    )
                break

        return message, {
            "tool_iterations": iteration,
            "search_rounds": search_rounds,
            "forced_finalize": forced_finalize,
        }

    @staticmethod
    def _dedup_search_result(result: Dict[str, Any], seen: set) -> Dict[str, Any]:
        """Strip already-seen search hits to avoid context bloat."""
        inner = result.get("result")
        if not isinstance(inner, dict) or "results" not in inner:
            return result
        deduped = {}
        for modality, items in inner["results"].items():
            if not isinstance(items, list):
                deduped[modality] = items
                continue
            unique = []
            for item in items:
                key = (
                    modality,
                    item.get("start_sec", item.get("timestamp_sec", "")),
                    item.get("end_sec", ""),
                )
                if key not in seen:
                    seen.add(key)
                    unique.append(item)
            deduped[modality] = unique
        new_total = sum(len(v) for v in deduped.values() if isinstance(v, list))
        return {
            **result,
            "result": {**inner, "results": deduped, "total_results": new_total},
        }

    async def _run_tool_loop_async(
        self,
        messages: list,
        message,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        video_id: str = None,
        tracer=None,
    ) -> Tuple[object, Dict[str, Any]]:
        """Async tool loop for API requests."""
        iteration = 0
        search_rounds = 0
        forced_finalize = False
        seen_search_hits: set = set()

        while message.tool_calls:
            iteration += 1
            tool_calls = message.tool_calls
            tool_names = [tool_call.function.name for tool_call in tool_calls]
            if "search_video" in tool_names:
                search_rounds += 1

            async def _exec_tool_async(tc):
                args = json.loads(tc.function.arguments)
                if tracer:
                    tracer.log_tool_call(iteration, tc.function.name, args)
                t0 = time.time()
                try:
                    result = await self.execute_tool_call_async(tc, video_id=video_id)
                except Exception as e:
                    logger.exception("Tool %s failed", tc.function.name)
                    result = {"success": False, "error": str(e)}
                    if tracer:
                        tracer.log_tool_error(iteration, tc.function.name, str(e))
                elapsed = time.time() - t0
                if tc.function.name == "search_video":
                    result = self._dedup_search_result(result, seen_search_hits)
                if tracer:
                    tracer.log_tool_result(iteration, tc.function.name, result, elapsed)
                return result

            if len(tool_calls) == 1:
                tool_call = tool_calls[0]
                tool_result = await _exec_tool_async(tool_call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result),
                    }
                )
                round_results = [tool_result]
            else:
                parallel_results = await asyncio.gather(
                    *(_exec_tool_async(tc) for tc in tool_calls)
                )
                for tool_call, tool_result in zip(tool_calls, parallel_results):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result),
                        }
                    )
                round_results = list(parallel_results)

            all_failed = all(
                isinstance(r, dict) and not r.get("success", True)
                for r in round_results
            )
            if all_failed:
                iteration -= 1
                if "search_video" in tool_names:
                    search_rounds -= 1

            tool_choice, should_finalize = self._tool_policy(
                tool_names, iteration, search_rounds
            )
            completion_messages = messages
            if should_finalize:
                forced_finalize = True
                completion_messages = messages + [FINALIZE_TOOL_LOOP_MESSAGE.copy()]

            total_chars = sum(
                len(json.dumps(m) if isinstance(m, dict) else m.model_dump_json())
                for m in completion_messages
            )
            if tracer:
                tracer.log_completion_request(iteration, tool_choice, len(completion_messages), total_chars)

            response = await self._create_completion_async(
                **self._completion_kwargs(
                    completion_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tool_choice=tool_choice,
                )
            )
            message = response.choices[0].message
            messages.append(message)

            reasoning = getattr(message, "reasoning_content", None) or (message.model_extra or {}).get("reasoning", "")
            if tracer:
                tracer.log_completion_response(iteration, message.content, message.tool_calls, reasoning)

            if tool_choice == "none":
                if message.tool_calls:
                    logger.warning(
                        "Model returned tool calls despite tool_choice='none'"
                    )
                break

        return message, {
            "tool_iterations": iteration,
            "search_rounds": search_rounds,
            "forced_finalize": forced_finalize,
        }

    def chat_with_messages(
        self,
        video_id: str,
        messages: List[Any],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Process a stateless API request using only request-local messages."""
        started_at = time.perf_counter()
        request_messages = self._build_stateless_messages(video_id, messages)

        try:
            response = self._create_completion(
                **self._completion_kwargs(
                    request_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tool_choice="auto",
                )
            )
            message = response.choices[0].message
            request_messages.append(message)

            message, metrics = self._run_tool_loop(
                request_messages,
                message,
                max_tokens=max_tokens,
                temperature=temperature,
                video_id=video_id,
            )
            self._log_chat_metrics(video_id, started_at, metrics)
            return _sanitize_response(message.content or "")
        except Exception:
            logger.exception("Error in chat for video_id=%s", video_id)
            raise

    async def chat_with_messages_async(
        self,
        video_id: str,
        messages: List[Any],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Async stateless chat path for FastAPI requests."""
        from utils.tracer import RequestTracer

        user_content = messages[-1]["content"] if messages else ""
        tracer = RequestTracer(video_id, user_content)

        started_at = time.perf_counter()
        request_messages = self._build_stateless_messages(video_id, messages)
        tracer.log_system_prompt(request_messages[0].get("content", "") if request_messages else "")

        try:
            response = await self._create_completion_async(
                **self._completion_kwargs(
                    request_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tool_choice="auto",
                )
            )
            message = response.choices[0].message
            request_messages.append(message)

            reasoning = getattr(message, "reasoning_content", None) or (message.model_extra or {}).get("reasoning", "")
            tracer.log_completion_response(0, message.content, message.tool_calls, reasoning)

            message, metrics = await self._run_tool_loop_async(
                request_messages,
                message,
                max_tokens=max_tokens,
                temperature=temperature,
                video_id=video_id,
                tracer=tracer,
            )
            self._log_chat_metrics(video_id, started_at, metrics)
            answer = _sanitize_response(message.content or "")
            tracer.log_answer(answer, metrics)
            return answer
        except Exception as e:
            tracer.log_error(str(e))
            logger.exception("Error in async chat for video_id=%s", video_id)
            raise

    def chat_with_video(
        self,
        video_id: str,
        user_question: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Process a user question using the stateful CLI conversation path."""
        messages = self._get_conversation_history(video_id)
        messages.append({"role": "user", "content": user_question})
        started_at = time.perf_counter()

        try:
            response = self._create_completion(
                **self._completion_kwargs(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tool_choice="auto",
                )
            )
            message = response.choices[0].message
            messages.append(message)

            message, metrics = self._run_tool_loop(
                messages,
                message,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            self._log_chat_metrics(video_id, started_at, metrics)
            return _sanitize_response(message.content or "")
        except Exception:
            logger.exception("Error in chat for video_id=%s", video_id)
            raise

    def test_connection(self) -> bool:
        """Test the connection to the vLLM server."""
        try:
            self._create_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )
            return True
        except Exception as e:
            logger.error("Connection test failed: %s", e)
            return False
