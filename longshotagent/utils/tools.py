"""
Tools for LLM integration with the video preprocessing pipeline.

All embedding models are accessed via vLLM embedding servers,
and refinement models via vLLM-served ALM/VLM servers.
All HTTP calls use connection-pooled sessions. Results are LRU-cached.
"""

import asyncio
import logging
import os
import json
import hashlib
import concurrent.futures
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Literal, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np
import ffmpeg
import httpx

from preprocessing import VectorStore
from .shared_cache import get_shared_cache
from .video_utils import VideoPathResolver

logger = logging.getLogger(__name__)


HTTP_CLIENT_LIMITS = httpx.Limits(max_connections=256, max_keepalive_connections=256)
EMBEDDING_REQUEST_TIMEOUT_SECONDS = 300.0
REFINEMENT_REQUEST_TIMEOUT_SECONDS = 300.0
SEARCH_RESULT_CACHE_SIZE = int(
    os.getenv("VIDEO_AGENT_SEARCH_RESULT_CACHE_SIZE", "1024")
)
BACKEND_REQUEST_MAX_RETRIES = int(
    os.getenv("VIDEO_AGENT_BACKEND_REQUEST_MAX_RETRIES", "2")
)
BACKEND_REQUEST_RETRY_BASE_SECONDS = float(
    os.getenv("VIDEO_AGENT_BACKEND_REQUEST_RETRY_BASE_SECONDS", "0.5")
)
EMBEDDING_BACKEND_CONCURRENCY_LIMIT = int(
    os.getenv("VIDEO_AGENT_EMBEDDING_CONCURRENCY_LIMIT", "32")
)
VLM_BACKEND_CONCURRENCY_LIMIT = int(os.getenv("VIDEO_AGENT_VLM_CONCURRENCY_LIMIT", "4"))
ALM_BACKEND_CONCURRENCY_LIMIT = int(os.getenv("VIDEO_AGENT_ALM_CONCURRENCY_LIMIT", "4"))
ENABLE_THINKING = False
SEARCH_RESULT_CACHE_TTL_SECONDS = float(
    os.getenv("VIDEO_AGENT_SEARCH_CACHE_TTL_SECONDS", "21600")
)
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
DB_QUERY_MAX_RETRIES = int(os.getenv("VIDEO_AGENT_DB_QUERY_MAX_RETRIES", "3"))
DB_QUERY_RETRY_BASE_SECONDS = float(
    os.getenv("VIDEO_AGENT_DB_QUERY_RETRY_BASE_SECONDS", "0.3")
)


def _retry_db_query(fn, description: str = "db query"):
    """Retry a ChromaDB query on transient pool/lock errors."""
    for attempt in range(DB_QUERY_MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            is_pool = "pool" in msg or "timed out" in msg or "busy" in msg
            if attempt >= DB_QUERY_MAX_RETRIES or not is_pool:
                raise
            delay = _retry_delay_seconds(attempt, DB_QUERY_RETRY_BASE_SECONDS)
            logger.warning(
                "Retrying %s after attempt %d/%d failed: %s",
                description,
                attempt + 1,
                DB_QUERY_MAX_RETRIES + 1,
                e,
            )
            time.sleep(delay)


async def _retry_db_query_async(fn, description: str = "db query"):
    """Async retry wrapper: runs each Chroma call in a thread, sleeps on the loop."""
    for attempt in range(DB_QUERY_MAX_RETRIES + 1):
        try:
            return await asyncio.to_thread(fn)
        except Exception as e:
            msg = str(e).lower()
            is_pool = "pool" in msg or "timed out" in msg or "busy" in msg
            if attempt >= DB_QUERY_MAX_RETRIES or not is_pool:
                raise
            delay = _retry_delay_seconds(attempt, DB_QUERY_RETRY_BASE_SECONDS)
            logger.warning(
                "Retrying %s after attempt %d/%d failed: %s",
                description,
                attempt + 1,
                DB_QUERY_MAX_RETRIES + 1,
                e,
            )
            await asyncio.sleep(delay)


def _timeout(seconds: float) -> httpx.Timeout:
    """Build a timeout budget that tolerates localhost queueing under load."""
    return httpx.Timeout(connect=5.0, read=seconds, write=seconds, pool=None)


def _retry_delay_seconds(attempt: int, base_delay: float) -> float:
    """Compute capped exponential backoff delay."""
    return min(base_delay * (2**attempt), 4.0)


def _is_retryable_http_error(error: Exception) -> bool:
    """Retry only transient transport/server failures."""
    if isinstance(error, httpx.TimeoutException):
        return True
    if isinstance(error, httpx.TransportError):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRYABLE_HTTP_STATUS_CODES
    return False


# ---------------------------------------------------------------------------
# Pydantic tool models
# ---------------------------------------------------------------------------


class VideoSearchTool(BaseModel):
    video_id: str = Field(
        description="The unique identifier of the video to search within"
    )
    query: str = Field(
        description="The search query or question about the video content"
    )
    modality: List[Literal["audio", "visual"]] = Field(
        description="List of modalities to search", default=["audio", "visual"]
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return per modality",
    )


class VerifyClaimTool(BaseModel):
    video_id: str = Field(description="The unique identifier of the video")
    start_time: float = Field(description="Start time in seconds", ge=0)
    end_time: float = Field(description="End time in seconds", gt=0)
    claim: str = Field(description="The claim to verify")


class RefineVideoTool(BaseModel):
    video_id: str = Field(description="The unique identifier of the video to refine")
    start_time: float = Field(description="Start time of the segment in seconds", ge=0)
    end_time: float = Field(description="End time of the segment in seconds", gt=0)
    modality: Literal["audio", "visual"] = Field(description="audio or visual")
    query: Optional[str] = Field(
        default=None, description="Optional query for visual analysis"
    )

    def model_post_init(self, __context) -> None:
        duration = self.end_time - self.start_time
        if duration <= 0:
            raise ValueError(f"Invalid segment duration ({duration:.1f}s)")
        # Audio: soft cap 60s, hard cap 300s (audio refinement is cheap).
        # Visual: hard cap 60s.
        if self.modality == "audio":
            hard_cap = 300.0
        else:
            hard_cap = 60.0
        if duration > hard_cap:
            logger.info(
                "Clamping %s refine_video segment from %.1fs to %.1fs [%ss-%ss]",
                self.modality,
                duration,
                hard_cap,
                self.start_time,
                self.end_time,
            )
            self.end_time = self.start_time + hard_cap


# ---------------------------------------------------------------------------
# Search executor — embedding queries + ChromaDB
# ---------------------------------------------------------------------------


class VideoSearchExecutor:
    """
    Executes vector search via vLLM embedding servers + ChromaDB.
    Uses connection-pooled HTTP sessions and per-query embedding cache.
    """

    TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_CACHE_SIZE = 256

    _siglip_model = None
    _siglip_tokenizer = None
    _siglip_device = None
    _siglip_lock = threading.Lock()

    @classmethod
    def _ensure_siglip(cls):
        """Load the open_clip SigLIP text encoder (shared across instances)."""
        if cls._siglip_tokenizer is not None:
            return
        with cls._siglip_lock:
            if cls._siglip_tokenizer is not None:
                return
            import open_clip
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-16-SigLIP-512", pretrained="webli", device=device,
            )
            model.eval()
            tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP-512")
            cls._siglip_device = device
            cls._siglip_model = model
            cls._siglip_tokenizer = tokenizer
            logger.info("Loaded SigLIP text encoder on %s for visual query embedding", device)

    def __init__(
        self,
        db_path: str = "./chroma_db",
        text_embedding_url: str = "http://localhost:8014/v1",
        visual_embedding_url: str = "http://localhost:8018/v1",
        vector_store: Optional[VectorStore] = None,
    ):
        self.vector_store = vector_store or VectorStore(db_path=db_path)
        self.text_embedding_url = text_embedding_url

        # Shared connection pools for sync and async callers (audio embeddings).
        self._text_client = httpx.Client(
            limits=HTTP_CLIENT_LIMITS,
            timeout=_timeout(EMBEDDING_REQUEST_TIMEOUT_SECONDS),
        )
        self._async_text_client = httpx.AsyncClient(
            limits=HTTP_CLIENT_LIMITS,
            timeout=_timeout(EMBEDDING_REQUEST_TIMEOUT_SECONDS),
        )

        # LRU embedding cache: query text → numpy array
        self._audio_embed_cache: Dict[str, np.ndarray] = {}
        self._visual_embed_cache: Dict[str, np.ndarray] = {}
        self._search_result_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        self._embedding_inflight: Dict[
            Tuple[str, str, str], concurrent.futures.Future
        ] = {}
        self._search_inflight: Dict[str, concurrent.futures.Future] = {}
        self._sync_embedding_slots = threading.BoundedSemaphore(
            EMBEDDING_BACKEND_CONCURRENCY_LIMIT
        )
        self._async_embedding_slots = asyncio.Semaphore(
            EMBEDDING_BACKEND_CONCURRENCY_LIMIT
        )
        self._shared_cache = get_shared_cache()

    def close(self):
        """Close sync HTTP clients."""
        self._text_client.close()

    async def aclose(self):
        """Close async HTTP clients."""
        await self._async_text_client.aclose()

    def _get_embedding_future(
        self,
        query: str,
        cache: Dict[str, np.ndarray],
        inflight_key: Tuple[str, str, str],
    ) -> Tuple[Optional[np.ndarray], concurrent.futures.Future, bool]:
        """Return a cache hit or a shared future for the in-flight request."""
        with self._cache_lock:
            cached = cache.get(query)
            if cached is not None:
                return cached, concurrent.futures.Future(), False

            future = self._embedding_inflight.get(inflight_key)
            if future is None:
                future = concurrent.futures.Future()
                self._embedding_inflight[inflight_key] = future
                return None, future, True

            return None, future, False

    def _finish_embedding_future(
        self,
        query: str,
        cache: Dict[str, np.ndarray],
        inflight_key: Tuple[str, str, str],
        future: concurrent.futures.Future,
        embedding: np.ndarray,
    ) -> None:
        """Store a completed embedding and wake any waiters."""
        with self._cache_lock:
            if len(cache) >= self.EMBEDDING_CACHE_SIZE:
                cache.pop(next(iter(cache)))
            cache[query] = embedding
            self._embedding_inflight.pop(inflight_key, None)
        future.set_result(embedding)

    def _fail_embedding_future(
        self,
        inflight_key: Tuple[str, str, str],
        future: concurrent.futures.Future,
        error: Exception,
    ) -> None:
        """Propagate an embedding failure to all waiters."""
        with self._cache_lock:
            self._embedding_inflight.pop(inflight_key, None)
        future.set_exception(error)

    @staticmethod
    def _search_cache_key(tool_params: VideoSearchTool) -> str:
        """Build a stable cache key for identical search requests."""
        return json.dumps(
            {
                "video_id": tool_params.video_id,
                "query": tool_params.query,
                "modality": sorted(tool_params.modality),
                "max_results": tool_params.max_results,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def _get_search_future(
        self, cache_key: str
    ) -> Tuple[Optional[Dict[str, Any]], concurrent.futures.Future, bool]:
        """Return a cache hit or a shared future for an in-flight search."""
        with self._cache_lock:
            cached = self._search_result_cache.get(cache_key)
            if cached is not None:
                return cached, concurrent.futures.Future(), False

            future = self._search_inflight.get(cache_key)
            if future is None:
                future = concurrent.futures.Future()
                self._search_inflight[cache_key] = future
                return None, future, True

            return None, future, False

    def _finish_search_future(
        self,
        cache_key: str,
        future: concurrent.futures.Future,
        result: Dict[str, Any],
    ) -> None:
        """Store a completed search result and wake any waiters."""
        with self._cache_lock:
            if len(self._search_result_cache) >= SEARCH_RESULT_CACHE_SIZE:
                self._search_result_cache.pop(next(iter(self._search_result_cache)))
            self._search_result_cache[cache_key] = result
            self._search_inflight.pop(cache_key, None)
        future.set_result(result)

    def _fail_search_future(
        self,
        cache_key: str,
        future: concurrent.futures.Future,
        error: Exception,
    ) -> None:
        """Propagate an unexpected search failure to all waiters."""
        with self._cache_lock:
            self._search_inflight.pop(cache_key, None)
        future.set_exception(error)

    def _get_cached_search_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check local memory first, then the shared cross-worker cache."""
        with self._cache_lock:
            cached = self._search_result_cache.get(cache_key)
        if cached is not None:
            return cached
        if self._shared_cache is None:
            return None
        cached = self._shared_cache.get_json("search_result", cache_key)
        if cached is None:
            return None
        with self._cache_lock:
            if len(self._search_result_cache) >= SEARCH_RESULT_CACHE_SIZE:
                self._search_result_cache.pop(next(iter(self._search_result_cache)))
            self._search_result_cache[cache_key] = cached
        return cached

    def _store_shared_search_result(
        self, cache_key: str, result: Dict[str, Any]
    ) -> None:
        """Persist a deterministic search result for other workers."""
        if self._shared_cache is None:
            return
        self._shared_cache.set_json(
            "search_result",
            cache_key,
            result,
            ttl_seconds=SEARCH_RESULT_CACHE_TTL_SECONDS,
        )

    def _encode_text_via_api(
        self,
        query: str,
        client: httpx.Client,
        base_url: str,
        model: str,
        cache: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Encode text via the embedding server with caching and in-flight dedupe."""
        inflight_key = (base_url, model, query)
        cached, future, is_leader = self._get_embedding_future(
            query, cache, inflight_key
        )
        if cached is not None:
            return cached
        if not is_leader:
            return future.result()

        try:
            response = None
            with self._sync_embedding_slots:
                for attempt in range(BACKEND_REQUEST_MAX_RETRIES + 1):
                    try:
                        response = client.post(
                            f"{base_url}/embeddings",
                            json={"model": model, "input": query},
                            timeout=_timeout(EMBEDDING_REQUEST_TIMEOUT_SECONDS),
                        )
                        response.raise_for_status()
                        break
                    except Exception as e:
                        if (
                            attempt >= BACKEND_REQUEST_MAX_RETRIES
                            or not _is_retryable_http_error(e)
                        ):
                            raise
                        delay = _retry_delay_seconds(
                            attempt, BACKEND_REQUEST_RETRY_BASE_SECONDS
                        )
                        logger.warning(
                            "Retrying embedding request after attempt %d/%d failed: %s",
                            attempt + 1,
                            BACKEND_REQUEST_MAX_RETRIES + 1,
                            str(e) or type(e).__name__,
                        )
                        time.sleep(delay)
            embedding = np.array(
                response.json()["data"][0]["embedding"], dtype=np.float32
            )
            self._finish_embedding_future(query, cache, inflight_key, future, embedding)
            return embedding
        except Exception as e:
            self._fail_embedding_future(inflight_key, future, e)
            raise

    async def _encode_text_via_api_async(
        self,
        query: str,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        cache: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Async embedding call with shared in-flight dedupe."""
        inflight_key = (base_url, model, query)
        cached, future, is_leader = self._get_embedding_future(
            query, cache, inflight_key
        )
        if cached is not None:
            return cached
        if not is_leader:
            return await asyncio.wrap_future(future)

        try:
            response = None
            async with self._async_embedding_slots:
                for attempt in range(BACKEND_REQUEST_MAX_RETRIES + 1):
                    try:
                        response = await client.post(
                            f"{base_url}/embeddings",
                            json={"model": model, "input": query},
                            timeout=_timeout(EMBEDDING_REQUEST_TIMEOUT_SECONDS),
                        )
                        response.raise_for_status()
                        break
                    except Exception as e:
                        if (
                            attempt >= BACKEND_REQUEST_MAX_RETRIES
                            or not _is_retryable_http_error(e)
                        ):
                            raise
                        delay = _retry_delay_seconds(
                            attempt, BACKEND_REQUEST_RETRY_BASE_SECONDS
                        )
                        logger.warning(
                            "Retrying async embedding request after attempt %d/%d failed: %s",
                            attempt + 1,
                            BACKEND_REQUEST_MAX_RETRIES + 1,
                            str(e) or type(e).__name__,
                        )
                        await asyncio.sleep(delay)
            embedding = np.array(
                response.json()["data"][0]["embedding"], dtype=np.float32
            )
            self._finish_embedding_future(query, cache, inflight_key, future, embedding)
            return embedding
        except Exception as e:
            self._fail_embedding_future(inflight_key, future, e)
            raise

    def _encode_query_for_audio(self, query: str) -> np.ndarray:
        return self._encode_text_via_api(
            query,
            self._text_client,
            self.text_embedding_url,
            self.TEXT_MODEL,
            self._audio_embed_cache,
        )

    def _encode_query_for_visual(self, query: str) -> np.ndarray:
        with self._cache_lock:
            cached = self._visual_embed_cache.get(query)
        if cached is not None:
            return cached
        self._ensure_siglip()
        import torch
        with torch.no_grad():
            tokens = self._siglip_tokenizer([query]).to(self._siglip_device)
            emb = self._siglip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            result = emb.cpu().numpy().flatten().astype(np.float32)
        with self._cache_lock:
            if len(self._visual_embed_cache) < self.EMBEDDING_CACHE_SIZE:
                self._visual_embed_cache[query] = result
        return result

    async def _encode_query_for_audio_async(self, query: str) -> np.ndarray:
        return await self._encode_text_via_api_async(
            query,
            self._async_text_client,
            self.text_embedding_url,
            self.TEXT_MODEL,
            self._audio_embed_cache,
        )

    async def _encode_query_for_visual_async(self, query: str) -> np.ndarray:
        return await asyncio.to_thread(self._encode_query_for_visual, query)

    def _query_audio_results(
        self, tool_params: VideoSearchTool, audio_embedding: np.ndarray
    ) -> List[Dict[str, Any]]:
        raw = _retry_db_query(
            lambda: self.vector_store.query_audio_embeddings(
                query_embedding=audio_embedding,
                video_id=tool_params.video_id,
                n_results=tool_params.max_results,
            ),
            "audio query",
        )
        return [
            {
                "text": r.content or "",
                "score": round(1.0 - r.distance, 3),
                "start_sec": round(r.metadata.get("start_time_ms", 0) / 1000, 2),
                "end_sec": round(r.metadata.get("end_time_ms", 0) / 1000, 2),
            }
            for r in raw
        ]

    def _query_visual_results(
        self, tool_params: VideoSearchTool, visual_embedding: np.ndarray
    ) -> List[Dict[str, Any]]:
        raw = _retry_db_query(
            lambda: self.vector_store.query_visual_embeddings(
                query_embedding=visual_embedding,
                video_id=tool_params.video_id,
                n_results=tool_params.max_results,
            ),
            "visual query",
        )
        return [
            {
                "score": round(1.0 - r.distance, 3),
                "timestamp_sec": round(r.metadata.get("timestamp_ms", 0) / 1000, 2),
                "frame_number": r.metadata.get("frame_number", 0),
            }
            for r in raw
        ]

    async def _query_audio_results_async(
        self, tool_params: VideoSearchTool, audio_embedding: np.ndarray
    ) -> List[Dict[str, Any]]:
        raw = await _retry_db_query_async(
            lambda: self.vector_store.query_audio_embeddings(
                query_embedding=audio_embedding,
                video_id=tool_params.video_id,
                n_results=tool_params.max_results,
            ),
            "audio query",
        )
        return [
            {
                "text": r.content or "",
                "score": round(1.0 - r.distance, 3),
                "start_sec": round(r.metadata.get("start_time_ms", 0) / 1000, 2),
                "end_sec": round(r.metadata.get("end_time_ms", 0) / 1000, 2),
            }
            for r in raw
        ]

    async def _query_visual_results_async(
        self, tool_params: VideoSearchTool, visual_embedding: np.ndarray
    ) -> List[Dict[str, Any]]:
        raw = await _retry_db_query_async(
            lambda: self.vector_store.query_visual_embeddings(
                query_embedding=visual_embedding,
                video_id=tool_params.video_id,
                n_results=tool_params.max_results,
            ),
            "visual query",
        )
        return [
            {
                "score": round(1.0 - r.distance, 3),
                "timestamp_sec": round(r.metadata.get("timestamp_ms", 0) / 1000, 2),
                "frame_number": r.metadata.get("frame_number", 0),
            }
            for r in raw
        ]

    def execute_search(self, tool_params: VideoSearchTool) -> Dict[str, Any]:
        """
        Execute search with pipelined encoding + parallel DB queries.

        Pipeline:
          1. Encode audio + visual queries in parallel (both are HTTP calls)
          2. Run ChromaDB queries in parallel using pre-computed embeddings
        """
        logger.debug(
            "Search: video_id='%s', query='%s'",
            tool_params.video_id,
            tool_params.query,
        )

        cache_key = self._search_cache_key(tool_params)
        cached = self._get_cached_search_result(cache_key)
        if cached is not None:
            return cached
        cached, future, is_leader = self._get_search_future(cache_key)
        if cached is not None:
            return cached
        if not is_leader:
            return future.result()

        try:
            want_audio = "audio" in tool_params.modality
            want_visual = "visual" in tool_params.modality

            # Phase 1: Encode queries in parallel
            audio_embedding = None
            visual_embedding = None

            if want_audio and want_visual:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    audio_fut = pool.submit(
                        self._encode_query_for_audio, tool_params.query
                    )
                    visual_fut = pool.submit(
                        self._encode_query_for_visual, tool_params.query
                    )
                    audio_embedding = audio_fut.result()
                    visual_embedding = visual_fut.result()
            elif want_audio:
                audio_embedding = self._encode_query_for_audio(tool_params.query)
            elif want_visual:
                visual_embedding = self._encode_query_for_visual(tool_params.query)

            # Phase 2: Run DB queries in parallel with pre-computed embeddings
            results: Dict[str, list] = {}

            if want_audio and want_visual:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    af = pool.submit(
                        self._query_audio_results, tool_params, audio_embedding
                    )
                    vf = pool.submit(
                        self._query_visual_results, tool_params, visual_embedding
                    )
                    results["audio"] = af.result()
                    results["visual"] = vf.result()
            elif want_audio:
                results["audio"] = self._query_audio_results(
                    tool_params, audio_embedding
                )
            elif want_visual:
                results["visual"] = self._query_visual_results(
                    tool_params, visual_embedding
                )

            total = sum(len(v) for v in results.values())
            logger.debug("Search completed: %d results", total)

            result = {
                "video_id": tool_params.video_id,
                "query": tool_params.query,
                "results": results,
                "total_results": total,
            }
            self._store_shared_search_result(cache_key, result)
            self._finish_search_future(cache_key, future, result)
            return result
        except Exception as e:
            self._fail_search_future(cache_key, future, e)
            raise

    async def execute_search_async(
        self, tool_params: VideoSearchTool
    ) -> Dict[str, Any]:
        """Async search path for API requests."""
        logger.debug(
            "Search: video_id='%s', query='%s'",
            tool_params.video_id,
            tool_params.query,
        )

        cache_key = self._search_cache_key(tool_params)
        cached = await asyncio.to_thread(self._get_cached_search_result, cache_key)
        if cached is not None:
            return cached
        cached, future, is_leader = self._get_search_future(cache_key)
        if cached is not None:
            return cached
        if not is_leader:
            return await asyncio.wrap_future(future)

        try:
            want_audio = "audio" in tool_params.modality
            want_visual = "visual" in tool_params.modality

            audio_embedding = None
            visual_embedding = None

            if want_audio and want_visual:
                audio_embedding, visual_embedding = await asyncio.gather(
                    self._encode_query_for_audio_async(tool_params.query),
                    self._encode_query_for_visual_async(tool_params.query),
                )
            elif want_audio:
                audio_embedding = await self._encode_query_for_audio_async(
                    tool_params.query
                )
            elif want_visual:
                visual_embedding = await self._encode_query_for_visual_async(
                    tool_params.query
                )

            results: Dict[str, list] = {}

            if want_audio and want_visual:
                audio_results, visual_results = await asyncio.gather(
                    self._query_audio_results_async(tool_params, audio_embedding),
                    self._query_visual_results_async(tool_params, visual_embedding),
                )
                results["audio"] = audio_results
                results["visual"] = visual_results
            elif want_audio:
                results["audio"] = await self._query_audio_results_async(
                    tool_params, audio_embedding
                )
            elif want_visual:
                results["visual"] = await self._query_visual_results_async(
                    tool_params, visual_embedding
                )

            total = sum(len(v) for v in results.values())
            logger.debug("Search completed: %d results", total)

            result = {
                "video_id": tool_params.video_id,
                "query": tool_params.query,
                "results": results,
                "total_results": total,
            }
            await asyncio.to_thread(
                self._store_shared_search_result, cache_key, result
            )
            self._finish_search_future(cache_key, future, result)
            return result
        except Exception as e:
            self._fail_search_future(cache_key, future, e)
            raise


# Global executor instance
_search_executor = None
_search_executor_lock = threading.Lock()


def get_search_executor(
    db_path: str = "./chroma_db",
    text_embedding_url: str = "http://localhost:8014/v1",
    visual_embedding_url: str = "http://localhost:8018/v1",
    vector_store: Optional[VectorStore] = None,
) -> VideoSearchExecutor:
    """Get or create the global search executor instance."""
    global _search_executor
    with _search_executor_lock:
        if _search_executor is None:
            _search_executor = VideoSearchExecutor(
                db_path=db_path,
                text_embedding_url=text_embedding_url,
                visual_embedding_url=visual_embedding_url,
                vector_store=vector_store,
            )
        elif vector_store is not None:
            _search_executor.vector_store = vector_store
    return _search_executor


# ---------------------------------------------------------------------------
# Tool definitions for OpenAI function calling
# ---------------------------------------------------------------------------

VIDEO_SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_video",
        "description": "Search through video content including audio transcriptions and visual frames using semantic similarity.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query about the video",
                },
                "modality": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["audio", "visual"]},
                    "description": "Modalities to search",
                    "default": ["audio", "visual"],
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "Max results per modality",
                },
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Video refiner — segment extraction + ALM/VLM analysis
# ---------------------------------------------------------------------------


class VideoRefiner:
    """
    Extracts and analyzes video segments via ALM (audio) and VLM (visual).
    Uses connection-pooled sessions, LRU caches for duration/path lookups,
    and result caching for identical refinement requests.
    """

    REFINEMENT_CACHE_SIZE = 128

    def __init__(
        self,
        vlm_base_url: str = "http://localhost:8011/v1",
        vlm_model_name: str = "Qwen/Qwen3.6-35B-A3B",
        alm_base_url: str = "http://localhost:8013/v1",
        alm_model_name: str = "nvidia/audio-flamingo-3-hf",
        video_search_paths: list = None,
    ):
        self.vlm_base_url = vlm_base_url
        self.vlm_model_name = vlm_model_name
        self.alm_base_url = alm_base_url
        self.alm_model_name = alm_model_name
        self.video_search_paths = video_search_paths or ["./videos"]
        self._path_resolver = VideoPathResolver(self.video_search_paths)

        # Shared connection pools for sync and async callers.
        self._vlm_client = httpx.Client(
            limits=HTTP_CLIENT_LIMITS,
            timeout=_timeout(REFINEMENT_REQUEST_TIMEOUT_SECONDS),
        )
        self._alm_client = httpx.Client(
            limits=HTTP_CLIENT_LIMITS,
            timeout=_timeout(REFINEMENT_REQUEST_TIMEOUT_SECONDS),
        )
        self._async_vlm_client = httpx.AsyncClient(
            limits=HTTP_CLIENT_LIMITS,
            timeout=_timeout(REFINEMENT_REQUEST_TIMEOUT_SECONDS),
        )
        self._async_alm_client = httpx.AsyncClient(
            limits=HTTP_CLIENT_LIMITS,
            timeout=_timeout(REFINEMENT_REQUEST_TIMEOUT_SECONDS),
        )

        # LRU caches
        self._duration_cache: Dict[str, float] = {}  # video_path → duration
        self._refinement_cache: Dict[str, Dict] = {}  # cache_key → result
        self._cache_lock = threading.RLock()
        self._refinement_inflight: Dict[str, concurrent.futures.Future] = {}
        self._sync_vlm_slots = threading.BoundedSemaphore(VLM_BACKEND_CONCURRENCY_LIMIT)
        self._async_vlm_slots = asyncio.Semaphore(VLM_BACKEND_CONCURRENCY_LIMIT)
        self._sync_alm_slots = threading.BoundedSemaphore(ALM_BACKEND_CONCURRENCY_LIMIT)
        self._async_alm_slots = asyncio.Semaphore(ALM_BACKEND_CONCURRENCY_LIMIT)

        # Tmp directory for cached segments — cleared on init
        self.tmp_dir = Path("./tmp/video_segments")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._clear_tmp_dir()

    def close(self):
        """Close sync HTTP clients."""
        self._vlm_client.close()
        self._alm_client.close()

    async def aclose(self):
        """Close async HTTP clients."""
        await self._async_vlm_client.aclose()
        await self._async_alm_client.aclose()

    def _clear_tmp_dir(self):
        """Remove all cached segments on startup."""
        try:
            count = 0
            for entry in self.tmp_dir.iterdir():
                if entry.is_file():
                    entry.unlink(missing_ok=True)
                    count += 1
            if count:
                logger.info("Cleared %d cached segments from %s", count, self.tmp_dir)
        except Exception as e:
            logger.warning("Temp cleanup error: %s", e)

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration with cache."""
        with self._cache_lock:
            cached_duration = self._duration_cache.get(video_path)
        if cached_duration is not None:
            return cached_duration
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(
                probe["streams"][0].get("duration", 0) or probe["format"]["duration"]
            )
        except Exception:
            duration = 0.0
        with self._cache_lock:
            self._duration_cache[video_path] = duration
        return duration

    def _find_video_file(
        self, video_id: str, original_path: str = None
    ) -> Optional[str]:
        """Find video file with cache, delegating to shared VideoPathResolver."""
        return self._path_resolver.find_video_file(video_id, original_path)

    _MIN_AUDIO_DURATION = 2.0

    def _extract_audio_segment(
        self, video_path: str, start_time: float, end_time: float
    ) -> str:
        """Extract audio segment with caching."""
        video_dur = self._get_video_duration(video_path)
        if video_dur > 0:
            if start_time >= video_dur:
                logger.warning(
                    "Clamping audio start_time %.1fs to %.1fs (video duration %.1fs)",
                    start_time, max(video_dur - 10, 0), video_dur,
                )
                start_time = max(video_dur - 10, 0)
            if end_time > video_dur:
                end_time = video_dur
        duration = end_time - start_time
        if duration <= 0:
            raise ValueError(f"Invalid duration: {duration}s")
        if duration < self._MIN_AUDIO_DURATION:
            video_dur = self._get_video_duration(video_path)
            needed = self._MIN_AUDIO_DURATION - duration
            end_time = min(end_time + needed, video_dur) if video_dur > 0 else end_time + needed
            if end_time - start_time < self._MIN_AUDIO_DURATION:
                start_time = max(0.0, end_time - self._MIN_AUDIO_DURATION)
            duration = end_time - start_time

        seg_hash = hashlib.md5(
            f"audio_{video_path}_{start_time}_{end_time}".encode()
        ).hexdigest()[:8]
        cached_path = self.tmp_dir / f"audio_{start_time}s-{end_time}s_{seg_hash}.wav"

        if cached_path.exists():
            return str(cached_path)

        (
            ffmpeg.input(video_path, ss=start_time, t=duration)
            .output(
                str(cached_path),
                acodec="pcm_s16le",
                ac=1,
                ar=16000,
                **{"threads": 4, "avoid_negative_ts": "make_zero"},
            )
            .overwrite_output()
            .run(quiet=True, capture_stdout=True)
        )
        if not cached_path.exists() or cached_path.stat().st_size == 0:
            raise RuntimeError("Audio extraction failed")
        return str(cached_path)

    def _extract_video_segment(
        self, video_path: str, start_time: float, end_time: float, video_id: str
    ) -> str:
        """Extract and re-encode a video segment to h264/yuv420p, cached."""
        duration = end_time - start_time
        if duration <= 0:
            raise ValueError(f"Invalid duration: {duration}s")

        video_duration = self._get_video_duration(video_path)
        if video_duration > 0:
            if start_time >= video_duration:
                start_time = max(video_duration - 10, 0)
            if end_time > video_duration:
                end_time = video_duration
            duration = end_time - start_time
            if duration <= 0:
                start_time = max(video_duration - 10, 0)
                end_time = video_duration
                duration = end_time - start_time

        seg_hash = hashlib.md5(
            f"{video_id}_{start_time}_{end_time}".encode()
        ).hexdigest()[:8]
        segment_path = (
            self.tmp_dir / f"{video_id}_{start_time}s-{end_time}s_{seg_hash}.mp4"
        )

        if segment_path.exists():
            return str(segment_path)

        (
            ffmpeg.input(video_path, ss=start_time, t=duration)
            .output(
                str(segment_path),
                vcodec="libx264",
                acodec="aac",
                pix_fmt="yuv420p",
                **{
                    "crf": 28,
                    "preset": "ultrafast",
                    "threads": 4,
                    "movflags": "+faststart",
                    "avoid_negative_ts": "make_zero",
                },
            )
            .overwrite_output()
            .run(capture_stdout=True, quiet=True)
        )
        return str(segment_path)

    def _refinement_cache_key(
        self,
        video_id: str,
        start: float,
        end: float,
        modality: str,
        query: Optional[str] = None,
    ) -> str:
        """Deterministic cache key for refinement results."""
        return f"{video_id}|{start}|{end}|{modality}|{query or ''}"

    def _get_refinement_future(
        self, cache_key: str
    ) -> Tuple[Optional[Dict[str, Any]], concurrent.futures.Future, bool]:
        """Return a cache hit or a shared future for the in-flight refinement."""
        with self._cache_lock:
            cached = self._refinement_cache.get(cache_key)
            if cached is not None:
                return cached, concurrent.futures.Future(), False

            future = self._refinement_inflight.get(cache_key)
            if future is None:
                future = concurrent.futures.Future()
                self._refinement_inflight[cache_key] = future
                return None, future, True

            return None, future, False

    def _finish_refinement_future(
        self,
        cache_key: str,
        future: concurrent.futures.Future,
        result: Dict[str, Any],
    ) -> None:
        """Store a completed refinement and wake any waiters."""
        with self._cache_lock:
            if result.get("success"):
                if len(self._refinement_cache) >= self.REFINEMENT_CACHE_SIZE:
                    self._refinement_cache.pop(next(iter(self._refinement_cache)))
                self._refinement_cache[cache_key] = result
            self._refinement_inflight.pop(cache_key, None)
        future.set_result(result)

    def _fail_refinement_future(
        self,
        cache_key: str,
        future: concurrent.futures.Future,
        error: Exception,
    ) -> None:
        """Propagate a refinement failure to all waiters."""
        with self._cache_lock:
            self._refinement_inflight.pop(cache_key, None)
        future.set_exception(error)

    def refine_audio(
        self, video_path: str, start_time: float, end_time: float,
        query: str = None,
    ) -> Dict[str, Any]:
        """Refine audio via ALM with connection-pooled session."""
        try:
            audio_path = os.path.abspath(
                self._extract_audio_segment(video_path, start_time, end_time)
            )

            if query:
                audio_prompt = f"Listen to this audio and answer: {query}\nDescribe all sounds: speech, music, background noise, instruments."
            else:
                audio_prompt = "Describe all sounds in this audio: speech (transcribe it), music, instruments, background noise. Be comprehensive."

            response = None
            with self._sync_alm_slots:
                for attempt in range(BACKEND_REQUEST_MAX_RETRIES + 1):
                    try:
                        response = self._alm_client.post(
                            f"{self.alm_base_url}/chat/completions",
                            json={
                                "model": self.alm_model_name,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "audio_url",
                                                "audio_url": {
                                                    "url": f"file://{audio_path}"
                                                },
                                            },
                                            {
                                                "type": "text",
                                                "text": audio_prompt,
                                            },
                                        ],
                                    }
                                ],
                                "max_tokens": 1024,
                                "temperature": 0.0,
                            },
                            timeout=_timeout(REFINEMENT_REQUEST_TIMEOUT_SECONDS),
                        )
                        response.raise_for_status()
                        break
                    except Exception as e:
                        if (
                            attempt >= BACKEND_REQUEST_MAX_RETRIES
                            or not _is_retryable_http_error(e)
                        ):
                            raise
                        delay = _retry_delay_seconds(
                            attempt, BACKEND_REQUEST_RETRY_BASE_SECONDS
                        )
                    logger.warning(
                        "Retrying audio refinement after attempt %d/%d failed: %s",
                        attempt + 1,
                        BACKEND_REQUEST_MAX_RETRIES + 1,
                        str(e) or type(e).__name__,
                    )
                    time.sleep(delay)
            transcription = response.json()["choices"][0]["message"]["content"]

            return {
                "success": True,
                "transcription": transcription,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
            }
        except httpx.HTTPStatusError as e:
            detail = e.response.text.strip()
            if len(detail) > 500:
                detail = detail[:500] + "..."
            logger.error("Audio refinement HTTP error: %s | body=%s", e, detail)
            return {
                "success": False,
                "error": f"{e} | body={detail}",
                "start_time": start_time,
                "end_time": end_time,
            }
        except Exception as e:
            logger.error("Audio refinement error: %s", e)
            return {
                "success": False,
                "error": str(e),
                "start_time": start_time,
                "end_time": end_time,
            }

    async def refine_audio_async(
        self, video_path: str, start_time: float, end_time: float,
        query: str = None,
    ) -> Dict[str, Any]:
        """Async audio refinement via ALM."""
        try:
            audio_path = os.path.abspath(
                await asyncio.to_thread(
                    self._extract_audio_segment, video_path, start_time, end_time
                )
            )

            if query:
                audio_prompt = f"Listen to this audio and answer: {query}\nDescribe all sounds: speech, music, background noise, instruments."
            else:
                audio_prompt = "Describe all sounds in this audio: speech (transcribe it), music, instruments, background noise. Be comprehensive."

            response = None
            async with self._async_alm_slots:
                for attempt in range(BACKEND_REQUEST_MAX_RETRIES + 1):
                    try:
                        response = await self._async_alm_client.post(
                            f"{self.alm_base_url}/chat/completions",
                            json={
                                "model": self.alm_model_name,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "audio_url",
                                                "audio_url": {
                                                    "url": f"file://{audio_path}"
                                                },
                                            },
                                            {
                                                "type": "text",
                                                "text": audio_prompt,
                                            },
                                        ],
                                    }
                                ],
                                "max_tokens": 1024,
                                "temperature": 0.0,
                            },
                            timeout=_timeout(REFINEMENT_REQUEST_TIMEOUT_SECONDS),
                        )
                        response.raise_for_status()
                        break
                    except Exception as e:
                        if (
                            attempt >= BACKEND_REQUEST_MAX_RETRIES
                            or not _is_retryable_http_error(e)
                        ):
                            raise
                        delay = _retry_delay_seconds(
                            attempt, BACKEND_REQUEST_RETRY_BASE_SECONDS
                        )
                        logger.warning(
                            "Retrying async audio refinement after attempt %d/%d failed: %s",
                            attempt + 1,
                            BACKEND_REQUEST_MAX_RETRIES + 1,
                            str(e) or type(e).__name__,
                        )
                        await asyncio.sleep(delay)
            transcription = response.json()["choices"][0]["message"]["content"]

            return {
                "success": True,
                "transcription": transcription,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
            }
        except httpx.HTTPStatusError as e:
            detail = e.response.text.strip()
            if len(detail) > 500:
                detail = detail[:500] + "..."
            logger.error("Audio refinement HTTP error: %s | body=%s", e, detail)
            return {
                "success": False,
                "error": f"{e} | body={detail}",
                "start_time": start_time,
                "end_time": end_time,
            }
        except Exception as e:
            logger.error("Audio refinement error: %s", e)
            return {
                "success": False,
                "error": str(e),
                "start_time": start_time,
                "end_time": end_time,
            }

    def refine_visual(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        video_id: str,
        query: Optional[str] = None,
        use_original: bool = False,
    ) -> Dict[str, Any]:
        """Refine visual via VLM with connection-pooled session."""
        try:
            if use_original:
                segment_video = os.path.abspath(video_path)
            else:
                segment_video = os.path.abspath(
                    self._extract_video_segment(video_path, start_time, end_time, video_id)
                )

            prompt = (
                query
                if query
                else "Please provide a detailed description of everything visible in this video segment, "
                "including objects, people, actions, scenes, text, and any other relevant visual information."
            )

            response = None
            with self._sync_vlm_slots:
                for attempt in range(BACKEND_REQUEST_MAX_RETRIES + 1):
                    try:
                        response = self._vlm_client.post(
                            f"{self.vlm_base_url}/chat/completions",
                            json={
                                "model": self.vlm_model_name,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "video_url",
                                                "video_url": {
                                                    "url": f"file://{segment_video}"
                                                },
                                            },
                                            {"type": "text", "text": prompt},
                                        ],
                                    }
                                ],
                                "max_tokens": 2048,
                                "temperature": 0.0,
                                "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
                            },
                            timeout=_timeout(REFINEMENT_REQUEST_TIMEOUT_SECONDS),
                        )
                        response.raise_for_status()
                        break
                    except Exception as e:
                        if (
                            attempt >= BACKEND_REQUEST_MAX_RETRIES
                            or not _is_retryable_http_error(e)
                        ):
                            raise
                        delay = _retry_delay_seconds(
                            attempt, BACKEND_REQUEST_RETRY_BASE_SECONDS
                        )
                        logger.warning(
                            "Retrying visual refinement after attempt %d/%d failed: %s",
                            attempt + 1,
                            BACKEND_REQUEST_MAX_RETRIES + 1,
                            str(e) or type(e).__name__,
                        )
                    time.sleep(delay)
            description = response.json()["choices"][0]["message"]["content"]

            return {
                "success": True,
                "description": description,
                "query": query,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
            }
        except httpx.HTTPStatusError as e:
            detail = e.response.text.strip()
            if len(detail) > 500:
                detail = detail[:500] + "..."
            logger.error("Visual refinement HTTP error: %s | body=%s", e, detail)
            return {
                "success": False,
                "error": f"{e} | body={detail}",
                "start_time": start_time,
                "end_time": end_time,
                "query": query,
            }
        except Exception as e:
            logger.error("Visual refinement error: %s", e)
            return {
                "success": False,
                "error": str(e),
                "start_time": start_time,
                "end_time": end_time,
                "query": query,
            }

    @staticmethod
    def _is_avcodec_error(error: httpx.HTTPStatusError) -> bool:
        """Check if a VLM 400 error is an avcodec decoding issue."""
        return error.response.status_code == 400 and "avcodec" in error.response.text

    async def _vlm_request_async(self, segment_video: str, prompt: str):
        """Send a single VLM request, return the response."""
        async with self._async_vlm_slots:
            for attempt in range(BACKEND_REQUEST_MAX_RETRIES + 1):
                try:
                    response = await self._async_vlm_client.post(
                        f"{self.vlm_base_url}/chat/completions",
                        json={
                            "model": self.vlm_model_name,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "video_url",
                                            "video_url": {
                                                "url": f"file://{segment_video}"
                                            },
                                        },
                                        {"type": "text", "text": prompt},
                                    ],
                                }
                            ],
                            "max_tokens": 2048,
                            "temperature": 0.0,
                            "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
                        },
                        timeout=_timeout(REFINEMENT_REQUEST_TIMEOUT_SECONDS),
                    )
                    response.raise_for_status()
                    return response
                except Exception as e:
                    if (
                        attempt >= BACKEND_REQUEST_MAX_RETRIES
                        or not _is_retryable_http_error(e)
                    ):
                        raise
                    delay = _retry_delay_seconds(
                        attempt, BACKEND_REQUEST_RETRY_BASE_SECONDS
                    )
                    logger.warning(
                        "Retrying async VLM request after attempt %d/%d failed: %s",
                        attempt + 1,
                        BACKEND_REQUEST_MAX_RETRIES + 1,
                        str(e) or type(e).__name__,
                    )
                    await asyncio.sleep(delay)

    async def refine_visual_async(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        video_id: str,
        query: Optional[str] = None,
        use_original: bool = False,
    ) -> Dict[str, Any]:
        """Async visual refinement via VLM."""
        try:
            if use_original:
                segment_video = os.path.abspath(video_path)
            else:
                segment_video = os.path.abspath(
                    await asyncio.to_thread(
                        self._extract_video_segment,
                        video_path, start_time, end_time, video_id,
                    )
                )

            prompt = (
                query
                if query
                else "Please provide a detailed description of everything visible in this video segment, "
                "including objects, people, actions, scenes, text, and any other relevant visual information."
            )

            try:
                response = await self._vlm_request_async(segment_video, prompt)
            except httpx.HTTPStatusError as e:
                if self._is_avcodec_error(e) and not use_original:
                    logger.warning("Avcodec error for %s, re-extracting segment", segment_video)
                    Path(segment_video).unlink(missing_ok=True)
                    segment_video = os.path.abspath(
                        await asyncio.to_thread(
                            self._extract_video_segment,
                            video_path, start_time, end_time, video_id,
                        )
                    )
                    response = await self._vlm_request_async(segment_video, prompt)
                else:
                    raise

            description = response.json()["choices"][0]["message"]["content"]

            return {
                "success": True,
                "description": description,
                "query": query,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
            }
        except httpx.HTTPStatusError as e:
            detail = e.response.text.strip()
            if len(detail) > 500:
                detail = detail[:500] + "..."
            logger.error("Visual refinement HTTP error: %s | body=%s", e, detail)
            return {
                "success": False,
                "error": f"{e} | body={detail}",
                "start_time": start_time,
                "end_time": end_time,
                "query": query,
            }
        except Exception as e:
            logger.error("Visual refinement error: %s", e)
            return {
                "success": False,
                "error": str(e),
                "start_time": start_time,
                "end_time": end_time,
                "query": query,
            }

    SHORT_VIDEO_THRESHOLD = 60.0

    def _is_short_video(self, video_path: str) -> bool:
        """Check if a video is short enough to send in full without extraction."""
        duration = self._get_video_duration(video_path)
        return 0 < duration <= self.SHORT_VIDEO_THRESHOLD

    _VERIFY_PROMPT = (
        "Describe everything visible in this video in detail: "
        "all objects, people, their clothing, actions, setting, and any text or symbols."
    )

    def execute_verify_claim(
        self, tool_params: VerifyClaimTool, vector_store=None
    ) -> Dict[str, Any]:
        """Verify a claim by getting an unbiased description (no leading questions)."""
        original_video_path = (
            vector_store.get_video_path(tool_params.video_id)
            if vector_store is not None else None
        )
        video_path = self._find_video_file(tool_params.video_id, original_video_path)
        if not video_path:
            raise FileNotFoundError(f"Video file not found for '{tool_params.video_id}'")

        short = self._is_short_video(video_path)
        if short:
            result = self.refine_visual(
                video_path, 0.0, self._get_video_duration(video_path),
                tool_params.video_id, self._VERIFY_PROMPT, use_original=True,
            )
        else:
            result = self.refine_visual(
                video_path, tool_params.start_time, tool_params.end_time,
                tool_params.video_id, self._VERIFY_PROMPT,
            )
        return {
            "claim": tool_params.claim,
            "description": result.get("description", result.get("error", "")),
            "start_time": tool_params.start_time,
            "end_time": tool_params.end_time,
        }

    async def execute_verify_claim_async(
        self, tool_params: VerifyClaimTool, vector_store=None
    ) -> Dict[str, Any]:
        """Async verify a claim by getting an unbiased description."""
        original_video_path = None
        if vector_store is not None:
            original_video_path = await asyncio.to_thread(
                vector_store.get_video_path, tool_params.video_id
            )
        video_path = await asyncio.to_thread(
            self._find_video_file, tool_params.video_id, original_video_path
        )
        if not video_path:
            raise FileNotFoundError(f"Video file not found for '{tool_params.video_id}'")

        short = await asyncio.to_thread(self._is_short_video, video_path)
        if short:
            duration = await asyncio.to_thread(self._get_video_duration, video_path)
            result = await self.refine_visual_async(
                video_path, 0.0, duration,
                tool_params.video_id, self._VERIFY_PROMPT, use_original=True,
            )
        else:
            result = await self.refine_visual_async(
                video_path, tool_params.start_time, tool_params.end_time,
                tool_params.video_id, self._VERIFY_PROMPT,
            )
        return {
            "claim": tool_params.claim,
            "description": result.get("description", result.get("error", "")),
            "start_time": tool_params.start_time,
            "end_time": tool_params.end_time,
        }

    def execute_refinement(
        self, tool_params: RefineVideoTool, vector_store=None
    ) -> Dict[str, Any]:
        """Execute refinement with result caching."""
        cache_key = self._refinement_cache_key(
            tool_params.video_id,
            tool_params.start_time,
            tool_params.end_time,
            tool_params.modality,
            tool_params.query,
        )
        cached, future, is_leader = self._get_refinement_future(cache_key)
        if cached is not None:
            logger.debug("Refinement cache hit: %s", cache_key)
            return cached
        if not is_leader:
            return future.result()

        try:
            original_video_path = (
                vector_store.get_video_path(tool_params.video_id)
                if vector_store is not None
                else None
            )
            video_path = self._find_video_file(
                tool_params.video_id, original_video_path
            )
            if not video_path:
                result = {
                    "success": False,
                    "error": f"Video file not found for '{tool_params.video_id}'",
                }
            else:
                short = self._is_short_video(video_path)
                start_time = tool_params.start_time
                end_time = tool_params.end_time
                if short:
                    duration = self._get_video_duration(video_path)
                    start_time, end_time = 0.0, duration
                logger.debug(
                    "Refining %s [%ss-%ss]%s",
                    tool_params.modality,
                    start_time,
                    end_time,
                    " (full)" if short else "",
                )
                if tool_params.modality == "audio":
                    result = self.refine_audio(
                        video_path, start_time, end_time,
                        query=tool_params.query,
                    )
                else:
                    if short:
                        result = self.refine_visual(
                            video_path, start_time, end_time,
                            tool_params.video_id, tool_params.query,
                            use_original=True,
                        )
                    else:
                        result = self.refine_visual(
                            video_path, start_time, end_time,
                            tool_params.video_id, tool_params.query,
                        )

            self._finish_refinement_future(cache_key, future, result)
            return result
        except Exception as e:
            self._fail_refinement_future(cache_key, future, e)
            raise

    async def execute_refinement_async(
        self, tool_params: RefineVideoTool, vector_store=None
    ) -> Dict[str, Any]:
        """Async refinement path for API requests."""
        cache_key = self._refinement_cache_key(
            tool_params.video_id,
            tool_params.start_time,
            tool_params.end_time,
            tool_params.modality,
            tool_params.query,
        )
        cached, future, is_leader = self._get_refinement_future(cache_key)
        if cached is not None:
            logger.debug("Refinement cache hit: %s", cache_key)
            return cached
        if not is_leader:
            return await asyncio.wrap_future(future)

        try:
            original_video_path = None
            if vector_store is not None:
                original_video_path = await asyncio.to_thread(
                    vector_store.get_video_path, tool_params.video_id
                )
            video_path = await asyncio.to_thread(
                self._find_video_file, tool_params.video_id, original_video_path
            )
            if not video_path:
                result = {
                    "success": False,
                    "error": f"Video file not found for '{tool_params.video_id}'",
                }
            else:
                short = await asyncio.to_thread(self._is_short_video, video_path)
                start_time = tool_params.start_time
                end_time = tool_params.end_time
                if short:
                    duration = await asyncio.to_thread(self._get_video_duration, video_path)
                    start_time, end_time = 0.0, duration
                logger.debug(
                    "Refining %s [%ss-%ss]%s",
                    tool_params.modality,
                    start_time,
                    end_time,
                    " (full)" if short else "",
                )
                if tool_params.modality == "audio":
                    result = await self.refine_audio_async(
                        video_path, start_time, end_time,
                        query=tool_params.query,
                    )
                else:
                    if short:
                        result = await self.refine_visual_async(
                            video_path, start_time, end_time,
                            tool_params.video_id, tool_params.query,
                            use_original=True,
                        )
                    else:
                        result = await self.refine_visual_async(
                            video_path, start_time, end_time,
                            tool_params.video_id, tool_params.query,
                        )

            self._finish_refinement_future(cache_key, future, result)
            return result
        except Exception as e:
            self._fail_refinement_future(cache_key, future, e)
            raise


# Global refiner instance
_video_refiner = None
_video_refiner_lock = threading.Lock()


def get_video_refiner(
    vlm_base_url: str = "http://localhost:8011/v1",
    vlm_model_name: str = "google/gemma-4-31B-it",
    alm_base_url: str = "http://localhost:8013/v1",
    alm_model_name: str = "nvidia/audio-flamingo-3-hf",
    video_search_paths: list = None,
) -> VideoRefiner:
    """Get or create the global video refiner instance."""
    global _video_refiner
    with _video_refiner_lock:
        if _video_refiner is None:
            _video_refiner = VideoRefiner(
                vlm_base_url=vlm_base_url,
                vlm_model_name=vlm_model_name,
                alm_base_url=alm_base_url,
                alm_model_name=alm_model_name,
                video_search_paths=video_search_paths,
            )
    return _video_refiner


VERIFY_CLAIM_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "verify_claim",
        "description": "Verify whether a specific object exists in a video segment by getting an unbiased scene description. ONLY for object existence questions. NOT for audio, counting, spatial, emotion, or action questions. Call at most once or twice per request.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_time": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Start time in seconds",
                },
                "end_time": {
                    "type": "number",
                    "minimum": 0,
                    "description": "End time in seconds",
                },
                "claim": {
                    "type": "string",
                    "description": "The object or entity to check for",
                },
            },
            "required": ["start_time", "end_time", "claim"],
        },
    },
}

REFINE_VIDEO_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "refine_video",
        "description": "Extract and analyze a specific video segment in detail. Audio: speech, music, sounds. Visual: objects, people, actions, scenes. Max 60s per segment.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_time": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Start time in seconds",
                },
                "end_time": {
                    "type": "number",
                    "minimum": 0,
                    "description": "End time in seconds",
                },
                "modality": {
                    "type": "string",
                    "enum": ["audio", "visual"],
                    "description": "audio or visual",
                },
                "query": {
                    "type": "string",
                    "description": "Question or focus for the analysis",
                },
            },
            "required": ["start_time", "end_time", "modality"],
        },
    },
}
