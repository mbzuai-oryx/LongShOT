"""
Qdrant-backed vector store. Mirrors the public API of the original
Chroma-based VectorStore so call sites do not change.

Two collections are used:
  - audio_embeddings  (text embedding of transcripts, e.g. 384-dim MiniLM)
  - visual_embeddings (SigLIP, 768-dim)

Distance reported to callers is kept as `1 - cosine_similarity` so downstream
code that computes `score = 1.0 - distance` continues to produce cosine
similarity in [-1, 1] (practically [0, 1] for normalised embeddings).
"""

import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Represents a query result with metadata."""

    id: str
    distance: float
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    content: Optional[str] = None


def _point_id(raw_id: str) -> str:
    """Qdrant point IDs must be uint64 or UUID. Hash the string ID into a UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))


class QdrantVectorStore:
    """Qdrant-backed replacement for the Chroma VectorStore."""

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_path: Optional[str] = None,
        db_path: str = "./chroma_db",
        audio_collection_name: str = "audio_embeddings",
        visual_collection_name: str = "visual_embeddings",
        audio_vector_size: int = 384,
        visual_vector_size: int = 768,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
    ):
        # Local embedded mode takes priority if configured; otherwise connect
        # to a Qdrant server. Env vars allow swapping without code changes.
        self.qdrant_path = qdrant_path or os.getenv("QDRANT_PATH")
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        if not self.qdrant_path and not self.qdrant_url:
            self.qdrant_url = "http://localhost:6333"
        api_key = api_key or os.getenv("QDRANT_API_KEY") or None
        self.audio_collection_name = audio_collection_name
        self.visual_collection_name = visual_collection_name
        self.audio_vector_size = audio_vector_size
        self.visual_vector_size = visual_vector_size

        # Video metadata JSON lives alongside whatever db_path the caller used.
        # Keep the same layout so existing ./chroma_db/video_metadata.json is
        # reused after migration.
        self.db_path = Path(db_path)
        self._metadata_file = self.db_path / "video_metadata.json"
        self._metadata_lock = threading.RLock()
        self._video_metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._video_metadata_mtime_ns: Optional[int] = None

        if self.qdrant_path:
            Path(self.qdrant_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=self.qdrant_path)
            endpoint = f"local:{self.qdrant_path}"
        else:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
                timeout=60,
            )
            endpoint = self.qdrant_url
        self._initialize_collections()
        self._refresh_video_metadata_cache(force=True)
        logger.info("Qdrant client initialized successfully (%s)", endpoint)

    # ------------------------------------------------------------------
    # Collection setup
    # ------------------------------------------------------------------
    def _ensure_collection(self, name: str, vector_size: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if name in existing:
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config=rest.VectorParams(
                size=vector_size, distance=rest.Distance.COSINE
            ),
        )
        # Payload indexes speed up filtered queries by video_id.
        self.client.create_payload_index(
            collection_name=name,
            field_name="video_id",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=name,
            field_name="type",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )

    def _initialize_collections(self) -> None:
        self._ensure_collection(self.audio_collection_name, self.audio_vector_size)
        self._ensure_collection(self.visual_collection_name, self.visual_vector_size)

    # ------------------------------------------------------------------
    # Video metadata JSON (same format and file as before)
    # ------------------------------------------------------------------
    def _refresh_video_metadata_cache(
        self, force: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        with self._metadata_lock:
            if not self._metadata_file.exists():
                self._video_metadata_cache = {}
                self._video_metadata_mtime_ns = None
                return self._video_metadata_cache
            try:
                mtime_ns = self._metadata_file.stat().st_mtime_ns
                if force or self._video_metadata_mtime_ns != mtime_ns:
                    with open(self._metadata_file, "r") as f:
                        self._video_metadata_cache = json.load(f) or {}
                    self._video_metadata_mtime_ns = mtime_ns
            except Exception as e:
                logger.error(f"Error refreshing video metadata cache: {e}")
            return self._video_metadata_cache

    def _write_video_metadata_cache(self) -> None:
        with self._metadata_lock:
            self.db_path.mkdir(parents=True, exist_ok=True)
            with open(self._metadata_file, "w") as f:
                json.dump(self._video_metadata_cache, f, indent=2)
            self._video_metadata_mtime_ns = self._metadata_file.stat().st_mtime_ns

    def add_video_metadata(
        self,
        video_id: str,
        video_path: str,
        additional_metadata: Dict[str, Any] = None,
    ) -> None:
        metadata = {
            "video_id": video_id,
            "video_path": video_path,
            "stored_at": np.datetime64("now").astype(str),
        }
        if additional_metadata:
            metadata.update(additional_metadata)
        with self._metadata_lock:
            all_metadata = self._refresh_video_metadata_cache()
            all_metadata[video_id] = metadata
            self._write_video_metadata_cache()
        logger.info(f"Stored metadata for video {video_id} at path: {video_path}")

    def get_video_path(self, video_id: str) -> Optional[str]:
        try:
            all_metadata = self._refresh_video_metadata_cache()
            video_metadata = all_metadata.get(video_id)
            if video_metadata:
                return video_metadata.get("video_path")
        except Exception as e:
            logger.error(f"Error retrieving video path for {video_id}: {e}")
        return None

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def add_audio_embeddings(
        self,
        video_id: str,
        segments: List,
        embeddings: List[np.ndarray],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        if len(segments) != len(embeddings):
            raise ValueError("Number of segments must match number of embeddings")

        points: List[rest.PointStruct] = []
        for segment, embedding in zip(segments, embeddings):
            raw_id = f"{video_id}_audio_{segment.start_time}_{segment.end_time}"
            vec = (
                embedding.tolist()
                if isinstance(embedding, np.ndarray)
                else list(embedding)
            )
            payload = {
                "raw_id": raw_id,
                "video_id": video_id,
                "type": "audio",
                "start_time_ms": int(segment.start_time * 1000),
                "end_time_ms": int(segment.end_time * 1000),
                "duration_ms": int((segment.end_time - segment.start_time) * 1000),
                "language": segment.language,
                "language_probability": segment.language_probability,
                "embedding_model": embedding_model,
                "text": segment.text,
            }
            points.append(
                rest.PointStruct(id=_point_id(raw_id), vector=vec, payload=payload)
            )

        self._upsert_batched(self.audio_collection_name, points)
        logger.info(
            f"Successfully added all {len(points)} audio embeddings for video {video_id}"
        )

    def add_visual_embeddings(
        self,
        video_id: str,
        frame_embeddings: List,
        embedding_model: str = "siglip",
    ) -> None:
        points: List[rest.PointStruct] = []
        for frame_emb in frame_embeddings:
            raw_id = f"{video_id}_visual_{frame_emb.timestamp}"
            payload = {
                "raw_id": raw_id,
                "video_id": video_id,
                "type": "visual",
                "timestamp_ms": int(frame_emb.timestamp * 1000),
                "frame_number": frame_emb.frame_number,
                "image_width": frame_emb.image_size[0],
                "image_height": frame_emb.image_size[1],
                "embedding_model": embedding_model,
            }
            points.append(
                rest.PointStruct(
                    id=_point_id(raw_id),
                    vector=frame_emb.embedding.tolist(),
                    payload=payload,
                )
            )

        self._upsert_batched(self.visual_collection_name, points)
        logger.info(
            f"Successfully added all {len(points)} visual embeddings for video {video_id}"
        )

    def _upsert_batched(
        self, collection: str, points: List[rest.PointStruct], batch_size: int = 1024
    ) -> None:
        for i in range(0, len(points), batch_size):
            chunk = points[i : i + batch_size]
            self.client.upsert(collection_name=collection, points=chunk, wait=False)

    def flush_to_disk(self) -> None:
        # Qdrant persists WAL continuously; no-op for compatibility.
        return None

    def sync_video_completion(self, video_id: str) -> None:
        logger.info(f"✅ Video {video_id} completely saved to Qdrant")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @staticmethod
    def _video_filter(video_id: Optional[str], modality: str) -> rest.Filter:
        must: List[rest.FieldCondition] = [
            rest.FieldCondition(
                key="type", match=rest.MatchValue(value=modality)
            )
        ]
        if video_id:
            must.append(
                rest.FieldCondition(
                    key="video_id", match=rest.MatchValue(value=video_id)
                )
            )
        return rest.Filter(must=must)

    def query_audio_embeddings(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        video_id: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> List[QueryResult]:
        hits = self.client.query_points(
            collection_name=self.audio_collection_name,
            query=query_embedding.tolist(),
            query_filter=self._video_filter(video_id, "audio"),
            limit=n_results,
            with_payload=True,
        ).points
        results: List[QueryResult] = []
        for hit in hits:
            md = hit.payload or {}
            if time_range is not None:
                start_ms, end_ms = (
                    int(time_range[0] * 1000),
                    int(time_range[1] * 1000),
                )
                seg_start = md.get("start_time_ms", 0)
                seg_end = md.get("end_time_ms", 0)
                if not (seg_start <= end_ms and seg_end >= start_ms):
                    continue
            results.append(
                QueryResult(
                    id=md.get("raw_id", str(hit.id)),
                    distance=1.0 - float(hit.score),
                    metadata=md,
                    content=md.get("text"),
                )
            )
        return results

    def query_visual_embeddings(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        video_id: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> List[QueryResult]:
        hits = self.client.query_points(
            collection_name=self.visual_collection_name,
            query=query_embedding.tolist(),
            query_filter=self._video_filter(video_id, "visual"),
            limit=n_results,
            with_payload=True,
        ).points
        results: List[QueryResult] = []
        for hit in hits:
            md = hit.payload or {}
            if time_range is not None:
                start_ms, end_ms = (
                    int(time_range[0] * 1000),
                    int(time_range[1] * 1000),
                )
                ts = md.get("timestamp_ms", 0)
                if not (start_ms <= ts <= end_ms):
                    continue
            results.append(
                QueryResult(
                    id=md.get("raw_id", str(hit.id)),
                    distance=1.0 - float(hit.score),
                    metadata=md,
                )
            )
        return results

    def query_multimodal(
        self,
        audio_query: Optional[np.ndarray] = None,
        visual_query: Optional[np.ndarray] = None,
        n_results: int = 10,
        video_id: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
        combine_scores: bool = True,
    ) -> Dict[str, List[QueryResult]]:
        results: Dict[str, List[QueryResult]] = {}
        if audio_query is not None:
            results["audio"] = self.query_audio_embeddings(
                audio_query, n_results, video_id, time_range
            )
        if visual_query is not None:
            results["visual"] = self.query_visual_embeddings(
                visual_query, n_results, video_id, time_range
            )
        return results

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------
    def get_video_metadata(self, video_id: str) -> Dict[str, Any]:
        audio_count = self._count_by_video(self.audio_collection_name, video_id)
        visual_count = self._count_by_video(self.visual_collection_name, video_id)
        return {
            "video_id": video_id,
            "audio_segments": audio_count,
            "visual_frames": visual_count,
            "total_embeddings": audio_count + visual_count,
        }

    def _count_by_video(self, collection: str, video_id: str) -> int:
        res = self.client.count(
            collection_name=collection,
            count_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="video_id", match=rest.MatchValue(value=video_id)
                    )
                ]
            ),
            exact=True,
        )
        return int(res.count)

    def delete_video_embeddings(self, video_id: str) -> None:
        video_filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="video_id", match=rest.MatchValue(value=video_id)
                )
            ]
        )
        self.client.delete(
            collection_name=self.audio_collection_name,
            points_selector=rest.FilterSelector(filter=video_filter),
        )
        self.client.delete(
            collection_name=self.visual_collection_name,
            points_selector=rest.FilterSelector(filter=video_filter),
        )
        logger.info(f"Deleted all embeddings for video {video_id}")

    def list_videos(self) -> List[str]:
        if self._metadata_file.exists():
            try:
                all_metadata = self._refresh_video_metadata_cache()
                return sorted(all_metadata.keys())
            except Exception:
                pass
        video_ids: set = set()
        for collection in (self.audio_collection_name, self.visual_collection_name):
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=collection,
                    limit=1024,
                    offset=offset,
                    with_payload=["video_id"],
                    with_vectors=False,
                )
                for p in points:
                    vid = (p.payload or {}).get("video_id")
                    if vid:
                        video_ids.add(vid)
                if offset is None:
                    break
        return sorted(video_ids)

    def get_collection_stats(self) -> Dict[str, Any]:
        audio_count = int(
            self.client.count(self.audio_collection_name, exact=True).count
        )
        visual_count = int(
            self.client.count(self.visual_collection_name, exact=True).count
        )
        return {
            "audio_embeddings": audio_count,
            "visual_embeddings": visual_count,
            "total_embeddings": audio_count + visual_count,
            "unique_videos": len(self.list_videos()),
            "database_path": self.qdrant_path or self.qdrant_url,
        }
