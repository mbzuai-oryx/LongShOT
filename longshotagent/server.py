#!/usr/bin/env python3
"""
FastAPI server for the Video Understanding Agent system.

Chat requests use `async def` so upstream LLM/tool HTTP waits do not pin
worker threads, while video preprocessing stays lazy and request-driven.
"""

import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

from config import load_config
from agent import VideoAgent
from main import VideoPreprocessingPipeline
from preprocessing import VectorStore

# Configure logging
log_handlers = [logging.StreamHandler(sys.stdout)]
server_log_file = os.getenv("VIDEO_SERVER_LOG_FILE", "logs/server.log")
if server_log_file:
    Path(server_log_file).parent.mkdir(parents=True, exist_ok=True)
    log_handlers.append(logging.FileHandler(server_log_file))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=log_handlers,
    force=True,
)
# Re-attach in case uvicorn's logging config later resets the root logger.
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
for _h in log_handlers:
    _h.setFormatter(_formatter)
    if _h not in _root_logger.handlers:
        _root_logger.addHandler(_h)
logger = logging.getLogger(__name__)

# Silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Pydantic models for API requests/responses


class ChatMessage(BaseModel):
    """Individual chat message."""

    role: str = Field(
        ..., description="Role of the message sender (user, assistant, system)"
    )
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(default="video-agent", description="Model identifier")
    messages: List[ChatMessage] = Field(
        ..., description="List of messages in the conversation"
    )
    video_id: str = Field(..., description="Unique identifier of the video")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Sampling temperature")


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""

    index: int = Field(0, description="Choice index")
    message: ChatMessage = Field(..., description="Generated message")
    finish_reason: str = Field("stop", description="Reason for completion")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(..., description="Unique completion ID")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field("video-agent", description="Model identifier")
    choices: List[ChatCompletionChoice] = Field(..., description="Generated choices")


class ProcessVideoRequest(BaseModel):
    """Request model for video processing."""

    video_path: str = Field(..., description="Path to the video file")
    video_id: Optional[str] = Field(None, description="Custom video identifier")
    force_reprocess: bool = Field(
        False, description="Force reprocessing even if cached"
    )
    language: Optional[str] = Field(None, description="Language code for transcription")


class ProcessVideoResponse(BaseModel):
    """Response model for video processing."""

    video_id: str
    video_path: str
    from_cache: bool
    audio_segments: int
    visual_frames: int
    processing_time: float
    audio_duration: Optional[float] = None
    language: Optional[str] = None


class VideoInfo(BaseModel):
    """Video information model."""

    video_id: str
    video_path: str
    audio_segments: int
    visual_frames: int
    total_embeddings: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    video_agent_connected: bool
    database_accessible: bool
    models_loaded: bool


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "video-agent-system"
    permission: List[dict] = []
    root: str
    parent: Optional[str] = None


class ModelsResponse(BaseModel):
    """OpenAI-compatible models list response."""

    object: str = "list"
    data: List[ModelInfo]


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None


# Global instances (initialized on startup)
app_config = None
video_agent = None
preprocessing_pipeline = None
vector_store = None
preprocessing_pipeline_lock = threading.Lock()


def _get_preprocessing_pipeline() -> VideoPreprocessingPipeline:
    """Lazily initialize preprocessing so chat workers avoid loading heavy models."""
    global preprocessing_pipeline

    if preprocessing_pipeline is not None:
        return preprocessing_pipeline

    if app_config is None or vector_store is None:
        raise HTTPException(
            status_code=503, detail="Server configuration not initialized"
        )

    with preprocessing_pipeline_lock:
        if preprocessing_pipeline is None:
            preprocessing_db_path = app_config.preprocessing.db_path
            if preprocessing_db_path != app_config.agent.db_path:
                logger.warning(
                    "preprocessing.db_path (%s) differs from agent.db_path (%s); reusing the shared agent VectorStore for server throughput",
                    preprocessing_db_path,
                    app_config.agent.db_path,
                )

            logger.info(
                "Initializing preprocessing pipeline on first /process request..."
            )
            preprocessing_pipeline = VideoPreprocessingPipeline(
                whisper_model=app_config.preprocessing.whisper_model,
                siglip_model=app_config.preprocessing.siglip_model,
                cache_dir=app_config.preprocessing.cache_dir,
                db_path=preprocessing_db_path,
                vector_store=vector_store,
                device=app_config.preprocessing.device,
                batch_size=getattr(app_config.preprocessing, "batch_size", 256),
                audio_batch_size=getattr(
                    app_config.preprocessing, "audio_batch_size", 32
                ),
                text_model_instances=1,
                image_model_instances=1,
                max_workers=getattr(app_config.preprocessing, "max_workers", 16),
                frame_extraction_workers=getattr(
                    app_config.preprocessing, "frame_extraction_workers", 2
                ),
                audio_processing_workers=getattr(
                    app_config.preprocessing, "audio_processing_workers", 2
                ),
                database_storage_workers=getattr(
                    app_config.preprocessing, "database_storage_workers", 2
                ),
                chunk_size_multiplier=getattr(
                    app_config.preprocessing, "chunk_size_multiplier", 1.0
                ),
                enable_memory_optimization=getattr(
                    app_config.preprocessing, "enable_memory_optimization", True
                ),
                clear_cache_between_videos=getattr(
                    app_config.preprocessing, "clear_cache_between_videos", False
                ),
                immediate_database_sync=getattr(
                    app_config.preprocessing, "immediate_database_sync", True
                ),
            )

    return preprocessing_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    global app_config, video_agent, preprocessing_pipeline, vector_store

    from utils.tracer import init_session
    init_session()

    logger.info("Starting Video Understanding Agent API server...")

    try:
        # Load configuration
        config_path = os.getenv("CONFIG_PATH", "config.yaml")
        logger.info(f"Loading configuration from: {config_path}")
        app_config = load_config(config_path=config_path)

        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore(db_path=app_config.agent.db_path)

        # Initialize video agent
        logger.info("Initializing video agent...")
        video_agent = VideoAgent(
            vllm_base_url=app_config.agent.vllm_base_url,
            model_name=app_config.agent.model_name,
            db_path=app_config.agent.db_path,
            vlm_base_url=app_config.agent.vlm_base_url,
            vlm_model_name=app_config.agent.vlm_model_name,
            alm_base_url=getattr(
                app_config.agent, "alm_base_url", "http://localhost:8013/v1"
            ),
            alm_model_name=getattr(
                app_config.agent, "alm_model_name", "nvidia/audio-flamingo-3-hf"
            ),
            text_embedding_url=getattr(
                app_config.agent, "text_embedding_url", "http://localhost:8014/v1"
            ),
            visual_embedding_url=getattr(
                app_config.agent, "visual_embedding_url", "http://localhost:8018/v1"
            ),
            videos_dir=app_config.agent.videos_dir,
            video_search_paths=getattr(
                app_config.agent,
                "video_search_paths",
                [app_config.agent.videos_dir],
            ),
            vector_store=vector_store,
        )

        logger.info(
            "Deferring preprocessing pipeline initialization until /process is used"
        )

        # Eagerly initialize all components and health check backends
        logger.info("Initializing all agent components...")

        # 1. LLM server
        if not video_agent.test_connection():
            raise RuntimeError(
                f"LLM server not reachable at {app_config.agent.vllm_base_url}"
            )
        logger.info("LLM server connected")

        # 2. Search executor (loads SigLIP + connects to text embedding server)
        search_executor = video_agent._get_search_executor()
        logger.info("Search executor initialized")

        # 3. Eagerly load SigLIP text encoder
        from utils.tools import VideoSearchExecutor
        VideoSearchExecutor._ensure_siglip()
        logger.info("SigLIP text encoder loaded")

        # 4. Text embedding server health check
        import httpx as _httpx
        for name, url in [
            ("Text embedding", getattr(app_config.agent, "text_embedding_url", "http://localhost:8014/v1")),
            ("VLM", app_config.agent.vlm_base_url),
            ("ALM", getattr(app_config.agent, "alm_base_url", "http://localhost:8013/v1")),
        ]:
            try:
                r = _httpx.get(f"{url}/models", timeout=10)
                r.raise_for_status()
                logger.info("%s server connected at %s", name, url)
            except Exception as e:
                raise RuntimeError(f"{name} server not reachable at {url}: {e}")

        # 5. Video refiner
        video_agent._get_video_refiner()
        logger.info("Video refiner initialized")

        logger.info("All components initialized and healthy!")

    except Exception as e:
        logger.error(f"Failed to initialize server components: {e}")
        raise

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down Video Understanding Agent API server...")
    if video_agent:
        await video_agent.aclose()


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Video Understanding Agent API",
    description="REST API for video preprocessing and interactive video querying with multi-turn conversations",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for web frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models", response_model=ModelsResponse)
def list_models():
    """
    OpenAI-compatible models endpoint.
    Lists available models for the video understanding system.
    """
    try:
        created_timestamp = int(time.time())

        models = [
            ModelInfo(
                id="video-agent",
                created=created_timestamp,
                root="video-agent",
                owned_by="video-agent-system",
            )
        ]

        # Add specific model info if agent is available
        if video_agent and app_config:
            models.append(
                ModelInfo(
                    id=app_config.agent.model_name,
                    created=created_timestamp,
                    root=app_config.agent.model_name,
                    owned_by="video-agent-system",
                )
            )

            models.append(
                ModelInfo(
                    id=app_config.agent.vlm_model_name,
                    created=created_timestamp,
                    root=app_config.agent.vlm_model_name,
                    owned_by="video-agent-system",
                )
            )

        return ModelsResponse(data=models)

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    try:
        agent_connected = False
        if video_agent:
            agent_connected = video_agent.test_connection()

        db_accessible = False
        if vector_store:
            try:
                vector_store.get_collection_stats()
                db_accessible = True
            except Exception:
                db_accessible = False

        models_loaded = False
        if video_agent:
            models_loaded = (
                video_agent.search_executor is not None
                and video_agent.video_refiner is not None
            )

        status = "healthy" if agent_connected and db_accessible else "degraded"

        return HealthResponse(
            status=status,
            video_agent_connected=agent_connected,
            database_accessible=db_accessible,
            models_loaded=models_loaded,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint (non-streaming, async).
    """
    try:
        if not video_agent:
            raise HTTPException(status_code=503, detail="Video agent not initialized")

        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(
                status_code=400, detail="No user message found in request"
            )

        logger.debug(f"Processing chat completion for video_id='{request.video_id}'")

        response_text = await video_agent.chat_with_messages_async(
            video_id=request.video_id,
            messages=[
                {"role": msg.role, "content": msg.content} for msg in request.messages
            ],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:16]}",
            created=int(time.time()),
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=response_text)
                )
            ],
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in chat completions endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/process", response_model=ProcessVideoResponse)
def process_video(request: ProcessVideoRequest):
    """Process a video through the preprocessing pipeline."""
    try:
        if not Path(request.video_path).exists():
            raise HTTPException(
                status_code=404, detail=f"Video file not found: {request.video_path}"
            )

        pipeline = _get_preprocessing_pipeline()

        logger.debug(f"Processing video: {request.video_path}")

        results = pipeline.process_video(
            video_path=request.video_path,
            video_id=request.video_id,
            force_reprocess=request.force_reprocess,
            language=request.language,
        )

        return ProcessVideoResponse(
            video_id=results["video_id"],
            video_path=results["video_path"],
            from_cache=results["from_cache"],
            audio_segments=results["audio_segments"],
            visual_frames=results["visual_frames"],
            processing_time=results["processing_time"],
            audio_duration=results.get("audio_duration"),
            language=results.get("language"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in process video endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/videos", response_model=List[VideoInfo])
def list_processed_videos():
    """List all processed videos in the database."""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")

        stats = vector_store.get_collection_stats()

        videos = []

        # TODO: Implement get_all_videos_info() in VectorStore
        if stats.get("unique_videos", 0) > 0:
            videos.append(
                VideoInfo(
                    video_id="summary",
                    video_path="multiple",
                    audio_segments=stats.get("audio_embeddings", 0),
                    visual_frames=stats.get("visual_embeddings", 0),
                    total_embeddings=stats.get("total_embeddings", 0),
                )
            )

        return videos

    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{video_id}")
def clear_conversation(video_id: str):
    """Clear conversation history for a specific video."""
    try:
        if not video_agent:
            raise HTTPException(status_code=503, detail="Video agent not initialized")

        video_agent.clear_conversation_history(video_id)
        logger.info(f"Cleared conversation history for video_id: {video_id}")

        return {"message": f"Conversation history cleared for video_id: {video_id}"}

    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations")
def clear_all_conversations():
    """Clear all conversation histories."""
    try:
        if not video_agent:
            raise HTTPException(status_code=503, detail="Video agent not initialized")

        video_agent.clear_conversation_history()
        logger.info("Cleared all conversation histories")

        return {"message": "All conversation histories cleared"}

    except Exception as e:
        logger.error(f"Error clearing all conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_system_stats():
    """Get system statistics including database and cache info."""
    try:
        stats = {}

        if vector_store:
            stats["database"] = vector_store.get_collection_stats()

        if preprocessing_pipeline and preprocessing_pipeline.cache_manager:
            stats["cache"] = preprocessing_pipeline.cache_manager.get_cache_stats()

        if video_agent:
            stats["conversations"] = {
                "active_conversations": len(video_agent.conversation_histories),
                "video_ids": list(video_agent.conversation_histories.keys()),
            }

        return stats

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video Understanding Agent API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8012, help="Port to bind to")
    parser.add_argument(
        "--config", default="config.yaml", help="Configuration file path"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for concurrent requests",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable shared tool/search result cache",
    )
    args = parser.parse_args()

    os.environ["CONFIG_PATH"] = args.config
    if args.no_cache:
        os.environ["VIDEO_AGENT_DISABLE_SHARED_CACHE"] = "1"
        logger.info("Shared tool/search result cache disabled via --no-cache")

    logger.info(
        f"Starting server on {args.host}:{args.port} with {args.workers} workers"
    )
    logger.info(f"Using config file: {args.config}")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        workers=1 if args.reload else args.workers,
        reload=args.reload,
        log_level=args.log_level,
        access_log=False,
        log_config=None,
    )
