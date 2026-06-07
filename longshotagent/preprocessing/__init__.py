"""
Video Preprocessing Pipeline for Long Video Understanding

This package provides tools for extracting and embedding audio transcriptions
and visual features from videos for storage in a vector database.
"""

from .audio_processor import AudioProcessor
from .video_processor import VideoProcessor
from .image_embedder import ImageEmbedder
from .qdrant_vector_store import QdrantVectorStore as VectorStore
from .cache_manager import CacheManager

__version__ = "1.0.0"
__all__ = [
    "AudioProcessor",
    "VideoProcessor",
    "ImageEmbedder",
    "VectorStore",
    "CacheManager"
]