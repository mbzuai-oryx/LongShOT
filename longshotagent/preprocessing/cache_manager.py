"""
Cache management system for storing and retrieving preprocessed video data
with hash-based storage and reprocessing capabilities.
"""

import logging
import hashlib
import pickle
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import os
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    video_hash: str
    video_path: str
    cache_timestamp: datetime
    audio_segments_count: int
    visual_frames_count: int
    processing_config: Dict[str, Any]
    file_size: int
    duration: Optional[float] = None


class CacheManager:
    """
    Manages caching of preprocessed video data including transcriptions,
    embeddings, and vector database indices.
    """

    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize the CacheManager.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache subdirectories
        self.audio_cache_dir = self.cache_dir / "audio"
        self.visual_cache_dir = self.cache_dir / "visual"
        self.embeddings_cache_dir = self.cache_dir / "embeddings"
        self.metadata_cache_dir = self.cache_dir / "metadata"

        # Create subdirectories
        for subdir in [self.audio_cache_dir, self.visual_cache_dir,
                      self.embeddings_cache_dir, self.metadata_cache_dir]:
            subdir.mkdir(exist_ok=True)

        # Cache index file
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()

    def _load_cache_index(self) -> Dict[str, CacheEntry]:
        """Load the cache index from disk."""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r') as f:
                    data = json.load(f)

                # Convert back to CacheEntry objects
                cache_index = {}
                for video_hash, entry_dict in data.items():
                    # Convert timestamp string back to datetime
                    entry_dict['cache_timestamp'] = datetime.fromisoformat(entry_dict['cache_timestamp'])
                    cache_index[video_hash] = CacheEntry(**entry_dict)

                return cache_index
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                return {}
        return {}

    def _save_cache_index(self) -> None:
        """Save the cache index to disk."""
        try:
            # Convert CacheEntry objects to dictionaries for JSON serialization
            data = {}
            for video_hash, entry in self.cache_index.items():
                entry_dict = asdict(entry)
                # Convert datetime to string
                entry_dict['cache_timestamp'] = entry.cache_timestamp.isoformat()
                data[video_hash] = entry_dict

            with open(self.cache_index_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    def _get_video_hash(self, video_path: str) -> str:
        """
        Generate a hash for the video file based on its content.

        Args:
            video_path: Path to the video file

        Returns:
            SHA256 hash of the video file
        """
        hash_sha256 = hashlib.sha256()

        try:
            with open(video_path, "rb") as f:
                # Read file in chunks for memory efficiency
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)

            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash video file {video_path}: {e}")
            raise

    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes."""
        return os.path.getsize(file_path)

    def is_cached(self, video_path: str, config: Dict[str, Any]) -> bool:
        """
        Check if video is already cached with the given configuration.

        Args:
            video_path: Path to the video file
            config: Processing configuration

        Returns:
            True if cached, False otherwise
        """
        try:
            video_hash = self._get_video_hash(video_path)

            if video_hash not in self.cache_index:
                return False

            cache_entry = self.cache_index[video_hash]

            # Check if configuration matches
            if cache_entry.processing_config != config:
                logger.info(f"Cache exists but config mismatch for video {video_path}")
                return False

            # Check if all cache files exist
            cache_files = self._get_cache_file_paths(video_hash)
            for file_path in cache_files.values():
                if not file_path.exists():
                    logger.warning(f"Cache file missing: {file_path}")
                    return False

            logger.info(f"Video {video_path} found in cache")
            return True

        except Exception as e:
            logger.error(f"Error checking cache for {video_path}: {e}")
            return False

    def _get_cache_file_paths(self, video_hash: str) -> Dict[str, Path]:
        """Get all cache file paths for a video hash."""
        return {
            'audio_segments': self.audio_cache_dir / f"{video_hash}_segments.pkl",
            'audio_info': self.audio_cache_dir / f"{video_hash}_info.json",
            'visual_frames': self.visual_cache_dir / f"{video_hash}_frames.pkl",
            'audio_embeddings': self.embeddings_cache_dir / f"{video_hash}_audio_emb.pkl",
            'visual_embeddings': self.embeddings_cache_dir / f"{video_hash}_visual_emb.pkl",
            'metadata': self.metadata_cache_dir / f"{video_hash}_metadata.json"
        }

    def save_to_cache(
        self,
        video_path: str,
        audio_segments: List,
        audio_info: Dict[str, Any],
        visual_frames: List,
        audio_embeddings: List,
        visual_embeddings: List,
        config: Dict[str, Any]
    ) -> str:
        """
        Save processed video data to cache.

        Args:
            video_path: Path to the video file
            audio_segments: List of AudioSegment objects
            audio_info: Audio transcription info
            visual_frames: List of VideoFrame objects
            audio_embeddings: List of audio embedding arrays
            visual_embeddings: List of visual embedding arrays
            config: Processing configuration

        Returns:
            Video hash used for caching
        """
        try:
            video_hash = self._get_video_hash(video_path)
            cache_files = self._get_cache_file_paths(video_hash)

            # Save audio segments
            with open(cache_files['audio_segments'], 'wb') as f:
                pickle.dump(audio_segments, f)

            # Save audio info
            with open(cache_files['audio_info'], 'w') as f:
                json.dump(audio_info, f, indent=2)

            # Save visual frames metadata (without the actual images)
            frame_metadata = []
            for frame in visual_frames:
                if frame.image is not None:
                    img_size = (frame.image.width, frame.image.height)
                else:
                    img_size = (512, 512)  # Default SigLIP frame size
                frame_metadata.append({
                    'timestamp': frame.timestamp,
                    'frame_number': frame.frame_number,
                    'image_size': img_size
                })

            with open(cache_files['visual_frames'], 'wb') as f:
                pickle.dump(frame_metadata, f)

            # Save embeddings
            with open(cache_files['audio_embeddings'], 'wb') as f:
                pickle.dump(audio_embeddings, f)

            with open(cache_files['visual_embeddings'], 'wb') as f:
                pickle.dump(visual_embeddings, f)

            # Save general metadata
            metadata = {
                'video_path': video_path,
                'video_hash': video_hash,
                'cache_timestamp': datetime.now().isoformat(),
                'config': config,
                'file_size': self._get_file_size(video_path),
                'duration': audio_info.get('duration')
            }

            with open(cache_files['metadata'], 'w') as f:
                json.dump(metadata, f, indent=2)

            # Update cache index
            cache_entry = CacheEntry(
                video_hash=video_hash,
                video_path=video_path,
                cache_timestamp=datetime.now(),
                audio_segments_count=len(audio_segments),
                visual_frames_count=len(visual_frames),
                processing_config=config,
                file_size=self._get_file_size(video_path),
                duration=audio_info.get('duration')
            )

            self.cache_index[video_hash] = cache_entry
            self._save_cache_index()

            logger.info(f"Cached video data for {video_path} with hash {video_hash}")
            return video_hash

        except Exception as e:
            logger.error(f"Failed to save cache for {video_path}: {e}")
            raise

    def load_from_cache(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Load processed video data from cache.

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary containing cached data or None if not found
        """
        try:
            video_hash = self._get_video_hash(video_path)

            if video_hash not in self.cache_index:
                return None

            cache_files = self._get_cache_file_paths(video_hash)

            # Load all cached data
            cached_data = {}

            # Load audio segments
            with open(cache_files['audio_segments'], 'rb') as f:
                cached_data['audio_segments'] = pickle.load(f)

            # Load audio info
            with open(cache_files['audio_info'], 'r') as f:
                cached_data['audio_info'] = json.load(f)

            # Load visual frames metadata
            with open(cache_files['visual_frames'], 'rb') as f:
                cached_data['visual_frames'] = pickle.load(f)

            # Load embeddings
            with open(cache_files['audio_embeddings'], 'rb') as f:
                cached_data['audio_embeddings'] = pickle.load(f)

            with open(cache_files['visual_embeddings'], 'rb') as f:
                cached_data['visual_embeddings'] = pickle.load(f)

            # Load metadata
            with open(cache_files['metadata'], 'r') as f:
                cached_data['metadata'] = json.load(f)

            logger.info(f"Loaded cached data for {video_path}")
            return cached_data

        except Exception as e:
            logger.error(f"Failed to load cache for {video_path}: {e}")
            return None

    def clear_cache(self, video_path: Optional[str] = None) -> None:
        """
        Clear cache for a specific video or all cached data.

        Args:
            video_path: Path to specific video (None to clear all)
        """
        try:
            if video_path:
                # Clear cache for specific video
                video_hash = self._get_video_hash(video_path)
                if video_hash in self.cache_index:
                    cache_files = self._get_cache_file_paths(video_hash)

                    # Remove cache files
                    for file_path in cache_files.values():
                        if file_path.exists():
                            file_path.unlink()

                    # Remove from index
                    del self.cache_index[video_hash]
                    self._save_cache_index()

                    logger.info(f"Cleared cache for {video_path}")
            else:
                # Clear all cache
                for subdir in [self.audio_cache_dir, self.visual_cache_dir,
                              self.embeddings_cache_dir, self.metadata_cache_dir]:
                    for file_path in subdir.glob("*"):
                        file_path.unlink()

                self.cache_index.clear()
                self._save_cache_index()

                logger.info("Cleared all cache data")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.

        Returns:
            Dictionary containing cache statistics
        """
        total_size = 0

        # Calculate total cache size
        for subdir in [self.audio_cache_dir, self.visual_cache_dir,
                      self.embeddings_cache_dir, self.metadata_cache_dir]:
            for file_path in subdir.glob("*"):
                total_size += file_path.stat().st_size

        return {
            'total_videos_cached': len(self.cache_index),
            'total_cache_size_mb': total_size / (1024 * 1024),
            'cache_directory': str(self.cache_dir),
            'oldest_entry': min(
                (entry.cache_timestamp for entry in self.cache_index.values()),
                default=None
            ),
            'newest_entry': max(
                (entry.cache_timestamp for entry in self.cache_index.values()),
                default=None
            )
        }

    def list_cached_videos(self) -> List[Dict[str, Any]]:
        """
        Get list of all cached videos with their information.

        Returns:
            List of dictionaries containing video information
        """
        return [
            {
                'video_path': entry.video_path,
                'video_hash': entry.video_hash,
                'cache_timestamp': entry.cache_timestamp,
                'audio_segments': entry.audio_segments_count,
                'visual_frames': entry.visual_frames_count,
                'duration': entry.duration,
                'file_size_mb': entry.file_size / (1024 * 1024)
            }
            for entry in self.cache_index.values()
        ]