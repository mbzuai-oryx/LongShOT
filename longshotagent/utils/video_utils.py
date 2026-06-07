"""
Shared video file utilities.

Provides common video path resolution used by the VideoRefiner pipeline.
"""

import glob
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = [
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
]


class VideoPathResolver:
    """Resolves video IDs to file paths with caching."""

    def __init__(self, video_search_paths: List[str]):
        self.video_search_paths = video_search_paths
        self._path_cache: Dict[str, Optional[str]] = {}
        self._cache_lock = threading.Lock()

    def find_video_file(
        self, video_id: str, original_path: Optional[str] = None
    ) -> Optional[str]:
        """Find a video file by ID, searching configured paths.

        Args:
            video_id: The video identifier (typically filename stem).
            original_path: Optional original path hint from metadata.

        Returns:
            Absolute path to the video file, or None if not found.
        """
        with self._cache_lock:
            cached = self._path_cache.get(video_id)
        if cached is not None:
            if cached and Path(cached).exists():
                return cached

        if original_path and Path(original_path).exists():
            with self._cache_lock:
                self._path_cache[video_id] = original_path
            return original_path

        for search_path in self.video_search_paths:
            for ext in VIDEO_EXTENSIONS:
                matches = glob.glob(
                    os.path.join(search_path, "**", f"{video_id}{ext}"),
                    recursive=True,
                )
                if matches:
                    with self._cache_lock:
                        self._path_cache[video_id] = matches[0]
                    return matches[0]

        with self._cache_lock:
            self._path_cache[video_id] = None
        return None
