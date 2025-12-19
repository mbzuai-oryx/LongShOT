"""
Video downloader module for the Arabic video dataset.
This module handles downloading Arabic videos from YouTube using video IDs from a file.
"""

import os
import json
import time
import logging
import threading
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import yt_dlp

# Import project configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from caption_pipeline.utils.rich_console import get_console
from config import (
    VIDEO_DIR, METADATA_DIR, VIDEO_IDS_FILE,
    MIN_VIDEO_DURATION, MAX_VIDEO_DURATION
)

# Set up logging
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'downloader.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
rich_console = get_console()

class VideoDownloader:
    """Class to download Arabic videos from YouTube using video IDs."""
    
    def __init__(self):
        """Initialize the VideoDownloader."""
        self.video_dir = VIDEO_DIR
        self.metadata_dir = METADATA_DIR
        self.video_ids_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), VIDEO_IDS_FILE)
        
        # Create directories if they don't exist
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Initialize the metadata DataFrame
        self.metadata_file = os.path.join(self.metadata_dir, 'video_metadata.csv')
        if os.path.exists(self.metadata_file):
            self.metadata_df = pd.read_csv(self.metadata_file)
        else:
            self.metadata_df = pd.DataFrame(columns=[
                'video_id', 'title', 'channel', 'duration', 'view_count',
                'publish_date', 'description', 'tags', 'download_date',
                'file_path', 'status'
            ])
        
        # Add a threading lock for metadata access
        self.metadata_lock = threading.Lock()

    def read_video_ids(self) -> List[str]:
        """
        Read video IDs from the video_ids.txt file.
        
        Returns:
            List of video IDs
        """
        if not os.path.exists(self.video_ids_file):
            rich_console.print_error(f"Video IDs file not found: {self.video_ids_file}")
            return []
        
        video_ids = []
        try:
            with open(self.video_ids_file, 'r') as f:
                for line in f:
                    # Skip comments and empty lines
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # Extract video ID (first word in the line)
                    video_id = line.split()[0]
                    video_ids.append(video_id)
            
                return video_ids
        
        except Exception as e:
            rich_console.print_error(f"Error reading video IDs file: {e}")
            return []

    def get_video_info(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a video using yt-dlp.
        
        Args:
            video_id: The YouTube video ID
            
        Returns:
            Dictionary containing video metadata, or None if retrieval failed
        """
        try:
            info_opts = {
                'quiet': True,
                'skip_download': True,
                'ignoreerrors': True
            }
            
            with yt_dlp.YoutubeDL(info_opts) as ydl:
                video_info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                
                if not video_info:
                    return None
                
                # Check duration
                duration = video_info.get('duration', 0)
                if not (MIN_VIDEO_DURATION <= duration <= MAX_VIDEO_DURATION):
                    rich_console.print_warning(f"Video {video_id} duration ({duration}s) is outside allowed range")
                    return None
                
                # Extract metadata
                metadata = {
                    'video_id': video_id,
                    'title': video_info.get('title', ''),
                    'channel': video_info.get('uploader', ''),
                    'duration': duration,
                    'view_count': video_info.get('view_count', 0),
                    'publish_date': video_info.get('upload_date', ''),
                    'description': video_info.get('description', ''),
                    'tags': ','.join(video_info.get('tags', [])),
                    'download_date': None,
                    'file_path': None,
                    'status': 'found'
                }
                
                return metadata
        
        except Exception as e:
            rich_console.print_warning(f"Error getting info for video {video_id}: {e}")
            return None

    def download_video(self, video_id: str) -> Optional[str]:
        """
        Download a video by its ID.
        
        Args:
            video_id: The YouTube video ID
            
        Returns:
            Path to the downloaded video file, or None if download failed
        """
        if self._is_video_downloaded(video_id):
            file_path = self._get_video_path(video_id)
            if file_path and os.path.exists(file_path):
                rich_console.print_info(f"Video {video_id} already downloaded.")
                return file_path
            else:
                # File doesn't exist despite metadata saying it's downloaded
                rich_console.print_warning(f"Video {video_id} marked as downloaded but file missing. Re-downloading...")
        
        rich_console.print_info(f"Downloading video: {video_id}")
        
        try:
            # Set up yt-dlp options
            output_path = os.path.join(self.video_dir, f"%(id)s.%(ext)s")
            ydl_opts = {
                'format': 'best[ext=mp4]/best',  # Prefer MP4 but accept other formats if needed
                'outtmpl': output_path,
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': False,  # Don't ignore errors during download
                'progress': True,  # Show progress bar
                'writethumbnail': False,
                'writesubtitles': False,
                'cookiefile': '../../cookies.txt', 
                # Speed optimization options
                'buffersize': 50331648,  # 48MB buffer size (default is 8KB)
                'concurrent_fragment_downloads': 10,  # Download fragments in parallel
                'throttledratelimit': None,  # No rate limit for maximum speed
                'noresizebuffer': True,  # Don't resize the buffer - maintain max size
                'retries': 10,  # Retry up to 10 times on connection errors
                'file_access_retries': 5,  # Retry up to 5 times on file access errors
                'fragment_retries': 10,  # Retry up to 10 times on fragment download errors
                'skip_unavailable_fragments': False,  # Don't skip unavailable fragments
                'keepvideo': False,  # Don't keep video files after post-processing
                'sleep_interval': 30,  # Sleep 1 second between retries
                'sleep_interval_request' : 10,  # Sleep 10 seconds between requests
                'legacyserverconnect' : True,  # Use legacy server connection method
                'geo_bypass': True,  # Bypass geographic restrictions
                'geo_bypass_country': 'UK',  # Assume US location for geo bypass
                'geo_bypass_ip_block': 'auto',  # Auto-detect IP block for geo bypass
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',  # Convert to MP4 if needed
                }]
            }
            
            # Download the video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                rich_console.print_info(f"Starting download of video {video_id}")
                video_info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
                
                if not video_info:
                    rich_console.print_error(f"Failed to download video {video_id}")
                    return None
                
                # Get the actual file path after download
                file_path = os.path.join(self.video_dir, f"{video_id}.mp4")
                
                # Verify file exists and is not empty
                if not os.path.exists(file_path):
                    rich_console.print_error(f"Downloaded file not found: {file_path}")
                    return None
                    
                if os.path.getsize(file_path) == 0:
                    rich_console.print_error(f"Downloaded file is empty: {file_path}")
                    os.remove(file_path)
                    return None
                
                rich_console.print_info(f"Successfully downloaded video to {file_path}")
                
                # Update metadata
                video_idx = self.metadata_df[self.metadata_df['video_id'] == video_id].index
                if len(video_idx) > 0:
                    self.metadata_df.loc[video_idx, 'download_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
                    self.metadata_df.loc[video_idx, 'file_path'] = file_path
                    self.metadata_df.loc[video_idx, 'status'] = 'downloaded'
                else:
                    # Add new row if not already in metadata
                    new_row = {
                        'video_id': video_id,
                        'title': video_info.get('title', ''),
                        'channel': video_info.get('uploader', ''),
                        'duration': video_info.get('duration', 0),
                        'view_count': video_info.get('view_count', 0),
                        'publish_date': video_info.get('upload_date', ''),
                        'description': video_info.get('description', ''),
                        'tags': ','.join(video_info.get('tags', [])),
                        'download_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                        'file_path': file_path,
                        'status': 'downloaded'
                    }
                    self.metadata_df = pd.concat([self.metadata_df, pd.DataFrame([new_row])], ignore_index=True)
                
                # Save metadata
                self.metadata_df.to_csv(self.metadata_file, index=False)
                
                return file_path
        
        except Exception as e:
            rich_console.print_error(f"Error downloading video {video_id}: {str(e)}")
            
            # Mark as failed in metadata
            video_idx = self.metadata_df[self.metadata_df['video_id'] == video_id].index
            if len(video_idx) > 0:
                self.metadata_df.loc[video_idx, 'status'] = 'download_failed'
                self.metadata_df.to_csv(self.metadata_file, index=False)
            
            return None

    def update_metadata(self, video_id: str, updates: Dict[str, Any]):
        """
        Thread-safe update of metadata for a video.
        
        Args:
            video_id: The YouTube video ID
            updates: Dictionary containing fields to update
        """
        with self.metadata_lock:
            video_idx = self.metadata_df[self.metadata_df['video_id'] == video_id].index
            if len(video_idx) > 0:
                for key, value in updates.items():
                    self.metadata_df.loc[video_idx, key] = value
            else:
                # Add new row if not already in metadata
                new_row = {'video_id': video_id, **updates}
                self.metadata_df = pd.concat([self.metadata_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save metadata
            self.metadata_df.to_csv(self.metadata_file, index=False)

    def process_video(self, video_id: str) -> Tuple[str, Optional[str]]:
        """
        Process a single video: fetch info, download, and update metadata.
        
        This function is designed to be called by multiple threads.
        
        Args:
            video_id: The YouTube video ID
            
        Returns:
            Tuple of (video_id, file_path) where file_path is None if download failed
        """
        # Skip if already downloaded and file exists
        if self._is_video_downloaded(video_id):
            file_path = self._get_video_path(video_id)
            if file_path and os.path.exists(file_path):
                rich_console.print_info(f"Video {video_id} already downloaded.")
                return video_id, file_path
        
        # Get video info if not already in metadata
        if not self._is_video_in_dataset(video_id):
            video_info = self.get_video_info(video_id)
            if video_info:
                self.update_metadata(video_id, video_info)
            else:
                rich_console.print_warning(f"Could not get info for video {video_id}, skipping...")
                return video_id, None

        # Download the video
        file_path = self.download_video(video_id)
        return video_id, file_path

    def run_pipeline(self, max_downloads: Optional[int] = None, workers: int = 4) -> List[str]:
        """
        Run the full pipeline: read video IDs from file and download them in parallel.
        
        Args:
            max_downloads: Maximum number of videos to download
            workers: Number of parallel download workers
            
        Returns:
            List of paths to successfully downloaded video files
        """
        # Read video IDs from file
        video_ids = self.read_video_ids()
        if not video_ids:
            rich_console.print_error("No video IDs found to process")
            return []

        if max_downloads:
            video_ids = video_ids[:max_downloads]

        rich_console.print_info(f"Starting downloads for {len(video_ids)} videos using {workers} workers...")
        
        downloaded_paths = []
        failed_ids = []
        
        # Create a progress bar that will be updated by all threads
        pbar = tqdm(total=len(video_ids), desc="Processing videos")
        
        # Use a ThreadPoolExecutor to process videos in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all video processing tasks
            future_to_video = {executor.submit(self.process_video, video_id): video_id for video_id in video_ids}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_video):
                video_id = future_to_video[future]
                try:
                    _, file_path = future.result()
                    if file_path and os.path.exists(file_path):
                        downloaded_paths.append(file_path)
                    else:
                        failed_ids.append(video_id)
                except Exception as e:
                    rich_console.print_error(f"Exception processing video {video_id}: {str(e)}")
                    failed_ids.append(video_id)
                
                # Update progress bar
                pbar.update(1)
        
        # Close the progress bar
        pbar.close()

        total = len(video_ids)
        success = len(downloaded_paths)
        failed = len(failed_ids)
        rich_console.print_info(f"Download pipeline completed:")
        rich_console.print_info(f"  Total videos processed: {total}")
        rich_console.print_info(f"  Successfully downloaded: {success}")
        rich_console.print_info(f"  Failed: {failed}")
        
        if failed > 0:
            rich_console.print_info("Failed video IDs:")
            for video_id in failed_ids:
                rich_console.print_info(f"  - {video_id}")

        return downloaded_paths

    def _is_video_in_dataset(self, video_id: str) -> bool:
        """Check if a video ID is already in our dataset."""
        with self.metadata_lock:
            return video_id in self.metadata_df['video_id'].values
    
    def _is_video_downloaded(self, video_id: str) -> bool:
        """Check if a video has already been downloaded."""
        with self.metadata_lock:
            video_data = self.metadata_df[self.metadata_df['video_id'] == video_id]
            if len(video_data) == 0:
                # Check if video file exists on disk (for offline videos)
                video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
                if os.path.exists(video_path):
                    # Register the offline video in metadata
                    self._register_offline_video(video_id, video_path)
                    return True
                return False
            return video_data.iloc[0]['status'] == 'downloaded'
    
    def _register_offline_video(self, video_id: str, video_path: str):
        """Register an offline video in metadata."""
        try:
            # Get basic file info
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                rich_console.print_warning(f"Offline video {video_id} has zero size, skipping registration")
                return
                
            # Create metadata entry for offline video
            new_row = {
                'video_id': video_id,
                'title': f'Offline Video {video_id}',
                'channel': 'Unknown',
                'duration': 0,  # Duration will be determined during preprocessing
                'view_count': 0,
                'publish_date': '',
                'description': 'Offline video file',
                'tags': '',
                'download_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'file_path': video_path,
                'status': 'downloaded'
            }
            
            # Add to metadata DataFrame
            self.metadata_df = pd.concat([self.metadata_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save metadata
            self.metadata_df.to_csv(self.metadata_file, index=False)
            
            rich_console.print_info(f"Registered offline video: {video_id}")
            
        except Exception as e:
            rich_console.print_error(f"Error registering offline video {video_id}: {e}")

    def _get_video_path(self, video_id: str) -> Optional[str]:
        """Get the file path for a downloaded video."""
        with self.metadata_lock:
            video_data = self.metadata_df[self.metadata_df['video_id'] == video_id]
            if len(video_data) == 0:
                # Check if video file exists on disk (for offline videos)
                video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
                if os.path.exists(video_path):
                    return video_path
                return None
            elif video_data.iloc[0]['status'] != 'downloaded':
                return None
            return video_data.iloc[0]['file_path']


def main():
    """Main function to run the downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download Arabic videos from YouTube')
    parser.add_argument('--max-videos', type=int, default=None, 
                        help='Maximum number of videos to download')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Number of parallel download workers')
    
    args = parser.parse_args()
    
    downloader = VideoDownloader()
    downloader.run_pipeline(max_downloads=args.max_videos, workers=args.workers)


if __name__ == "__main__":
    main()