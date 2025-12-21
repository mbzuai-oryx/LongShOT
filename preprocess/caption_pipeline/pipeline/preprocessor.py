"""
Video preprocessor module for the LongShOT video dataset.
This module handles preprocessing of downloaded videos, including audio extraction.
"""

import os
import logging
import subprocess
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import shutil

# Import project configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from caption_pipeline.utils.rich_console import get_console
from config import VIDEO_DIR, AUDIO_DIR, METADATA_DIR

# Set up logging
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,  # Suppress info messages for cleaner console
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'preprocessor.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
rich_console = get_console()

# Try to import ffmpeg-python, but prepare for fallback
try:
    import ffmpeg
    HAS_FFMPEG_PY = hasattr(ffmpeg, 'input')
except (ImportError, AttributeError):
    HAS_FFMPEG_PY = False
    rich_console.print_warning("ffmpeg-python not properly installed. Using subprocess fallback.")


class VideoPreprocessor:
    """Class to preprocess videos, including extracting audio."""
    
    def __init__(self):
        """Initialize the VideoPreprocessor."""
        self.video_dir = VIDEO_DIR
        self.audio_dir = AUDIO_DIR
        self.metadata_dir = METADATA_DIR
        
        # Create directories if they don't exist
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Check if ffmpeg is installed
        if not self._check_ffmpeg_installed():
            rich_console.print_error("FFmpeg is not installed or not in PATH. Please install FFmpeg.")
            raise RuntimeError("FFmpeg not found")
        
        # Initialize the metadata DataFrame
        self.metadata_file = os.path.join(self.metadata_dir, 'video_metadata.csv')

        # Define required columns
        required_columns = [
            'video_id', 'title', 'channel', 'duration', 'view_count',
            'publish_date', 'description', 'tags', 'download_date',
            'file_path', 'status', 'audio_path', 'caption_path'
        ]

        try:
            if os.path.exists(self.metadata_file) and os.path.getsize(self.metadata_file) > 0:
                self.metadata_df = pd.read_csv(self.metadata_file)
                rich_console.print_info(f"Loaded metadata for {len(self.metadata_df)} videos")
            else:
                # Create an empty metadata file with the required columns
                rich_console.print_warning(f"Metadata file not found or empty. Creating metadata file at {self.metadata_file}")
                self.metadata_df = pd.DataFrame(columns=required_columns)
                self.metadata_df.to_csv(self.metadata_file, index=False)
                rich_console.print_info("Created empty metadata file")
        except pd.errors.EmptyDataError:
            # Handle empty or malformed CSV file
            rich_console.print_warning("Metadata file was empty or malformed. Recreating...")
            self.metadata_df = pd.DataFrame(columns=required_columns)
            self.metadata_df.to_csv(self.metadata_file, index=False)
            rich_console.print_info("Recreated metadata file")
        except Exception as e:
            rich_console.print_error(f"Error initializing metadata: {e}")
            raise
    
    def _check_ffmpeg_installed(self) -> bool:
        """Check if ffmpeg is installed on the system."""
        return shutil.which('ffmpeg') is not None
    
    def _extract_audio_with_subprocess(self, video_path: str, output_path: str, output_format: str) -> bool:
        """
        Extract audio using subprocess to call FFmpeg directly.
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the output audio file
            output_format: Format of the output audio (wav, mp3)
            
        Returns:
            True if extraction was successful, False otherwise
        """
        try:
            # Build the ffmpeg command based on the output format
            if output_format == 'wav':
                cmd = [
                    'ffmpeg', '-i', video_path, 
                    '-vn', '-acodec', 'pcm_s16le', 
                    '-ar', '16000', '-ac', '1',
                    '-y', output_path
                ]
            else:  # mp3
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-vn', '-acodec', 'libmp3lame',
                    '-ar', '16000', '-ac', '1',
                    '-y', output_path
                ]
            
            # Execute the command
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if process.returncode != 0:
                rich_console.print_error(f"FFmpeg error: {process.stderr}")
                return False
            
            return True
            
        except Exception as e:
            rich_console.print_error(f"Error running FFmpeg via subprocess: {e}")
            return False
    
    def _extract_audio_with_ffmpeg_python(self, video_path: str, output_path: str, output_format: str) -> bool:
        """
        Extract audio using ffmpeg-python wrapper.
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the output audio file
            output_format: Format of the output audio (wav, mp3)
            
        Returns:
            True if extraction was successful, False otherwise
        """
        try:
            # Use ffmpeg-python to extract audio
            stream = ffmpeg.input(video_path)
            
            if output_format == 'wav':
                audio = stream.audio.output(output_path, acodec='pcm_s16le', ar=16000, ac=1)
            else:  # mp3
                audio = stream.audio.output(output_path, acodec='libmp3lame', ar=16000, ac=1)
            
            audio.run(quiet=True, overwrite_output=True)
            return True
            
        except Exception as e:
            rich_console.print_error(f"Error using ffmpeg-python: {e}")
            return False
    
    def extract_audio(self, video_id: str, output_format: str = 'wav') -> Optional[str]:
        """
        Extract audio from a video file.
        
        Args:
            video_id: The YouTube video ID
            output_format: The audio format to save (wav, mp3, etc.)
            
        Returns:
            Path to the extracted audio file, or None if extraction failed
        """
        # Check if audio already extracted
        audio_path = os.path.join(self.audio_dir, f"{video_id}.{output_format}")
        if os.path.exists(audio_path):
            rich_console.print_info(f"Audio for video {video_id} already extracted.")
            return audio_path
        
        # Get video path from metadata
        video_data = self.metadata_df[self.metadata_df['video_id'] == video_id]
        if len(video_data) == 0 or video_data.iloc[0]['status'] != 'downloaded':
            rich_console.print_error(f"Video {video_id} not found or not downloaded.")
            return None
        
        video_path = video_data.iloc[0]['file_path']
        if not os.path.exists(video_path):
            rich_console.print_error(f"Video file {video_path} does not exist.")
            return None
        
        rich_console.print_info(f"Extracting audio from video: {video_id}")
        
        try:
            # Extract audio using ffmpeg
            output_path = os.path.join(self.audio_dir, f"{video_id}.{output_format}")
            
            # Try to use ffmpeg-python if available, otherwise use subprocess
            success = False
            if HAS_FFMPEG_PY:
                success = self._extract_audio_with_ffmpeg_python(video_path, output_path, output_format)
            
            # If ffmpeg-python failed or not available, fall back to subprocess
            if not success:
                success = self._extract_audio_with_subprocess(video_path, output_path, output_format)
                
            if not success:
                raise Exception("Both ffmpeg extraction methods failed")
            
            rich_console.print_info(f"Successfully extracted audio for video {video_id} to {output_path}")
            return output_path
        
        except Exception as e:
            rich_console.print_error(f"Error extracting audio for video {video_id}: {e}")
            return None
    
    def _extract_audio_worker(self, video_data: Tuple[str, str], output_format: str = 'wav') -> Tuple[str, str, Optional[str]]:
        """
        Worker function for parallel audio extraction.
        
        Args:
            video_data: Tuple containing (video_id, status)
            output_format: The audio format to save
            
        Returns:
            Tuple of (video_id, new_status, audio_path or None)
        """
        video_id, _ = video_data
        audio_path = self.extract_audio(video_id, output_format)
        if audio_path:
            return (video_id, 'audio_extracted', audio_path)
        else:
            return (video_id, 'audio_extraction_failed', None)
    
    def batch_extract_audio(self, max_videos: int = None, workers: int = 1, output_format: str = 'wav') -> List[str]:
        """
        Extract audio from all downloaded videos, optionally in parallel.
        
        Args:
            max_videos: Maximum number of videos to process
            workers: Number of parallel workers (default: 1)
            output_format: The audio format to save (wav, mp3, etc.)
            
        Returns:
            List of paths to successfully extracted audio files
        """
        # Get all downloaded videos
        downloaded_videos = self.metadata_df[self.metadata_df['status'] == 'downloaded']
        
        if max_videos:
            downloaded_videos = downloaded_videos.head(max_videos)
        
        video_ids = [(row['video_id'], row['status']) for _, row in downloaded_videos.iterrows()]
        total_videos = len(video_ids)
        
        if total_videos == 0:
            rich_console.print_info("No downloaded videos found for audio extraction.")
            return []
        
        rich_console.print_info(f"Extracting audio from {total_videos} videos using {workers} workers")
        
        audio_paths = []
        updated_metadata = []
        
        if workers > 1 and total_videos > 1:
            # Parallel processing
            worker_fn = partial(self._extract_audio_worker, output_format=output_format)
            
            with mp.Pool(processes=workers) as pool:
                # Process videos in parallel and update metadata in batch
                results = list(tqdm(
                    pool.imap(worker_fn, video_ids),
                    total=total_videos,
                    desc="Extracting audio"
                ))
                
                # Collect results
                for video_id, status, audio_path in results:
                    if audio_path:
                        audio_paths.append(audio_path)
                    updated_metadata.append((video_id, status, audio_path))
        else:
            # Serial processing
            for video_id, _ in tqdm(video_ids, total=total_videos, desc="Extracting audio"):
                audio_path = self.extract_audio(video_id, output_format)
                if audio_path:
                    audio_paths.append(audio_path)
                    updated_metadata.append((video_id, 'audio_extracted', audio_path))
                else:
                    updated_metadata.append((video_id, 'audio_extraction_failed', None))
        
        # Update metadata in batch
        self._update_metadata(updated_metadata)
        
        rich_console.print_info(f"Successfully extracted audio for {len(audio_paths)} videos")
        return audio_paths
    
    def _update_metadata(self, updates: List[Tuple[str, str, Optional[str]]]):
        """
        Update metadata in batch to avoid race conditions.
        
        Args:
            updates: List of (video_id, status, audio_path) tuples
        """
        # Re-read the metadata to avoid conflicts with other processes
        self.metadata_df = pd.read_csv(self.metadata_file)
        
        for video_id, status, audio_path in updates:
            video_idx = self.metadata_df[self.metadata_df['video_id'] == video_id].index
            if len(video_idx) > 0:
                self.metadata_df.loc[video_idx, 'status'] = status
                if audio_path:
                    self.metadata_df.loc[video_idx, 'audio_path'] = audio_path
        
        # Save updated metadata
        self.metadata_df.to_csv(self.metadata_file, index=False)
    
    def run_pipeline(self, max_videos: int = None, workers: int = 1, output_format: str = 'wav'):
        """
        Run the full preprocessing pipeline on all downloaded videos.
        
        Args:
            max_videos: Maximum number of videos to process
            workers: Number of parallel workers (default: 1)
            output_format: The audio format to save (wav, mp3, etc.)
            
        Returns:
            List of paths to successfully extracted audio files
        """
        rich_console.print_info(f"Running preprocessing pipeline with {workers} workers")
        audio_paths = self.batch_extract_audio(max_videos=max_videos, workers=workers, output_format=output_format)
        return audio_paths


def main():
    """Main function to run the preprocessor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Preprocessor')
    parser.add_argument('--max-videos', type=int, default=None, help='Maximum number of videos to process')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--output-format', type=str, default='wav', choices=['wav', 'mp3'], 
                        help='Audio output format')
    
    args = parser.parse_args()
    
    preprocessor = VideoPreprocessor()
    preprocessor.run_pipeline(max_videos=args.max_videos, workers=args.workers, output_format=args.output_format)


if __name__ == "__main__":
    main()
