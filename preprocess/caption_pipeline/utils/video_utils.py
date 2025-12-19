"""
Utility functions for video processing.
"""

import os
import subprocess
import tempfile
import logging
import json
from typing import Dict, Any, Optional, Tuple

from caption_pipeline.utils.rich_console import get_console

# Try to import ffmpeg-python, but prepare for fallback
try:
    import ffmpeg
    HAS_FFMPEG_PY = hasattr(ffmpeg, 'input')
except (ImportError, AttributeError):
    HAS_FFMPEG_PY = False

logger = logging.getLogger(__name__)
rich_console = get_console()


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Get metadata from a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video metadata
    """
    try:
        if HAS_FFMPEG_PY:
            # Use ffmpeg-python
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            # Extract useful metadata
            duration = float(probe['format']['duration'])
            width = int(video_info['width'])
            height = int(video_info['height'])
            
            # Calculate frame rate
            frame_rate = 0
            if 'avg_frame_rate' in video_info:
                fr_parts = video_info['avg_frame_rate'].split('/')
                if len(fr_parts) == 2 and int(fr_parts[1]) != 0:
                    frame_rate = int(fr_parts[0]) / int(fr_parts[1])
            
            # Get total frames
            total_frames = 0
            if 'nb_frames' in video_info:
                total_frames = int(video_info['nb_frames'])
            elif frame_rate > 0:
                total_frames = int(duration * frame_rate)
            
            return {
                'duration': duration,
                'width': width,
                'height': height,
                'frame_rate': frame_rate,
                'total_frames': total_frames,
                'format': probe['format']['format_name'],
                'bitrate': int(probe['format']['bit_rate']) if 'bit_rate' in probe['format'] else 0,
            }
        else:
            # Fallback to subprocess
            return _get_video_metadata_subprocess(video_path)
    
    except Exception as e:
        rich_console.print_error(f"Error getting video metadata: {e}")
        return {
            'duration': 0,
            'width': 0,
            'height': 0,
            'frame_rate': 0,
            'total_frames': 0,
            'format': 'unknown',
            'bitrate': 0
        }


def _get_video_metadata_subprocess(video_path: str) -> Dict[str, Any]:
    """
    Get video metadata using subprocess fallback.
    """
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        
        video_info = next((s for s in probe_data['streams'] if s['codec_type'] == 'video'), None)
        if not video_info:
            raise ValueError("No video stream found")
        
        # Extract useful metadata
        duration = float(probe_data['format']['duration'])
        width = int(video_info['width'])
        height = int(video_info['height'])
        
        # Calculate frame rate
        frame_rate = 0
        if 'avg_frame_rate' in video_info:
            fr_parts = video_info['avg_frame_rate'].split('/')
            if len(fr_parts) == 2 and int(fr_parts[1]) != 0:
                frame_rate = int(fr_parts[0]) / int(fr_parts[1])
        
        # Get total frames
        total_frames = 0
        if 'nb_frames' in video_info:
            total_frames = int(video_info['nb_frames'])
        elif frame_rate > 0:
            total_frames = int(duration * frame_rate)
        
        return {
            'duration': duration,
            'width': width,
            'height': height,
            'frame_rate': frame_rate,
            'total_frames': total_frames,
            'format': probe_data['format']['format_name'],
            'bitrate': int(probe_data['format']['bit_rate']) if 'bit_rate' in probe_data['format'] else 0,
        }
    
    except Exception as e:
        rich_console.print_error(f"Error getting video metadata with subprocess: {e}")
        return {
            'duration': 0,
            'width': 0,
            'height': 0,
            'frame_rate': 0,
            'total_frames': 0,
            'format': 'unknown',
            'bitrate': 0
        }


def extract_frame(video_path: str, time_point: float, output_path: Optional[str] = None) -> Optional[str]:
    """
    Extract a frame from a video at a specific time point.
    
    Args:
        video_path: Path to the video file
        time_point: Time point in seconds to extract frame from
        output_path: Path to save the extracted frame, or None to use a temporary file
        
    Returns:
        Path to the extracted frame image, or None if extraction failed
    """
    try:
        if output_path is None:
            # Create temporary file
            fd, output_path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
        
        if HAS_FFMPEG_PY:
            # Extract frame using ffmpeg-python
            (
                ffmpeg
                .input(video_path, ss=time_point)
                .output(output_path, vframes=1)
                .run(quiet=True, overwrite_output=True)
            )
        else:
            # Fallback to subprocess
            cmd = [
                'ffmpeg', '-i', video_path, '-ss', str(time_point),
                '-vframes', '1', '-y', output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
    
    except Exception as e:
        rich_console.print_error(f"Error extracting frame: {e}")
        return None


def extract_segment(video_path: str, start_time: float, end_time: float, output_path: Optional[str] = None) -> Optional[str]:
    """
    Extract a video segment between specified start and end times.
    
    Args:
        video_path: Path to the video file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Path to save the extracted segment, or None to use a temporary file
        
    Returns:
        Path to the extracted video segment, or None if extraction failed
    """
    try:
        if output_path is None:
            # Create temporary file
            fd, output_path = tempfile.mkstemp(suffix='.mp4')
            os.close(fd)
        
        if HAS_FFMPEG_PY:
            # Extract segment using ffmpeg-python
            (
                ffmpeg
                .input(video_path, ss=start_time, to=end_time)
                .output(output_path, c='copy')
                .run(quiet=True, overwrite_output=True)
            )
        else:
            # Fallback to subprocess
            cmd = [
                'ffmpeg', '-i', video_path, 
                '-ss', str(start_time), '-to', str(end_time),
                '-c', 'copy', '-y', output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
    
    except Exception as e:
        rich_console.print_error(f"Error extracting segment: {e}")
        return None


def compress_video(video_path: str, output_path: str, target_size_mb: int = 10, video_codec: str = 'libx264') -> Optional[str]:
    """
    Compress a video to a target file size.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the compressed video
        target_size_mb: Target size in megabytes
        video_codec: Video codec to use
        
    Returns:
        Path to the compressed video, or None if compression failed
    """
    try:
        # Get video metadata
        metadata = get_video_metadata(video_path)
        duration = metadata['duration']
        
        # Calculate target bitrate (bits per second)
        target_size_bits = target_size_mb * 8 * 1024 * 1024
        bitrate = int(target_size_bits / duration)
        
        if HAS_FFMPEG_PY:
            # Compress video using ffmpeg-python
            (
                ffmpeg
                .input(video_path)
                .output(
                    output_path,
                    vcodec=video_codec,
                    audio_bitrate='128k',
                    video_bitrate=str(bitrate),
                    maxrate=str(bitrate),
                    bufsize=str(int(bitrate / 2))
                )
                .run(quiet=True, overwrite_output=True)
            )
        else:
            # Fallback to subprocess
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vcodec', video_codec,
                '-b:a', '128k',
                '-b:v', str(bitrate),
                '-maxrate', str(bitrate),
                '-bufsize', str(int(bitrate / 2)),
                '-y', output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
    
    except Exception as e:
        rich_console.print_error(f"Error compressing video: {e}")
        return None


def get_scene_changes(video_path: str, threshold: float = 0.4) -> list:
    """
    Detect scene changes in a video.
    
    Args:
        video_path: Path to the video file
        threshold: Threshold for scene change detection (0.0-1.0)
        
    Returns:
        List of timestamps (in seconds) where scene changes occur
    """
    try:
        # Create temporary file for ffmpeg output
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Run ffmpeg scene detection
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', video_path,
            '-filter:v', f'select=\'gt(scene,{threshold})\',showinfo',
            '-f', 'null',
            '-'
        ]
        
        result = subprocess.run(
            ffmpeg_cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Parse output to find scene changes
        scene_changes = []
        for line in result.stderr.split('\n'):
            if 'pts_time' in line:
                time_parts = line.split('pts_time:')
                if len(time_parts) > 1:
                    timestamp = float(time_parts[1].split()[0])
                    scene_changes.append(timestamp)
        
        return scene_changes
    
    except Exception as e:
        rich_console.print_error(f"Error detecting scene changes: {e}")
        return []


def generate_thumbnail(video_path: str, output_path: str, time_offset: float = None, width: int = 320) -> bool:
    """
    Generate a thumbnail image from a video file at a specific time offset.
    
    Args:
        video_path: Path to the video file
        output_path: Path where the thumbnail should be saved
        time_offset: Time offset in seconds for the thumbnail (default: 15% of video duration)
        width: Width of the thumbnail in pixels (height will be adjusted to maintain aspect ratio)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get video metadata
        metadata = get_video_metadata(video_path)
        duration = metadata.get('duration', 0)
        
        # If no time offset is specified, use 15% of the video duration
        # This typically skips intros and gets to more representative content
        if time_offset is None:
            if duration > 30:
                # For longer videos, use 15% of duration
                time_offset = duration * 0.15
            elif duration > 5:
                # For medium videos, use 1.5 seconds in
                time_offset = 1.5
            else:
                # For very short videos, use 0.5 seconds
                time_offset = 0.5
            
        # Ensure the time offset is valid
        if time_offset >= duration:
            time_offset = max(0.5, duration * 0.15)
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if HAS_FFMPEG_PY:
            # Generate thumbnail using ffmpeg-python with better quality settings
            (
                ffmpeg
                .input(video_path, ss=time_offset)
                .output(
                    output_path, 
                    vframes=1, 
                    format='image2', 
                    vf=f'scale={width}:-1:flags=lanczos', # Better scaling algorithm
                    q=2  # Quality factor (2-31, lower is better)
                )
                .global_args('-y')  # Overwrite output files without asking
                .global_args('-loglevel', 'error')  # Suppress ffmpeg output
                .run(capture_stdout=True, capture_stderr=True)
            )
        else:
            # Fallback to subprocess
            cmd = [
                'ffmpeg', '-i', video_path, '-ss', str(time_offset),
                '-vframes', '1', '-vf', f'scale={width}:-1:flags=lanczos',
                '-q:v', '2', '-y', '-loglevel', 'error', output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        
        return os.path.exists(output_path)
    except Exception as e:
        rich_console.print_error(f"Error generating thumbnail for {video_path}: {str(e)}")
        return False


def generate_thumbnails_for_all_videos(videos_dir: str, thumbnails_dir: str) -> Dict[str, str]:
    """
    Generate thumbnails for all videos in a directory.
    
    Args:
        videos_dir: Directory containing video files
        thumbnails_dir: Directory where thumbnails should be saved
        
    Returns:
        Dictionary mapping video IDs to thumbnail paths
    """
    thumbnails = {}
    
    # Ensure thumbnails directory exists
    os.makedirs(thumbnails_dir, exist_ok=True)
    
    # Get all video files
    video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]
    
    for video_file in video_files:
        video_id = os.path.splitext(video_file)[0]
        video_path = os.path.join(videos_dir, video_file)
        thumbnail_path = os.path.join(thumbnails_dir, f"{video_id}.jpg")
        
        # Generate thumbnail if it doesn't already exist
        if not os.path.exists(thumbnail_path):
            success = generate_thumbnail(video_path, thumbnail_path)
            if success:
                rich_console.print_info(f"Generated thumbnail for {video_id}")
                thumbnails[video_id] = thumbnail_path
            else:
                rich_console.print_error(f"Failed to generate thumbnail for {video_id}")
        else:
            thumbnails[video_id] = thumbnail_path
            
    return thumbnails
