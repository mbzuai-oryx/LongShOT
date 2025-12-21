#!/usr/bin/env python
"""
Standalone script for running audio descriptions using Audio Flamingo 3.

This script processes videos that have completed video descriptions to generate
audio descriptions for non-speech content using Audio Flamingo 3.
"""

import os
import sys
import logging
import time
from typing import List, Optional


# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project components
from caption_pipeline.pipeline.audio_descriptor import AudioDescriptor
from caption_pipeline.utils.rich_console import get_console
from config import AUDIO_FLAMINGO_MODEL_PATH, VIDEO_DESCRIPTIONS_DIR

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/audio_descriptions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize rich console
rich_console = get_console()

def find_videos_needing_audio_descriptions(max_videos: Optional[int] = None) -> List[str]:
    """
    Find video IDs that have video descriptions but need audio descriptions.
    
    Args:
        max_videos: Maximum number of videos to return
        
    Returns:
        List of video IDs that need audio description processing
    """
    if not os.path.exists(VIDEO_DESCRIPTIONS_DIR):
        rich_console.print_warning(f"Video descriptions directory not found: {VIDEO_DESCRIPTIONS_DIR}")
        return []
    
    # Get all video description files
    video_ids = []
    audio_descriptions_dir = os.path.join(os.path.dirname(VIDEO_DESCRIPTIONS_DIR), 'audio_descriptions')
    
    for filename in os.listdir(VIDEO_DESCRIPTIONS_DIR):
        if filename.endswith('_descriptions.json'):
            video_id = filename.replace('_descriptions.json', '')
            
            # Check if audio descriptions already exist (either aligned or original)
            audio_desc_file = os.path.join(audio_descriptions_dir, f"{video_id}_audio_descriptions.json")
            aligned_audio_desc_file = os.path.join(audio_descriptions_dir, f"{video_id}_audio_descriptions_aligned.json")
            
            if not os.path.exists(audio_desc_file) and not os.path.exists(aligned_audio_desc_file):
                video_ids.append(video_id)
    
    if max_videos and len(video_ids) > max_videos:
        video_ids = video_ids[:max_videos]
    
    rich_console.print_info(f"Found {len(video_ids)} videos needing audio descriptions")
    return video_ids

def run_audio_descriptions(video_ids: Optional[List[str]] = None,
                          max_videos: Optional[int] = None,
                          model_path: str = AUDIO_FLAMINGO_MODEL_PATH,
                          batch_size: int = 8,
                          num_gpus: Optional[int] = None) -> bool:
    """
    Run audio descriptions for specified videos or all videos needing processing.

    Args:
        video_ids: Optional list of specific video IDs to process
        max_videos: Maximum number of videos to process
        model_path: Path to Audio Flamingo 3 model
        batch_size: Batch size for processing
        num_gpus: Number of GPUs to use

    Returns:
        True if processing was successful, False otherwise
    """
    start_time = time.time()

    # Determine videos to process
    if video_ids is None:
        video_ids = find_videos_needing_audio_descriptions(max_videos)

    if not video_ids:
        rich_console.print_info("No videos require audio description processing")
        return True

    # Initialize audio descriptor
    rich_console.print_component_header("Audio Description Setup",
                                      f"Initializing Audio Flamingo 3 for {len(video_ids)} videos")

    try:
        audio_descriptor = AudioDescriptor(
            model_path=model_path,
            batch_size=batch_size,
            num_gpus=num_gpus
        )
        
        # Process videos
        rich_console.print_info(f"Starting audio description processing for {len(video_ids)} videos")
        results = audio_descriptor.process_videos_batch(video_ids)
        
        # Calculate statistics
        successful_count = len([r for r in results.values() if r is not None])
        failed_count = len(video_ids) - successful_count
        processing_time = time.time() - start_time
        
        # Print final summary
        rich_console.print_completion_message("Audio Description Processing", {
            'total': len(video_ids),
            'successful': successful_count,
            'duration': processing_time
        })
        
        if failed_count > 0:
            failed_videos = [vid for vid, result in results.items() if result is None]
            rich_console.print_warning(f"Failed videos: {', '.join(failed_videos[:5])}")
            if len(failed_videos) > 5:
                rich_console.print_warning(f"... and {len(failed_videos) - 5} more")
        
        return successful_count > 0
        
    except Exception as e:
        processing_time = time.time() - start_time
        rich_console.print_error(f"Audio description processing failed after {processing_time:.2f}s: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            if 'audio_descriptor' in locals():
                audio_descriptor.cleanup()
        except Exception as e:
            rich_console.print_warning(f"Cleanup warning: {e}")

