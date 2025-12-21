#!/usr/bin/env python
"""
Standalone script for running audio description alignment.

This script processes videos that have audio descriptions to apply temporal and spatial 
alignment for environmental continuity using multiple previous segments as context.
"""

import os
import sys
import time
import logging
from typing import List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project components
from caption_pipeline.pipeline.audio_description_aligner import AudioDescriptionAligner
from caption_pipeline.utils.rich_console import get_console
from config import LLM_MODEL, LLM_SERVER_URL, AUDIO_DESCRIPTIONS_DIR

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/audio_description_alignment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize rich console
rich_console = get_console()

def find_videos_needing_audio_alignment(max_videos: Optional[int] = None) -> List[str]:
    """
    Find video IDs that have audio descriptions but need alignment.
    
    Args:
        max_videos: Maximum number of videos to return
        
    Returns:
        List of video IDs that need audio description alignment
    """
    if not os.path.exists(AUDIO_DESCRIPTIONS_DIR):
        rich_console.print_warning(f"Audio descriptions directory not found: {AUDIO_DESCRIPTIONS_DIR}")
        return []
    
    # Get all audio description files that are not already aligned
    video_ids = []
    
    for filename in os.listdir(AUDIO_DESCRIPTIONS_DIR):
        if filename.endswith('_audio_descriptions.json') and not filename.endswith('_aligned.json'):
            video_id = filename.replace('_audio_descriptions.json', '')
            
            # Check if aligned version doesn't already exist
            aligned_file = os.path.join(AUDIO_DESCRIPTIONS_DIR, f"{video_id}_audio_descriptions_aligned.json")
            if not os.path.exists(aligned_file):
                video_ids.append(video_id)
    
    if max_videos and len(video_ids) > max_videos:
        video_ids = video_ids[:max_videos]
    
    rich_console.print_info(f"Found {len(video_ids)} videos needing audio description alignment")
    return video_ids

def run_audio_description_alignment(video_ids: Optional[List[str]] = None,
                                  max_videos: Optional[int] = None,
                                  model_name: str = LLM_MODEL,
                                  api_base: str = LLM_SERVER_URL,
                                  max_workers: int = 8) -> bool:
    """
    Run audio description alignment for specified videos or all videos needing processing.

    Args:
        video_ids: Optional list of specific video IDs to process
        max_videos: Maximum number of videos to process
        model_name: Language model to use for alignment
        api_base: API base URL for the language model server
        max_workers: Maximum number of concurrent workers

    Returns:
        True if processing was successful, False otherwise
    """
    start_time = time.time()

    # Determine videos to process
    if video_ids is None:
        video_ids = find_videos_needing_audio_alignment(max_videos)

    if not video_ids:
        rich_console.print_info("No videos require audio description alignment")
        return True

    # Initialize audio description aligner (context_segments hardcoded to 3)
    rich_console.print_component_header("Audio Description Alignment Setup",
                                      f"Initializing LLM alignment for {len(video_ids)} videos")

    try:
        aligner = AudioDescriptionAligner(
            model_name=model_name,
            api_base=api_base,
            max_workers=max_workers,
            context_segments=3
        )
        
        # Test connection to the model server
        aligner._test_connection()
        
        # Process videos
        rich_console.print_info(f"Starting audio description alignment for {len(video_ids)} videos")
        aligned_files = aligner.batch_align_audio_descriptions(video_ids=video_ids)
        
        # Calculate statistics
        successful_count = len(aligned_files)
        failed_count = len(video_ids) - successful_count
        processing_time = time.time() - start_time
        
        # Print final summary
        rich_console.print_completion_message("Audio Description Alignment", {
            'total': len(video_ids),
            'successful': successful_count,
            'duration': processing_time
        })
        
        if failed_count > 0:
            rich_console.print_warning(f"Failed to align {failed_count} videos")
        
        return successful_count > 0
        
    except Exception as e:
        processing_time = time.time() - start_time
        rich_console.print_error(f"Audio description alignment failed after {processing_time:.2f}s: {e}")
        return False

