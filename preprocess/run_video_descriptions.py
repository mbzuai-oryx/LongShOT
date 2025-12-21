#!/usr/bin/env python
"""
Standalone script to run video description generation for the LongShOT video dataset.
This script processes videos that already have completed audio captions and generates
visual descriptions using Qwen 2.5-VL via vLLM.
"""

import os
import sys
import logging
import time
from typing import List, Dict

from caption_pipeline.utils.rich_console import get_console

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from caption_pipeline.pipeline.video_descriptor import VideoDescriptor
from caption_pipeline.pipeline.metadata_generator import VideoMetadataGenerator
from config import (ENABLE_VIDEO_DESCRIPTIONS, VIDEO_DESCRIPTION_MODEL, VLLM_SERVER_URL)

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/video_descriptions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
rich_console = get_console()


def run_video_descriptions(video_ids: List[str] = None, max_videos: int = None, descriptor: VideoDescriptor = None, max_workers: int = 8, auto_metadata: bool = True) -> Dict[str, str]:
    """
    Run video description generation for specified videos or all available videos.
    
    Args:
        video_ids: List of specific video IDs to process, or None for all available
        max_videos: Maximum number of videos to process
        descriptor: Pre-configured VideoDescriptor instance, or None to create default
        max_workers: Maximum number of concurrent workers
        auto_metadata: Automatically generate metadata after video descriptions complete
        
    Returns:
        Dictionary mapping video IDs to result file paths (None for failures)
    """
    start_time = time.time()
    
    # Check if video descriptions are enabled
    if not ENABLE_VIDEO_DESCRIPTIONS:
        rich_console.print_error("Video descriptions are disabled in configuration. Please set ENABLE_VIDEO_DESCRIPTIONS=True")
        return {}
    
    # Initialize video descriptor if not provided
    if descriptor is None:
        try:
            descriptor = VideoDescriptor(
                model_name=VIDEO_DESCRIPTION_MODEL,
                api_base=VLLM_SERVER_URL,
                max_workers=max_workers
            )
            # Reduced logging - only essential info
            rich_console.print_info(f"Video descriptor initialized with model: {VIDEO_DESCRIPTION_MODEL}")
        except Exception as e:
            rich_console.print_error(f"Failed to initialize video descriptor: {e}")
            return {}
    
    # Process videos
    results = {}
    
    if video_ids:
        # Process specific video IDs
        results = descriptor.process_video_batch(video_ids)
    else:
        # Process all available videos
        # Use the batch generation method
        try:
            description_files = descriptor.batch_generate_descriptions(max_videos=max_videos)
            results = {os.path.basename(f).replace('_descriptions.json', ''): f 
                      for f in description_files}
        except Exception as e:
            rich_console.print_error(f"Error during batch processing: {e}")
            return {}
    
    # Calculate and log statistics
    execution_time = time.time() - start_time
    successful_count = sum(1 for result in results.values() if result is not None)
    total_count = len(results)
    
    # Brief summary only
    if total_count > successful_count:
        rich_console.print_warning(f"Video descriptions: {successful_count}/{total_count} successful")
    else:
        rich_console.print_success(f"Video descriptions: {successful_count}/{total_count} successful")
    
    # Automatically trigger metadata generation if enabled and we have successful results
    if auto_metadata and successful_count > 0:
        rich_console.print_info("Automatically triggering metadata generation for videos with descriptions...")
        try:
            successful_video_ids = [vid for vid, result in results.items() if result is not None]
            trigger_metadata_generation(successful_video_ids, max_workers=max_workers)
        except Exception as e:
            rich_console.print_warning(f"Failed to automatically generate metadata: {e}")
            rich_console.print_info("You can manually run metadata generation later with: python run_metadata_generation.py")
    
    return results


def trigger_metadata_generation(video_ids: List[str], max_workers: int = 8) -> bool:
    """
    Trigger metadata generation for videos that have completed video descriptions.
    
    Args:
        video_ids: List of video IDs to generate metadata for
        max_workers: Maximum number of concurrent workers
        
    Returns:
        True if metadata generation was successful, False otherwise
    """
    try:
        rich_console.print_info(f"Starting automatic metadata generation for {len(video_ids)} videos...")
        
        # Initialize metadata generator
        metadata_generator = VideoMetadataGenerator(
            api_base=VLLM_SERVER_URL,
            max_workers=max_workers
        )
        
        # Generate metadata for the successful videos
        metadata_results = metadata_generator.process_videos(
            video_ids=video_ids
        )
        
        # Log results
        successful_metadata = sum(1 for result in metadata_results.values() if result)
        total_metadata = len(metadata_results)
        
        rich_console.print_info(f"Metadata generation completed: {successful_metadata}/{total_metadata} videos processed successfully")
        
        return successful_metadata > 0
        
    except Exception as e:
        rich_console.print_error(f"Error during automatic metadata generation: {e}")
        return False


