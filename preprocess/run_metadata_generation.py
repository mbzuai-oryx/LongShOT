#!/usr/bin/env python
"""
Standalone script to run enhanced metadata generation for videos with descriptions.
This script processes videos that already have completed visual descriptions and generates
summaries, categories, and other metadata using LLMs via vLLM.
"""

import os
import sys
import argparse
import logging
import time
from datetime import timedelta
from typing import List, Dict

from caption_pipeline.utils.rich_console import get_console

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from caption_pipeline.pipeline.metadata_generator import VideoMetadataGenerator
from config import VLLM_SERVER_URL

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/metadata_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

rich_console = get_console()


def run_metadata_generation(video_ids: List[str] = None, max_videos: int = None, 
                         use_concurrent: bool = True, max_workers: int = 8, 
                         model_name: str = None, api_base: str = None) -> Dict[str, bool]:
    """
    Run metadata generation for videos with descriptions.
    
    Args:
        video_ids: List of specific video IDs to process, or None for all available
        max_videos: Maximum number of videos to process
        use_concurrent: Enable concurrent processing for faster generation
        max_workers: Maximum number of concurrent workers
        model_name: Name of the model to use (defaults to config)
        api_base: vLLM server API base URL (defaults to config)
        
    Returns:
        Dictionary mapping video IDs to success status (True/False)
    """
    start_time = time.time()
    
    # Use provided parameters or fall back to config defaults
    server_url = api_base or VLLM_SERVER_URL
    
    # Initialize metadata generator
    try:
        generator = VideoMetadataGenerator(
            api_base=server_url,
            max_workers=max_workers
        )
        # Reduced logging - initialization is handled by the generator itself
    except Exception as e:
        rich_console.print_error(f"Failed to initialize metadata generator: {e}")
        return {}
    
    # Process videos
    if video_ids:
        # Process specific video IDs
        results = generator.process_videos(video_ids, use_concurrent=use_concurrent)
    else:
        # Process all available videos
        video_ids_with_descriptions = None  # The generator will find all videos with descriptions
        if max_videos:
            # If we have a limit, get all videos with descriptions first then limit
            descriptions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'video_descriptions')
            all_description_files = [f[:-17] for f in os.listdir(descriptions_dir) if f.endswith('_descriptions_aligned.json')]
            video_ids_with_descriptions = all_description_files[:max_videos]
        
        # Generate metadata
        results = generator.process_videos(video_ids_with_descriptions, use_concurrent=use_concurrent)
    
    # Calculate and log statistics
    execution_time = time.time() - start_time
    successful_count = sum(1 for result in results.values() if result)
    total_count = len(results)
    
    # Brief summary only
    if total_count > successful_count:
        rich_console.print_warning(f"Metadata generation: {successful_count}/{total_count} successful")
    else:
        rich_console.print_success(f"Metadata generation: {successful_count}/{total_count} successful")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate enhanced metadata for videos with descriptions')
    parser.add_argument('--video-id', type=str, nargs='*', help='Specific video ID(s) to process')
    parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of concurrent workers')
    
    args = parser.parse_args()
    
    run_metadata_generation(
        video_ids=args.video_id,
        max_videos=args.max_videos,
        use_concurrent=True,
        max_workers=args.max_workers
    )
