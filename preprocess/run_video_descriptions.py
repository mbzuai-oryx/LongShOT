#!/usr/bin/env python
"""
Standalone script to run video description generation for the Arabic video dataset.
This script processes videos that already have completed audio captions and generates
visual descriptions using Qwen 2.5-VL via vLLM.
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
        results = descriptor.process_video_batch(video_ids, auto_metadata=auto_metadata)
    else:
        # Process all available videos
        # Use the batch generation method
        try:
            description_files = descriptor.batch_generate_descriptions(max_videos=max_videos, auto_metadata=auto_metadata)
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


def main():
    """Parse arguments and run video description generation."""
    parser = argparse.ArgumentParser(description='Generate video descriptions for Arabic videos')
    
    # Video selection options
    parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to process')
    parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    
    # Model configuration options
    parser.add_argument('--model', type=str, default=VIDEO_DESCRIPTION_MODEL, 
                       help=f'Vision model to use (default: {VIDEO_DESCRIPTION_MODEL})')
    parser.add_argument('--api-base', type=str, default=VLLM_SERVER_URL,
                       help=f'vLLM server API base URL (default: {VLLM_SERVER_URL})')
    
    # Concurrent processing options
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of concurrent workers (default: 4)')
    
    # Metadata generation options
    parser.add_argument('--no-auto-metadata', action='store_true',
                       help='Disable automatic metadata generation after video descriptions complete')
    
    # Logging options
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Override configuration if custom parameters provided
    model_name = VIDEO_DESCRIPTION_MODEL
    api_base = VLLM_SERVER_URL
    
    if args.model != VIDEO_DESCRIPTION_MODEL:
        model_name = args.model
        rich_console.print_info(f"Using custom model: {model_name}")
    
    if args.api_base != VLLM_SERVER_URL:
        api_base = args.api_base
        rich_console.print_info(f"Using custom API base: {api_base}")
    
    # Run video description generation with potentially custom settings
    from caption_pipeline.pipeline.video_descriptor import VideoDescriptor
    
    # Initialize with custom parameters and enhanced parallelism
    descriptor = VideoDescriptor(
        model_name=model_name,
        api_base=api_base,
        max_workers=args.max_workers
    )
    
    # Process videos with the custom-configured descriptor
    try:
        results = run_video_descriptions(
            video_ids=args.video_ids,
            max_videos=args.max_videos,
            descriptor=descriptor,
            max_workers=args.max_workers,
            auto_metadata=not args.no_auto_metadata,
        )
        
        if results:
            rich_console.print_info("Video description generation completed successfully")
            
            # Print results summary
            successful = [vid for vid, result in results.items() if result is not None]
            failed = [vid for vid, result in results.items() if result is None]
            
            if successful:
                rich_console.print_info(f"Successfully processed videos: {', '.join(successful)}")
            
            if failed:
                rich_console.print_warning(f"Failed to process videos: {', '.join(failed)}")
        else:
            rich_console.print_error("No videos were processed")
            sys.exit(1)
            
    except Exception as e:
        rich_console.print_error(f"Video description generation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
