#!/usr/bin/env python
"""
Standalone script to run comprehensive multimodal video understanding generation.
This script processes videos that have captions, video descriptions, and optionally audio descriptions
to create unified, comprehensive segment understanding using Qwen 2.5-VL via vLLM.
"""

import os
import sys
import argparse
import logging
import time
from typing import List, Dict

from caption_pipeline.utils.rich_console import get_console

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from caption_pipeline.pipeline.multimodal_understanding import MultimodalVideoUnderstanding
from config import LLM_MODEL, LLM_SERVER_URL

# Set up logging with reduced verbosity
os.makedirs('logs', exist_ok=True)

# Custom filter to reduce verbosity
class VerbosityFilter(logging.Filter):
    def filter(self, record):
        # Suppress certain verbose messages
        suppress_messages = [
            "Caption files found",
            "Video description files found", 
            "Existing multimodal files",
            "Eligible videos (intersection)",
            "Videos needing processing"
        ]
        return not any(msg in record.getMessage() for msg in suppress_messages)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/multimodal_understanding.log'),
        # Only add StreamHandler with filter to reduce console verbosity
    ]
)

# Add filter to reduce verbosity for multimodal understanding logger specifically
multimodal_logger = logging.getLogger('caption_pipeline.pipeline.multimodal_understanding')
multimodal_logger.addFilter(VerbosityFilter())

logger = logging.getLogger(__name__)
rich_console = get_console()


def run_multimodal_understanding(video_ids: List[str] = None, max_videos: int = None, 
                                max_workers: int = 8, model_name: str = None,
                                api_base: str = None, use_concurrent: bool = True) -> Dict[str, bool]:
    """
    Run comprehensive multimodal understanding generation for specified videos or all available videos.
    
    Args:
        video_ids: List of specific video IDs to process, or None for all available
        max_videos: Maximum number of videos to process
        max_workers: Maximum number of concurrent workers
        model_name: Name of the model to use (defaults to config)
        api_base: vLLM server API base URL (defaults to config)
        use_concurrent: Whether to use concurrent processing
        
    Returns:
        Dictionary mapping video IDs to success status (True/False)
    """
    start_time = time.time()
    
    # Use defaults from config if not specified
    model_name = model_name or LLM_MODEL
    api_base = api_base or LLM_SERVER_URL
    
    # Initialize multimodal understanding processor
    init_start = time.time()
    try:
        processor = MultimodalVideoUnderstanding(
            model_name=model_name,
            api_base=api_base,
            max_workers=max_workers
        )
        init_time = time.time() - init_start
        rich_console.print_success(f"✨ Multimodal understanding processor initialized in {init_time:.1f}s")
    except Exception as e:
        rich_console.print_error(f"Failed to initialize multimodal understanding processor: {e}")
        return {}
    
    # Determine scope of processing
    if video_ids:
        scope_msg = f"specific video IDs ({len(video_ids)} videos)"
    elif max_videos:
        scope_msg = f"up to {max_videos} videos with required multimodal data"
    else:
        scope_msg = "all videos with required multimodal data"
    
    # Print component header
    rich_console.print_component_header(
        "Multimodal Understanding Generation",
        f"Processing {scope_msg} (max_workers={max_workers})"
    )
    
    # Process videos
    results = processor.process_videos(
        video_ids=video_ids,
        max_videos=max_videos,
        use_concurrent=use_concurrent
    )
    
    # Calculate and log statistics
    execution_time = time.time() - start_time
    successful_count = sum(1 for result in results.values() if result)
    total_count = len(results)
    
    # Print completion message using rich console
    rich_console.print_completion_message("Multimodal Understanding", {
        'successful': successful_count,
        'total': total_count,
        'duration': execution_time
    })
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate comprehensive multimodal video understanding')
    parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to process')
    parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    parser.add_argument('--max-workers', type=int, default=8, help='Maximum number of concurrent workers (default: 8)')
    parser.add_argument('--model', type=str, help='Vision model to use (defaults to config)')
    parser.add_argument('--api-base', type=str, help='vLLM server API base URL (defaults to config)')
    parser.add_argument('--sequential', action='store_true', help='Process videos sequentially instead of concurrently')
    parser.add_argument('--list-only', action='store_true', help='Only list videos that need processing')
    
    args = parser.parse_args()
    
    if args.list_only:
        # Show videos that need processing
        try:
            processor = MultimodalVideoUnderstanding()
            videos_needing_processing = processor.get_videos_needing_processing()
            
            if videos_needing_processing:
                rich_console.print_info(f"📋 Videos needing multimodal understanding processing ({len(videos_needing_processing)}):")
                for i, video_id in enumerate(videos_needing_processing, 1):
                    rich_console.console.print(f"  [dim]{i:2d}.[/] [cyan]{video_id}[/]")
            else:
                rich_console.print_success("✅ No videos need multimodal understanding processing")
                
        except Exception as e:
            rich_console.print_error(f"Error checking videos: {e}")
    else:
        # Run processing
        results = run_multimodal_understanding(
            video_ids=args.video_ids,
            max_videos=args.max_videos,
            max_workers=args.max_workers,
            model_name=args.model,
            api_base=args.api_base,
            use_concurrent=not args.sequential
        )
        
        # Print final summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        if successful == total:
            rich_console.print_success(f"🎉 Final Results: {successful}/{total} videos processed successfully")
        else:
            rich_console.print_warning(f"⚠️  Final Results: {successful}/{total} videos processed successfully, {total - successful} failed")
