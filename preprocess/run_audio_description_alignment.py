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
import argparse
from typing import List, Optional

# Set multiprocessing start method before other imports
import multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

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
                                  max_workers: int = 8,
                                  context_segments: int = 3) -> bool:
    """
    Run audio description alignment for specified videos or all videos needing processing.
    
    Args:
        video_ids: Optional list of specific video IDs to process
        max_videos: Maximum number of videos to process
        model_name: Language model to use for alignment
        api_base: API base URL for the language model server
        max_workers: Maximum number of concurrent workers
        context_segments: Number of previous segments to use as context
        
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
    
    # Initialize audio description aligner
    rich_console.print_component_header("Audio Description Alignment Setup", 
                                      f"Initializing LLM alignment for {len(video_ids)} videos")
    
    try:
        aligner = AudioDescriptionAligner(
            model_name=model_name,
            api_base=api_base,
            max_workers=max_workers,
            context_segments=context_segments
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

def main():
    """Main entry point for the audio description alignment script."""
    parser = argparse.ArgumentParser(
        description='Align audio descriptions for temporal and spatial continuity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos needing audio description alignment
  python run_audio_description_alignment.py
  
  # Process specific videos
  python run_audio_description_alignment.py --video-ids video1 video2 video3
  
  # Process up to 10 videos with custom model
  python run_audio_description_alignment.py --max-videos 10 --model Qwen/Qwen3-30B-A3B-Instruct-2507
  
  # Use more context segments and workers
  python run_audio_description_alignment.py --context-segments 5 --max-workers 12
        """
    )
    
    # Video selection arguments
    parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to process')
    parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    
    # Model and processing arguments
    parser.add_argument('--model', type=str, 
                       default=LLM_MODEL,
                       help='Language model to use for alignment')
    parser.add_argument('--api-base', type=str,
                       default=LLM_SERVER_URL,
                       help='API base URL for the language model server')
    parser.add_argument('--max-workers', type=int, default=8,
                       help='Maximum number of concurrent workers (default: 8)')
    parser.add_argument('--context-segments', type=int, default=3,
                       help='Number of previous segments to use as context (default: 3)')
    
    # Output options
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    parser.add_argument('--list-only', action='store_true', 
                       help='Only list videos that need processing, do not process them')
    
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Handle list-only mode
    if args.list_only:
        video_ids = find_videos_needing_audio_alignment(args.max_videos)
        if video_ids:
            rich_console.print_info(f"Videos needing audio description alignment:")
            for video_id in video_ids:
                rich_console.print_info(f"  • {video_id}")
        else:
            rich_console.print_info("No videos need audio description alignment")
        return
    
    # Print startup information
    rich_console.print_header(
        "Audio Description Alignment",
        "Aligning audio descriptions for temporal and spatial continuity"
    )
    
    # Show configuration
    rich_console.print_info("Configuration:")
    rich_console.print_info(f"  • Model: {args.model}")
    rich_console.print_info(f"  • API base: {args.api_base}")
    rich_console.print_info(f"  • Max workers: {args.max_workers}")
    rich_console.print_info(f"  • Context segments: {args.context_segments}")
    if args.max_videos:
        rich_console.print_info(f"  • Max videos: {args.max_videos}")
    if args.video_ids:
        rich_console.print_info(f"  • Specific videos: {len(args.video_ids)} videos")
    rich_console.print_info("")
    
    # Run processing
    success = run_audio_description_alignment(
        video_ids=args.video_ids,
        max_videos=args.max_videos,
        model_name=args.model,
        api_base=args.api_base,
        max_workers=args.max_workers,
        context_segments=args.context_segments
    )
    
    if success:
        rich_console.print_success("Audio description alignment completed successfully")
        sys.exit(0)
    else:
        rich_console.print_error("Audio description alignment failed")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rich_console.print_warning("Audio description alignment interrupted by user")
        sys.exit(130)
    except Exception as e:
        rich_console.print_error(f"Unexpected error: {e}")
        sys.exit(1)
