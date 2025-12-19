#!/usr/bin/env python
"""
Standalone script for running audio descriptions using Audio Flamingo 3.

This script processes videos that have completed video descriptions to generate
audio descriptions for non-speech content using Audio Flamingo 3.
"""

import os
import sys
import argparse
import logging
import time
from typing import List, Optional

# Set multiprocessing start method before other imports
import multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

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

def main():
    """Main entry point for the audio descriptions script."""
    parser = argparse.ArgumentParser(
        description='Generate audio descriptions for videos using Audio Flamingo 3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos needing audio descriptions
  python run_audio_descriptions.py
  
  # Process specific videos
  python run_audio_descriptions.py --video-ids video1 video2 video3
  
  # Process up to 10 videos with custom model path
  python run_audio_descriptions.py --max-videos 10 --model-path /path/to/model
  
  # Use multiple GPUs with larger batch size
  python run_audio_descriptions.py --num-gpus 2 --batch-size 16
        """
    )
    
    # Video selection arguments
    parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to process')
    parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    
    # Model and processing arguments
    parser.add_argument('--model-path', type=str, 
                       default=AUDIO_FLAMINGO_MODEL_PATH,
                       help='Path to Audio Flamingo 3 model')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing (default: 8)')
    parser.add_argument('--num-gpus', type=int, help='Number of GPUs to use (default: auto-detect)')
    
    
    # Processing options
    parser.add_argument('--text-prompt', type=str,
                       default="Please describe the audio in detail, focusing on music, sounds, ambience, and any non-speech audio content.",
                       help='Text prompt for audio description')
    
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
        video_ids = find_videos_needing_audio_descriptions(args.max_videos)
        if video_ids:
            print("Videos needing audio descriptions:")
            for video_id in video_ids:
                print(f"  - {video_id}")
        else:
            print("No videos need audio descriptions")
        return
    
    # Print startup information
    rich_console.print_header(
        "Audio Description Generation",
        "Processing videos with Audio Flamingo 3 for non-speech audio content"
    )
    
    # Show configuration
    rich_console.print_info("Configuration:")
    rich_console.print_info(f"  • Model path: {args.model_path}")
    rich_console.print_info(f"  • Batch size: {args.batch_size}")
    rich_console.print_info(f"  • GPUs: {args.num_gpus or 'auto-detect'}")
    if args.max_videos:
        rich_console.print_info(f"  • Max videos: {args.max_videos}")
    if args.video_ids:
        rich_console.print_info(f"  • Specific videos: {len(args.video_ids)} videos")
    rich_console.print_info("")
    
    # Run processing
    success = run_audio_descriptions(
        video_ids=args.video_ids,
        max_videos=args.max_videos,
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus
    )
    
    if success:
        rich_console.print_success("Audio description processing completed successfully")
        sys.exit(0)
    else:
        rich_console.print_error("Audio description processing failed")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rich_console.print_warning("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        rich_console.print_error(f"Unexpected error: {e}")
        sys.exit(1)