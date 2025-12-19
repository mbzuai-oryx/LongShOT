#!/usr/bin/env python
"""
Standalone script to run key events generation from aligned descriptions.

This script generates key events for video segments by analyzing aligned descriptions
from multiple sources (video descriptions, audio descriptions, multimodal understanding).
"""

import os
import sys
import time
import argparse
import logging

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from config import LLM_MODEL, LLM_SERVER_URL, KEY_EVENTS_DIR
from caption_pipeline.pipeline.key_events_generator import KeyEventsGenerator
from caption_pipeline.utils.rich_console import get_console

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/key_events_generation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
rich_console = get_console()


def run_key_events_generation(video_ids=None, max_videos=None, model_name=LLM_MODEL, 
                             api_base=LLM_SERVER_URL, max_workers=8):
    """Run key events generation for videos.
    
    Args:
        video_ids: List of specific video IDs to process
        max_videos: Maximum number of videos to process
        model_name: Language model to use
        api_base: vLLM server API base URL
        max_workers: Maximum number of concurrent workers
    """
    rich_console.print_header("Key Events Generation from Aligned Descriptions")
    
    # Create output directory
    os.makedirs(KEY_EVENTS_DIR, exist_ok=True)
    
    start_time = time.time()
    
    # Initialize generator
    generator = KeyEventsGenerator(
        model_name=model_name,
        api_base=api_base,
        max_workers=max_workers
    )
    
    # Process key events generation
    key_events_files = generator.batch_generate_key_events(
        video_ids=video_ids,
        max_videos=max_videos
    )
    
    # Print summary
    processing_time = time.time() - start_time
    
    rich_console.print_completion_message("Key Events Generation", {
        'total': len(key_events_files),
        'successful': len(key_events_files),
        'duration': processing_time / 60
    })
    
    if key_events_files:
        rich_console.print_info("Generated key events files:")
        for key_events_file in key_events_files:
            rich_console.print_info(f"  - {key_events_file}")
    
    return key_events_files


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate key events from aligned descriptions')
    parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to process')
    parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    parser.add_argument('--model', type=str, default=LLM_MODEL, help='Language model to use')
    parser.add_argument('--api-base', type=str, default=LLM_SERVER_URL, help='vLLM server API base URL')
    parser.add_argument('--max-workers', type=int, default=8, help='Maximum number of concurrent workers')
    
    args = parser.parse_args()
    
    # Run key events generation
    key_events_files = run_key_events_generation(
        video_ids=args.video_ids,
        max_videos=args.max_videos,
        model_name=args.model,
        api_base=args.api_base,
        max_workers=args.max_workers
    )
    
    rich_console.print_info(f"Key events generation completed. Generated files for {len(key_events_files)} videos.")


if __name__ == "__main__":
    main()
