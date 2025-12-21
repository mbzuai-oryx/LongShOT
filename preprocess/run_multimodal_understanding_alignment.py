#!/usr/bin/env python
"""
Standalone script to run multimodal understanding alignment for temporal continuity.
This script processes already-generated multimodal understanding files to improve
temporal continuity between consecutive segments.
"""

import os
import sys
import logging
import time
from typing import List

from caption_pipeline.utils.rich_console import get_console

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from caption_pipeline.pipeline.multimodal_understanding_aligner import MultimodalUnderstandingAligner
from config import LLM_MODEL, LLM_SERVER_URL

# Set up logging with reduced verbosity
os.makedirs('logs', exist_ok=True)

# Custom filter to reduce verbosity
class VerbosityFilter(logging.Filter):
    def filter(self, record):
        # Suppress certain verbose messages
        suppress_messages = [
            "Multimodal understanding files found",
            "Existing aligned files",
            "Files needing alignment"
        ]
        return not any(msg in record.getMessage() for msg in suppress_messages)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/multimodal_understanding_alignment.log'),
        # Only add StreamHandler with filter to reduce console verbosity
    ]
)

# Add filter to reduce verbosity for multimodal understanding aligner logger specifically
aligner_logger = logging.getLogger('caption_pipeline.pipeline.multimodal_understanding_aligner')
aligner_logger.addFilter(VerbosityFilter())

logger = logging.getLogger(__name__)
rich_console = get_console()


def run_multimodal_understanding_alignment(video_ids: List[str] = None, max_videos: int = None, 
                                         max_workers: int = 8, model_name: str = None,
                                         api_base: str = None) -> List[str]:
    """
    Run multimodal understanding alignment for specified videos or all available videos.
    
    Args:
        video_ids: List of specific video IDs to process, or None for all available
        max_videos: Maximum number of videos to process
        max_workers: Maximum number of concurrent workers
        model_name: Name of the model to use (defaults to config)
        api_base: vLLM server API base URL (defaults to config)
        
    Returns:
        List of aligned multimodal understanding file paths
    """
    start_time = time.time()
    
    # Use defaults from config if not specified
    model_name = model_name or LLM_MODEL
    
    # Determine API base URL
    if api_base is None:
        api_base = LLM_SERVER_URL 
    
    # Initialize multimodal understanding aligner
    init_start = time.time()
    try:
        aligner = MultimodalUnderstandingAligner(
            model_name=model_name,
            api_base=api_base,
            max_workers=max_workers
        )
        init_time = time.time() - init_start
        rich_console.print_success(f"Multimodal understanding aligner initialized in {init_time:.1f}s")
    except Exception as e:
        rich_console.print_error(f"Failed to initialize multimodal understanding aligner: {e}")
        return []
    
    # Determine scope of processing
    if video_ids:
        scope_msg = f"specific video IDs ({len(video_ids)} videos)"
    elif max_videos:
        scope_msg = f"up to {max_videos} videos with multimodal understanding"
    else:
        scope_msg = "all videos with multimodal understanding"
    
    # Print component header
    rich_console.print_component_header(
        "Multimodal Understanding Alignment",
        f"Processing {scope_msg} (max_workers={max_workers})"
    )
    
    # Process alignment
    aligned_files = aligner.batch_align_multimodal_understanding(
        video_ids=video_ids,
        max_videos=max_videos
    )
    
    # Calculate and log statistics
    execution_time = time.time() - start_time
    successful_count = len(aligned_files)
    
    # Print completion message using rich console
    rich_console.print_completion_message("Multimodal Understanding Alignment", {
        'successful': successful_count,
        'total': successful_count,  # All processed files are successful (failed ones aren't returned)
        'duration': execution_time
    })
    
    return aligned_files


