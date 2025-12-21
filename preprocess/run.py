#!/usr/bin/env python
"""
Main script to run the complete pipeline for the LongShOT video dataset.
"""

import os
import argparse
import logging
import torch.multiprocessing as mp
import time
from datetime import timedelta

from config import AUDIO_FLAMINGO_MODEL_PATH

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

from caption_pipeline.pipeline.orchestrator import PipelineOrchestrator

# Import rich console utilities
from caption_pipeline.utils.rich_console import get_console

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors on console
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/caption_pipeline.log'),
        logging.StreamHandler()  # Rich will handle most console output
    ]
)

# Suppress verbose logging from external libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('faster_whisper').setLevel(logging.WARNING)
logging.getLogger('caption_pipeline.pipeline.caption_generator').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Dictionary to store execution times
execution_times = {}

# Initialize rich console
rich_console = get_console()


def get_processing_args(args, defaults=None):
    """Extract common processing arguments from args with defaults."""
    defaults = defaults or {}
    return {
        'video_ids': getattr(args, 'video_ids', None),
        'max_videos': getattr(args, 'max_videos', None),
        'max_workers': getattr(args, 'max_workers', defaults.get('max_workers', 8)),
    }


def build_scope_message(video_ids, max_videos, context: str = "") -> str:
    """Build scope description message for processing stages."""
    if video_ids:
        return f"specific video IDs ({len(video_ids)} videos)"
    elif max_videos:
        return f"up to {max_videos} videos{' with ' + context if context else ''}"
    return f"all videos{' with ' + context if context else ''}"

def run_video_descriptions(args):
    """Run video description generation with parallel or sequential processing."""
    start_time = time.time()

    from run_video_descriptions import run_video_descriptions as run_desc

    proc = get_processing_args(args)
    scope_msg = build_scope_message(proc['video_ids'], proc['max_videos'], "completed captions")
    rich_console.print_info(f"Starting video description generation for {scope_msg} (concurrent, max_workers={proc['max_workers']})...")
    
    # Disable auto_metadata to prevent duplicate metadata generation since we call it explicitly later
    run_desc(video_ids=proc['video_ids'], max_videos=proc['max_videos'], max_workers=proc['max_workers'], auto_metadata=False)
    
    execution_time = time.time() - start_time
    execution_times['video_descriptions'] = execution_time


def run_audio_descriptions(args):
    """Run audio description generation using Audio Flamingo 3."""
    start_time = time.time()

    from run_audio_descriptions import run_audio_descriptions as run_audio

    proc = get_processing_args(args)
    scope_msg = build_scope_message(proc['video_ids'], proc['max_videos'], "video descriptions")
    rich_console.print_info(f"Starting audio description generation for {scope_msg}...")
    
    # Get model path and processing args
    model_path = getattr(args, 'audio_model_path', AUDIO_FLAMINGO_MODEL_PATH)
    batch_size = getattr(args, 'audio_batch_size', 8)
    num_gpus = getattr(args, 'audio_gpus', None)

    run_audio(video_ids=proc['video_ids'], max_videos=proc['max_videos'],
              model_path=model_path, batch_size=batch_size, num_gpus=num_gpus)
    
    execution_time = time.time() - start_time
    execution_times['audio_descriptions'] = execution_time


def run_video_description_alignment(args, model_name=None):
    """Run video description alignment for temporal continuity."""
    start_time = time.time()
    
    from caption_pipeline.pipeline.video_description_aligner import VideoDescriptionAligner
    from config import VIDEO_DESCRIPTION_MODEL, LLM_MODEL, VLLM_SERVER_URL, LLM_SERVER_URL
    
    video_ids = args.video_ids if hasattr(args, 'video_ids') else None
    max_videos = args.max_videos if hasattr(args, 'max_videos') else None
    max_workers = getattr(args, 'max_workers', 8)
    # Use provided model_name or fall back to VIDEO_DESCRIPTION_MODEL
    model_name = model_name or getattr(args, 'model', VIDEO_DESCRIPTION_MODEL)
    # Use LLM server URL if using LLM model, otherwise VLM server URL
    api_base = LLM_SERVER_URL if model_name == LLM_MODEL else VLLM_SERVER_URL
    
    # Consolidated startup message
    scope_msg = ""
    if video_ids:
        scope_msg = f"specific video IDs ({len(video_ids)} videos)"
    elif max_videos:
        scope_msg = f"up to {max_videos} videos with descriptions"
    else:
        scope_msg = "all videos with descriptions"
    
    rich_console.print_info(f"Starting video description alignment for {scope_msg} (max_workers={max_workers})...")
    
    # Initialize aligner
    aligner = VideoDescriptionAligner(
        model_name=model_name,
        api_base=api_base,
        max_workers=max_workers
    )
    
    # Process alignments
    aligned_files = aligner.batch_align_descriptions(
        video_ids=video_ids,
        max_videos=max_videos
    )
    
    execution_time = time.time() - start_time
    execution_times['video_description_alignment'] = execution_time
    
    rich_console.print_success(f"Video description alignment completed for {len(aligned_files)} videos in {execution_time/60:.1f} minutes")


def run_multimodal_understanding(args, model_name=None):
    """Run comprehensive multimodal understanding generation."""
    start_time = time.time()
    
    from run_multimodal_understanding import run_multimodal_understanding as run_multimodal
    from config import LLM_MODEL, LLM_SERVER_URL
    
    video_ids = args.video_ids if hasattr(args, 'video_ids') else None
    max_videos = args.max_videos if hasattr(args, 'max_videos') else None
    max_workers = getattr(args, 'max_workers', 8)
    # Use provided model_name or fall back to VIDEO_DESCRIPTION_MODEL
    model_to_use =  LLM_MODEL
    
    # Consolidated startup message
    scope_msg = ""
    if video_ids:
        scope_msg = f"specific video IDs ({len(video_ids)} videos)"
    elif max_videos:
        scope_msg = f"up to {max_videos} videos with video and audio descriptions"
    else:
        scope_msg = "all videos with video and audio descriptions"
    
    rich_console.print_info(f"Starting comprehensive multimodal understanding for {scope_msg} (max_workers={max_workers}, model={model_to_use})...")
    
    # Get appropriate server URL
    from config import LLM_SERVER_URL, VLLM_SERVER_URL, LLM_MODEL
    api_base = LLM_SERVER_URL if model_to_use == LLM_MODEL else VLLM_SERVER_URL
    
    # Pass model name and API base to the function
    run_multimodal(video_ids=video_ids, max_videos=max_videos, max_workers=max_workers, 
                   model_name=model_to_use, api_base=api_base)
    
    execution_time = time.time() - start_time
    execution_times['multimodal_understanding'] = execution_time


def run_multimodal_understanding_alignment(args, model_name=None):
    """Run multimodal understanding alignment for temporal continuity."""
    start_time = time.time()
    
    from run_multimodal_understanding_alignment import run_multimodal_understanding_alignment as run_alignment
    from config import  LLM_MODEL, LLM_SERVER_URL
    
    video_ids = args.video_ids if hasattr(args, 'video_ids') else None
    max_videos = args.max_videos if hasattr(args, 'max_videos') else None
    max_workers = getattr(args, 'max_workers', 8)
    # Use provided model_name or fall back to VIDEO_DESCRIPTION_MODEL
    model_to_use = model_name or getattr(args, 'model', LLM_MODEL)
    
    # Consolidated startup message
    scope_msg = ""
    if video_ids:
        scope_msg = f"specific video IDs ({len(video_ids)} videos)"
    elif max_videos:
        scope_msg = f"up to {max_videos} videos with multimodal understanding"
    else:
        scope_msg = "all videos with multimodal understanding"
    
    rich_console.print_info(f"Starting multimodal understanding alignment for {scope_msg} (max_workers={max_workers}, model={model_to_use})...")
    
    # Get appropriate server URL
    api_base = LLM_SERVER_URL 
    
    # Run alignment
    aligned_files = run_alignment(video_ids=video_ids, max_videos=max_videos, max_workers=max_workers, 
                                 model_name=model_to_use, api_base=api_base)
    
    execution_time = time.time() - start_time
    execution_times['multimodal_understanding_alignment'] = execution_time
    
    rich_console.print_success(f"Multimodal understanding alignment completed for {len(aligned_files)} videos in {execution_time/60:.1f} minutes")


def run_audio_description_alignment(args, model_name=None):
    """Run audio description alignment for temporal and spatial continuity."""
    start_time = time.time()

    from run_audio_description_alignment import run_audio_description_alignment as run_alignment
    from config import LLM_MODEL, LLM_SERVER_URL

    video_ids = args.video_ids if hasattr(args, 'video_ids') else None
    max_videos = args.max_videos if hasattr(args, 'max_videos') else None
    max_workers = getattr(args, 'max_workers', 8)
    # Use provided model_name or fall back to LLM_MODEL
    model_to_use = model_name or getattr(args, 'model', LLM_MODEL)

    # Consolidated startup message
    scope_msg = ""
    if video_ids:
        scope_msg = f"specific videos ({len(video_ids)} videos)"
    elif max_videos:
        scope_msg = f"up to {max_videos} videos"
    else:
        scope_msg = "all available videos"

    rich_console.print_info(f"Starting audio description alignment for {scope_msg} (max_workers={max_workers}, model={model_to_use})...")

    # Get appropriate server URL
    api_base = LLM_SERVER_URL

    # Run alignment (context_segments hardcoded to 3)
    success = run_alignment(video_ids=video_ids, max_videos=max_videos,
                           model_name=model_to_use, api_base=api_base,
                           max_workers=max_workers)
    
    execution_time = time.time() - start_time
    execution_times['audio_description_alignment'] = execution_time
    
    if success:
        rich_console.print_success(f"Audio description alignment completed in {execution_time/60:.1f} minutes")
    else:
        rich_console.print_warning(f"Audio description alignment completed with some issues in {execution_time/60:.1f} minutes")


def run_key_events_generation(args, model_name=None):
    """Run key events generation from multimodal understanding data."""
    start_time = time.time()
    
    from run_key_events_generation import run_key_events_generation as run_generation
    from config import LLM_MODEL, LLM_SERVER_URL
    
    video_ids = args.video_ids if hasattr(args, 'video_ids') else None
    max_videos = args.max_videos if hasattr(args, 'max_videos') else None
    max_workers = getattr(args, 'max_workers', 8)
    # Use provided model_name or fall back to LLM_MODEL
    model_to_use = model_name or getattr(args, 'model', LLM_MODEL)
    
    # Consolidated startup message
    scope_msg = ""
    if video_ids:
        scope_msg = f"specific videos ({len(video_ids)} videos)"
    elif max_videos:
        scope_msg = f"up to {max_videos} videos"
    else:
        scope_msg = "all available videos"
    
    rich_console.print_info(f"Starting key events generation for {scope_msg} (max_workers={max_workers}, model={model_to_use})...")
    
    # Get appropriate server URL
    api_base = LLM_SERVER_URL 
    
    # Run key events generation
    key_events_files = run_generation(video_ids=video_ids, max_videos=max_videos, 
                                     model_name=model_to_use, api_base=api_base, 
                                     max_workers=max_workers)
    
    execution_time = time.time() - start_time
    execution_times['key_events_generation'] = execution_time
    
    if key_events_files:
        rich_console.print_success(f"Key events generation completed for {len(key_events_files)} videos in {execution_time/60:.1f} minutes")
    else:
        rich_console.print_warning(f"Key events generation completed with issues in {execution_time/60:.1f} minutes")


def run_metadata_generation(args, model_name=None):
    """Run enhanced metadata generation with parallel processing by default."""
    start_time = time.time()

    from run_metadata_generation import run_metadata_generation as run_meta
    from config import VIDEO_DESCRIPTION_MODEL

    video_ids = args.video_ids if hasattr(args, 'video_ids') else None
    max_videos = args.max_videos if hasattr(args, 'max_videos') else None
    max_workers = getattr(args, 'max_workers', 8)
    # Use provided model_name or fall back to VIDEO_DESCRIPTION_MODEL
    model_to_use = model_name or getattr(args, 'model', VIDEO_DESCRIPTION_MODEL)
    
    # Consolidated startup message
    scope_msg = ""
    if video_ids:
        scope_msg = f"specific video IDs ({len(video_ids)} videos)"
    elif max_videos:
        scope_msg = f"up to {max_videos} videos with descriptions"
    else:
        scope_msg = "all videos with descriptions"
    
    rich_console.print_info(f"Starting enhanced metadata generation for {scope_msg} (max_workers={max_workers}, model={model_to_use})...")
    
    # Get appropriate server URL
    from config import LLM_SERVER_URL, VLLM_SERVER_URL, LLM_MODEL
    api_base = LLM_SERVER_URL if model_to_use == LLM_MODEL else VLLM_SERVER_URL

    # Pass API base to the function
    run_meta(video_ids=video_ids, max_videos=max_videos,
             use_concurrent=True, max_workers=max_workers,
             api_base=api_base)
    
    execution_time = time.time() - start_time
    execution_times['metadata_generation'] = execution_time


def run_final_consolidation(args):
    """Run final consolidation to create training-ready JSONL files."""
    start_time = time.time()

    from caption_pipeline.pipeline.final_consolidator import FinalConsolidator

    proc = get_processing_args(args)
    scope_msg = build_scope_message(proc['video_ids'], proc['max_videos'], "processed")
    rich_console.print_info(f"Starting final consolidation for {scope_msg}...")

    # Initialize consolidator and process videos
    consolidator = FinalConsolidator()
    consolidator.consolidate_all_videos(video_ids=proc['video_ids'], max_videos=proc['max_videos'])
    
    execution_time = time.time() - start_time
    execution_times['final_consolidation'] = execution_time


def generate_dataset_summary(args):
    """Generate comprehensive dataset summary and statistics."""
    start_time = time.time()
    
    from caption_pipeline.utils.dataset_analyzer import DatasetAnalyzer
    from config import DATASET_DIR
    
    rich_console.print_info("Generating comprehensive dataset summary...")
    
    # Initialize analyzer and generate summary
    analyzer = DatasetAnalyzer(DATASET_DIR)
    summary_file = analyzer.generate_and_save_summary()
    
    rich_console.print_success(f"Dataset summary saved to: {summary_file}")
    
    execution_time = time.time() - start_time
    execution_times['dataset_summary'] = execution_time


def print_execution_summary():
    """Print a summary of execution times for all components."""
    if execution_times:
        rich_console.print_pipeline_summary(execution_times)


def run_vlm_stages(args):
    """Run the first set of pipeline stages using VLM model (up to audio descriptions)."""
    full_start_time = time.time()
    
    from config import VIDEO_DESCRIPTION_MODEL
    
    # Print pipeline startup information
    rich_console.print_info("Starting VLM stages (main pipeline + video descriptions + audio descriptions)...")
    rich_console.print_info(f"Using VLM model: {VIDEO_DESCRIPTION_MODEL}")
    rich_console.print_info(f"Command line arguments: --max-videos={getattr(args, 'max_videos', None)}")
    
    # Use the parallel pipeline orchestrator for main pipeline
    rich_console.print_info("Creating pipeline orchestrator with specified configuration...")
    orchestrator = PipelineOrchestrator(
        max_videos=getattr(args, 'max_videos', None),
        download_workers=getattr(args, 'download_workers', 1),
        preprocess_workers=getattr(args, 'preprocess_workers', 4),
        caption_workers=getattr(args, 'caption_workers', 2),
        output_format=getattr(args, 'output_format', 'wav'),
        whisper_model_size=getattr(args, 'model_size', 'large-v3'),
        whisper_compute_type=getattr(args, 'compute_type', 'float16'),
        whisper_batch_size=getattr(args, 'batch_size', 16)
    )
    rich_console.print_success("Pipeline orchestrator created successfully")
    
    # Run the main pipeline (download, preprocess, captions)
    rich_console.print_info("Starting main pipeline execution...")
    total_videos, completed, failed = orchestrator.run_pipeline()
    
    # Run video descriptions if completed videos exist
    if completed > 0:
        rich_console.print_info(f"Main pipeline completed. Starting video description generation for {completed} videos...")
        # Set up args for video descriptions with proper defaults
        if not hasattr(args, 'max_workers'):
            args.max_workers = 8
        
        # Use parallel cleanup - clean up main pipeline models while initializing video descriptions
        def init_video_descriptions():
            return run_video_descriptions(args)
        
        orchestrator.cleanup_and_prepare_for_next_stage(init_video_descriptions)
    elif completed == 0:
        rich_console.print_warning("No videos completed main pipeline - skipping video descriptions")
    
    # Run audio descriptions for completed videos
    if completed > 0:
        rich_console.print_info("Starting audio description generation for completed videos...")
        run_audio_descriptions(args)

        # Clean up Audio Flamingo resources after completion
        try:
            from caption_pipeline.utils.model_cleanup import model_cleanup_manager
            rich_console.print_info("Cleaning up Audio Flamingo 3 resources...")
            model_cleanup_manager._force_gpu_cleanup()
            rich_console.print_success("Audio Flamingo 3 cleanup completed")
        except Exception as e:
            rich_console.print_warning(f"Warning during Audio Flamingo cleanup: {e}")
    else:
        rich_console.print_warning("No videos completed main pipeline - skipping audio descriptions")
    
    # Calculate execution time
    full_execution_time = time.time() - full_start_time
    execution_times['vlm_stages'] = full_execution_time
    
    # Print completion summary
    rich_console.print_success(f"VLM stages completed in {timedelta(seconds=full_execution_time)}")
    rich_console.print_info(f"VLM stages results: {completed} completed, {failed} failed out of {total_videos} total videos")
    
    # Print execution summary
    print_execution_summary()


def run_llm_stages(args):
    """Run the second set of pipeline stages using LLM model (alignment + multimodal + metadata + consolidation)."""
    full_start_time = time.time()
    
    from config import LLM_MODEL
    
    # Print pipeline startup information
    rich_console.print_info("Starting LLM stages (alignment + multimodal understanding + metadata + consolidation)...")
    rich_console.print_info(f"Using LLM model: {LLM_MODEL}")
    rich_console.print_info(f"Command line arguments: --max-videos={getattr(args, 'max_videos', None)}")
    
    # Count eligible videos for processing
    from config import VIDEO_DESCRIPTIONS_DIR
    
    # Find description files to process (similar to video_description_aligner logic)
    files_to_process = []
    video_ids = getattr(args, 'video_ids', None)
    max_videos = getattr(args, 'max_videos', None)
    
    if video_ids:
        # Process specific video IDs
        for video_id in video_ids:
            desc_file = os.path.join(VIDEO_DESCRIPTIONS_DIR, f"{video_id}_descriptions.json")
            if os.path.exists(desc_file):
                files_to_process.append(video_id)
            else:
                rich_console.print_warning(f"Description file not found for {video_id}")
    else:
        # Find all description files
        if os.path.exists(VIDEO_DESCRIPTIONS_DIR):
            for filename in os.listdir(VIDEO_DESCRIPTIONS_DIR):
                if filename.endswith('_descriptions.json'):
                    video_id = filename.replace('_descriptions.json', '')
                    files_to_process.append(video_id)
    
    if max_videos:
        files_to_process = files_to_process[:max_videos]
    
    if not files_to_process:
        rich_console.print_warning("No videos with video descriptions found - cannot proceed with LLM stages")
        return
    
    rich_console.print_info(f"Found {len(files_to_process)} videos ready for LLM processing")
    
    # Set up default max_workers if not provided
    if not hasattr(args, 'max_workers'):
        args.max_workers = 8
    
    # Run video description alignment using LLM
    rich_console.print_info(f"Starting video description alignment for {len(files_to_process)} videos...")
    run_video_description_alignment(args, model_name=LLM_MODEL)
    
    # Run audio description alignment using LLM (if audio descriptions exist)
    rich_console.print_info(f"Starting audio description alignment for {len(files_to_process)} videos...")
    run_audio_description_alignment(args, model_name=LLM_MODEL)
    
    # Run multimodal understanding using LLM
    rich_console.print_info(f"Starting comprehensive multimodal understanding for {len(files_to_process)} videos...")
    run_multimodal_understanding(args, model_name=LLM_MODEL)
    
    # Run multimodal understanding alignment using LLM
    rich_console.print_info(f"Starting multimodal understanding alignment for {len(files_to_process)} videos...")
    run_multimodal_understanding_alignment(args, model_name=LLM_MODEL)
    
    # Run key events generation using LLM
    rich_console.print_info(f"Starting key events generation for {len(files_to_process)} videos...")
    run_key_events_generation(args, model_name=LLM_MODEL)
    
    # Run metadata generation using LLM
    rich_console.print_info(f"Starting metadata generation for {len(files_to_process)} videos...")
    run_metadata_generation(args, model_name=LLM_MODEL)
    
    # Run final consolidation
    rich_console.print_info(f"Starting final consolidation for {len(files_to_process)} videos...")
    run_final_consolidation(args)
    
    # Generate dataset summary
    rich_console.print_info(f"Generating comprehensive dataset summary for {len(files_to_process)} videos...")
    generate_dataset_summary(args)
    
    # Calculate execution time
    full_execution_time = time.time() - full_start_time
    execution_times['llm_stages'] = full_execution_time
    
    # Print completion summary
    rich_console.print_success(f"LLM stages completed in {timedelta(seconds=full_execution_time)}")
    rich_console.print_info(f"LLM stages processed {len(files_to_process)} videos")
    
    # Print execution summary
    print_execution_summary()


def main():
    """Parse arguments and run the appropriate pipeline component."""
    parser = argparse.ArgumentParser(description='LongShOT Video Dataset Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Pipeline component to run')

    # ========== Parent Parsers ==========

    # Common parent parser for video ID filtering (all commands)
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to process')
    common_parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')

    # Parent parser for LLM-based stages (alignments, multimodal, metadata)
    llm_parser = argparse.ArgumentParser(add_help=False)
    llm_parser.add_argument('--model', type=str, help='Language model to use')
    llm_parser.add_argument('--api-base', type=str, help='vLLM server API base URL')
    llm_parser.add_argument('--max-workers', type=int, default=8, help='Maximum concurrent workers (default: 8)')

    # Parent parser for VLM pipeline stages (Whisper, preprocessing)
    vlm_parser = argparse.ArgumentParser(add_help=False)
    vlm_parser.add_argument('--download-workers', type=int, default=1, help='Parallel download workers')
    vlm_parser.add_argument('--preprocess-workers', type=int, default=4, help='Parallel preprocessing workers')
    vlm_parser.add_argument('--caption-workers', type=int, default=2, help='Parallel captioning workers')
    vlm_parser.add_argument('--output-format', type=str, default='wav', choices=['wav', 'mp3'], help='Audio output format')
    vlm_parser.add_argument('--model-size', type=str, default='large-v3',
                           choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'distil-large-v3'],
                           help='Whisper model size')
    vlm_parser.add_argument('--compute-type', type=str, default='float16',
                           choices=['float32', 'float16', 'int8_float16', 'int8'], help='Whisper compute type')
    vlm_parser.add_argument('--batch-size', type=int, default=16, help='Batch size for processing')

    # Parent parser for audio description stages
    audio_parser = argparse.ArgumentParser(add_help=False)
    audio_parser.add_argument('--audio-model-path', type=str, default=AUDIO_FLAMINGO_MODEL_PATH, help='Audio Flamingo 3 model path')
    audio_parser.add_argument('--audio-batch-size', type=int, default=8, help='Batch size for audio processing')
    audio_parser.add_argument('--audio-gpus', type=int, help='Number of GPUs for audio processing')

    # ========== Subparsers ==========

    # Video descriptions parser
    video_desc_parser = subparsers.add_parser('video-descriptions', parents=[common_parser, llm_parser],
                                              help='Generate video descriptions')

    # Video description alignment parser
    subparsers.add_parser('video-description-alignment', parents=[common_parser, llm_parser],
                          help='Align video descriptions for temporal continuity')

    # Audio description alignment parser
    subparsers.add_parser('audio-description-alignment', parents=[common_parser, llm_parser],
                          help='Align audio descriptions for temporal and spatial continuity')

    # Audio descriptions parser
    audio_desc_parser = subparsers.add_parser('audio-descriptions', parents=[common_parser],
                                              help='Generate audio descriptions using Audio Flamingo 3')
    audio_desc_parser.add_argument('--model-path', type=str, default=AUDIO_FLAMINGO_MODEL_PATH, help='Path to Audio Flamingo 3 model')
    audio_desc_parser.add_argument('--batch-size', type=int, default=8, help='Batch size for processing (default: 8)')
    audio_desc_parser.add_argument('--num-gpus', type=int, help='Number of GPUs to use (default: auto-detect)')

    # Multimodal understanding parser
    subparsers.add_parser('multimodal-understanding', parents=[common_parser, llm_parser],
                          help='Generate comprehensive multimodal understanding')

    # Multimodal understanding alignment parser
    subparsers.add_parser('multimodal-understanding-alignment', parents=[common_parser, llm_parser],
                          help='Align multimodal understanding for temporal continuity')

    # Metadata generation parser
    subparsers.add_parser('metadata', parents=[common_parser, llm_parser],
                          help='Generate enhanced metadata')

    # Final consolidation parser
    subparsers.add_parser('consolidate', parents=[common_parser],
                          help='Create training-ready JSONL dataset files')

    # Dataset summary parser
    summary_parser = subparsers.add_parser('summary', help='Generate comprehensive dataset summary and statistics')
    summary_parser.add_argument('--output-file', type=str, help='Custom output file path for summary')

    # VLM stages parser (main pipeline + video descriptions + audio descriptions)
    vlm_stages_parser = subparsers.add_parser('vlm-stages', parents=[common_parser, vlm_parser, audio_parser],
                                              help='Run VLM stages: main pipeline + video descriptions + audio descriptions')
    vlm_stages_parser.add_argument('--max-workers', type=int, default=8, help='Concurrent workers for video descriptions')

    # LLM stages parser (alignment + multimodal + metadata + consolidation)
    subparsers.add_parser('llm-stages', parents=[common_parser, llm_parser],
                          help='Run LLM stages: alignment + multimodal + metadata + consolidation')

    args = parser.parse_args()

    if args.command == 'video-descriptions':
        run_video_descriptions(args)
    elif args.command == 'video-description-alignment':
        run_video_description_alignment(args)
    elif args.command == 'audio-description-alignment':
        run_audio_description_alignment(args)
    elif args.command == 'audio-descriptions':
        run_audio_descriptions(args)
    elif args.command == 'multimodal-understanding':
        run_multimodal_understanding(args)
    elif args.command == 'multimodal-understanding-alignment':
        run_multimodal_understanding_alignment(args)
    elif args.command == 'metadata':
        run_metadata_generation(args)
    elif args.command == 'consolidate':
        run_final_consolidation(args)
    elif args.command == 'summary':
        generate_dataset_summary(args)
    elif args.command == 'vlm-stages':
        run_vlm_stages(args)
    elif args.command == 'llm-stages':
        run_llm_stages(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
