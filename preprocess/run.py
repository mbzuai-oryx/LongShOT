#!/usr/bin/env python
"""
Main script to run the complete pipeline for the Arabic video dataset.
"""

import os
import argparse
import logging
import torch.multiprocessing as mp
import time
from datetime import timedelta

from config import AUDIO_FLAMINGO_MODEL_PATH

# Set multiprocessing start method to 'spawn' before importing other modules
# This is critical for CUDA support in multiprocessing
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

from caption_pipeline.pipeline.orchestrator import PipelineOrchestrator
from caption_pipeline.interface.app import run_app, create_app
from caption_pipeline.interface.routes.auth import create_admin_user

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
logging.getLogger('caption_pipeline.models.movie_caption_enhancer').setLevel(logging.WARNING)
logging.getLogger('caption_pipeline.pipeline.caption_generator').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Dictionary to store execution times
execution_times = {}

# Initialize rich console
rich_console = get_console()




def run_video_descriptions(args):
    """Run video description generation with parallel or sequential processing."""
    start_time = time.time()
    
    from run_video_descriptions import run_video_descriptions as run_desc
    
    video_ids = args.video_ids if hasattr(args, 'video_ids') else None
    max_videos = args.max_videos if hasattr(args, 'max_videos') else None
    max_workers = args.max_workers if hasattr(args, 'max_workers') else 8
    
    # Consolidated startup message
    scope_msg = ""
    if video_ids:
        scope_msg = f"specific video IDs ({len(video_ids)} videos)"
    elif max_videos:
        scope_msg = f"up to {max_videos} videos with completed captions"
    else:
        scope_msg = "all videos with completed captions"
    
    processing_mode = f"concurrent (max_workers={max_workers})"
    rich_console.print_info(f"Starting video description generation for {scope_msg} ({processing_mode})...")
    
    # Disable auto_metadata to prevent duplicate metadata generation since we call it explicitly later
    run_desc(video_ids=video_ids, max_videos=max_videos, max_workers=max_workers, auto_metadata=False)
    
    execution_time = time.time() - start_time
    execution_times['video_descriptions'] = execution_time


def run_audio_descriptions(args):
    """Run audio description generation using Audio Flamingo 3."""  
    start_time = time.time()
    
    from run_audio_descriptions import run_audio_descriptions as run_audio
    
    video_ids = args.video_ids if hasattr(args, 'video_ids') else None
    max_videos = args.max_videos if hasattr(args, 'max_videos') else None
    
    # Consolidated startup message
    scope_msg = ""
    if video_ids:
        scope_msg = f"specific video IDs ({len(video_ids)} videos)"
    elif max_videos:
        scope_msg = f"up to {max_videos} videos with video descriptions"
    else:
        scope_msg = "all videos with video descriptions"
    
    rich_console.print_info(f"Starting audio description generation for {scope_msg}...")
    
    # Get model path and processing args
    model_path = getattr(args, 'audio_model_path', AUDIO_FLAMINGO_MODEL_PATH)
    batch_size = getattr(args, 'audio_batch_size', 8)
    num_gpus = getattr(args, 'audio_gpus', None)
    
    run_audio(video_ids=video_ids, max_videos=max_videos, 
              model_path=model_path, batch_size=batch_size, num_gpus=num_gpus)
    
    execution_time = time.time() - start_time
    execution_times['audio_descriptions'] = execution_time


def run_parallel_video_and_audio_descriptions(args):
    """
    Run video descriptions and audio descriptions in sequence (parallel processing disabled due to GPU OOM issues).
    
    This function provides sequential processing to avoid GPU memory conflicts.
    """
    start_time = time.time()
    
    rich_console.print_warning("⚠️ Parallel processing temporarily disabled due to GPU memory issues")
    rich_console.print_info("� Running video and audio descriptions sequentially")
    
    # Run video descriptions first
    rich_console.print_info("1️⃣ Starting video description generation...")
    run_video_descriptions(args)
    
    # Run audio descriptions after video descriptions complete
    if getattr(args, 'audio_descriptions', False):
        rich_console.print_info("2️⃣ Starting audio description generation...")
        run_audio_descriptions(args)
    
    execution_time = time.time() - start_time
    execution_times['sequential_video_audio_descriptions'] = execution_time
    
    rich_console.print_success(f"✓ Sequential processing completed in {execution_time/60:.1f} minutes")

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
    context_segments = getattr(args, 'context_segments', 3)
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
    
    rich_console.print_info(f"Starting audio description alignment for {scope_msg} (max_workers={max_workers}, context_segments={context_segments}, model={model_to_use})...")
    
    # Get appropriate server URL
    api_base = LLM_SERVER_URL 
    
    # Run alignment
    success = run_alignment(video_ids=video_ids, max_videos=max_videos, 
                           model_name=model_to_use, api_base=api_base, 
                           max_workers=max_workers, context_segments=context_segments)
    
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
    max_workers = args.workers if hasattr(args, 'workers') else 8
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
    
    # Pass model name and API base to the function
    run_meta(video_ids=video_ids, max_videos=max_videos, 
             use_concurrent=True, max_workers=max_workers, 
             model_name=model_to_use, api_base=api_base)
    
    execution_time = time.time() - start_time
    execution_times['metadata_generation'] = execution_time


def run_final_consolidation(args):
    """Run final consolidation to create training-ready JSONL files."""
    start_time = time.time()
    
    from caption_pipeline.pipeline.final_consolidator import FinalConsolidator
    
    video_ids = args.video_ids if hasattr(args, 'video_ids') else None
    max_videos = args.max_videos if hasattr(args, 'max_videos') else None
    
    # Consolidated startup message
    scope_msg = ""
    if video_ids:
        scope_msg = f"specific video IDs ({len(video_ids)} videos)"
    elif max_videos:
        scope_msg = f"up to {max_videos} processed videos"
    else:
        scope_msg = "all processed videos"
    
    rich_console.print_info(f"Starting final consolidation for {scope_msg}...")
    
    # Initialize consolidator and process videos
    consolidator = FinalConsolidator()
    consolidator.consolidate_all_videos(video_ids=video_ids, max_videos=max_videos)
    
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
    
    rich_console.print_success(f"📊 Dataset summary saved to: {summary_file}")
    
    execution_time = time.time() - start_time
    execution_times['dataset_summary'] = execution_time


def run_web_interface(args):
    """Run the web interface."""
    rich_console.print_info("Starting web interface...")
    start_time = time.time()
    
    app = create_app()
    
    # Create admin user if it doesn't exist
    create_admin_user(app)
    
    # Record setup time before running the app
    setup_time = time.time() - start_time
    execution_times['web_setup'] = setup_time
    rich_console.print_info(f"Web interface setup completed in {timedelta(seconds=setup_time)}")
    
    # Run the flask app
    run_app()


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
        whisper_batch_size=getattr(args, 'batch_size', 16),
        enhanced_captions=getattr(args, 'enhanced', True),
        movie_style=getattr(args, 'movie_style', True),
        visual_context=getattr(args, 'visual_context', True),
        enable_video_descriptions=False  # Handle video descriptions separately
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
    
    # Run audio descriptions if enabled and completed videos exist
    if getattr(args, 'audio_descriptions', False) and completed > 0:
        rich_console.print_info("Starting audio description generation for completed videos...")
        run_audio_descriptions(args)
        
        # Clean up Audio Flamingo resources after completion
        try:
            from caption_pipeline.utils.model_cleanup import model_cleanup_manager
            rich_console.print_info("🧹 Cleaning up Audio Flamingo 3 resources...")
            model_cleanup_manager._force_gpu_cleanup()
            rich_console.print_success("✓ Audio Flamingo 3 cleanup completed")
        except Exception as e:
            rich_console.print_warning(f"Warning during Audio Flamingo cleanup: {e}")
    elif getattr(args, 'audio_descriptions', False) and completed == 0:
        rich_console.print_warning("Audio descriptions requested but no videos completed main pipeline")
    
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


def run_full_pipeline(args):
    """Run the complete pipeline using parallel processing."""
    full_start_time = time.time()
    
    # Print pipeline startup information
    rich_console.print_info("Initializing parallel video processing pipeline...")
    rich_console.print_info(f"Command line arguments: --max-videos={args.max_videos}, --video-descriptions={args.video_descriptions}")
    
    # Use the parallel pipeline orchestrator
    rich_console.print_info("Creating pipeline orchestrator with specified configuration...")
    orchestrator = PipelineOrchestrator(
        max_videos=args.max_videos,
        download_workers=args.download_workers,
        preprocess_workers=args.preprocess_workers,
        caption_workers=args.caption_workers,
        output_format=args.output_format,
        whisper_model_size=args.model_size,
        whisper_compute_type=args.compute_type,
        whisper_batch_size=args.batch_size,
        enhanced_captions=args.enhanced,
        movie_style=args.movie_style,
        visual_context=args.visual_context,
        enable_video_descriptions=False  # Disable in orchestrator, handle separately
    )
    rich_console.print_success("Pipeline orchestrator created successfully")
    
    # Run the parallel pipeline
    rich_console.print_info("Starting main pipeline execution...")
    total_videos, completed, failed = orchestrator.run_pipeline()
    
    # Detect if both video and audio descriptions are enabled for parallel processing
    both_descriptions_enabled = (args.video_descriptions and 
                                getattr(args, 'audio_descriptions', False) and
                                not getattr(args, 'disable_parallel_processing', False) and
                                completed > 0)
    
    if both_descriptions_enabled:
        # PARALLEL MODE: Process video and audio descriptions simultaneously at segment level
        rich_console.print_warning("� Parallel processing mode detected but currently disabled due to GPU memory conflicts")
        rich_console.print_info("🔄 Automatically switching to sequential processing for stability")
        rich_console.print_info(f"💡 To force parallel processing, use --disable-parallel-processing=false (not recommended)")
        
        # Force sequential processing for now
        both_descriptions_enabled = False
        
        # Set up args for parallel processing
        if not hasattr(args, 'max_workers'):
            args.max_workers = 8
        
        # Use parallel cleanup while initializing parallel processing
        def init_parallel_processing():
            return run_parallel_video_and_audio_descriptions(args)
        
        try:
            orchestrator.cleanup_and_prepare_for_next_stage(init_parallel_processing)
            
            # Clean up Audio Flamingo resources after parallel completion
            try:
                from caption_pipeline.utils.model_cleanup import model_cleanup_manager
                rich_console.print_info("🧹 Cleaning up Audio Flamingo 3 resources...")
                model_cleanup_manager._force_gpu_cleanup()
                rich_console.print_success("✓ Audio Flamingo 3 cleanup completed")
            except Exception as e:
                rich_console.print_warning(f"Warning during Audio Flamingo cleanup: {e}")
        except Exception as e:
            rich_console.print_error(f"Parallel processing failed: {e}")
            rich_console.print_info("Falling back to sequential processing...")
            # Fall back to sequential processing without parallel cleanup
            if args.video_descriptions:
                run_video_descriptions(args)
            if getattr(args, 'audio_descriptions', False):
                run_audio_descriptions(args)
    
    else:
        # SEQUENTIAL MODE: Process video and audio descriptions separately
        
        # Run video descriptions if requested
        if args.video_descriptions and completed > 0:
            rich_console.print_info(f"Main pipeline completed. Starting video description generation for {completed} videos...")
            # Set up args for video descriptions with proper defaults
            if not hasattr(args, 'max_workers'):
                args.max_workers = 8
            
            # Use parallel cleanup - clean up main pipeline models while initializing video descriptions
            def init_video_descriptions():
                return run_video_descriptions(args)
            
            orchestrator.cleanup_and_prepare_for_next_stage(init_video_descriptions)
        elif args.video_descriptions and completed == 0:
            rich_console.print_warning("Video descriptions requested but no videos completed main pipeline")
        
        # Run audio descriptions if enabled (only if not already done in parallel mode)
        if getattr(args, 'audio_descriptions', False) and completed > 0:
            rich_console.print_info("Starting audio description generation for completed videos...")
            run_audio_descriptions(args)
            
            # Clean up Audio Flamingo resources after completion
            try:
                from caption_pipeline.utils.model_cleanup import model_cleanup_manager
                rich_console.print_info("🧹 Cleaning up Audio Flamingo 3 resources...")
                # Force cleanup of any remaining GPU memory 
                model_cleanup_manager._force_gpu_cleanup()
                rich_console.print_success("✓ Audio Flamingo 3 cleanup completed")
            except Exception as e:
                rich_console.print_warning(f"Warning during Audio Flamingo cleanup: {e}")
        elif getattr(args, 'audio_descriptions', False) and completed == 0:
            rich_console.print_warning("Audio descriptions requested but no videos completed main pipeline")
    
    # Run comprehensive multimodal understanding for all completed videos with descriptions
    if completed > 0 and args.video_descriptions:
        rich_console.print_info(f"Starting comprehensive multimodal understanding for {completed} completed videos...")
        run_multimodal_understanding(args)
        
        # Run multimodal understanding alignment for improved temporal continuity
        rich_console.print_info(f"Starting multimodal understanding alignment for {completed} completed videos...")
        run_multimodal_understanding_alignment(args)
    elif completed > 0 and not args.video_descriptions:
        rich_console.print_warning("Skipping multimodal understanding - video descriptions not enabled")
    else:
        rich_console.print_warning("No videos completed - skipping multimodal understanding")
    
    # Run metadata generation for all completed videos
    if completed > 0:
        rich_console.print_info(f"Starting metadata generation for {completed} completed videos...")
        run_metadata_generation(args)
    else:
        rich_console.print_warning("No videos completed - skipping metadata generation")
    
    # Run final consolidation for all completed videos (mandatory step)
    if completed > 0:
        rich_console.print_info(f"Starting final consolidation for {completed} completed videos...")
        run_final_consolidation(args)
    else:
        rich_console.print_warning("No videos completed - skipping final consolidation")
    
    # Generate comprehensive dataset summary (final step)
    if completed > 0:
        rich_console.print_info(f"Generating comprehensive dataset summary for {completed} completed videos...")
        generate_dataset_summary(args)
    else:
        rich_console.print_warning("No videos completed - skipping dataset summary generation")
    
    # Run web interface if requested
    if args.web:
        run_web_interface(args)
    
    # Calculate full pipeline time
    full_execution_time = time.time() - full_start_time
    execution_times['full_pipeline'] = full_execution_time
    
    # Print pipeline completion summary
    rich_console.print_success(f"Full pipeline execution completed in {timedelta(seconds=full_execution_time)}")
    rich_console.print_info(f"Pipeline results: {completed} completed, {failed} failed out of {total_videos} total videos")
    
    # Print execution summary
    print_execution_summary()


def main():
    """Parse arguments and run the appropriate pipeline component."""
    parser = argparse.ArgumentParser(description='Arabic Video Dataset Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Pipeline component to run')
    
    
    # Web interface parser
    subparsers.add_parser('web', help='Run web interface')
    
    # Video descriptions parser
    video_desc_parser = subparsers.add_parser('video-descriptions', help='Generate video descriptions')
    video_desc_parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to process')
    video_desc_parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    video_desc_parser.add_argument('--model', type=str, help='Vision model to use')
    video_desc_parser.add_argument('--api-base', type=str, help='vLLM server API base URL')
    video_desc_parser.add_argument('--max-workers', type=int, default=8, help='Maximum number of concurrent workers (default: 8)')
    video_desc_parser.add_argument('--offline', action='store_true', help='Use offline vLLM model')
    video_desc_parser.add_argument('--max-frames', type=int, help='Maximum frames per segment')
    
    # Video description alignment parser
    alignment_parser = subparsers.add_parser('video-description-alignment', help='Align video descriptions for temporal continuity')
    alignment_parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to align')
    alignment_parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    alignment_parser.add_argument('--model', type=str, help='Language model to use for alignment')
    alignment_parser.add_argument('--api-base', type=str, help='vLLM server API base URL')
    alignment_parser.add_argument('--max-workers', type=int, default=8, help='Maximum number of concurrent workers (default: 8)')
    
    # Audio description alignment parser
    audio_alignment_parser = subparsers.add_parser('audio-description-alignment', help='Align audio descriptions for temporal and spatial continuity')
    audio_alignment_parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to align')
    audio_alignment_parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    audio_alignment_parser.add_argument('--model', type=str, help='Language model to use for alignment')
    audio_alignment_parser.add_argument('--api-base', type=str, help='vLLM server API base URL')
    audio_alignment_parser.add_argument('--max-workers', type=int, default=8, help='Maximum number of concurrent workers (default: 8)')
    audio_alignment_parser.add_argument('--context-segments', type=int, default=3, help='Number of previous segments to use as context (default: 3)')
    audio_alignment_parser.add_argument('--list-only', action='store_true', help='Only list videos that need alignment processing')
    
    # Audio descriptions parser
    audio_desc_parser = subparsers.add_parser('audio-descriptions', help='Generate audio descriptions using Audio Flamingo 3')
    audio_desc_parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to process')
    audio_desc_parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    audio_desc_parser.add_argument('--model-path', type=str, default=AUDIO_FLAMINGO_MODEL_PATH, help='Path to Audio Flamingo 3 model')
    audio_desc_parser.add_argument('--batch-size', type=int, default=8, help='Batch size for processing (default: 8)')
    audio_desc_parser.add_argument('--num-gpus', type=int, help='Number of GPUs to use (default: auto-detect)')
    audio_desc_parser.add_argument('--text-prompt', type=str, help='Custom text prompt for audio description')
    audio_desc_parser.add_argument('--list-only', action='store_true', help='Only list videos that need processing')
    
    # Multimodal understanding parser
    multimodal_parser = subparsers.add_parser('multimodal-understanding', help='Generate comprehensive multimodal understanding')
    multimodal_parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to process')
    multimodal_parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    multimodal_parser.add_argument('--max-workers', type=int, default=6, help='Maximum number of concurrent workers (default: 8)')
    multimodal_parser.add_argument('--model', type=str, help='Language model to use')
    multimodal_parser.add_argument('--api-base', type=str, help='vLLM server API base URL')
    multimodal_parser.add_argument('--sequential', action='store_true', help='Process videos sequentially instead of concurrently')
    multimodal_parser.add_argument('--list-only', action='store_true', help='Only list videos that need processing')
    
    # Multimodal understanding alignment parser
    multimodal_alignment_parser = subparsers.add_parser('multimodal-understanding-alignment', help='Align multimodal understanding for temporal continuity')
    multimodal_alignment_parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to align')
    multimodal_alignment_parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    multimodal_alignment_parser.add_argument('--model', type=str, help='Language model to use for alignment')
    multimodal_alignment_parser.add_argument('--api-base', type=str, help='vLLM server API base URL')
    multimodal_alignment_parser.add_argument('--max-workers', type=int, default=8, help='Maximum number of concurrent workers (default: 8)')
    multimodal_alignment_parser.add_argument('--list-only', action='store_true', help='Only list videos that need alignment processing')
    
    # Metadata generation parser
    metadata_parser = subparsers.add_parser('metadata', help='Generate enhanced metadata')
    metadata_parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to process')
    metadata_parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    metadata_parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    metadata_parser.add_argument('--model', type=str, help='Language model to use')
    metadata_parser.add_argument('--api-base', type=str, help='vLLM server API base URL')
    
    # Final consolidation parser
    consolidation_parser = subparsers.add_parser('consolidate', help='Create training-ready JSONL dataset files')
    consolidation_parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to consolidate')
    consolidation_parser.add_argument('--max-videos', type=int, help='Maximum number of videos to consolidate')
    
    # Dataset summary parser
    summary_parser = subparsers.add_parser('summary', help='Generate comprehensive dataset summary and statistics')
    summary_parser.add_argument('--output-file', type=str, help='Custom output file path for summary (default: dataset/dataset_summary.json)')
    
    # VLM stages parser (first set: main pipeline + video descriptions + audio descriptions)
    vlm_stages_parser = subparsers.add_parser('vlm-stages', help='Run first set of pipeline stages using VLM model (main pipeline + video descriptions + audio descriptions)')
    vlm_stages_parser.add_argument('--max-videos', type=int, default=None, help='Maximum number of videos to process')
    vlm_stages_parser.add_argument('--download-workers', type=int, default=1, help='Number of parallel download workers')
    vlm_stages_parser.add_argument('--preprocess-workers', type=int, default=4, help='Number of parallel preprocessing workers')
    vlm_stages_parser.add_argument('--caption-workers', type=int, default=2, help='Number of parallel captioning workers')
    vlm_stages_parser.add_argument('--output-format', type=str, default='wav', choices=['wav', 'mp3'], help='Audio output format for preprocessor')
    vlm_stages_parser.add_argument('--model-size', type=str, default='large-v3', choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'distil-large-v3'], help='Whisper model size')
    vlm_stages_parser.add_argument('--compute-type', type=str, default='float16', choices=['float32', 'float16', 'int8_float16', 'int8'], help='Compute type for Whisper model')
    vlm_stages_parser.add_argument('--language', type=str, default='en', help='Language code (default: en)')
    vlm_stages_parser.add_argument('--batch-size', type=int, default=16, help='Batch size for parallel processing (default: 16)')
    vlm_stages_parser.add_argument('--enhanced', action='store_true', help='Use enhanced caption generation', default=True)
    vlm_stages_parser.add_argument('--movie-style', action='store_true', default=True, help='Add movie-style captions like [music], [effects], etc.')
    vlm_stages_parser.add_argument('--visual-context', action='store_true', default=True, help='Add visual scene descriptions using CLIP')
    vlm_stages_parser.add_argument('--audio-descriptions', action='store_true', default=True, help='Generate audio descriptions for non-speech content using Audio Flamingo 3')
    vlm_stages_parser.add_argument('--audio-model-path', type=str, default=AUDIO_FLAMINGO_MODEL_PATH, help='Path to Audio Flamingo 3 model')
    vlm_stages_parser.add_argument('--audio-batch-size', type=int, default=8, help='Batch size for audio description processing')
    vlm_stages_parser.add_argument('--audio-gpus', type=int, help='Number of GPUs for audio processing')
    vlm_stages_parser.add_argument('--max-workers', type=int, default=8, help='Maximum number of concurrent workers for video descriptions')
    
    # LLM stages parser (second set: alignment + multimodal + metadata + consolidation)
    llm_stages_parser = subparsers.add_parser('llm-stages', help='Run second set of pipeline stages using LLM model (alignment + multimodal + metadata + consolidation)')
    llm_stages_parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to process')
    llm_stages_parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    llm_stages_parser.add_argument('--max-workers', type=int, default=8, help='Maximum number of concurrent workers')
    llm_stages_parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers for metadata generation')
    
    # Full pipeline parser with Faster Whisper options
    full_parser = subparsers.add_parser('full', help='Run full pipeline')
    full_parser.add_argument('--max-videos', type=int, default=None, help='Maximum number of videos to process')
    full_parser.add_argument('--web', action='store_true', help='Start web interface after processing')
    full_parser.add_argument('--download-workers', type=int, default=1, help='Number of parallel download workers')
    full_parser.add_argument('--preprocess-workers', type=int, default=4, help='Number of parallel preprocessing workers')
    full_parser.add_argument('--caption-workers', type=int, default=2, help='Number of parallel captioning workers')
    full_parser.add_argument('--output-format', type=str, default='wav', choices=['wav', 'mp3'], 
                            help='Audio output format for preprocessor')
    full_parser.add_argument('--model-size', type=str, default='large-v3', 
                           choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'distil-large-v3'],
                           help='Whisper model size')
    full_parser.add_argument('--compute-type', type=str, default='float16',
                           choices=['float32', 'float16', 'int8_float16', 'int8'],
                           help='Compute type for Whisper model')
    full_parser.add_argument('--language', type=str, default='en', help='Language code (default: ar for Arabic)')
    full_parser.add_argument('--batch-size', type=int, default=16, 
                           help='Batch size for parallel processing (default: 16)')
    full_parser.add_argument('--enhanced', action='store_true', help='Use enhanced caption generation', default=True)
    full_parser.add_argument('--movie-style', action='store_true', default=True,
                           help='Add movie-style captions like [music], [effects], etc.')
    full_parser.add_argument('--visual-context', action='store_true', default=True,
                           help='Add visual scene descriptions using CLIP')
    full_parser.add_argument('--video-descriptions', action='store_true', default=True,
                           help='Generate visual descriptions for video segments using Qwen 2.5-VL')
    full_parser.add_argument('--audio-descriptions', action='store_true', default=True,
                           help='Generate audio descriptions for non-speech content using Audio Flamingo 3')
    full_parser.add_argument('--audio-model-path', type=str, default=AUDIO_FLAMINGO_MODEL_PATH,
                           help='Path to Audio Flamingo 3 model')
    full_parser.add_argument('--audio-batch-size', type=int, default=8,
                           help='Batch size for audio description processing')
    full_parser.add_argument('--audio-gpus', type=int, help='Number of GPUs for audio processing')
    full_parser.add_argument('--workers', type=int, default=4, 
                           help='Number of parallel workers for metadata generation')
    full_parser.add_argument('--disable-parallel-processing', action='store_true', default=True,
                           help='Disable parallel processing of video and audio descriptions')
    full_parser.add_argument('--max-workers', type=int, default=8,
                           help='Maximum number of concurrent workers for video descriptions')
    
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
    elif args.command == 'web':
        run_web_interface(args)
    elif args.command == 'full':
        run_full_pipeline(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
