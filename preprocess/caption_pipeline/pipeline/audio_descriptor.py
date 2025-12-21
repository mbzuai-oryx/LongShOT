"""
Audio Description Processor for Caption Pipeline

This module generates audio descriptions for non-speech segments using Audio Flamingo 3,
integrating with the existing caption pipeline between video descriptions and metadata generation.
"""

import os
import sys
import json
import logging
import time
import tempfile
import shutil
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import torch
import queue
import threading

# Add audio_flamingo3 to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'audio_flamingo3'))

# Import project configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import VIDEO_DIR, VIDEO_DESCRIPTIONS_DIR, METADATA_DIR

# Import pipeline utilities
from caption_pipeline.utils.audio_segment_extractor import AudioSegmentExtractor
from caption_pipeline.utils.rich_console import get_console

# Import Audio Flamingo 3
from audio_flamingo3.inference import SimpleInferenceSystem

# Import common VLM utilities
from caption_pipeline.utils.vlm_common import (
    load_json_file, save_json_file, safe_remove_file,
    calculate_progress_interval, ThrottledErrorLogger
)

# Set up logging
logger = logging.getLogger(__name__)
rich_console = get_console()

# Constants
DEFAULT_MODEL_PATH = "../../MODELS/audio-flamingo-3"

# Magic numbers extracted as constants
MEMORY_PER_SEGMENT_GB = 2.5  # GPU memory per segment (conservative estimate)
GPU_UTILIZATION_RATIO = 0.6  # Target GPU memory utilization (60%)
MAX_SAFE_GPUS = 4  # Maximum GPUs to use to prevent OOM
EARLY_EXTRACTION_PHASE_RATIO = 0.3  # First 30% of segments use smaller batches
ERROR_LOG_SUPPRESSION_THRESHOLD = 3  # Number of errors to log before suppression
MAX_EXTRACTION_WORKERS = 8  # Maximum concurrent audio extraction threads
DEFAULT_TEXT_PROMPT = """You are an expert audio analyst providing domain-agnostic audio environment descriptions for video understanding benchmarks. Analyze this audio segment focusing on non-speech elements using precise, factual language.

**ANALYSIS FRAMEWORK:**

**Environmental Audio:**
- Acoustic space characteristics (indoor/outdoor, size, reverb properties)
- Background atmosphere and ambient sounds
- Recording perspective and audio quality

**Sound Elements:**
- Music: Identify instruments, tempo, style, and arrangement patterns
- Sound effects: Identify specific sound sources and their characteristics
- Environmental noise: Mechanical sounds, nature sounds, urban sounds
- Audio transitions: Changes in volume, frequency, or sound composition

**Production Characteristics:**
- Audio clarity, dynamic range, and technical quality
- Spatial positioning and stereo/mono characteristics
- Any processing effects or post-production elements

**DOMAIN-AGNOSTIC LANGUAGE:**
- Avoid genre-specific terminology or assumptions about content type
- Use neutral, technical audio descriptions applicable to any video domain
- Focus on observable sonic characteristics rather than contextual interpretation
- Ensure language works universally (sports, news, documentaries, tutorials, entertainment)

**LANGUAGE VARIATION:**
Vary descriptive patterns and avoid repetitive phrasing. Use alternatives like:
- "Audio features...", "The soundscape includes...", "Sonic environment contains...", "Audio characteristics show..."
- "Background presents...", "Ambiance reveals...", "Sound elements demonstrate..."

**CRITICAL OUTPUT REQUIREMENTS:**
Provide exactly 80-150 words describing ONLY the audio environment (excluding speech/dialogue).
Focus on factual sonic observations using consistent but varied language patterns.
If audio is minimal or silent, describe this briefly and factually.

Audio Environment Analysis:"""
DEFAULT_BATCH_SIZE = 8
DEFAULT_TEMP_DIR = tempfile.gettempdir()

class AudioDescriptor:
    """
    Audio description processor that integrates Audio Flamingo 3 with the caption pipeline.
    
    This class processes video segments to generate descriptions of non-speech audio content,
    complementing the existing speech transcription and visual description components.
    """
    
    def __init__(self, 
                 model_path: str = DEFAULT_MODEL_PATH,
                 text_prompt: str = DEFAULT_TEXT_PROMPT,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 temp_dir: Optional[str] = None,
                 sample_rate: int = 16000,
                 num_gpus: Optional[int] = None,
                 ):
        """
        Initialize the Audio Descriptor.
        
        Args:
            model_path: Path to Audio Flamingo 3 model
            text_prompt: Text prompt for audio description
            batch_size: Number of segments to process in each batch
            temp_dir: Directory for temporary files
            sample_rate: Sample rate for extracted audio
            num_gpus: Number of GPUs to use (None for auto-detect)
        """
        self.model_path = model_path
        self.text_prompt = text_prompt
        self.batch_size = batch_size
        self.temp_dir = temp_dir or os.path.join(DEFAULT_TEMP_DIR, 'audio_descriptions')
        self.sample_rate = sample_rate
        # Use all available GPUs by default if not specified, but limit to prevent OOM
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        # Limit to maximum GPUs to prevent OOM in parallel scenarios
        max_safe = min(available_gpus, MAX_SAFE_GPUS)
        self.num_gpus = num_gpus if num_gpus is not None else max_safe
        
        # Optimize batch size based on GPU memory and count
        self.batch_size = self._optimize_batch_size(batch_size)
        
        # Initialize directories
        self.video_dir = VIDEO_DIR
        self.video_descriptions_dir = VIDEO_DESCRIPTIONS_DIR
        self.audio_descriptions_dir = os.path.join(os.path.dirname(VIDEO_DESCRIPTIONS_DIR), 'audio_descriptions')
        self.metadata_dir = METADATA_DIR
        
        # Create directories
        os.makedirs(self.audio_descriptions_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize components
        self.audio_extractor = AudioSegmentExtractor(temp_dir=self.temp_dir, sample_rate=sample_rate)
        self.inference_system = None  # Lazy initialization
        
        # Load metadata
        self.metadata_file = os.path.join(self.metadata_dir, 'video_metadata.csv')
        self._load_metadata()
        
        # Consolidated initialization message
        rich_console.print_info(f"AudioDescriptor initialized: {self.num_gpus} GPUs, batch_size={self.batch_size}, sample_rate={sample_rate}Hz")
    
    def _load_metadata(self):
        """Load video metadata for finding video files."""
        if os.path.exists(self.metadata_file):
            try:
                self.metadata_df = pd.read_csv(self.metadata_file)
                # Only log if there are issues, reduce verbosity
            except Exception as e:
                rich_console.print_error(f"Error loading metadata: {e}")
                self.metadata_df = pd.DataFrame()
        else:
            # Use default paths silently
            self.metadata_df = pd.DataFrame()
    
    def _optimize_batch_size(self, requested_batch_size: int) -> int:
        """Optimize batch size based on GPU count and memory."""
        if not torch.cuda.is_available():
            return requested_batch_size
        
        try:
            # Get GPU memory for first GPU (assuming similar across GPUs)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Calculate optimal batch size based on GPU memory and count
            # Audio Flamingo 3 roughly needs memory per segment, using conservative estimate
            max_segments_per_gpu = max(1, int((gpu_memory_gb * GPU_UTILIZATION_RATIO) / MEMORY_PER_SEGMENT_GB))
            
            # Calculate total batch size across all GPUs
            memory_based_batch_size = max_segments_per_gpu * self.num_gpus
            
            # Use performance-optimized batch size: prefer multiples of GPU count for even distribution
            optimal_batch_size = max(self.num_gpus, min(requested_batch_size, memory_based_batch_size))
            
            # Round to nearest multiple of GPU count for optimal distribution
            optimal_batch_size = ((optimal_batch_size + self.num_gpus - 1) // self.num_gpus) * self.num_gpus
            
            # Apply heuristics based on GPU count - be more conservative
            if self.num_gpus >= 4:
                # For 4+ GPUs, use smaller batches to avoid OOM in parallel scenarios
                optimal_batch_size = max(optimal_batch_size, self.num_gpus * 2)  # Reduced from 3 to 2 per GPU
            
            # Cap at more conservative maximum to avoid OOM
            max_reasonable_batch = self.num_gpus * 4  # Reduced from 6 to 4 per GPU
            optimal_batch_size = min(optimal_batch_size, max_reasonable_batch)
            
            if optimal_batch_size != requested_batch_size:
                rich_console.print_info(f"Optimized batch size: {requested_batch_size} -> {optimal_batch_size} (GPU memory: {gpu_memory_gb:.1f}GB)")
            
            return optimal_batch_size
            
        except Exception as e:
            rich_console.print_warning(f"Batch size optimization failed: {e}, using requested size: {requested_batch_size}")
            return requested_batch_size
    
    def _get_video_path(self, video_id: str) -> Optional[str]:
        """Get the video file path for a video ID."""
        # Check metadata first
        if not self.metadata_df.empty:
            video_data = self.metadata_df[self.metadata_df['video_id'] == video_id]
            if len(video_data) > 0 and 'file_path' in video_data.columns:
                video_path = video_data.iloc[0]['file_path']
                if pd.notna(video_path) and os.path.exists(video_path):
                    return video_path
        
        # Check default location
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        if os.path.exists(video_path):
            return video_path
        
        return None
    
    def _get_video_descriptions_path(self, video_id: str) -> Optional[str]:
        """Get the video descriptions file path for a video ID."""
        descriptions_path = os.path.join(self.video_descriptions_dir, f"{video_id}_descriptions.json")
        return descriptions_path if os.path.exists(descriptions_path) else None
    
    def _init_inference_system(self):
        """Initialize the Audio Flamingo 3 inference system with persistent model."""
        if self.inference_system is None:
            try:
                # Initialize inference system
                self.inference_system = SimpleInferenceSystem(
                    model_path=self.model_path,
                    num_gpus=self.num_gpus,  # Use all available GPUs
                    persistent_model=True  # Enable model persistence
                )
                rich_console.print_success("Audio Flamingo 3 inference system ready")
            except Exception as e:
                rich_console.print_error(f"Failed to initialize Audio Flamingo 3: {e}")
                raise
    
    def _load_video_descriptions(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load video descriptions data for a video."""
        descriptions_path = self._get_video_descriptions_path(video_id)
        if not descriptions_path:
            rich_console.print_warning(f"Video descriptions not found for {video_id}")
            return None
        
        try:
            with open(descriptions_path, 'r', encoding='utf-8') as f:
                descriptions_data = json.load(f)
            
            segments = descriptions_data.get('segments', [])
            if not segments:
                rich_console.print_warning(f"No segments found in video descriptions for {video_id}")
                return None
            
            # Successfully loaded segments - details will be shown in filtering step
            return descriptions_data
            
        except Exception as e:
            rich_console.print_error(f"Error loading video descriptions for {video_id}: {e}")
            return None
    
    def _filter_segments_for_audio_description(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Return all segments for audio description generation.
        
        Process all segments regardless of speech content to capture background audio,
        music, sound effects, and ambient sounds in addition to any speech content.
        """
        rich_console.print_info(f"Processing all {len(segments)} segments for audio description")
        return segments
    
    def _save_audio_descriptions(self, video_id: str, video_path: str, 
                                audio_descriptions: List[Dict[str, Any]]) -> str:
        """Save audio descriptions to JSON file."""
        output_file = os.path.join(self.audio_descriptions_dir, f"{video_id}_audio_descriptions.json")
        
        # Prepare output data
        output_data = {
            'video_id': video_id,
            'video_path': video_path,
            'model_used': 'Audio-Flamingo-3',
            'model_path': self.model_path,
            'text_prompt': self.text_prompt,
            'processing_method': 'audio_flamingo_3_pipeline_integration',
            'sample_rate': self.sample_rate,
            'batch_size': self.batch_size,
            'total_segments': len(audio_descriptions),
            'successful_segments': len([d for d in audio_descriptions if not d.get('error')]),
            'failed_segments': len([d for d in audio_descriptions if d.get('error')]),
            'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'segments': sorted(audio_descriptions, key=lambda x: x.get('start', 0))
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            rich_console.print_success(f"Saved audio descriptions: {output_file}")
            return output_file
            
        except Exception as e:
            rich_console.print_error(f"Failed to save audio descriptions: {e}")
            raise
    
    def generate_audio_descriptions_pipeline(self, video_id: str) -> Optional[str]:
        """
        Generate audio descriptions for a single video using pipeline processing.
        
        This method implements parallel processing where:
        1. Model loading starts immediately in background
        2. Audio extraction happens in parallel using multiple threads
        3. Segments are sent to GPU inference as soon as they're extracted
        4. Maximum GPU utilization with minimal idle time
        
        Args:
            video_id: ID of the video to process
            
        Returns:
            Path to the generated audio descriptions file, or None if failed
        """
        rich_console.print_component_header("Audio Description Pipeline", f"Processing {video_id}")
        start_time = time.time()
        
        # Check if audio descriptions already exist
        output_file = os.path.join(self.audio_descriptions_dir, f"{video_id}_audio_descriptions.json")
        if os.path.exists(output_file):
            rich_console.print_info(f"Audio descriptions already exist for {video_id}")
            return output_file
        
        try:
            # Get video path
            video_path = self._get_video_path(video_id)
            if not video_path:
                rich_console.print_error(f"Video file not found for {video_id}")
                return None
            
            # Load video descriptions
            descriptions_data = self._load_video_descriptions(video_id)
            if not descriptions_data:
                return None
            
            segments = descriptions_data.get('segments', [])
            
            # Filter segments for audio description
            audio_segments = self._filter_segments_for_audio_description(segments)
            if not audio_segments:
                rich_console.print_warning(f"No segments require audio description for {video_id}")
                # Create empty audio descriptions file
                empty_descriptions = []
                return self._save_audio_descriptions(video_id, video_path, empty_descriptions)
            
            # Start model loading in background thread
            model_loading_event = threading.Event()
            model_loading_thread = threading.Thread(
                target=self._background_model_loading,
                args=(model_loading_event,)
            )
            model_loading_thread.start()
            
            # Create queues for pipeline processing
            extraction_queue = queue.Queue()
            results_queue = queue.Queue()
            
            # Start audio extraction in parallel
            rich_console.print_info(f"Starting pipeline processing for {len(audio_segments)} segments")
            
            extraction_thread = threading.Thread(
                target=self._parallel_audio_extraction,
                args=(video_path, audio_segments, video_id, extraction_queue)
            )
            extraction_thread.start()
            
            # Start inference processing
            inference_thread = threading.Thread(
                target=self._pipeline_inference_processing,
                args=(extraction_queue, results_queue, model_loading_event, len(audio_segments))
            )
            inference_thread.start()
            
            # Collect results
            audio_descriptions = []
            processed_count = 0
            last_progress_update = 0
            
            while processed_count < len(audio_segments):
                try:
                    result = results_queue.get(timeout=600)  # 10 minute timeout per segment
                    audio_descriptions.append(result)
                    processed_count += 1
                    
                    # Show progress every 10% or for final segment
                    progress_interval = calculate_progress_interval(len(audio_segments))
                    if (processed_count - last_progress_update >= progress_interval) or processed_count == len(audio_segments):
                        if result.get('error'):
                            rich_console.print_warning(f"Progress: {processed_count}/{len(audio_segments)} segments (some failed)")
                        else:
                            rich_console.print_info(f"Progress: {processed_count}/{len(audio_segments)} segments processed")
                        last_progress_update = processed_count
                    
                except queue.Empty:
                    rich_console.print_error(f"Timeout waiting for segment results")
                    break
            
            # Wait for threads to complete
            extraction_thread.join(timeout=60)
            inference_thread.join(timeout=60)
            model_loading_thread.join(timeout=60)
            
            # Save results
            output_file = self._save_audio_descriptions(video_id, video_path, audio_descriptions)
            
            # Performance stats
            processing_time = time.time() - start_time
            successful_count = len([d for d in audio_descriptions if not d.get('error')])
            
            rich_console.print_success(f"Pipeline audio descriptions completed for {video_id} in {processing_time:.2f}s")
            rich_console.print_info(f"  - Processed: {successful_count}/{len(audio_descriptions)} segments")
            
            return output_file
            
        except Exception as e:
            processing_time = time.time() - start_time
            rich_console.print_error(f"Pipeline audio description failed for {video_id} after {processing_time:.2f}s: {e}")
            return None
        
        finally:
            # Cleanup will be handled by individual threads
            pass

    def generate_audio_descriptions(self, video_id: str) -> Optional[str]:
        """
        Generate audio descriptions for a single video using pipeline processing.
        
        This method automatically uses the optimized pipeline approach for better performance.
        
        Args:
            video_id: ID of the video to process
            
        Returns:
            Path to the generated audio descriptions file, or None if failed
        """
        # Use the optimized pipeline version
        return self.generate_audio_descriptions_pipeline(video_id)
    
    def generate_audio_descriptions_persistent(self, video_id: str) -> Optional[str]:
        """
        Generate audio descriptions for a single video using persistent workers (no shutdown).
        
        This method is optimized for batch processing where workers should remain active
        between videos to avoid model reloading overhead.
        
        Args:
            video_id: ID of the video to process
            
        Returns:
            Path to the generated audio descriptions file, or None if failed
        """
        rich_console.print_component_header("Audio Description Pipeline", f"Processing {video_id}")
        start_time = time.time()
        
        # Check if audio descriptions already exist
        output_file = os.path.join(self.audio_descriptions_dir, f"{video_id}_audio_descriptions.json")
        if os.path.exists(output_file):
            rich_console.print_info(f"Audio descriptions already exist for {video_id}")
            return output_file
        
        try:
            # Get video path
            video_path = self._get_video_path(video_id)
            if not video_path:
                rich_console.print_error(f"Video file not found for {video_id}")
                return None
            
            # Load video descriptions
            descriptions_data = self._load_video_descriptions(video_id)
            if not descriptions_data:
                return None
            
            segments = descriptions_data.get('segments', [])
            
            # Filter segments for audio description
            audio_segments = self._filter_segments_for_audio_description(segments)
            if not audio_segments:
                rich_console.print_warning(f"No segments require audio description for {video_id}")
                # Create empty audio descriptions file
                empty_descriptions = []
                return self._save_audio_descriptions(video_id, video_path, empty_descriptions)
            
            # Process segments using persistent workers (no worker shutdown)
            rich_console.print_info(f"Processing {len(audio_segments)} segments with persistent workers")
            
            # Extract and process audio segments
            all_audio_descriptions = []
            
            # Extract audio segments
            extracted_segments = []
            for segment_idx, segment in enumerate(audio_segments):
                try:
                    segment_output_dir = os.path.join(self.temp_dir, video_id)
                    os.makedirs(segment_output_dir, exist_ok=True)
                    
                    # Create segment data for extractor
                    segment_data = {
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'segment_index': segment_idx
                    }
                    
                    # Extract audio segment
                    extracted_segment = self.audio_extractor.extract_single_segment(
                        video_path, segment_data, segment_output_dir
                    )
                    
                    if extracted_segment:
                        extracted_segment['original_segment'] = segment
                        extracted_segment['segment_index'] = segment_idx
                        extracted_segments.append(extracted_segment)
                    
                except Exception as e:
                    rich_console.print_warning(f"Failed to extract segment {segment_idx}: {e}")
            
            # Process extracted segments with persistent workers
            if extracted_segments:
                successful_segments = [seg for seg in extracted_segments if seg.get('extraction_success')]
                if successful_segments:
                    # Prepare batch for inference
                    batch_audio_paths = [seg['audio_path'] for seg in successful_segments]
                    
                    # Use batch inference with persistent workers (auto_shutdown=False)
                    batch_results = self.inference_system.batch_inference(
                        batch_audio_paths, self.text_prompt, auto_shutdown=False
                    )
                    
                    # Process results
                    for segment, result in zip(successful_segments, batch_results):
                        if result['success']:
                            audio_description = {
                                'segment_index': segment.get('segment_index', 0),
                                'start': segment.get('start', 0),
                                'end': segment.get('end', 0),
                                'duration': segment.get('duration', 0),
                                'audio_description': result['response'],
                                'processing_method': 'audio_flamingo_3_persistent_batch',
                                'inference_time': result.get('inference_time', 0),
                                'gpu_id': result.get('gpu_id', -1),
                                'sample_rate': result.get('sample_rate', self.sample_rate),
                                'original_segment': segment.get('original_segment', {})
                            }
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            audio_description = {
                                'segment_index': segment.get('segment_index', 0),
                                'start': segment.get('start', 0),
                                'end': segment.get('end', 0),
                                'duration': segment.get('duration', 0),
                                'audio_description': f"Inference failed: {error_msg}",
                                'processing_method': 'audio_flamingo_3_error',
                                'error': error_msg,
                                'gpu_id': result.get('gpu_id', -1),
                                'original_segment': segment.get('original_segment', {})
                            }
                        
                        all_audio_descriptions.append(audio_description)
                        
                        # Clean up audio file immediately
                        safe_remove_file(segment.get('audio_path'))
            
            # Save results
            output_file = self._save_audio_descriptions(video_id, video_path, all_audio_descriptions)
            
            # Performance stats
            processing_time = time.time() - start_time
            successful_count = len([d for d in all_audio_descriptions if not d.get('error')])
            
            rich_console.print_success(f"Persistent audio descriptions completed for {video_id} in {processing_time:.2f}s")
            rich_console.print_info(f"  - Processed: {successful_count}/{len(all_audio_descriptions)} segments")
            
            return output_file
            
        except Exception as e:
            processing_time = time.time() - start_time
            rich_console.print_error(f"Persistent audio description failed for {video_id} after {processing_time:.2f}s: {e}")
            return None
    
    def process_videos_batch(self, video_ids: List[str]) -> Dict[str, Optional[str]]:
        """
        Process multiple videos to generate audio descriptions.
        
        Args:
            video_ids: List of video IDs to process
            
        Returns:
            Dictionary mapping video IDs to output file paths (or None if failed)
        """
        rich_console.print_component_header("Batch Audio Description", 
                                          f"Processing {len(video_ids)} videos")
        
        results = {}
        failed_videos = []
        
        # Initialize inference system once for all videos
        try:
            self._init_inference_system()
            self.inference_system.start_workers()
            rich_console.print_info(f"Initialized persistent Audio Flamingo 3 for batch processing of {len(video_ids)} videos")
        except Exception as e:
            rich_console.print_error(f"Failed to initialize inference system: {e}")
            return {video_id: None for video_id in video_ids}
        
        try:
            for i, video_id in enumerate(video_ids, 1):
                rich_console.print_info(f"Processing video {i}/{len(video_ids)}: {video_id}")
                
                try:
                    # Use persistent batch method that doesn't shutdown workers
                    output_file = self.generate_audio_descriptions_persistent(video_id)
                    results[video_id] = output_file
                    
                    if output_file:
                        rich_console.print_success(f"Completed audio descriptions for {video_id} ({i}/{len(video_ids)})")
                    else:
                        failed_videos.append(video_id)
                        rich_console.print_error(f"Failed audio descriptions for {video_id} ({i}/{len(video_ids)})")
                        
                except Exception as e:
                    failed_videos.append(video_id)
                    results[video_id] = None
                    rich_console.print_error(f"Exception processing {video_id}: {e}")
        
        finally:
            # Shutdown workers only once at the end of all videos
            try:
                if self.inference_system and hasattr(self.inference_system, '_workers_started') and self.inference_system._workers_started:
                    self.inference_system.shutdown_workers()
                    rich_console.print_info("GPU workers shut down after batch processing")
            except Exception as e:
                rich_console.print_warning(f"Error shutting down GPU workers: {e}")
        
        # Print summary
        successful_count = len([r for r in results.values() if r is not None])
        rich_console.print_completion_message("Audio Description Generation", {
            'total': len(video_ids),
            'successful': successful_count,
            'duration': 0  # Duration calculated externally
        })
        
        if failed_videos:
            rich_console.print_warning(f"Failed videos: {', '.join(failed_videos[:5])}")
            if len(failed_videos) > 5:
                rich_console.print_warning(f"... and {len(failed_videos) - 5} more")
        
        return results
    
    def _background_model_loading(self, model_loading_event: threading.Event):
        """Initialize the inference system in background while audio extraction is happening."""
        try:
            rich_console.print_info("Loading Audio Flamingo 3 in background...")
            # Only initialize the inference system, don't start workers yet
            # Workers will be started in the inference processing thread
            self._init_inference_system()
            model_loading_event.set()
            # Success logging will be handled by the main thread
        except Exception as e:
            rich_console.print_error(f"Background model loading failed: {e}")
            model_loading_event.set()  # Set event anyway to unblock other threads
    
    def _parallel_audio_extraction(self, video_path: str, audio_segments: List[Dict[str, Any]], 
                                 video_id: str, extraction_queue: queue.Queue):
        """Extract audio segments in parallel using ThreadPoolExecutor."""
        rich_console.print_info(f"Starting parallel audio extraction for {len(audio_segments)} segments")
        
        try:
            # Use ThreadPoolExecutor for parallel audio extraction
            max_workers = min(MAX_EXTRACTION_WORKERS, len(audio_segments))
            extracted_count = 0
            failed_count = 0
            last_progress_update = 0
            error_logger = ThrottledErrorLogger(threshold=ERROR_LOG_SUPPRESSION_THRESHOLD, console=rich_console)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all extraction tasks
                future_to_segment = {}
                
                for i, segment in enumerate(audio_segments):
                    future = executor.submit(
                        self._extract_single_segment,
                        video_path, segment, video_id, i
                    )
                    future_to_segment[future] = (i, segment)
                
                # Process completed extractions as they finish
                for future in as_completed(future_to_segment):
                    segment_index, original_segment = future_to_segment[future]
                    
                    try:
                        extracted_segment = future.result()
                        if extracted_segment and extracted_segment.get('extraction_success'):
                            # Send to inference queue immediately
                            extraction_queue.put(extracted_segment)
                            extracted_count += 1
                            
                            # Show progress every 25% or minimum every 20 segments
                            progress_interval = calculate_progress_interval(len(audio_segments), target_updates=4, minimum=20)
                            if (extracted_count - last_progress_update >= progress_interval) or extracted_count == len(audio_segments):
                                rich_console.print_info(f"Extraction progress: {extracted_count}/{len(audio_segments)} segments ({extracted_count/len(audio_segments)*100:.0f}%)")
                                last_progress_update = extracted_count
                        else:
                            failed_count += 1
                            # Still queue failed extractions for error handling
                            extraction_queue.put(extracted_segment or {
                                'segment_index': segment_index,
                                'extraction_success': False,
                                'error': 'Extraction failed',
                                'original_segment': original_segment
                            })
                    except Exception as e:
                        failed_count += 1
                        error_logger.log(f"Error extracting segment {segment_index}: {e}")
                        extraction_queue.put({
                            'segment_index': segment_index,
                            'extraction_success': False,
                            'error': str(e),
                            'original_segment': original_segment
                        })
            
            # Signal extraction completion
            extraction_queue.put(None)
            
            # Final summary
            if failed_count > 0:
                rich_console.print_warning(f"Audio extraction completed: {extracted_count} successful, {failed_count} failed")
            else:
                rich_console.print_success(f"All {extracted_count} audio segments extracted successfully")
            
        except Exception as e:
            rich_console.print_error(f"Parallel audio extraction failed: {e}")
            extraction_queue.put(None)  # Signal completion even on error
    
    def _extract_single_segment(self, video_path: str, segment: Dict[str, Any], 
                              video_id: str, segment_index: int) -> Optional[Dict[str, Any]]:
        """Extract a single audio segment."""
        try:
            segment_output_dir = os.path.join(self.temp_dir, video_id)
            os.makedirs(segment_output_dir, exist_ok=True)
            
            # Create segment data for extractor
            segment_data = {
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'segment_index': segment_index
            }
            
            # Extract audio segment
            extracted_segment = self.audio_extractor.extract_single_segment(
                video_path, segment_data, segment_output_dir
            )
            
            if extracted_segment:
                extracted_segment['original_segment'] = segment
                extracted_segment['segment_index'] = segment_index
                return extracted_segment
            else:
                return {
                    'segment_index': segment_index,
                    'extraction_success': False,
                    'error': 'Failed to extract audio segment',
                    'original_segment': segment
                }
                
        except Exception as e:
            return {
                'segment_index': segment_index,
                'extraction_success': False,
                'error': str(e),
                'original_segment': segment
            }
    
    def _pipeline_inference_processing(self, extraction_queue: queue.Queue, 
                                     results_queue: queue.Queue,
                                     model_loading_event: threading.Event,
                                     total_segments: int):
        """Process extracted segments through GPU inference as they become available using multi-GPU workers."""
        rich_console.print_info("Starting GPU inference pipeline...")
        
        # Wait for model to be loaded (basic initialization)
        model_loading_event.wait(timeout=300)  # 5 minute timeout
        
        if not model_loading_event.is_set():
            rich_console.print_error("Model loading timeout - proceeding anyway")
        else:
            rich_console.print_success("Model loading completed")
        
        # Start GPU workers
        self.inference_system.start_workers()
        rich_console.print_success("GPU workers started")
        
        processed_count = 0
        
        try:
            while processed_count < total_segments:
                try:
                    # Collect segments for batch processing
                    batch_segments = []
                    batch_audio_paths = []
                    
                    # Progressive batching: start with smaller batches early, larger batches later
                    early_extraction_phase = processed_count < (total_segments * EARLY_EXTRACTION_PHASE_RATIO)
                    if early_extraction_phase:
                        # Use smaller batches during extraction phase for faster startup
                        effective_batch_size = max(self.num_gpus, self.batch_size // 2)  # At least 1 per GPU
                        batch_timeout = 1.0  # Shorter timeout for early batches
                    else:
                        # Use full batch size once extraction is well underway
                        effective_batch_size = self.batch_size
                        batch_timeout = 2.0  # Standard timeout
                    
                    # Get segments up to effective batch size or until queue is empty/timeout
                    while len(batch_segments) < effective_batch_size and processed_count + len(batch_segments) < total_segments:
                        try:
                            extracted_segment = extraction_queue.get(timeout=batch_timeout)
                            
                            if extracted_segment is None:
                                # End of extraction signal - continue with current batch
                                rich_console.print_info("Extraction completed signal received")
                                break
                            
                            if extracted_segment.get('extraction_success'):
                                batch_segments.append(extracted_segment)
                                batch_audio_paths.append(extracted_segment['audio_path'])
                            else:
                                # Handle extraction failure immediately
                                audio_description = {
                                    'segment_index': extracted_segment.get('segment_index', processed_count),
                                    'start': extracted_segment.get('start', 0),
                                    'end': extracted_segment.get('end', 0),
                                    'duration': extracted_segment.get('duration', 0),
                                    'audio_description': f"Extraction failed: {extracted_segment.get('error', 'Unknown error')}",
                                    'processing_method': 'extraction_failed',
                                    'error': extracted_segment.get('error', 'Unknown extraction error'),
                                    'original_segment': extracted_segment.get('original_segment', {})
                                }
                                results_queue.put(audio_description)
                                processed_count += 1
                                
                        except queue.Empty:
                            # No more segments available right now, process current batch
                            break
                    
                    # Process batch if we have segments
                    if batch_segments:
                        batch_start_time = time.time()
                        batch_num = processed_count//self.batch_size + 1
                        
                        # Run batch inference on multiple GPUs (without auto-shutdown)
                        batch_results = self.inference_system.batch_inference(batch_audio_paths, self.text_prompt, auto_shutdown=False)
                        
                        batch_successful = 0
                        batch_failed = 0
                        
                        # Process results and send to results queue
                        for segment, result in zip(batch_segments, batch_results):
                            if result['success']:
                                batch_successful += 1
                                audio_description = {
                                    'segment_index': segment.get('segment_index', 0),
                                    'start': segment.get('start', 0),
                                    'end': segment.get('end', 0),
                                    'duration': segment.get('duration', 0),
                                    'audio_description': result['response'],
                                    'processing_method': 'audio_flamingo_3_multi_gpu_pipeline',
                                    'inference_time': result.get('inference_time', 0),
                                    'gpu_id': result.get('gpu_id', -1),
                                    'sample_rate': result.get('sample_rate', self.sample_rate),
                                    'original_segment': segment.get('original_segment', {})
                                }
                            else:
                                batch_failed += 1
                                error_msg = result.get('error', 'Unknown error')
                                audio_description = {
                                    'segment_index': segment.get('segment_index', 0),
                                    'start': segment.get('start', 0),
                                    'end': segment.get('end', 0),
                                    'duration': segment.get('duration', 0),
                                    'audio_description': f"Inference failed: {error_msg}",
                                    'processing_method': 'audio_flamingo_3_error',
                                    'error': error_msg,
                                    'gpu_id': result.get('gpu_id', -1),
                                    'original_segment': segment.get('original_segment', {})
                                }
                            
                            results_queue.put(audio_description)
                            processed_count += 1
                            
                            # Clean up audio file immediately
                            safe_remove_file(segment.get('audio_path'))

                        batch_time = time.time() - batch_start_time
                        progress_pct = (processed_count / total_segments) * 100
                        throughput = len(batch_segments) / batch_time  # segments per second
                        
                        # Enhanced batch progress logging with performance metrics
                        batch_type = "Early" if effective_batch_size < self.batch_size else "Full"
                        if batch_failed > 0:
                            rich_console.print_warning(f"{batch_type} Batch {batch_num}: {batch_successful}/{len(batch_segments)} successful ({batch_time:.1f}s, {throughput:.1f} seg/s) | Overall: {processed_count}/{total_segments} ({progress_pct:.0f}%)")
                        else:
                            rich_console.print_info(f"{batch_type} Batch {batch_num}: {batch_successful} segments processed ({batch_time:.1f}s, {throughput:.1f} seg/s) | Overall: {processed_count}/{total_segments} ({progress_pct:.0f}%)")
                    
                    # If no segments in current batch and we haven't processed everything, continue waiting
                    elif processed_count < total_segments:
                        # Wait a bit more for segments to become available
                        continue
                    else:
                        # All segments processed
                        break
                        
                except Exception as e:
                    rich_console.print_error(f"Error in batch processing: {e}")
                    # Create error result for any remaining unprocessed segments
                    error_result = {
                        'segment_index': processed_count,
                        'audio_description': f"Batch processing error: {str(e)}",
                        'processing_method': 'batch_error',
                        'error': str(e),
                        'original_segment': {}
                    }
                    results_queue.put(error_result)
                    processed_count += 1
            
            rich_console.print_success(f"GPU inference pipeline completed for {processed_count} segments")
            
        except Exception as e:
            rich_console.print_error(f"Pipeline inference processing failed: {e}")
        
        finally:
            # Shutdown GPU workers
            try:
                self.inference_system.shutdown_workers()
                rich_console.print_info("GPU workers shut down")
            except Exception as e:
                rich_console.print_warning(f"Error shutting down GPU workers: {e}")
    
    def cleanup(self):
        """Clean up resources and temporary files."""
        try:
            # Clean up inference system
            if self.inference_system:
                self.inference_system.cleanup_persistent_model()
                self.inference_system = None
            
            # Clean up temp directory
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            
            rich_console.print_info("Audio descriptor cleanup completed")
            
        except Exception as e:
            rich_console.print_warning(f"Error during cleanup: {e}")