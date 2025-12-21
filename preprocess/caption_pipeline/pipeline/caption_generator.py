"""
Caption generator module using Faster Whisper with built-in batching support.
"""

# Constants and imports
import sys
import os
import json
import time
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import logging
import numpy as np

# Import BatchedInferencePipeline for better performance
from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.vad import VadOptions

# Import project configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from caption_pipeline.utils.rich_console import get_console
from config import AUDIO_DIR, CAPTIONS_DIR, METADATA_DIR, GPU_DEVICE

# Import common VLM utilities
from caption_pipeline.utils.vlm_common import safe_remove_file

# Set up logging
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,  # Suppress info messages for cleaner console
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'caption_generator.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
rich_console = get_console()

# Add constants for model configuration
DEFAULT_MODEL_SIZE = "large-v3"  # Options: tiny, base, small, medium, large, large-v2, large-v3
DEFAULT_LANGUAGE = "en"  # Language code
DEFAULT_COMPUTE_TYPE = "float16"  # Options: float32, float16, int8_float16, int8
DEFAULT_DEVICE = "cuda"  # Use GPU if available

# Add constants for inference
DEFAULT_BATCH_SIZE = 16  # Default batch size for parallel processing
DEFAULT_BEAM_SIZE = 5    # Default beam size for transcription

# Constants for VAD settings
VAD_THRESHOLD = 0.5      # Higher values require louder audio to be considered speech
VAD_MIN_SPEECH_DURATION_MS = 250  # Minimum duration for a speech segment
VAD_MAX_SPEECH_DURATION_S = 30.0  # Maximum duration for a speech segment
VAD_MIN_SILENCE_DURATION_MS = 500  # Minimum duration for a silence segment

# Set maximum processing time
MAX_PROCESSING_TIME = 3600   # Maximum processing time per video (1 hour)

# Constants for validation
MAX_TIMESTAMP_GAP = 5.0  # Alert threshold for large gaps between segments
MIN_COVERAGE_RATIO = 0.1  # Minimum expected transcription coverage ratio
MIN_COVERAGE_SECONDS = 60.0  # Minimum expected transcription coverage in seconds


class CaptionGenerator:
    """Class to generate automatic captions for videos using Faster Whisper."""
    
    def __init__(self, 
                 model_size: str = DEFAULT_MODEL_SIZE,
                 device: str = DEFAULT_DEVICE,
                 compute_type: str = DEFAULT_COMPUTE_TYPE,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        """Initialize the caption generator."""
        # Initialize paths from config
        self.audio_dir = AUDIO_DIR
        self.captions_dir = CAPTIONS_DIR
        self.metadata_dir = METADATA_DIR
        self.metadata_file = os.path.join(self.metadata_dir, 'video_metadata.csv')
        
        # Load metadata
        self._reload_metadata()
        
        # Whisper model configuration
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        
        # Initialize Whisper model
        rich_console.print_info(f"Loading Faster Whisper model: {model_size} on {device} with {compute_type}, batch size {batch_size}")
        try:
            # Initialize base model
            self.base_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                device_index=0,
                num_workers=8
            )
            
            # Create batched inference pipeline for better performance if on GPU
            if device == "cuda":
                rich_console.print_info(f"Creating batched inference pipeline with batch_size={batch_size}")
                self.model = BatchedInferencePipeline(
                    model=self.base_model,
                )
            else:
                rich_console.print_info("Using standard inference model")
                self.model = self.base_model
                
            rich_console.print_info("Faster Whisper model loaded successfully")
        except Exception as e:
            rich_console.print_error(f"Error loading Faster Whisper model: {str(e)}")
            raise
            
        # Ensure caption directory exists
        os.makedirs(self.captions_dir, exist_ok=True)
    
    def _reload_metadata(self):
        """Reload metadata to ensure we have the latest updates."""
        if os.path.exists(self.metadata_file):
            try:
                self.metadata_df = pd.read_csv(
                    self.metadata_file,
                    dtype={'audio_path': str, 'caption_path': str, 'file_path': str}
                )
                rich_console.print_info(f"Loaded metadata for {len(self.metadata_df)} videos")
            except Exception as e:
                rich_console.print_error(f"Error loading metadata from {self.metadata_file}: {str(e)}")
                self.metadata_df = pd.DataFrame()
        else:
            rich_console.print_warning(f"Metadata file not found: {self.metadata_file}")
            self.metadata_df = pd.DataFrame()

    def _handle_caption_error(self, video_id: str, status: str, reason: str) -> None:
        """Handle common error cleanup and metadata update for caption failures.

        Args:
            video_id: ID of the video
            status: Status to set in metadata (e.g., 'caption_timeout', 'caption_generation_failed')
            reason: Reason for failure (for logging)
        """
        # Clean up partial files
        partial_path = os.path.join(self.captions_dir, f"{video_id}_partial.json")
        if safe_remove_file(partial_path):
            rich_console.print_info(f"[{video_id}] Cleaned up partial caption file due to {reason}")

        # Mark as failed in metadata
        self._reload_metadata()
        video_idx = self.metadata_df[self.metadata_df['video_id'] == video_id].index
        if len(video_idx) > 0:
            self.metadata_df.loc[video_idx, 'status'] = status
            self.metadata_df.to_csv(self.metadata_file, index=False)

    def _log_audio_details(self, video_id: str, audio_path: str, video_duration: float) -> None:
        """Log audio file details for debugging.

        Args:
            video_id: ID of the video
            audio_path: Path to audio file
            video_duration: Duration of video in seconds
        """
        logger.error(f"[{video_id}] Video details - Duration: {video_duration}s, Audio path: {audio_path}")
        if os.path.exists(audio_path):
            audio_size = os.path.getsize(audio_path)
            logger.error(f"[{video_id}] Audio file size: {audio_size} bytes")
        else:
            logger.error(f"[{video_id}] Audio file does not exist!")

    def _find_audio_path(self, video_id: str) -> Optional[str]:
        """Find the audio path for a video, checking multiple sources."""
        # First check if we already have the audio path in metadata
        video_data = self.metadata_df[self.metadata_df['video_id'] == video_id]
        if len(video_data) > 0 and 'audio_path' in video_data.columns and pd.notna(video_data.iloc[0]['audio_path']):
            audio_path = video_data.iloc[0]['audio_path']
            if os.path.exists(audio_path):
                return audio_path
        
        # Check common audio file patterns
        for ext in ['wav', 'mp3']:
            audio_path = os.path.join(self.audio_dir, f"{video_id}.{ext}")
            if os.path.exists(audio_path):
                rich_console.print_info(f"Found audio file for {video_id} using direct path check: {audio_path}")
                return audio_path
        
        rich_console.print_error(f"Audio for video {video_id} not found after checking all possible locations")
        return None

    def _save_partial_results(self, video_id: str, partial_data: Dict, progress: float = 0.5) -> None:
        """Save partial transcription results to a temporary file."""
        # Create partial captions data with progress information
        partial_captions = {
            **partial_data,
            "processing_stats": {
                **partial_data.get("processing_stats", {}),
                "processing_status": "in_progress",
                "progress": progress,
            }
        }
        
        # Save to partial file
        partial_path = os.path.join(self.captions_dir, f"{video_id}_partial.json")
        with open(partial_path, 'w', encoding='utf-8') as f:
            json.dump(partial_captions, f, ensure_ascii=False, indent=2)
            
        rich_console.print_info(f"[{video_id}] Progress: {progress:.1%} complete")

    def generate_captions(self, video_id: str) -> Optional[str]:
        """Generate captions for a video using Faster Whisper."""
        # Reload metadata to get the latest status
        self._reload_metadata()
        
        # Check if captions already generated
        caption_path = os.path.join(self.captions_dir, f"{video_id}.json")
        if os.path.exists(caption_path):
            rich_console.print_info(f"Captions for video {video_id} already generated.")
            return caption_path
        
        # Get audio path - use direct file checking as a fallback
        audio_path = self._find_audio_path(video_id)
        if not audio_path:
            rich_console.print_error(f"Audio for video {video_id} not found or not extracted.")
            return None
        
        # Get video metadata for additional info
        video_data = self.metadata_df[self.metadata_df['video_id'] == video_id]
        video_title = video_data.iloc[0]['title'] if len(video_data) > 0 and 'title' in video_data.columns else "Unknown"
        video_duration = float(video_data.iloc[0]['duration']) if len(video_data) > 0 and 'duration' in video_data.columns else 0.0
        
        rich_console.print_info(f"Generating captions for video: {video_id} (Duration: {video_duration:.1f}s)")
        start_time = time.time()
        
        # Set a timeout to prevent endless processing
        timeout_time = start_time + MAX_PROCESSING_TIME
        
        try:
            # Check if we're approaching timeout before starting
            if time.time() > timeout_time:
                raise TimeoutError("Processing timeout reached before transcription could begin")
                
            # Save partial result indicating we've started
            initial_data = {
                "video_id": video_id,
                "title": video_title,
                "duration": video_duration,
                "processing_stats": {
                    "processing_time_seconds": 0,
                    "model": {
                        "name": "faster-whisper",
                        "size": self.model_size,
                        "device": self.device,
                        "compute_type": self.compute_type,
                        "batch_size": self.batch_size
                    }
                }
            }
            self._save_partial_results(video_id, initial_data, progress=0.01)
            
            # Configure VAD options for more reliable speech detection
            vad_options = VadOptions(
                threshold=VAD_THRESHOLD,
                min_speech_duration_ms=VAD_MIN_SPEECH_DURATION_MS,
                max_speech_duration_s=VAD_MAX_SPEECH_DURATION_S,
                min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
                speech_pad_ms=400
            )
            
            # Process the audio file
            rich_console.print_info(f"[{video_id}] Transcribing audio file: {os.path.basename(audio_path)}")
            
            # Check timeout before starting transcription
            if time.time() > timeout_time:
                raise TimeoutError("Processing timeout reached before transcription")
            
            # First attempt with VAD filtering
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=DEFAULT_BEAM_SIZE,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=vad_options,
                batch_size=self.batch_size,
                initial_prompt=None,  # Let model auto-detect without bias
                language=None,  # Auto-detect language
                condition_on_previous_text=True,
                temperature=0.0  # Deterministic output
            )
            
            # Convert generator to list with timeout checking
            segments_list = []
            last_check_time = time.time()
            
            rich_console.print_info(f"[{video_id}] Processing transcription results...")
            for i, segment in enumerate(segments):
                segments_list.append(segment)
                
                # Check timeout every 10 segments to avoid excessive checking
                if i % 10 == 0:
                    current_time = time.time()
                    if current_time > timeout_time:
                        raise TimeoutError(f"Processing timeout reached while processing segment {i}")
                    
                    # Progress update every 30 seconds
                    if current_time - last_check_time > 30:
                        self._save_partial_results(video_id, initial_data, progress=0.2)
                        last_check_time = current_time
                        rich_console.print_info(f"[{video_id}] Processed {i} segments so far...")
            
            language = info.language
            language_probability = info.language_probability
            
            rich_console.print_info(f"[{video_id}] Completed processing {len(segments_list)} segments")
            
            # Check timeout after processing all segments
            if time.time() > timeout_time:
                raise TimeoutError("Processing timeout reached after processing segments")
            
            # If we got very few segments or the duration is much shorter than the video,
            # try again without VAD filtering
            total_transcribed_duration = sum(segment.end - segment.start for segment in segments_list)
            expected_min_duration = min(video_duration * MIN_COVERAGE_RATIO, MIN_COVERAGE_SECONDS)
            
            if not segments_list or total_transcribed_duration < expected_min_duration:
                rich_console.print_warning(f"[{video_id}] Initial transcription too short ({total_transcribed_duration:.2f}s), " +
                            f"retrying without VAD filtering for video of duration {video_duration:.2f}s")
                
                # Check timeout before retry
                if time.time() > timeout_time:
                    raise TimeoutError("Processing timeout reached before retry transcription")
                
                # Update progress
                self._save_partial_results(video_id, initial_data, progress=0.3)
                
                # Retry without VAD filtering
                segments, info = self.model.transcribe(
                    audio_path,
                    beam_size=DEFAULT_BEAM_SIZE,
                    word_timestamps=True,
                    vad_filter=True,  # Disable VAD filtering
                    initial_prompt=None,  # Let model auto-detect without bias
                    language=None,  # Auto-detect language
                    temperature=0.0,  # Deterministic output
                    condition_on_previous_text=True,
                    batch_size=self.batch_size
                )
                
                # Get all segments with timeout checking
                segments_list = []
                rich_console.print_info(f"[{video_id}] Processing retry transcription results...")
                for i, segment in enumerate(segments):
                    segments_list.append(segment)
                    
                    # Check timeout every 10 segments
                    if i % 10 == 0 and time.time() > timeout_time:
                        raise TimeoutError(f"Processing timeout reached during retry at segment {i}")
                
                language = info.language
                language_probability = info.language_probability
                rich_console.print_info(f"[{video_id}] Retry resulted in {len(segments_list)} segments")
            
            # Check if we got any segments
            if not segments_list:
                rich_console.print_error(f"[{video_id}] No transcription segments were generated")
                raise Exception("Failed to get any valid transcriptions")
                
            # Format segments from Segment objects to our standard format
            transcript_segments = [{
                'id': i,
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'words': [{
                    'word': word.word,
                    'start': word.start,
                    'end': word.end,
                    'probability': -1
                } for word in segment.words] if hasattr(segment, 'words') else []
            } for i, segment in enumerate(segments_list)]
                
            # Log segments count and duration
            total_segments = len(transcript_segments)
            total_duration = sum(segment['end'] - segment['start'] for segment in transcript_segments)
            rich_console.print_info(f"[{video_id}] Transcribed {total_segments} segments covering {total_duration:.2f}s " +
                       f"of {video_duration:.2f}s audio")
            
            # Format the result in our desired structure
            transcript = {
                'text': ' '.join(segment['text'] for segment in transcript_segments),
                'segments': transcript_segments,
                'language': language,
                'language_probability': language_probability
            }
            
            # Validate the transcript
            if not transcript['segments']:
                raise Exception("Failed to get any valid transcriptions")
            
            # Check for any abnormal timestamp patterns
            self._validate_timestamps(transcript['segments'], video_id)
            
            # Create captions data
            captions_data = {
                "video_id": video_id,
                "title": video_title,
                "duration": video_duration,
                "language": transcript['language'],
                "language_confidence": float(transcript['language_probability']),
                "transcript": transcript,
                "verification_status": "unverified",
                "processing_stats": {
                    "processing_time_seconds": time.time() - start_time,
                    "processing_status": "completed",
                    "progress": 1.0,
                    "transcribed_duration": total_duration,
                    "segments_count": total_segments,
                    "model": {
                        "name": "faster-whisper",
                        "size": self.model_size,
                        "device": self.device,
                        "compute_type": self.compute_type,
                        "batch_size": self.batch_size
                    }
                }
            }
            
            # Save captions to file
            with open(caption_path, 'w', encoding='utf-8') as f:
                json.dump(captions_data, f, ensure_ascii=False, indent=2)
            
            # Remove partial file if it exists
            partial_path = os.path.join(self.captions_dir, f"{video_id}_partial.json")
            safe_remove_file(partial_path)
            
            # Update metadata
            self._reload_metadata()  # Reload to avoid overwriting other changes
            video_idx = self.metadata_df[self.metadata_df['video_id'] == video_id].index
            if len(video_idx) > 0:
                self.metadata_df.loc[video_idx, 'caption_path'] = caption_path
                self.metadata_df.loc[video_idx, 'status'] = 'captioned'
                self.metadata_df.to_csv(self.metadata_file, index=False)
            
            processing_time = time.time() - start_time
            rich_console.print_info(f"[{video_id}] Caption generation completed in {processing_time:.2f}s")
            return caption_path
            
        except TimeoutError as e:
            processing_time = time.time() - start_time
            rich_console.print_error(f"[{video_id}] Processing timeout after {processing_time:.1f}s: {e}")
            rich_console.print_error(f"[{video_id}] Video duration: {video_duration}s, Timeout limit: {MAX_PROCESSING_TIME}s")

            # Log timeout details
            logger.error(f"[{video_id}] Caption generation timed out after {processing_time:.1f}s")
            self._log_audio_details(video_id, audio_path, video_duration)
            self._handle_caption_error(video_id, 'caption_timeout', 'timeout')
            return None

        except Exception as e:
            import traceback

            # Print comprehensive error information
            rich_console.print_error(f"[{video_id}] Error generating captions: {e}")
            rich_console.print_error(f"[{video_id}] Error type: {type(e).__name__}")
            rich_console.print_error(f"[{video_id}] Full traceback:")
            rich_console.print_error(traceback.format_exc())

            # Log to file as well
            logger.error(f"[{video_id}] Caption generation failed with error: {e}")
            logger.error(f"[{video_id}] Full traceback: {traceback.format_exc()}")
            self._log_audio_details(video_id, audio_path, video_duration)
            self._handle_caption_error(video_id, 'caption_generation_failed', 'error')
            return None
            
    def _validate_timestamps(self, segments: List[Dict], video_id: str = None) -> None:
        """Validate timestamps for consistency and log any issues."""
        if not segments:
            return
            
        # Get min and max timestamps
        min_time = min(seg['start'] for seg in segments)
        max_time = max(seg['end'] for seg in segments)
        
        # Check for negative timestamps
        has_negative = any(seg['start'] < 0 or seg['end'] < 0 for seg in segments)
        
        # Format video_id prefix for logs if provided
        vid_prefix = f"[{video_id}] " if video_id else ""
        
        # Log validation results
        rich_console.print_info(f"{vid_prefix}Timestamps: {min_time:.2f}s to {max_time:.2f}s" +
                   (", Contains negative timestamps" if has_negative else ""))
        
        # Check for large gaps in timestamps
        if len(segments) > 1:
            # Sort segments by start time
            sorted_segments = sorted(segments, key=lambda x: x['start'])
            
            # Find the largest gap
            largest_gap = 0
            for i in range(len(sorted_segments) - 1):
                gap = sorted_segments[i+1]['start'] - sorted_segments[i]['end']
                largest_gap = max(largest_gap, gap)
                
            if largest_gap > MAX_TIMESTAMP_GAP:
                rich_console.print_warning(f"{vid_prefix}Large gap detected: {largest_gap:.2f}s")