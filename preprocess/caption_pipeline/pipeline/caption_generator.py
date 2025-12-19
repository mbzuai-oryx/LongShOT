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

# Import MovieCaptionEnhancer for enhanced captions
from caption_pipeline.models.movie_caption_enhancer import MovieCaptionEnhancer

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
DEFAULT_LANGUAGE = "en"  # Arabic language code
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


class CaptionGenerator:
    """Class to generate automatic captions for Arabic videos using Faster Whisper."""
    
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
                self.metadata_df = pd.read_csv(self.metadata_file)
                rich_console.print_info(f"Loaded metadata for {len(self.metadata_df)} videos")
            except Exception as e:
                rich_console.print_error(f"Error loading metadata from {self.metadata_file}: {str(e)}")
                self.metadata_df = pd.DataFrame()
        else:
            rich_console.print_warning(f"Metadata file not found: {self.metadata_file}")
            self.metadata_df = pd.DataFrame()

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
            expected_min_duration = min(video_duration * 0.1, 60.0)  # At least 10% of video or 60 seconds
            
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
            if os.path.exists(partial_path):
                os.remove(partial_path)
            
            # Update metadata
            self._reload_metadata()  # Reload to avoid overwriting other changes
            video_idx = self.metadata_df[self.metadata_df['video_id'] == video_id].index
            if len(video_idx) > 0:
                self.metadata_df.loc[video_idx, 'caption_path'] = caption_path
                self.metadata_df.loc[video_idx, 'status'] = 'captioned'
                self.metadata_df.to_csv(self.metadata_file, index=False)
            
            processing_time = time.time() - start_time
            rich_console.print_info(f"[{video_id}] ✓ Caption generation completed in {processing_time:.2f}s")
            return caption_path
            
        except TimeoutError as e:
            processing_time = time.time() - start_time
            rich_console.print_error(f"[{video_id}] ✗ Processing timeout after {processing_time:.1f}s: {e}")
            rich_console.print_error(f"[{video_id}] Video duration: {video_duration}s, Timeout limit: {MAX_PROCESSING_TIME}s")
            
            # Log timeout details
            logger.error(f"[{video_id}] Caption generation timed out after {processing_time:.1f}s")
            logger.error(f"[{video_id}] Video details - Duration: {video_duration}s, Audio path: {audio_path}")
            if os.path.exists(audio_path):
                audio_size = os.path.getsize(audio_path)
                logger.error(f"[{video_id}] Audio file size: {audio_size} bytes")
            
            # Always clean up partial files on timeout
            partial_path = os.path.join(self.captions_dir, f"{video_id}_partial.json")
            if os.path.exists(partial_path):
                try:
                    os.remove(partial_path)
                    rich_console.print_info(f"[{video_id}] Cleaned up partial caption file due to timeout")
                except:
                    pass
            
            # Mark as failed in metadata
            self._reload_metadata()
            video_idx = self.metadata_df[self.metadata_df['video_id'] == video_id].index
            if len(video_idx) > 0:
                self.metadata_df.loc[video_idx, 'status'] = 'caption_timeout'
                self.metadata_df.to_csv(self.metadata_file, index=False)
            
            return None
            
        except Exception as e:
            import traceback
            
            # Print comprehensive error information
            rich_console.print_error(f"[{video_id}] ✗ Error generating captions: {e}")
            rich_console.print_error(f"[{video_id}] Error type: {type(e).__name__}")
            rich_console.print_error(f"[{video_id}] Full traceback:")
            rich_console.print_error(traceback.format_exc())
            
            # Log to file as well
            logger.error(f"[{video_id}] Caption generation failed with error: {e}")
            logger.error(f"[{video_id}] Full traceback: {traceback.format_exc()}")
            
            # Log video details for debugging
            logger.error(f"[{video_id}] Video details - Duration: {video_duration}s, Audio path: {audio_path}")
            if os.path.exists(audio_path):
                audio_size = os.path.getsize(audio_path)
                logger.error(f"[{video_id}] Audio file size: {audio_size} bytes")
            else:
                logger.error(f"[{video_id}] Audio file does not exist!")
            
            # Clean up partial files on error
            partial_path = os.path.join(self.captions_dir, f"{video_id}_partial.json")
            if os.path.exists(partial_path):
                try:
                    os.remove(partial_path)
                    rich_console.print_info(f"[{video_id}] Cleaned up partial caption file due to error")
                except:
                    pass
            
            # Mark as failed in metadata
            self._reload_metadata()
            video_idx = self.metadata_df[self.metadata_df['video_id'] == video_id].index
            if len(video_idx) > 0:
                self.metadata_df.loc[video_idx, 'status'] = 'caption_generation_failed'
                self.metadata_df.to_csv(self.metadata_file, index=False)
            
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
                
            if largest_gap > 5.0:  # Alert if gap > 5 seconds
                rich_console.print_warning(f"{vid_prefix}Large gap detected: {largest_gap:.2f}s")

    def process_audio_batch(self, audio_batch: List[Dict]) -> List[Dict]:
        """Process a batch of audio files in parallel."""
        start_time = time.time()
        rich_console.print_info(f"Processing batch of {len(audio_batch)} videos")
        
        # Create initial partial results for all videos in batch
        for item in audio_batch:
            video_id = item['video_id']
            initial_data = {
                "video_id": video_id,
                "title": item['metadata']['title'],
                "duration": float(item['metadata']['duration']),
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
        
        results = []
        
        for item in audio_batch:
            video_id = item['video_id']
            try:
                rich_console.print_info(f"[{video_id}] Starting transcription...")
                
                # Configure VAD options for more reliable speech detection
                vad_options = VadOptions(
                    threshold=VAD_THRESHOLD,
                    min_speech_duration_ms=VAD_MIN_SPEECH_DURATION_MS,
                    max_speech_duration_s=VAD_MAX_SPEECH_DURATION_S,
                    min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
                    speech_pad_ms=400
                )
                
                # Initial transcription with VAD filtering
                segments, info = self.model.transcribe(
                    item['audio_path'],
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
                
                # Convert generator to list and check if we got enough results
                segments_list = list(segments)
                language = info.language
                language_probability = info.language_probability
                
                # Get video duration from metadata
                video_duration = float(item['metadata'].get('duration', 0.0))
                
                # If we got very few segments or the duration is much shorter than the video,
                # try again without VAD filtering
                total_transcribed_duration = sum(segment.end - segment.start for segment in segments_list)
                expected_min_duration = min(video_duration * 0.1, 60.0)  # At least 10% of video or 60 seconds
                
                if not segments_list or total_transcribed_duration < expected_min_duration:
                    rich_console.print_warning(f"[{video_id}] Initial transcription too short ({total_transcribed_duration:.2f}s), " +
                                f"retrying without VAD filtering for video of duration {video_duration:.2f}s")
                    
                    # Retry without VAD filtering
                    segments, info = self.model.transcribe(
                        item['audio_path'],
                        beam_size=DEFAULT_BEAM_SIZE,
                        word_timestamps=True,
                        vad_filter=True,  # Disable VAD filtering
                        initial_prompt=None,  # Let model auto-detect without bias
                        language=None,  # Auto-detect language
                        temperature=0.0,  # Deterministic output
                        condition_on_previous_text=True,
                        batch_size=self.batch_size
                    )
                    
                    # Get all segments
                    segments_list = list(segments)
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
                
                # Format the result
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
                
                results.append({
                    'video_id': video_id,
                    'metadata': item['metadata'],
                    'transcript': transcript,
                    'success': True
                })
                
                # Update partial results
                self._save_partial_results(
                    video_id, 
                    {
                        "video_id": video_id,
                        "title": item['metadata']['title'],
                        "duration": float(item['metadata']['duration']),
                        "transcript": transcript,
                        "processing_stats": {
                            "processing_time_seconds": time.time() - start_time,
                        }
                    }, 
                    progress=0.9
                )
                
            except Exception as e:
                rich_console.print_error(f"[{video_id}] ✗ Error transcribing audio: {str(e)}")
                results.append({
                    'video_id': video_id,
                    'metadata': item['metadata'],
                    'error': str(e),
                    'success': False
                })
        
        batch_time = time.time() - start_time
        rich_console.print_info(f"Batch processing completed in {batch_time:.2f}s")
        return results

    def batch_generate_captions(self, max_videos: int = None, batch_size: int = None) -> List[str]:
        """Generate captions for all videos with extracted audio in parallel batches."""
        # Reload metadata to get the latest status
        self._reload_metadata()
        
        # Get all videos with extracted audio
        videos_with_audio = self.metadata_df[self.metadata_df['status'] == 'audio_extracted']
        
        if max_videos:
            videos_with_audio = videos_with_audio.head(max_videos)
        
        # Determine batch size
        actual_batch_size = batch_size or (self.batch_size if self.device == "cuda" else 1)
            
        total_videos = len(videos_with_audio)
        rich_console.print_info(f"Starting caption generation for {total_videos} videos in batches of {actual_batch_size}")
        start_time = time.time()
        
        caption_paths = []
        skipped_videos = 0
        
        # Create batches
        all_batches = []
        current_batch = []
        
        for _, row in videos_with_audio.iterrows():
            # Skip if captions already exist
            caption_path = os.path.join(self.captions_dir, f"{row['video_id']}.json")
            if os.path.exists(caption_path):
                rich_console.print_info(f"[{row['video_id']}] ↷ Skipping, captions already exist")
                caption_paths.append(caption_path)
                skipped_videos += 1
                continue
                
            # Find audio path with robust checks
            audio_path = self._find_audio_path(row['video_id'])
            if not audio_path:
                rich_console.print_error(f"[{row['video_id']}] ✗ Audio file not found")
                continue
                
            # Add to current batch
            current_batch.append({
                'video_id': row['video_id'],
                'audio_path': audio_path,
                'metadata': row.to_dict()
            })
            
            # If batch is full, add to all_batches and create new batch
            if len(current_batch) >= actual_batch_size:
                all_batches.append(current_batch)
                current_batch = []
        
        # Add any remaining videos to the last batch
        if current_batch:
            all_batches.append(current_batch)
            
        videos_to_process = total_videos - skipped_videos
        rich_console.print_info(f"Found {videos_to_process} videos to process in {len(all_batches)} batches")
            
        # Process batches with tqdm progress bar
        with tqdm(total=len(all_batches), desc="Processing batches", unit="batch") as pbar:
            for batch_idx, batch in enumerate(all_batches):
                rich_console.print_info(f"Batch [{batch_idx+1}/{len(all_batches)}]: Processing {len(batch)} videos")
                batch_results = self.process_audio_batch(batch)
                
                # Save results and update metadata
                successful_in_batch = 0
                for result in batch_results:
                    video_id = result['video_id']
                    if result['success']:
                        # Create final caption data
                        captions_data = {
                            "video_id": video_id,
                            "title": result['metadata']['title'],
                            "duration": float(result['metadata']['duration']),
                            "language": result['transcript']['language'],
                            "language_confidence": float(result['transcript']['language_probability']),
                            "transcript": result['transcript'],
                            "verification_status": "unverified",
                            "processing_stats": {
                                "processing_time_seconds": time.time() - start_time,
                                "processing_status": "completed",
                                "progress": 1.0,
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
                        caption_path = os.path.join(self.captions_dir, f"{video_id}.json")
                        with open(caption_path, 'w', encoding='utf-8') as f:
                            json.dump(captions_data, f, ensure_ascii=False, indent=2)
                        
                        # Remove partial file if it exists
                        partial_path = os.path.join(self.captions_dir, f"{video_id}_partial.json")
                        if os.path.exists(partial_path):
                            os.remove(partial_path)
                        
                        # Update metadata
                        self._reload_metadata()  # Reload to avoid overwriting other changes
                        video_idx = self.metadata_df[self.metadata_df['video_id'] == video_id].index
                        if len(video_idx) > 0:
                            self.metadata_df.loc[video_idx, 'caption_path'] = caption_path
                            self.metadata_df.loc[video_idx, 'status'] = 'captioned'
                        
                        caption_paths.append(caption_path)
                        rich_console.print_info(f"[{video_id}] ✓ Successfully generated captions")
                        successful_in_batch += 1
                    else:
                        # Mark as failed in metadata
                        self._reload_metadata()
                        video_idx = self.metadata_df[self.metadata_df['video_id'] == video_id].index
                        if len(video_idx) > 0:
                            self.metadata_df.loc[video_idx, 'status'] = 'caption_generation_failed'
                        rich_console.print_error(f"[{video_id}] ✗ Caption generation failed")
                
                # Save metadata after each batch
                self.metadata_df.to_csv(self.metadata_file, index=False)
                
                # Update progress bar with information
                completed = len(caption_paths) - skipped_videos
                pbar.set_postfix({
                    "completed": f"{completed}/{videos_to_process}",
                    "success_rate": f"{completed/max(1, videos_to_process):.1%}",
                    "batch_success": f"{successful_in_batch}/{len(batch)}"
                })
                pbar.update(1)
                
                # Show progress between batches
                if len(all_batches) > 1:
                    percent_done = (batch_idx + 1) / len(all_batches) * 100
                    rich_console.print_info(f"Progress: [{batch_idx+1}/{len(all_batches)}] {percent_done:.1f}% complete")
            
        total_time = time.time() - start_time
        successful_count = len(caption_paths) - skipped_videos
        rich_console.print_info(f"✓ Caption generation complete: {successful_count}/{videos_to_process} videos processed successfully in {total_time:.1f}s")
        
        return caption_paths


class EnhancedCaptionGenerator(CaptionGenerator):
    """Enhanced caption generator that adds movie-style descriptive elements and optional video descriptions."""
    
    def __init__(self, 
                 whisper_model: str = DEFAULT_MODEL_SIZE,
                 device: str = DEFAULT_DEVICE,
                 compute_type: str = DEFAULT_COMPUTE_TYPE,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 enable_movie_style: bool = True,
                 enable_segment_splitting: bool = True,
                 enable_video_descriptions: bool = False,
                 max_segment_length: int = 42,
                 min_segment_duration: float = 1.0):
        """Initialize the enhanced caption generator."""
        # Initialize base caption generator
        super().__init__(
            model_size=whisper_model,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size
        )
        
        # Enhancement parameters
        self.enable_movie_style = enable_movie_style
        self.enable_segment_splitting = enable_segment_splitting
        self.enable_video_descriptions = enable_video_descriptions
        self.max_segment_length = max_segment_length
        self.min_segment_duration = min_segment_duration
        
        # Initialize movie caption enhancer if movie style is enabled
        if self.enable_movie_style:
            self.enhancer = MovieCaptionEnhancer(
                enable_audio_analysis=True,
                enable_visual_analysis=True
            )
            rich_console.print_info("Movie caption enhancer initialized")
        else:
            self.enhancer = None
        
        # Initialize video descriptor if video descriptions are enabled
        if self.enable_video_descriptions:
            try:
                from config import (ENABLE_VIDEO_DESCRIPTIONS, VIDEO_DESCRIPTION_MODEL, 
                                  VLLM_SERVER_URL)
                from caption_pipeline.pipeline.video_descriptor import VideoDescriptor
                
                if ENABLE_VIDEO_DESCRIPTIONS:
                    self.video_descriptor = VideoDescriptor(
                        model_name=VIDEO_DESCRIPTION_MODEL,
                        api_base=VLLM_SERVER_URL
                    )
                    rich_console.print_info("Video descriptor initialized")
                else:
                    self.video_descriptor = None
                    rich_console.print_info("Video descriptions disabled in configuration")
            except Exception as e:
                rich_console.print_error(f"Failed to initialize video descriptor: {e}")
                self.video_descriptor = None
        else:
            self.video_descriptor = None
    
    def generate_captions(self, video_id: str) -> Optional[str]:
        """Generate enhanced captions with movie-style descriptive elements and optional video descriptions."""
        rich_console.print_info(f"Starting enhanced caption generation for video {video_id}")
        
        # First generate base captions
        caption_path = super().generate_captions(video_id)
        
        if not caption_path:
            rich_console.print_error(f"Failed to generate base captions for {video_id}")
            return None
        
        # Apply movie-style enhancements if enabled
        if self.enable_movie_style:
            caption_path = self._apply_movie_style_enhancements(video_id, caption_path)
        
        # Generate video descriptions if enabled
        if self.enable_video_descriptions and self.video_descriptor:
            try:
                rich_console.print_info(f"Generating video descriptions for {video_id}")
                description_file = self.video_descriptor.generate_video_descriptions(video_id)
                if description_file:
                    rich_console.print_info(f"Successfully generated video descriptions for {video_id}")
                    
                    # Automatically trigger metadata generation for this video
                    try:
                        rich_console.print_info(f"Automatically triggering metadata generation for {video_id}")
                        success = self.video_descriptor._trigger_metadata_generation([video_id])
                        if success:
                            rich_console.print_info(f"Successfully generated metadata for {video_id}")
                        else:
                            rich_console.print_warning(f"Failed to generate metadata for {video_id}")
                    except Exception as meta_e:
                        rich_console.print_warning(f"Failed to automatically generate metadata for {video_id}: {meta_e}")
                        rich_console.print_info("You can manually run metadata generation later with: python run_metadata_generation.py")
                else:
                    rich_console.print_warning(f"Failed to generate video descriptions for {video_id}")
            except Exception as e:
                rich_console.print_error(f"Error generating video descriptions for {video_id}: {e}")
        
        return caption_path
    
    def _apply_movie_style_enhancements(self, video_id: str, caption_path: str) -> str:
        """Apply movie-style enhancements to the generated captions."""
        if not self.enhancer:
            rich_console.print_info(f"Skipping movie-style enhancement for {video_id} - enhancer not available")
            return caption_path
        
        try:
            rich_console.print_info(f"Loading captions from {caption_path} for enhancement")
            # Load the generated captions
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption_data = json.load(f)
            
            # Get video and audio paths
            video_row = self.metadata_df[self.metadata_df['video_id'] == video_id]
            if len(video_row) == 0:
                rich_console.print_error(f"Video {video_id} not found in metadata")
                return caption_path
            
            video_path = video_row.iloc[0].get('file_path')
            audio_path = self._find_audio_path(video_id)
            
            rich_console.print_info(f"Video path: {video_path}, Audio path: {audio_path}")
            
            if not video_path or not os.path.exists(video_path):
                rich_console.print_warning(f"Video file not found for {video_id} at {video_path}, skipping enhancement")
                return caption_path
            
            # Enhance the segments
            segments = caption_data.get('segments', [])
            if segments:
                rich_console.print_info(f"Enhancing {len(segments)} segments for video {video_id}")
                enhanced_segments = self.enhancer.enhance_segments(segments, video_path, audio_path)
                
                # Convert enhanced segments back to standard format
                enhanced_caption_segments = []
                for enhanced_segment in enhanced_segments:
                    segment_dict = {
                        'start': enhanced_segment.start,
                        'end': enhanced_segment.end,
                        'text': enhanced_segment.enhanced_text or enhanced_segment.text
                    }
                    enhanced_caption_segments.append(segment_dict)
                
                # Update caption data
                caption_data['segments'] = enhanced_caption_segments
                caption_data['enhancement_info'] = {
                    'enhanced': True,
                    'movie_style': True,
                    'enhancement_timestamp': time.time()
                }
                
                # Save enhanced captions
                with open(caption_path, 'w', encoding='utf-8') as f:
                    json.dump(caption_data, f, ensure_ascii=False, indent=2)
                
                rich_console.print_info(f"Successfully enhanced captions for video {video_id}")
            else:
                rich_console.print_warning(f"No segments found for {video_id}")
            
        except Exception as e:
            rich_console.print_error(f"Error enhancing captions for video {video_id}: {e}")
            import traceback
            rich_console.print_error(traceback.format_exc())
            # Return the original captions if enhancement fails
        
        return caption_path
    
    def _split_segments(self, segments: List[Dict]) -> List[Dict]:
        """Split long segments into shorter ones for better readability."""
        if not self.enable_segment_splitting:
            return segments
        
        split_segments = []
        for segment in segments:
            duration = segment['end'] - segment['start']
            text_length = len(segment['text'])
            
            # Split if segment is too long (either by duration or text length)
            if duration > self.min_segment_duration * 3 or text_length > self.max_segment_length:
                # Simple text-based splitting for now
                words = segment['text'].split()
                if len(words) > 6:  # Only split if we have enough words
                    mid_point = len(words) // 2
                    mid_time = segment['start'] + (duration / 2)
                    
                    # Create two segments
                    first_segment = {
                        'start': segment['start'],
                        'end': mid_time,
                        'text': ' '.join(words[:mid_point])
                    }
                    second_segment = {
                        'start': mid_time,
                        'end': segment['end'],
                        'text': ' '.join(words[mid_point:])
                    }
                    
                    split_segments.extend([first_segment, second_segment])
                else:
                    split_segments.append(segment)
            else:
                split_segments.append(segment)
        
        return split_segments
    
    def batch_generate_video_descriptions(self, video_ids: List[str]) -> Dict[str, str]:
        """Generate video descriptions for a batch of videos that have completed captions."""
        if not self.enable_video_descriptions or not self.video_descriptor:
            rich_console.print_warning("Video descriptions are disabled or video descriptor not available")
            return {}
        
        rich_console.print_info(f"Starting batch video description generation for {len(video_ids)} videos")
        
        # Filter video IDs to only those that have completed captions
        videos_ready_for_descriptions = []
        for video_id in video_ids:
            caption_file = os.path.join(self.captions_dir, f"{video_id}.json")
            if os.path.exists(caption_file):
                videos_ready_for_descriptions.append(video_id)
            else:
                rich_console.print_warning(f"Skipping video {video_id} - no caption file found")
        
        if not videos_ready_for_descriptions:
            rich_console.print_warning("No videos ready for video description generation")
            return {}
        
        rich_console.print_info(f"Processing video descriptions for {len(videos_ready_for_descriptions)} videos with completed captions")
        
        # Use the video descriptor to process the batch
        try:
            results = self.video_descriptor.process_video_batch(videos_ready_for_descriptions, auto_metadata=True)
            
            successful_count = sum(1 for result in results.values() if result is not None)
            rich_console.print_info(f"Batch video description generation complete: {successful_count}/{len(videos_ready_for_descriptions)} videos processed successfully")
            
            return results
            
        except Exception as e:
            rich_console.print_error(f"Error during batch video description generation: {e}")
            return {}