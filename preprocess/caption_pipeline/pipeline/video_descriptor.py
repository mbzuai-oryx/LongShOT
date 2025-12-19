"""
Video descriptor module for generating visual descriptions using multimodal models.
This module processes video segments to create visual descriptions using Qwen 2.5-VL via vLLM.
"""

import os
import json
import logging
import base64
import time
import threading
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import pandas as pd
import sys
from io import BytesIO
from PIL import Image

# Import project configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import VIDEO_DESCRIPTION_MODEL, VIDEO_DIR, CAPTIONS_DIR, METADATA_DIR

# Import video utilities
from caption_pipeline.utils.video_utils import get_video_metadata

# Import rich console utilities
from caption_pipeline.utils.rich_console import get_console

# Import required components
from openai import OpenAI
import av

# Set up logging
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)
logger = logging.getLogger(__name__)
rich_console = get_console()

# Constants for video description
DEFAULT_MODEL = VIDEO_DESCRIPTION_MODEL
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_MAX_WORKERS = 8  # Increased for better throughput (adjust based on GPU memory)
MIN_SEGMENT_DURATION = 2.0  # Minimum segment duration for video extraction (shorter segments use single frame)
DEFAULT_MAX_FRAMES = 32  # Maximum frames per segment (balanced for performance and context)
DEFAULT_JPEG_QUALITY = 85  # JPEG compression quality (1-100, reduced for faster processing)
DEFAULT_SAMPLING_STRATEGY = 'adaptive'  # Fixed to adaptive for simplicity
GAP_MERGE_THRESHOLD = 5.0  # Threshold for intelligent gap merging (gaps < 5s merged, >= 5s become silent segments)
DESCRIPTION_PROMPT = """
You are an expert video analyst specializing in precise, objective descriptions for multimodal benchmarking across diverse video domains. Your task is to generate a detailed, temporally-aware visual description of the given video segment, capturing observable elements without inference or embellishment, applicable to any genre (e.g., sports, news, animation, documentaries).

Integrate the following aspects into coherent paragraphs:
Scene and Environment
Setting: Describe the location type (e.g., urban, studio, virtual), spatial layout, and key objects.
Atmosphere: Note lighting (e.g., natural, artificial), color tones, weather indicators (if applicable), and ambient visuals.
Composition: Detail placement of objects/subjects, depth, and framing.

Subjects and Actions
Appearance: Specify visible traits, attire, or defining features of people, characters, or entities.
Movements: Chronologically outline actions, gestures, or interactions.
Dynamics: Note positional shifts, groupings, or visible relationships between subjects.

Cinematic Elements
Camera: Describe angles (e.g., wide, close), movements, and transitions.
Visual Style: Cover color palette, balance, text overlays, or graphical effects.
Technical: Mention clarity, resolution, or production artifacts (e.g., compression, animation style).

Temporal Structure
Sequence: Follow the exact chronological flow of events within the segment.
Pacing: Indicate speed, rhythm, or notable shifts in tempo.
Evolution: Highlight changes from start to end, maintaining continuity with prior segments.


OUTPUT REQUIREMENTS
Produce a 100–200 word description that:
Uses neutral, factual paragraphs, avoiding lists, headings, or subjective language (e.g., no "dramatic," "beautiful").
Employs specific, descriptive terms (e.g., "red-shirted figure moves left" instead of "character dashes").
Adheres strictly to chronological order within the segment.
Prioritizes verifiable visuals for accurate reconstruction across domains.
Emphasizes unique segment details to ensure reproducibility for benchmarking.
Avoids domain-specific assumptions, ensuring applicability to any video type.
"""

class VideoDescriptor:
    """Class to generate visual descriptions for video segments using multimodal models.
    
    Uses enhanced video segment processing instead of individual frame extraction for:
    - Significantly faster processing (single ffmpeg call per segment vs multiple calls)
    - Reduced I/O operations and temporary file usage
    - Better temporal context preservation in video segments
    - More efficient vLLM server utilization
    
    Features intelligent gap elimination to ensure continuous temporal coverage:
    - Small gaps (< 5 seconds) are eliminated by extending segments backward
    - Large gaps (≥ 5 seconds) become separate silent segments  
    - Preserves all original audio content without merging
    - Eliminates timeline gaps for seamless video description coverage
    """
    
    def __init__(self, model_name=DEFAULT_MODEL, api_base=DEFAULT_API_BASE, max_workers=DEFAULT_MAX_WORKERS, 
                 max_frames=DEFAULT_MAX_FRAMES, jpeg_quality=DEFAULT_JPEG_QUALITY, 
                 segment_completion_callback=None):
        """Initialize VideoDescriptor with simplified configuration.
        
        Args:
            model_name: Name of the vision model to use
            api_base: Base URL for the vLLM server API
            max_workers: Max workers for parallel processing
            max_frames: Maximum frames to extract per segment
            jpeg_quality: JPEG compression quality (1-100)
            segment_completion_callback: Optional callback function called when each segment completes
                                       Should accept (video_id, video_path, segment_data) arguments
        """
        self.model_name = model_name
        self.api_base = api_base
        self.max_workers = max_workers
        self.max_frames = max_frames
        self.jpeg_quality = jpeg_quality
        self.segment_completion_callback = segment_completion_callback
        

        # Initialize paths from config
        self.video_dir = VIDEO_DIR
        self.captions_dir = CAPTIONS_DIR
        self.metadata_dir = METADATA_DIR
        self.video_descriptions_dir = os.path.join(os.path.dirname(CAPTIONS_DIR), 'video_descriptions')
        
        # Create video descriptions directory
        os.makedirs(self.video_descriptions_dir, exist_ok=True)
        
        # Load metadata
        self.metadata_file = os.path.join(self.metadata_dir, 'video_metadata.csv')
        self._load_metadata()
        
        # Test server connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to vLLM server."""
        try:
            client = OpenAI(api_key="EMPTY", base_url=self.api_base)
            client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": [{"type": "text", "text": "Test"}]}],
                max_tokens=5
            )
            rich_console.print_info(f"vLLM server connection successful at {self.api_base}")
        except Exception as e:
            rich_console.print_error(f"Failed to connect to vLLM server: {e}")
            raise
    
    def _load_metadata(self):
        """Load metadata to find video files."""
        if os.path.exists(self.metadata_file):
            try:
                self.metadata_df = pd.read_csv(self.metadata_file)
                rich_console.print_info(f"Loaded metadata for {len(self.metadata_df)} videos")
            except Exception as e:
                rich_console.print_error(f"Error loading metadata: {e}")
                self.metadata_df = pd.DataFrame()
        else:
            rich_console.print_warning("Metadata file not found")
            self.metadata_df = pd.DataFrame()
    
    def _find_video_path(self, video_id: str) -> Optional[str]:
        """Find the video path for a video ID."""
        # Check metadata first
        if not self.metadata_df.empty:
            video_data = self.metadata_df[self.metadata_df['video_id'] == video_id]
            if len(video_data) > 0 and 'file_path' in video_data.columns:
                video_path = video_data.iloc[0]['file_path']
                if pd.notna(video_path) and os.path.exists(video_path):
                    return video_path
        
        # Check for .mp4 file (most common format)
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        if os.path.exists(video_path):
            return video_path
        
        return None
    
    def _load_caption_segments(self, video_id: str) -> Optional[List[Dict]]:
        """Load caption segments from the generated captions file."""
        caption_file = os.path.join(self.captions_dir, f"{video_id}.json")
        
        if not os.path.exists(caption_file):
            rich_console.print_warning(f"Caption file not found for video {video_id}")
            return None
        
        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                caption_data = json.load(f)
            
            # Extract segments from the caption data
            if 'transcript' in caption_data and 'segments' in caption_data['transcript']:
                segments = caption_data['transcript']['segments']
                rich_console.print_info(f"Loaded {len(segments)} caption segments for video {video_id}")
                return segments
            else:
                rich_console.print_warning(f"No segments found in caption file for video {video_id}")
                return None
                
        except Exception as e:
            rich_console.print_error(f"Error loading caption segments for video {video_id}: {e}")
            return None
    
    

    def _extract_frames_as_base64(self, video_path: str, start_time: float, end_time: float, max_frames: int = 16, 
                                 jpeg_quality: int = 85) -> List[str]:
        """Extract frames from video segment using PyAV.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            max_frames: Maximum number of frames to extract
            jpeg_quality: JPEG compression quality (1-100)
        
        Returns:
            List of base64-encoded JPEG frames
        """
        duration = end_time - start_time
        
        # Process all segments, even very short ones (extract at least 1 frame)
        # Calculate number of frames (adaptive strategy)
        num_frames = self._calculate_optimal_frame_count(duration, max_frames)
        
        # Extract frames
        frame_base64_list = self._extract_frames_sequential(
            video_path, start_time, end_time, num_frames, jpeg_quality
        )
        
        return frame_base64_list

    def _calculate_optimal_frame_count(self, duration: float, max_frames: int) -> int:
        """Calculate optimal number of frames based on duration."""
        # Adaptive frame count based on duration, targeting 1-2fps
        # Always extract at least 1 frame, even for very short segments
        if duration <= 0.5:
            return 1
        elif duration <= 2.0:
            return min(4, max_frames)
        elif duration <= 5.0:
            return min(8, max_frames)
        elif duration <= 10.0:
            return min(16, max_frames)
        elif duration <= 20.0:
            return min(24, max_frames)
        else:
            return min(32, max_frames)

    def _extract_frames_sequential(self, video_path: str, start_time: float, end_time: float, 
                                 num_frames: int, jpeg_quality: int) -> List[str]:
        """Extract frames using sequential decoding."""
        frame_base64_list = []
        
        # Calculate target frame timestamps
        target_times = []
        for i in range(num_frames):
            if num_frames == 1:
                frame_time = start_time + (end_time - start_time) / 2
            else:
                frame_time = start_time + ((end_time - start_time) * i / (num_frames - 1))
            target_times.append(frame_time)
        
        try:
            with av.open(video_path) as container:
                video_stream = container.streams.video[0]
                
                # Seek to start time
                seek_timestamp = int(start_time / video_stream.time_base)
                container.seek(seek_timestamp, stream=video_stream, backward=True)
                
                # Decode frames
                target_index = 0
                tolerance = 0.2  # 200ms tolerance
                
                for frame in container.decode(video_stream):
                    if frame.pts is None:
                        continue
                        
                    frame_timestamp = float(frame.pts * frame.time_base)
                    
                    # Stop if past end time
                    if frame_timestamp > end_time + tolerance:
                        break
                    
                    # Skip frames before start time
                    if frame_timestamp < start_time - tolerance:
                        continue
                    
                    # Check if frame matches target time
                    if target_index < len(target_times):
                        time_diff = abs(frame_timestamp - target_times[target_index])
                        
                        if time_diff <= tolerance:
                            frame_base64 = self._frame_to_base64(frame, jpeg_quality)
                            if frame_base64:
                                frame_base64_list.append(frame_base64)
                                target_index += 1
                    
                    # Early exit if we have all frames
                    if target_index >= len(target_times):
                        break
                    
        except Exception as e:
            rich_console.print_error(f"Error extracting frames: {e}")
        
        return frame_base64_list


    def _frame_to_base64(self, frame, jpeg_quality: int) -> str:
        """Convert PyAV frame to base64 JPEG with optimizations."""
        try:
            # Convert frame to numpy array in RGB format
            rgb_array = frame.to_ndarray(format='rgb24')
            
            # Only resize if image is larger than 720p for faster model processing
            height, width = rgb_array.shape[:2]
            target_width, target_height = 1280, 720
            
            pil_image = Image.fromarray(rgb_array)
            
            # Only resize if the image is larger than 720p
            if height > target_height or width > target_width:
                # Calculate scale to maintain aspect ratio while fitting within 720p
                scale = min(target_width/width, target_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to base64 with optimized JPEG settings
            buffer = BytesIO()
            pil_image.save(buffer, format='JPEG', quality=jpeg_quality, optimize=True, progressive=True)
            frame_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return frame_base64
            
        except Exception as e:
            raise Exception(f"Frame conversion failed: {e}")


    
    def _save_segment_incrementally(self, output_file: str, video_id: str, video_path: str, 
                                  segment_data: Dict, is_first: bool = False) -> bool:
        """Save a single segment description incrementally to prevent data loss."""
        import tempfile
        
        try:
            if is_first:
                # Create backup of existing file if it exists
                if os.path.exists(output_file):
                    backup_file = f"{output_file}.backup_{int(time.time())}"
                    shutil.copy2(output_file, backup_file)
                    rich_console.print_info(f"Created backup: {backup_file}")
                
                # Initialize the JSON file with header information
                output_data = {
                    'video_id': video_id,
                    'video_path': video_path,
                    'model_used': self.model_name,
                    'generation_method': 'vllm_server_multi_frame_pyav_pipelined',
                    'max_workers': self.max_workers,
                    'segments': [segment_data],
                    'processing_info': {
                        'incremental_save': True,
                        'includes_silent_segments': True,
                        'uses_multi_frame_extraction': True,
                        'frame_extraction_method': 'PyAV_sequential',
                        'max_frames_per_segment': self.max_frames,
                        'jpeg_quality': self.jpeg_quality,
                        'sampling_strategy': 'adaptive',
                        'min_segment_duration': MIN_SEGMENT_DURATION,
                        'processing_started': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                }
                
                # Use atomic write with temporary file
                temp_file = output_file + '.tmp'
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                # Atomic move to final location
                shutil.move(temp_file, output_file)
                rich_console.print_info(f"Initialized incremental save file: {output_file}")
                
            else:
                # Read existing file and append new segment
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        output_data = json.load(f)
                except FileNotFoundError:
                    rich_console.print_error(f"Output file not found for incremental save: {output_file}")
                    return False
                except json.JSONDecodeError as e:
                    rich_console.print_error(f"Corrupted JSON file for incremental save: {output_file}, error: {e}")
                    return False
                
                # Append new segment
                output_data['segments'].append(segment_data)
                
                # Update last saved timestamp
                output_data['processing_info']['last_saved'] = time.strftime('%Y-%m-%d %H:%M:%S')
                
                # Use atomic write with temporary file
                temp_file = output_file + '.tmp'
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                # Atomic move to final location
                shutil.move(temp_file, output_file)
                
                # Trigger callback if this is a new segment (not first initialization)
                if not is_first and self.segment_completion_callback:
                    try:
                        self.segment_completion_callback(video_id, video_path, segment_data)
                    except Exception as e:
                        rich_console.print_warning(f"Segment completion callback error: {e}")
                
            return True
            
        except Exception as e:
            rich_console.print_error(f"Error saving segment incrementally: {e}")
            # Clean up temporary file if it exists
            temp_file = output_file + '.tmp'
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return False

    def _finalize_incremental_save(self, output_file: str) -> bool:
        """Finalize incremental save by updating completion info."""
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
            
            # Calculate segment statistics
            segments = output_data.get('segments', [])
            audio_segments = [s for s in segments if s.get('segment_type') == 'audio']
            silent_segments = [s for s in segments if s.get('segment_type') == 'silent']
            
            # Update final statistics
            output_data['total_segments'] = len(segments)
            output_data['audio_segments'] = len(audio_segments)
            output_data['silent_segments'] = len(silent_segments)
            output_data['processing_info'].update({
                'processing_completed': True,
                'completion_time': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Use atomic write with temporary file
            temp_file = output_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            # Atomic move to final location
            shutil.move(temp_file, output_file)
            rich_console.print_info(f"Finalized incremental save for {output_data.get('video_id', 'unknown')}")
            
            return True
            
        except Exception as e:
            rich_console.print_error(f"Error finalizing incremental save: {e}")
            # Clean up temporary file if it exists
            temp_file = output_file + '.tmp'
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return False

    def _check_existing_progress(self, output_file: str) -> Optional[Dict]:
        """Check if there's existing progress for incremental processing."""
        if not os.path.exists(output_file):
            return None
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if processing was completed
            processing_info = data.get('processing_info', {})
            if processing_info.get('processing_completed', False):
                rich_console.print_info(f"Processing already completed for {data.get('video_id', 'unknown')}")
                return data
            
            # Check if there are existing segments
            segments = data.get('segments', [])
            if segments:
                rich_console.print_info(f"Found existing progress: {len(segments)} segments already processed")
                return data
            
            return None
            
        except (json.JSONDecodeError, Exception) as e:
            rich_console.print_warning(f"Could not read existing progress file: {e}")
            return None

    def _resume_processing_from_checkpoint(self, output_file: str, all_segments: List[Dict]) -> List[Dict]:
        """Resume processing from existing checkpoint."""
        existing_data = self._check_existing_progress(output_file)
        if not existing_data:
            return []
        
        existing_segments = existing_data.get('segments', [])
        processed_count = len(existing_segments)
        
        rich_console.print_info(f"Resuming from checkpoint: {processed_count}/{len(all_segments)} segments already processed")
        
        return existing_segments


    def _generate_description_from_frames(self, frame_base64_list: List[str], duration: float, frame_count: int, previous_description: str = None) -> str:
        """Generate description from extracted frames using vLLM server."""
        try:
            client = OpenAI(api_key="EMPTY", base_url=self.api_base)
            
            # Create content for API call with multiple frames
            prompt_text = DESCRIPTION_PROMPT
            
            # Add previous segment description if available and this is not the first segment
            if previous_description:
                prompt_text += f"\n\n**Previous Segment Description:**\n{previous_description}\n\n"
            
            prompt_text += f"\n\nAnalyze these {frame_count} frames from a {duration:.1f}s video segment. Describe the temporal progression and any changes across the frames:"
            
            content = [{"type": "text", "text": prompt_text}]
            
            # Add each frame as an image
            for i, frame_base64 in enumerate(frame_base64_list):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
                })
            
            rich_console.print_info(f"Processing {frame_count} frames from {duration:.1f}s segment with vLLM")
            
            # Make request to vLLM server
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=1024,
                temperature=0.5
            )
            
            description = response.choices[0].message.content.strip()
            return description
            
        except Exception as e:
            rich_console.print_error(f"Error generating description from frames: {e}")
            return f"Error generating description: {str(e)}"

    def _generate_batch_descriptions(self, batch_requests: List[Dict]) -> List[str]:
        """Generate descriptions for multiple segments using concurrent requests to vLLM server."""
        descriptions = []
        
        if not batch_requests:
            return descriptions
        
        rich_console.print_info(f"Processing batch of {len(batch_requests)} segments with concurrent vLLM requests")
        
        # Use ThreadPoolExecutor to send concurrent requests - increased concurrency
        max_concurrent = min(len(batch_requests), self.max_workers)  # Use max_workers for consistency
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all requests concurrently
            future_to_index = {}
            for request in batch_requests:
                future = executor.submit(self._generate_single_description_from_request, request)
                future_to_index[future] = request.get('segment_index', 0)
            
            # Initialize results with None values
            descriptions = [None] * len(batch_requests)
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    description = future.result()
                    descriptions[index] = description
                    completed_count += 1
                    rich_console.print_info(f"Completed vLLM request {completed_count}/{len(batch_requests)}")
                except Exception as e:
                    rich_console.print_error(f"Error generating description for request {index}: {e}")
                    descriptions[index] = f"Error generating description: {str(e)}"
        
        return descriptions

    def _process_segments_optimized(self, video_path: str, video_id: str, segments: List[Dict], output_file: str, batch_size: int = 4) -> List[Dict]:
        """Optimized pipelined segment processing with overlapped frame extraction and vLLM requests."""
        described_segments = []
        
        # Always create progress bar for video description (no more concurrent check)
        progress = None
        task_id = None
        
        try:
            progress, task_id = rich_console.create_video_description_progress(video_id, len(segments))
        except Exception as e:
            rich_console.print_warning(f"Could not create progress bar for {video_id}: {e}")
        
        # Process segments with progress display
        rich_console.print_info(f"Processing {len(segments)} segments with pipelined frame extraction and vLLM requests (batch_size={batch_size})")
        
        # Use progress context if available
        if progress:
            with progress:
                described_segments = self._process_segments_pipelined_with_progress(
                    video_path, video_id, segments, output_file, progress, task_id
                )
        else:
            described_segments = self._process_segments_pipelined_without_progress(
                video_path, video_id, segments, output_file
            )
        
        return described_segments



    def _process_segments_pipelined_with_progress(self, video_path: str, video_id: str, segments: List[Dict], 
                                                output_file: str, progress, task_id) -> List[Dict]:
        """Pipelined processing: send individual requests to vLLM as workers become available."""
        described_segments = []
        
        # Worker status tracking
        frame_workers_busy = 0
        vllm_workers_busy = 0
        segments_completed = 0
        frames_extracted = 0
        vllm_requests_sent = 0
        
        rich_console.print_info("Starting pipelined processing with dynamic worker utilization")
        rich_console.print_info(f"Initialized {self.max_workers} frame workers and {self.max_workers} vLLM workers")
        
        # Start frame extraction and vLLM processing with dynamic worker pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as frame_executor:
            with ThreadPoolExecutor(max_workers=self.max_workers) as vllm_executor:
                # Submit frame extraction tasks
                frame_futures = [
                    frame_executor.submit(self._extract_segment_frames, video_path, segment, i)
                    for i, segment in enumerate(segments)
                ]
                frame_workers_busy = len(frame_futures)
                
                # Track vLLM futures and their corresponding segment data
                vllm_futures = {}
                completed_segments = {}
                
                self._print_worker_status("INITIAL", frame_workers_busy, vllm_workers_busy, frames_extracted, vllm_requests_sent, segments_completed, len(segments))
                
                # Process frame extraction results as they complete
                for frame_future in as_completed(frame_futures):
                    try:
                        frame_data = frame_future.result()
                        frame_workers_busy -= 1
                        frames_extracted += 1
                        
                        # Send to vLLM immediately if frames are available
                        if frame_data['frames']:
                            # Previous segment descriptions not available in concurrent processing
                            previous_description = None
                            
                            vllm_future = vllm_executor.submit(
                                self._generate_single_description_from_request,
                                {
                                    'frames': frame_data['frames'],
                                    'duration': frame_data['duration'],
                                    'frame_count': frame_data['frame_count'],
                                    'previous_description': previous_description
                                }
                            )
                            vllm_futures[vllm_future] = frame_data
                            vllm_workers_busy += 1
                            vllm_requests_sent += 1
                            
                            self._print_worker_status("FRAME→vLLM", frame_workers_busy, vllm_workers_busy, frames_extracted, vllm_requests_sent, segments_completed, len(segments))
                        else:
                            # Handle segments without frames immediately
                            no_frames_description = "No frames could be extracted from this video segment"
                            segment_result = self._create_segment_result(
                                frame_data, no_frames_description, 'no_frames_extracted'
                            )
                            completed_segments[frame_data['segment_index']] = segment_result
                            
                            
                            segments_completed += 1
                            frame_data['frames'].clear()
                            
                            self._print_worker_status("FRAME_DONE", frame_workers_busy, vllm_workers_busy, frames_extracted, vllm_requests_sent, segments_completed, len(segments))
                        
                    except Exception as e:
                        frame_workers_busy -= 1
                        rich_console.print_error(f"Frame extraction error: {e}")
                
                # Process vLLM results as they complete
                for vllm_future in as_completed(vllm_futures):
                    try:
                        frame_data = vllm_futures[vllm_future]
                        description = vllm_future.result()
                        vllm_workers_busy -= 1
                        segments_completed += 1
                        
                        # Create segment result
                        segment_result = self._create_segment_result(
                            frame_data, description, 'multi_frame_pipelined_concurrent'
                        )
                        completed_segments[frame_data['segment_index']] = segment_result
                        
                        
                        # Free memory immediately
                        frame_data['frames'].clear()
                        
                        self._print_worker_status("vLLM_DONE", frame_workers_busy, vllm_workers_busy, frames_extracted, vllm_requests_sent, segments_completed, len(segments))
                        
                    except Exception as e:
                        frame_data = vllm_futures[vllm_future]
                        vllm_workers_busy -= 1
                        segments_completed += 1
                        error_description = f"Error generating description: {e}"
                        segment_result = self._create_segment_result(
                            frame_data, error_description, 'generation_error'
                        )
                        completed_segments[frame_data['segment_index']] = segment_result
                        
                        
                        frame_data['frames'].clear()
                        rich_console.print_error(f"vLLM error: {e}")
                
                # Sort and save segments in order
                rich_console.print_info("Finalizing segments in chronological order")
                for i in range(len(segments)):
                    if i in completed_segments:
                        segment_result = completed_segments[i]
                        described_segments.append(segment_result)
                        
                        # Save incrementally and update progress
                        is_first = len(described_segments) == 1
                        if not self._save_segment_incrementally(output_file, video_id, video_path, segment_result, is_first):
                            rich_console.print_error(f"Failed to save segment incrementally for {video_id}. Stopping processing to prevent data loss.")
                            break
                        progress.update(task_id, advance=1)
                
                rich_console.print_success(f"Pipeline completed: {segments_completed}/{len(segments)} segments processed")
        
        return described_segments

    def _process_segments_pipelined_without_progress(self, video_path: str, video_id: str, segments: List[Dict], 
                                                   output_file: str) -> List[Dict]:
        """Pipelined processing: send individual requests to vLLM as workers become available."""
        described_segments = []
        
        # Worker status tracking
        frame_workers_busy = 0
        vllm_workers_busy = 0
        segments_completed = 0
        frames_extracted = 0
        vllm_requests_sent = 0
        
        rich_console.print_info("Starting pipelined processing with dynamic worker utilization")
        rich_console.print_info(f"Initialized {self.max_workers} frame workers and {self.max_workers} vLLM workers")
        
        # Start frame extraction and vLLM processing with dynamic worker pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as frame_executor:
            with ThreadPoolExecutor(max_workers=self.max_workers) as vllm_executor:
                # Submit frame extraction tasks
                frame_futures = [
                    frame_executor.submit(self._extract_segment_frames, video_path, segment, i)
                    for i, segment in enumerate(segments)
                ]
                frame_workers_busy = len(frame_futures)
                
                # Track vLLM futures and their corresponding segment data
                vllm_futures = {}
                completed_segments = {}
                
                self._print_worker_status("INITIAL", frame_workers_busy, vllm_workers_busy, frames_extracted, vllm_requests_sent, segments_completed, len(segments))
                
                # Process frame extraction results as they complete
                for frame_future in as_completed(frame_futures):
                    try:
                        frame_data = frame_future.result()
                        frame_workers_busy -= 1
                        frames_extracted += 1
                        
                        # Send to vLLM immediately if frames are available
                        if frame_data['frames']:
                            # Previous segment descriptions not available in concurrent processing
                            previous_description = None
                            
                            vllm_future = vllm_executor.submit(
                                self._generate_single_description_from_request,
                                {
                                    'frames': frame_data['frames'],
                                    'duration': frame_data['duration'],
                                    'frame_count': frame_data['frame_count'],
                                    'previous_description': previous_description
                                }
                            )
                            vllm_futures[vllm_future] = frame_data
                            vllm_workers_busy += 1
                            vllm_requests_sent += 1
                            
                            self._print_worker_status("FRAME→vLLM", frame_workers_busy, vllm_workers_busy, frames_extracted, vllm_requests_sent, segments_completed, len(segments))
                        else:
                            # Handle segments without frames immediately
                            no_frames_description = "No frames could be extracted from this video segment"
                            segment_result = self._create_segment_result(
                                frame_data, no_frames_description, 'no_frames_extracted'
                            )
                            completed_segments[frame_data['segment_index']] = segment_result
                            
                            
                            segments_completed += 1
                            frame_data['frames'].clear()
                            
                            self._print_worker_status("FRAME_DONE", frame_workers_busy, vllm_workers_busy, frames_extracted, vllm_requests_sent, segments_completed, len(segments))
                        
                    except Exception as e:
                        frame_workers_busy -= 1
                        rich_console.print_error(f"Frame extraction error: {e}")
                
                # Process vLLM results as they complete
                for vllm_future in as_completed(vllm_futures):
                    try:
                        frame_data = vllm_futures[vllm_future]
                        description = vllm_future.result()
                        vllm_workers_busy -= 1
                        segments_completed += 1
                        
                        # Create segment result
                        segment_result = self._create_segment_result(
                            frame_data, description, 'multi_frame_pipelined_concurrent'
                        )
                        completed_segments[frame_data['segment_index']] = segment_result
                        
                        
                        # Free memory immediately
                        frame_data['frames'].clear()
                        
                        self._print_worker_status("vLLM_DONE", frame_workers_busy, vllm_workers_busy, frames_extracted, vllm_requests_sent, segments_completed, len(segments))
                        
                    except Exception as e:
                        frame_data = vllm_futures[vllm_future]
                        vllm_workers_busy -= 1
                        segments_completed += 1
                        error_description = f"Error generating description: {e}"
                        segment_result = self._create_segment_result(
                            frame_data, error_description, 'generation_error'
                        )
                        completed_segments[frame_data['segment_index']] = segment_result
                        
                        
                        frame_data['frames'].clear()
                        rich_console.print_error(f"vLLM error: {e}")
                
                # Sort and save segments in order
                rich_console.print_info("Finalizing segments in chronological order")
                for i in range(len(segments)):
                    if i in completed_segments:
                        segment_result = completed_segments[i]
                        described_segments.append(segment_result)
                        
                        # Save incrementally
                        is_first = len(described_segments) == 1
                        if not self._save_segment_incrementally(output_file, video_id, video_path, segment_result, is_first):
                            rich_console.print_error(f"Failed to save segment incrementally for {video_id}. Stopping processing to prevent data loss.")
                            break
                
                rich_console.print_success(f"Pipeline completed: {segments_completed}/{len(segments)} segments processed")
        
        return described_segments


    def _print_worker_status(self, event: str, frame_busy: int, vllm_busy: int, frames_done: int, vllm_sent: int, completed: int, total: int):
        """Print simple worker status updates."""
        # Only print for key events to reduce verbosity
        if event in ["INITIAL", "vLLM_DONE"] or (completed > 0 and completed % 5 == 0):
            status_msg = f"Workers: Frame {frame_busy}/{self.max_workers}, vLLM {vllm_busy}/{self.max_workers} | Progress: {completed}/{total}"
            rich_console.print_info(status_msg)

    def _create_segment_result(self, frame_data: Dict, description: str, processing_method: str) -> Dict:
        """Create a segment result dictionary."""
        segment = frame_data['segment']
        return {
            'start': segment.get('start', 0),
            'end': segment.get('end', 0),
            'visual_description': description,
            'audio_text': segment.get('text', ''),
            'segment_type': segment.get('segment_type', 'audio'),
            'processing_method': processing_method,
            'frames_extracted': frame_data['frame_count']
        }


    def _generate_video_descriptions_concurrent(self, video_id: str, batch_size: int = 4, min_silent_duration: float = 2.0) -> Optional[str]:
        """Wrapper for generate_video_descriptions that marks the thread as concurrent."""
        # Mark this thread as concurrent processing
        threading.current_thread()._is_concurrent_video_processing = True
        try:
            return self.generate_video_descriptions(video_id, min_silent_duration, batch_size)
        finally:
            # Clean up the marker
            if hasattr(threading.current_thread(), '_is_concurrent_video_processing'):
                delattr(threading.current_thread(), '_is_concurrent_video_processing')

    def _extract_segment_frames(self, video_path: str, segment: Dict, segment_index: int) -> Dict:
        """Extract frames for a single segment and return frame data."""
        start_time = segment.get('start', 0)
        end_time = segment.get('end', start_time + 1)
        duration = end_time - start_time
        
        try:
            # Extract frames for this segment
            frame_base64_list = self._extract_frames_as_base64(
                video_path, start_time, end_time, 
                max_frames=self.max_frames,
                jpeg_quality=self.jpeg_quality
            )
            
            return {
                'segment_index': segment_index,
                'segment': segment,
                'frames': frame_base64_list,
                'frame_count': len(frame_base64_list),
                'duration': duration
            }
            
        except Exception as e:
            rich_console.print_error(f"Error extracting frames for segment {segment_index}: {e}")
            return {
                'segment_index': segment_index,
                'segment': segment,
                'frames': [],
                'frame_count': 0,
                'duration': duration,
                'error': str(e)
            }

    def _generate_single_description_from_request(self, request: Dict) -> str:
        """Generate description for a single request with frames."""
        try:
            client = OpenAI(api_key="EMPTY", base_url=self.api_base)
            
            frame_base64_list = request['frames']
            duration = request['duration']
            frame_count = request['frame_count']
            previous_description = request.get('previous_description', None)
            
            # Create content for this segment with optional previous description
            prompt_text = DESCRIPTION_PROMPT
            
            # Add previous segment description if available and this is not the first segment
            if previous_description:
                prompt_text += f"\n\n**Previous Segment Description:**\n{previous_description}\n\n"
            
            prompt_text += f"\n\nAnalyze these {frame_count} frames from a {duration:.1f}s video segment. Describe the temporal progression and any changes across the frames:"
            
            content = [{"type": "text", "text": prompt_text}]
            
            # Add each frame as an image
            for frame_base64 in frame_base64_list:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
                })
            
            # Make request to vLLM server
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=1024,
                temperature=0.5
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"Error generating description: {e}")


    def generate_video_descriptions(self, video_id: str, min_silent_duration: float = 2.0, batch_size: int = 8) -> Optional[str]:
        """Generate visual descriptions for all segments of a video using concurrent processing.
        
        Args:
            video_id: ID of the video to process
            min_silent_duration: Minimum duration (in seconds) for a silent segment to be included
            batch_size: Number of segments to process in each vLLM batch
        """
        rich_console.print_component_header("Video Description", f"Processing {video_id} with video segment extraction")
        processing_start_time = time.time()
        
        
        # Find video path
        video_path = self._find_video_path(video_id)
        if not video_path:
            rich_console.print_error(f"Video file not found for {video_id}")
            return None
        
        # Create complete segments list (audio + silent)
        segments = self._create_complete_segments_list(video_id, video_path, min_silent_duration)
        if not segments:
            rich_console.print_error(f"No segments found for {video_id}")
            return None
        
        rich_console.print_info(f"Processing {len(segments)} segments for video {video_id} with batch_size={batch_size}")
        
        # Prepare output file
        output_file = os.path.join(self.video_descriptions_dir, f"{video_id}_descriptions.json")
        
        # Check for existing progress or completed processing
        existing_progress = self._check_existing_progress(output_file)
        if existing_progress:
            # If processing was completed, return the file
            if existing_progress.get('processing_info', {}).get('processing_completed', False):
                return output_file
            else:
                rich_console.print_info(f"Resuming interrupted processing for {video_id}")
                # Continue with processing - the pipeline will handle resuming
        
        # Use optimized concurrent processing with batching
        # Ensure batch_size is reasonable for the number of segments
        batch_size = min(batch_size, len(segments)) if len(segments) > 0 else batch_size
        
        described_segments = self._process_segments_optimized(
            video_path, video_id, segments, output_file, batch_size=batch_size
        )
        
        # Sort segments by start time
        described_segments.sort(key=lambda x: x.get('start', 0))
        
        # Finalize incremental save
        if self._finalize_incremental_save(output_file):
            processing_time = time.time() - processing_start_time
            total_frames = sum(seg.get('frames_extracted', 0) for seg in described_segments)
            method = "PyAV-pipelined multi-frame concurrent"
            rich_console.print_success(f"Video descriptions completed for {video_id} in {processing_time:.1f}s using {method} extraction")
            rich_console.print_info(f"Processed {len(segments)} segments with {total_frames} total frames extracted")
            return output_file
        else:
            rich_console.print_error(f"Failed to finalize incremental save for {video_id}")
            return None

    def batch_generate_descriptions(self, max_videos: int = None, auto_metadata: bool = True, batch_size: int = 8) -> List[str]:
        """Generate descriptions for multiple videos sequentially with progress tracking.
        
        Uses PyAV for high-performance in-memory frame extraction (4-16 frames per segment) 
        instead of single frame or video segment processing for better temporal understanding.
        """
        
        rich_console.print_component_header("Batch Video Description", f"Processing multiple videos sequentially with enhanced video segment extraction (batch_size={batch_size})")
        
        # Get list of videos that have captions but no descriptions yet
        videos_to_process = []
        
        if not self.metadata_df.empty:
            for _, row in self.metadata_df.iterrows():
                video_id = row['video_id']
                
                # Check if captions exist
                caption_file = os.path.join(self.captions_dir, f"{video_id}.json")
                description_file = os.path.join(self.video_descriptions_dir, f"{video_id}_descriptions.json")
                
                if os.path.exists(caption_file) and not os.path.exists(description_file):
                    videos_to_process.append(video_id)
        
        if max_videos:
            videos_to_process = videos_to_process[:max_videos]
        
        extraction_method = "PyAV multi-frame pipelined"
        rich_console.print_info(f"Processing video descriptions for {len(videos_to_process)} videos using {extraction_method} extraction (sequential processing)")
        
        description_files = []
        failed_videos = []
        
        # Process videos with controlled concurrency for better performance
        max_concurrent_videos = min(2, len(videos_to_process))  # Process up to 2 videos concurrently
        rich_console.print_info(f"Processing {len(videos_to_process)} videos with max concurrency of {max_concurrent_videos}")
        
        if max_concurrent_videos > 1 and len(videos_to_process) > 1:
            # Use concurrent processing for multiple videos
            with ThreadPoolExecutor(max_workers=max_concurrent_videos) as executor:
                # Submit all video processing tasks
                video_futures = {
                    executor.submit(self.generate_video_descriptions, video_id, batch_size=batch_size): video_id
                    for video_id in videos_to_process
                }
                
                # Process results as they complete
                for i, future in enumerate(as_completed(video_futures), 1):
                    video_id = video_futures[future]
                    try:
                        description_file = future.result()
                        if description_file:
                            description_files.append(description_file)
                            rich_console.print_success(f"✓ Completed video descriptions for {video_id} ({i}/{len(videos_to_process)})")
                        else:
                            failed_videos.append(video_id)
                            rich_console.print_error(f"✗ Failed video descriptions for {video_id} ({i}/{len(videos_to_process)})")
                    except Exception as e:
                        failed_videos.append(video_id)
                        rich_console.print_error(f"✗ Error processing video {video_id}: {e}")
        else:
            # Sequential processing for single video or when concurrency is disabled
            for i, video_id in enumerate(videos_to_process, 1):
                rich_console.print_info(f"Processing video {i}/{len(videos_to_process)}: {video_id}")
                
                try:
                    description_file = self.generate_video_descriptions(video_id, batch_size=batch_size)
                    if description_file:
                        description_files.append(description_file)
                        rich_console.print_success(f"✓ Completed video descriptions for {video_id} ({i}/{len(videos_to_process)})")
                    else:
                        failed_videos.append(video_id)
                        rich_console.print_error(f"✗ Failed video descriptions for {video_id} ({i}/{len(videos_to_process)})")
                except Exception as e:
                    failed_videos.append(video_id)
                    rich_console.print_error(f"✗ Error processing video {video_id}: {e}")
                    continue
        
        # Print completion summary
        rich_console.print_completion_message("Video Description Generation", {
            'total': len(videos_to_process),
            'successful': len(description_files),
            'duration': 0  # Duration calculated externally
        })
        
        # Automatically trigger metadata generation if enabled and we have successful results
        if auto_metadata and description_files:
            rich_console.print_info("Automatically triggering metadata generation for videos with descriptions...")
            try:
                # Extract video IDs from description files
                video_ids = [os.path.basename(f).replace('_descriptions.json', '') for f in description_files]
                self._trigger_metadata_generation(video_ids)
            except Exception as e:
                rich_console.print_warning(f"Failed to automatically generate metadata: {e}")
                rich_console.print_info("You can manually run metadata generation later with: python run_metadata_generation.py")
        
        return description_files

    def process_video_batch(self, video_ids: List[str], auto_metadata: bool = True, batch_size: int = 8) -> Dict[str, str]:
        """Process a batch of videos to generate video descriptions sequentially."""
        
        results = {}
        rich_console.print_info(f"Processing video descriptions for {len(video_ids)} videos sequentially with batch_size={batch_size}")
        
        for i, video_id in enumerate(video_ids, 1):
            rich_console.print_info(f"Processing video {i}/{len(video_ids)}: {video_id}")
            
            try:
                description_file = self.generate_video_descriptions(video_id, batch_size=batch_size)
                if description_file:
                    results[video_id] = description_file
                    rich_console.print_success(f"✓ Successfully generated video descriptions for {video_id} ({i}/{len(video_ids)})")
                else:
                    results[video_id] = None
                    rich_console.print_error(f"✗ Failed to generate video descriptions for {video_id} ({i}/{len(video_ids)})")
            except Exception as e:
                rich_console.print_error(f"✗ Error processing video {video_id}: {e}")
                results[video_id] = None
        
        successful_count = sum(1 for result in results.values() if result is not None)
        rich_console.print_info(f"Video description generation complete: {successful_count}/{len(video_ids)} videos processed successfully")
        
        # Automatically trigger metadata generation if enabled and we have successful results
        if auto_metadata and successful_count > 0:
            rich_console.print_info("Automatically triggering metadata generation for videos with descriptions...")
            try:
                successful_video_ids = [vid for vid, result in results.items() if result is not None]
                self._trigger_metadata_generation(successful_video_ids)
            except Exception as e:
                rich_console.print_warning(f"Failed to automatically generate metadata: {e}")
                rich_console.print_info("You can manually run metadata generation later with: python run_metadata_generation.py")
        
        return results

    def _create_silent_segments(self, audio_segments: List[Dict], video_duration: float, min_silent_duration: float = 2.0) -> List[Dict]:
        """Create segments for silent parts of the video with improved gap handling."""
        if not audio_segments:
            # If no audio segments, create one segment for the entire video
            return [{
                'start': 0.0,
                'end': video_duration,
                'text': '',
                'segment_type': 'silent'
            }]

        silent_segments = []

        # Sort audio segments by start time
        audio_segments_sorted = sorted(audio_segments, key=lambda x: x.get('start', 0))

        # Check for silence at the beginning (only if significant)
        first_start = audio_segments_sorted[0].get('start', 0)
        if first_start >= GAP_MERGE_THRESHOLD:
            silent_segments.append({
                'start': 0.0,
                'end': first_start,
                'text': '',
                'segment_type': 'silent'
            })

        # Check for silence between segments (only for large gaps)
        for i in range(len(audio_segments_sorted) - 1):
            current_end = audio_segments_sorted[i].get('end', 0)
            next_start = audio_segments_sorted[i + 1].get('start', 0)
            gap_duration = next_start - current_end

            # Only create silent segments for large gaps
            # Small gaps will be handled by intelligent merging
            if gap_duration >= GAP_MERGE_THRESHOLD:
                silent_segments.append({
                    'start': current_end,
                    'end': next_start,
                    'text': '',
                    'segment_type': 'silent'
                })

        # Check for silence at the end (only if significant)
        last_end = audio_segments_sorted[-1].get('end', 0)
        final_gap = video_duration - last_end
        if final_gap >= GAP_MERGE_THRESHOLD:
            silent_segments.append({
                'start': last_end,
                'end': video_duration,
                'text': '',
                'segment_type': 'silent'
            })

        return silent_segments

    def _create_complete_segments_list(self, video_id: str, video_path: str, min_silent_duration: float = 2.0) -> Optional[List[Dict]]:
        """Create a complete list of segments including both audio and silent segments."""
        # Load audio segments
        audio_segments = self._load_caption_segments(video_id)
        if audio_segments is None:
            audio_segments = []

        # Add segment type to audio segments
        for segment in audio_segments:
            segment['segment_type'] = 'audio'

        # Get video duration
        try:
            video_metadata = get_video_metadata(video_path)
            video_duration = video_metadata.get('duration', 0)
            if video_duration <= 0:
                rich_console.print_error(f"Could not get video duration for {video_id}")
                return None
        except Exception as e:
            rich_console.print_error(f"Error getting video metadata for {video_id}: {e}")
            return None

        # Create silent segments
        silent_segments = self._create_silent_segments(audio_segments, video_duration, min_silent_duration)

        # Combine and sort all segments
        all_segments = audio_segments + silent_segments
        all_segments.sort(key=lambda x: x.get('start', 0))

        # Apply intelligent gap elimination by extending segments
        all_segments = self._eliminate_gaps_by_extending_segments(all_segments, min_silent_duration)

        rich_console.print_info(f"Created complete segments list: {len(audio_segments)} audio + {len(silent_segments)} silent = {len(all_segments)} total segments (after gap elimination)")

        return all_segments

    def _eliminate_gaps_by_extending_segments(self, segments: List[Dict], min_silent_duration: float = 2.0) -> List[Dict]:
        """
        Eliminate gaps between segments by extending segments backward or creating silent segments.
        
        Strategy:
        1. If gap < 5 seconds: extend the later segment backward to eliminate the gap
        2. If gap ≥ 5 seconds: create a separate silent segment to fill the gap
        
        Example:
        - Segment 1: 0-10.2s, Segment 2: 10.9-20s (gap: 0.7s) → Extend Segment 2 to 10.2-20s
        - Segment 1: 0-10s, Segment 2: 16s-20s (gap: 6s) → Create silent segment 10-16s
        
        Args:
            segments: List of segments sorted by start time
            min_silent_duration: Minimum duration for creating silent segments (not used in this approach)
            
        Returns:
            List of segments with gaps eliminated
        """
        if len(segments) <= 1:
            return segments
            
        result_segments = []
        gaps_extended = 0
        gaps_filled_with_silent = 0
        GAP_EXTEND_THRESHOLD = 5.0  # Extend segments for gaps < 5s, create silent segments for gaps ≥ 5s
        
        for i, segment in enumerate(segments):
            current_start = segment.get('start', 0)
            current_end = segment.get('end', 0)
            
            # Add the first segment as-is
            if i == 0:
                result_segments.append(segment.copy())
                continue
                
            # Get the previous segment from result list
            prev_segment = result_segments[-1]
            prev_end = prev_segment.get('end', 0)
            
            # Calculate gap between previous and current segment
            gap_duration = current_start - prev_end
            
            if gap_duration <= 0:
                # No gap or overlapping segments, just add current segment
                result_segments.append(segment.copy())
            elif gap_duration < GAP_EXTEND_THRESHOLD:
                # Small gap: extend current segment backward to eliminate the gap
                extended_segment = segment.copy()
                extended_segment['start'] = prev_end  # Extend backward to eliminate gap
                result_segments.append(extended_segment)
                gaps_extended += 1
                # rich_console.print_info(f"Extended segment backward: {current_start:.2f} → {prev_end:.2f} (eliminated {gap_duration:.2f}s gap)")
            else:
                # Large gap: create a silent segment to fill the gap
                gap_segment = {
                    'start': prev_end,
                    'end': current_start,
                    'text': '',
                    'segment_type': 'silent'
                }
                result_segments.append(gap_segment)
                gaps_filled_with_silent += 1
                # rich_console.print_info(f"Created silent segment for large gap: {prev_end:.2f} → {current_start:.2f} ({gap_duration:.2f}s)")
                
                # Add the current segment unchanged
                result_segments.append(segment.copy())
        
        # Log the gap elimination results
        original_count = len(segments)
        final_count = len(result_segments)
        rich_console.print_info(f"Gap elimination: {gaps_extended} segments extended, {gaps_filled_with_silent} silent segments created")
        rich_console.print_info(f"Final result: {original_count} original segments → {final_count} total segments")
        
        return result_segments
    def _trigger_metadata_generation(self, video_ids: List[str]) -> bool:
        """
        Trigger metadata generation for videos that have completed video descriptions.
        
        Args:
            video_ids: List of video IDs to generate metadata for
            
        Returns:
            True if metadata generation was successful, False otherwise
        """
        try:
            # Import here to avoid circular imports
            from caption_pipeline.pipeline.metadata_generator import VideoMetadataGenerator
            from config import VLLM_SERVER_URL
            
            rich_console.print_info(f"Starting automatic metadata generation for {len(video_ids)} videos...")
            
            # Initialize metadata generator
            metadata_generator = VideoMetadataGenerator(
                api_base=VLLM_SERVER_URL,
                max_workers=self.max_workers
            )
            
            # Generate metadata for the videos
            metadata_results = metadata_generator.process_videos(
                video_ids=video_ids
            )
            
            # Log results
            successful_metadata = sum(1 for result in metadata_results.values() if result)
            total_metadata = len(metadata_results)
            
            rich_console.print_info(f"Metadata generation completed: {successful_metadata}/{total_metadata} videos processed successfully")
            
            return successful_metadata > 0
            
        except Exception as e:
            rich_console.print_error(f"Error during automatic metadata generation: {e}")
            return False


def main():
    """Main function for testing the video descriptor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate video descriptions for Arabic videos')
    parser.add_argument('--video-id', type=str, help='Specific video ID to process')
    parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Vision model to use')
    parser.add_argument('--api-base', type=str, default=DEFAULT_API_BASE, help='vLLM server API base URL')
    parser.add_argument('--max-workers', type=int, default=DEFAULT_MAX_WORKERS, help='Maximum number of concurrent workers')
    parser.add_argument('--min-silent-duration', type=float, default=2.0, help='Minimum duration (seconds) for silent segments')
    parser.add_argument('--batch-size', type=int, default=8, help='Number of segments to process in each vLLM batch')
    
    args = parser.parse_args()
    
    rich_console = get_console()
    
    # Initialize video descriptor
    descriptor = VideoDescriptor(
        model_name=args.model,
        api_base=args.api_base,
        max_workers=args.max_workers
    )
    
    if args.video_id:
        # Process single video
        result = descriptor.generate_video_descriptions(
            args.video_id, 
            min_silent_duration=args.min_silent_duration,
            batch_size=args.batch_size
        )
        if result:
            rich_console.print_info(f"Successfully generated descriptions for {args.video_id}")
        else:
            rich_console.print_error(f"Failed to generate descriptions for {args.video_id}")
    else:
        # Process multiple videos
        results = descriptor.batch_generate_descriptions(
            max_videos=args.max_videos, 
            batch_size=args.batch_size
        )
        rich_console.print_info(f"Successfully processed {len(results)} videos")


if __name__ == "__main__":
    main()