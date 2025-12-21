"""
Parallel pipeline orchestrator for the LongShOT video dataset processing.
This module enables concurrent processing across all pipeline stages.
"""

import os
import sys
import time
import queue
import threading
import logging
from typing import List, Optional, Tuple
import pandas as pd

# Import other pipeline components
from caption_pipeline.pipeline.downloader import VideoDownloader
from caption_pipeline.pipeline.preprocessor import VideoPreprocessor
from caption_pipeline.pipeline.caption_generator import CaptionGenerator

# Import project configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import METADATA_DIR, VIDEO_DIR, AUDIO_DIR

# Import rich console utilities
from caption_pipeline.utils.rich_console import get_console

# Import model cleanup utilities
from caption_pipeline.utils.model_cleanup import model_cleanup_manager

# Import common VLM utilities
from caption_pipeline.utils.vlm_common import safe_remove_file

# Set up logging
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,  # Suppress info messages for cleaner console
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'orchestrator.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StageStatus:
    """Track processing status for each video at each stage."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Pipeline monitoring constants
STALL_DETECTION_THRESHOLD = 30  # Cycles before detecting stall (~5 minutes with 10s sleep)
COMPLETION_CHECK_CYCLES = 3  # Consecutive empty cycles before finishing
MONITOR_SLEEP_INTERVAL = 10  # Seconds between monitoring cycles


class PipelineOrchestrator:
    """Parallel pipeline orchestrator for processing videos through all stages concurrently."""
    
    def __init__(self,
                 max_videos: Optional[int] = None,
                 download_workers: int = 4,
                 preprocess_workers: int = 4,
                 caption_workers: int = 2,
                 output_format: str = 'wav',
                 whisper_model_size: str = 'large-v3',
                 whisper_compute_type: str = 'float16',
                 whisper_batch_size: int = 16,
                 **kwargs):
        """Initialize the pipeline orchestrator with configurable worker counts.

        Args:
            max_videos: Maximum number of videos to process (None for no limit)
            download_workers: Number of worker threads for downloading videos
            preprocess_workers: Number of worker threads for preprocessing (audio extraction)
            caption_workers: Number of worker threads for caption generation
            output_format: Audio output format (e.g., 'wav', 'mp3')
            whisper_model_size: Size of the Whisper model to use
            whisper_compute_type: Compute type for Whisper (e.g., 'float16', 'int8')
            whisper_batch_size: Batch size for Whisper captioning
        """
        # Configuration settings
        self.max_videos = max_videos
        self.download_workers = download_workers
        self.preprocess_workers = preprocess_workers
        self.caption_workers = caption_workers
        self.output_format = output_format
        
        # Create directories if they don't exist
        os.makedirs(METADATA_DIR, exist_ok=True)
        os.makedirs(VIDEO_DIR, exist_ok=True)
        os.makedirs(AUDIO_DIR, exist_ok=True)
        
        # Initialize downloader component
        self.downloader = VideoDownloader()
        
        # Set up Whisper model parameters
        self.whisper_model_size = whisper_model_size
        self.whisper_compute_type = whisper_compute_type
        self.whisper_batch_size = whisper_batch_size
        self.device = "cuda"
        
        # Lazy initialization flags
        self._preprocessor = None
        self._caption_generator = None
        
        # Initialize processing queues
        self.download_queue = queue.Queue()
        self.preprocess_queue = queue.Queue()
        self.caption_queue = queue.Queue()
        
        # Tracking structures (protected by locks)
        self.status_lock = threading.RLock()
        self.video_statuses = {}  # track status of each video in each stage
        self.active_videos = set()  # currently active videos
        self.completed_videos = set()  # fully processed videos
        self.failed_videos = set()  # videos that failed at any stage
        
        # Track videos that have finished downloading but haven't been added to the preprocessor metadata
        self.downloaded_videos = {}
        self.download_lock = threading.RLock()
        
        # Progress tracking
        self.total_videos = 0
        self.shutdown_event = threading.Event()
        self.rich_console = get_console()
        
        # Lock for component initialization
        self.init_lock = threading.RLock()
        
        # Metadata file path and DataFrame
        self.metadata_file = os.path.join(METADATA_DIR, 'video_metadata.csv')
        # Initialize metadata DataFrame - load from file if exists, else empty
        if os.path.exists(self.metadata_file):
            try:
                self.metadata_df = pd.read_csv(self.metadata_file)
            except Exception:
                self.metadata_df = pd.DataFrame()
        else:
            self.metadata_df = pd.DataFrame()

        # Add maximum processing time per stage
        self.max_stage_processing_time = {
            'download': 3600,  # 1 hour max for downloads
            'preprocess': 1800,  # 30 minutes max for preprocessing
            'caption': 3600,    # 1 hour max for caption generation
        }
        
        # Max total pipeline time to avoid indefinite hangs (4 hours)
        self.max_pipeline_time = 14400
        
        # Processing start times for each video at each stage
        self.processing_start_times = {}
        self.processing_times_lock = threading.RLock()
    
    @property
    def preprocessor(self):
        """Lazy initialization of the preprocessor."""
        with self.init_lock:
            if self._preprocessor is None:
                try:
                    # Make sure metadata file exists first
                    if not os.path.exists(self.metadata_file):
                        # Create an empty metadata file with required columns
                        self.rich_console.print_info("Creating initial metadata file for preprocessor")
                        columns = [
                            'video_id', 'title', 'channel', 'duration', 'view_count',
                            'publish_date', 'description', 'tags', 'download_date',
                            'file_path', 'status', 'audio_path', 'caption_path'
                        ]
                        pd.DataFrame(columns=columns).to_csv(self.metadata_file, index=False)
                    
                    # Now initialize the preprocessor
                    self.rich_console.print_info("Initializing video preprocessor...")
                    self._preprocessor = VideoPreprocessor()
                    self.rich_console.print_success("Video preprocessor initialized successfully")
                except Exception as e:
                    self.rich_console.print_error(f"Failed to initialize preprocessor: {e}")
                    raise
            return self._preprocessor
    
    @property
    def caption_generator(self):
        """Lazy initialization of the caption generator."""
        with self.init_lock:
            if self._caption_generator is None:
                try:
                    # Initialize caption generator
                    self.rich_console.print_info(f"Initializing Whisper model ({self.whisper_model_size}) - this may take a moment...")
                    self._caption_generator = CaptionGenerator(
                        model_size=self.whisper_model_size,
                        device=self.device,
                        compute_type=self.whisper_compute_type,
                        batch_size=self.whisper_batch_size
                    )
                    self.rich_console.print_success("Caption generator initialized and ready")
                    
                except Exception as e:
                    self.rich_console.print_error(f"Failed to initialize caption generator: {e}")
                    raise
            return self._caption_generator

    def _sync_metadata(self):
        """Reload metadata file to ensure we have the latest updates."""
        try:
            # Re-read metadata file (check size to avoid reading empty/truncated files)
            if not os.path.exists(self.metadata_file) or os.path.getsize(self.metadata_file) == 0:
                return pd.DataFrame()

            metadata_df = pd.read_csv(self.metadata_file)

            # Update any newly downloaded videos that might not be in the metadata yet
            with self.download_lock:
                for video_id, info in self.downloaded_videos.items():
                    video_idx = metadata_df[metadata_df['video_id'] == video_id].index
                    if len(video_idx) == 0:
                        # Add new row for this video
                        new_row = {
                            'video_id': video_id,
                            'file_path': info['file_path'],
                            'status': 'downloaded',
                            'download_date': pd.Timestamp.now().strftime('%Y-%m-%d')
                        }
                        metadata_df = pd.concat([metadata_df, pd.DataFrame([new_row])], ignore_index=True)
                    elif metadata_df.loc[video_idx[0], 'status'] != 'downloaded':
                        # Update existing row
                        metadata_df.loc[video_idx[0], 'file_path'] = info['file_path']
                        metadata_df.loc[video_idx[0], 'status'] = 'downloaded'
                        metadata_df.loc[video_idx[0], 'download_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')

                # Save updated metadata if changes were made
                if self.downloaded_videos:
                    metadata_df.to_csv(self.metadata_file, index=False)
                    self.downloaded_videos = {}  # Clear after saving

            # Return the updated metadata
            return metadata_df

        except pd.errors.EmptyDataError:
            # Handle empty or malformed CSV file gracefully
            return pd.DataFrame()
        except Exception as e:
            self.rich_console.print_error(f"Error syncing metadata: {e}")

        # Return empty DataFrame if we couldn't load metadata
        return pd.DataFrame()
    
    def _load_video_ids(self) -> List[str]:
        """Load video IDs from the file and filter based on configuration."""
        video_ids = self.downloader.read_video_ids()
        
        if self.max_videos and len(video_ids) > self.max_videos:
            video_ids = video_ids[:self.max_videos]
            
        return video_ids
    
    def _init_progress_tracking(self, video_ids: List[str]):
        """Initialize progress tracking for all videos."""
        with self.status_lock:
            self.total_videos = len(video_ids)
            self.active_videos = set(video_ids)
            
            # Initialize status for each video and stage
            for video_id in video_ids:
                self.video_statuses[video_id] = {
                    'download': StageStatus.PENDING,
                    'preprocess': StageStatus.PENDING,
                    'caption': StageStatus.PENDING,
                    'overall': StageStatus.PENDING
                }
            
            # Initialize rich console progress tracking
            stages = ['download', 'preprocess', 'caption']
            self.rich_console.start_pipeline_progress(self.total_videos, stages)
            self.rich_console.start_live_display()
    
    def _update_video_status(self, video_id: str, stage: str, status: str, error: Optional[str] = None):
        """Update status for a video at a specific stage (thread-safe)."""
        with self.status_lock:
            old_status = self.video_statuses.get(video_id, {}).get(stage)
            self.video_statuses[video_id][stage] = status
            
            # Track processing start times for timeout detection
            with self.processing_times_lock:
                if status == StageStatus.PROCESSING:
                    if video_id not in self.processing_start_times:
                        self.processing_start_times[video_id] = {}
                    self.processing_start_times[video_id][stage] = time.time()
                elif status in [StageStatus.COMPLETED, StageStatus.FAILED]:
                    # Clear start time when processing completes or fails
                    if video_id in self.processing_start_times and stage in self.processing_start_times[video_id]:
                        del self.processing_start_times[video_id][stage]
            
            # Update rich console progress for completed stages
            if old_status != StageStatus.COMPLETED and status == StageStatus.COMPLETED:
                self.rich_console.update_stage_progress(stage)
                self.rich_console.update_pipeline_progress()
            
            # Update overall status if all stages complete
            if stage == 'caption' and status == StageStatus.COMPLETED:
                self.video_statuses[video_id]['overall'] = StageStatus.COMPLETED
                self.completed_videos.add(video_id)
                self.active_videos.discard(video_id)
                # Update console stats
                self.rich_console.update_stats(completed=len(self.completed_videos), failed=len(self.failed_videos))
            
            # Handle failures
            if status == StageStatus.FAILED:
                if video_id not in self.failed_videos:
                    self.failed_videos.add(video_id)
                    self.active_videos.discard(video_id)
                    self.rich_console.print_error(f"Video {video_id} failed at {stage} stage")
                    # Update console stats
                    self.rich_console.update_stats(completed=len(self.completed_videos), failed=len(self.failed_videos))
            
            # Update console statistics
            self.rich_console.update_stats(
                completed=len(self.completed_videos),
                failed=len(self.failed_videos)
            )
    
    def _check_for_stuck_videos(self):
        """Check for videos that have been processing for too long and mark them as failed."""
        with self.processing_times_lock:
            current_time = time.time()
            stuck_videos = []
            
            for video_id, stages in self.processing_start_times.items():
                for stage, start_time in stages.items():
                    processing_time = current_time - start_time
                    max_time = self.max_stage_processing_time[stage]
                    
                    if processing_time > max_time:
                        self.rich_console.print_warning(f"Video {video_id} has been in {stage} stage for {processing_time:.1f}s, "
                                     f"exceeding maximum of {max_time}s. Marking as failed.")
                        stuck_videos.append((video_id, stage))
            
            # Release lock before updating status to avoid deadlock
        
        # Mark stuck videos as failed
        for video_id, stage in stuck_videos:
            self._update_video_status(video_id, stage, StageStatus.FAILED)
            self.rich_console.print_error(f"Marking video {video_id} as failed due to timeout in {stage} stage")
            
            # Force clear any partial results for caption generation
            if stage == 'caption':
                partial_path = os.path.join(self.caption_generator.captions_dir, f"{video_id}_partial.json")
                if safe_remove_file(partial_path):
                    self.rich_console.print_info(f"Removed partial caption file for timed out video: {video_id}")
        
        return len(stuck_videos) > 0
    
    def _print_status_summary(self):
        """Print a summary of the current status of all videos."""
        with self.status_lock:
            in_download = sum(1 for _, status in self.video_statuses.items() 
                            if status['download'] == StageStatus.PROCESSING)
            in_preprocess = sum(1 for _, status in self.video_statuses.items() 
                              if status['preprocess'] == StageStatus.PROCESSING)
            in_caption = sum(1 for _, status in self.video_statuses.items() 
                           if status['caption'] == StageStatus.PROCESSING)
            
            # Update stage active counts in rich console
            stage_counts = {
                'download': in_download,
                'preprocess': in_preprocess,
                'caption': in_caption
            }
            self.rich_console.update_stage_active_counts(stage_counts)
    
    def run_pipeline(self) -> Tuple[int, int, int]:
        """
        Run the full parallel pipeline.
        
        Returns:
            Tuple of (total videos, completed videos, failed videos)
        """
        start_time = time.time()
        
        # Print enhanced pipeline header with configuration details
        self.rich_console.print_header(
            "Parallel Video Processing Pipeline",
            "Processing videos through download -> preprocess -> caption stages"
        )

        # Initialize video IDs and tracking with detailed feedback
        self.rich_console.print_info("Discovering and validating video IDs...")
        video_ids = self._load_video_ids()
        if not video_ids:
            self.rich_console.print_error("No video IDs found to process")
            return (0, 0, 0)

        # Print video discovery results
        if self.max_videos and len(video_ids) > self.max_videos:
            self.rich_console.print_success(f"Found {len(video_ids)} video IDs, limited to {self.max_videos} as requested")
        else:
            self.rich_console.print_success(f"Found {len(video_ids)} video IDs to process")

        # Display pipeline configuration and existing progress
        self._print_startup_info(video_ids)
        
        self._init_progress_tracking(video_ids)
        self.rich_console.print_important_info(f"Processing {len(video_ids)} videos through the pipeline")
        
        # Start worker threads with detailed feedback
        self.rich_console.print_info("Initializing worker threads...")
        self._start_workers()
        
        # Queue all videos for download
        for video_id in video_ids:
            self.download_queue.put(video_id)
        
        try:
            # Processing loop - keep checking until all videos are processed
            timeout_counter = 0
            stalled_counter = 0
            last_completed_count = 0
            last_failed_count = 0
            
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Force termination if max pipeline time exceeded
                if elapsed_time > self.max_pipeline_time:
                    self.rich_console.print_warning(f"Maximum pipeline time of {self.max_pipeline_time}s exceeded. "
                                 f"Forcing termination after {elapsed_time:.1f}s")
                    break
                
                # Check for videos stuck in processing
                stuck_detected = self._check_for_stuck_videos()
                
                # Check if we're done (all videos completed or failed)
                with self.status_lock:
                    completed_count = len(self.completed_videos)
                    failed_count = len(self.failed_videos) 
                    total_processed = completed_count + failed_count
                    
                    # Check for progress stall
                    if completed_count == last_completed_count and failed_count == last_failed_count:
                        stalled_counter += 1
                    else:
                        stalled_counter = 0
                        last_completed_count = completed_count
                        last_failed_count = failed_count
                    
                    if stalled_counter >= STALL_DETECTION_THRESHOLD:
                        self.rich_console.print_warning(f"Pipeline appears stalled for {stalled_counter * MONITOR_SLEEP_INTERVAL}s. "
                                     f"Checking for stuck videos.")
                        self._print_status_summary()
                        
                        # Force timeout check with lower thresholds
                        if not stuck_detected:
                            # If we didn't detect any stuck videos using normal thresholds,
                            # manually check for any video in PROCESSING state and mark as failed
                            with self.status_lock:
                                for video_id, status in self.video_statuses.items():
                                    for stage in ['download', 'preprocess', 'caption']:
                                        if status[stage] == StageStatus.PROCESSING:
                                            self.rich_console.print_warning(f"Forcing timeout for {video_id} in {stage} stage due to pipeline stall")
                                            self._update_video_status(video_id, stage, StageStatus.FAILED)
                        
                        # Reset stalled counter
                        stalled_counter = 0
                    
                    if total_processed >= self.total_videos:
                        self.rich_console.print_info("All videos processed, exiting processing loop")
                        break
                
                # Check if all queues are empty and all stages complete
                download_empty = self.download_queue.empty()
                preprocess_empty = self.preprocess_queue.empty()
                caption_empty = self.caption_queue.empty()
                
                # Get counts of videos waiting to be processed
                pending_downloads, pending_preprocess, pending_captions = self._count_pending_items()
                
                # If everything is empty and nothing is pending, we might be done
                if (download_empty and preprocess_empty and caption_empty and
                    pending_downloads == 0 and pending_preprocess == 0 and pending_captions == 0):
                    
                    # Print status to help debug
                    self._print_status_summary()
                    
                    # Use the new method to double-check if all videos are truly processed
                    if self._are_all_videos_processed():
                        timeout_counter += 1
                        if timeout_counter >= COMPLETION_CHECK_CYCLES:
                            self.rich_console.print_info("All queues empty and no pending tasks, finishing pipeline")
                            break
                    else:
                        # Reset counter if we still have active tasks
                        self.rich_console.print_info("Videos still being processed, continuing pipeline")
                        timeout_counter = 0
                else:
                    timeout_counter = 0  # Reset timeout counter if any tasks are pending
                
                # Sync metadata with all downloaded videos
                self._sync_metadata()
                
                # Log current status less frequently (every 15 seconds)
                if elapsed_time % 15 < 5:  # Only print every 15 seconds
                    self._print_status_summary()
                
                # Sleep to avoid high CPU usage
                time.sleep(MONITOR_SLEEP_INTERVAL)
            
        except KeyboardInterrupt:
            self.rich_console.print_warning("Pipeline interrupted by user")
        finally:
            # Signal shutdown and wait for threads to finish
            self.shutdown_event.set()
            
            # Stop rich console progress display
            self.rich_console.stop_live_display()
            self.rich_console.finish_pipeline_progress()
            
            # Wait for threads to finish (with timeout)
            for thread in self.worker_threads:
                thread.join(timeout=2)
            
            # Final sync to save any downloaded videos not yet in metadata
            self._sync_metadata()
        
        # Calculate final statistics
        with self.status_lock:
            completed = len(self.completed_videos)
            failed = len(self.failed_videos)
            
        # Print summary using rich console
        duration = time.time() - start_time
        
        # Print video status table
        if self.video_statuses:
            self.rich_console.print_video_status_table(self.video_statuses)
        
        # Print completion summary
        self.rich_console.print_completion_message("Pipeline", {
            'total': self.total_videos,
            'successful': completed,
            'duration': duration
        })
        
        if failed > 0:
            self.rich_console.print_warning(f"Failed videos: {', '.join(list(self.failed_videos)[:5])}")
            if len(self.failed_videos) > 5:
                self.rich_console.print_warning(f"... and {len(self.failed_videos) - 5} more")
        
        # Cleanup main pipeline models before returning
        self._cleanup_main_pipeline_models()
        
        return (self.total_videos, completed, failed)
    
    def _download_worker(self):
        """Worker function for downloading videos."""
        while not self.shutdown_event.is_set():
            try:
                # Get next video_id from queue with timeout
                try:
                    video_id = self.download_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Skip if already downloaded
                if self._is_video_downloaded(video_id):
                    self.rich_console.print_info(f"[{video_id}] Already downloaded, skipping to preprocessing")
                    self._update_video_status(video_id, 'download', StageStatus.COMPLETED)
                    self.preprocess_queue.put(video_id)
                    self.download_queue.task_done()
                    continue
                
                # Update status
                self._update_video_status(video_id, 'download', StageStatus.PROCESSING)
                
                # Download the video
                self.rich_console.print_stage("download", video_id, "starting download")
                try:
                    file_path = self.downloader.download_video(video_id)
                    
                    if file_path and os.path.exists(file_path):
                        self.rich_console.print_stage("download", video_id, f"completed: {os.path.basename(file_path)}")
                        self._update_video_status(video_id, 'download', StageStatus.COMPLETED)
                        
                        # Track this downloaded video for metadata sync
                        with self.download_lock:
                            self.downloaded_videos[video_id] = {
                                'file_path': file_path,
                                'timestamp': time.time()
                            }
                        
                        # Add to preprocessing queue
                        self.preprocess_queue.put(video_id)
                    else:
                        self.rich_console.print_stage("download", video_id, "failed")
                        self._update_video_status(video_id, 'download', StageStatus.FAILED)
                        
                except Exception as e:
                    self.rich_console.print_stage("download", video_id, f"error: {str(e)}")
                    self._update_video_status(video_id, 'download', StageStatus.FAILED)
                
                # Mark task as done
                self.download_queue.task_done()
                
            except Exception as e:
                self.rich_console.print_error(f"Download worker error: {str(e)}")
    
    def _preprocess_worker(self):
        """Worker function for preprocessing videos (audio extraction)."""
        while not self.shutdown_event.is_set():
            try:
                # Get next video_id from queue with timeout
                try:
                    video_id = self.preprocess_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Make sure preprocessor is initialized
                try:
                    preprocessor = self.preprocessor  # This will initialize if needed
                except Exception as e:
                    self.rich_console.print_error(f"Could not initialize preprocessor: {e}")
                    self._update_video_status(video_id, 'preprocess', StageStatus.FAILED)
                    self.preprocess_queue.task_done()
                    continue
                
                # Skip if already preprocessed
                if self._is_audio_extracted(video_id):
                    self.rich_console.print_info(f"[{video_id}] Audio already extracted, skipping to captioning")
                    self._update_video_status(video_id, 'preprocess', StageStatus.COMPLETED)
                    self.caption_queue.put(video_id)
                    self.preprocess_queue.task_done()
                    continue
                
                # Update status
                self._update_video_status(video_id, 'preprocess', StageStatus.PROCESSING)
                
                # Sync metadata to make sure we have latest download information
                updated_metadata = self._sync_metadata()
                if not updated_metadata.empty:
                    # Update the preprocessor's metadata
                    preprocessor.metadata_df = updated_metadata
                
                # Verify video exists before extracting audio
                video_path = self._get_video_path(video_id)
                if not video_path or not os.path.exists(video_path):
                    self.rich_console.print_error(f"[{video_id}] Video file not found, cannot extract audio")
                    self._update_video_status(video_id, 'preprocess', StageStatus.FAILED)
                    self.preprocess_queue.task_done()
                    continue
                
                # Extract audio
                self.rich_console.print_stage("preprocess", video_id, "starting audio extraction")
                try:
                    audio_path = preprocessor.extract_audio(video_id, self.output_format)
                    
                    if audio_path and os.path.exists(audio_path):
                        self.rich_console.print_stage("preprocess", video_id, f"completed: {os.path.basename(audio_path)}")
                        self._update_video_status(video_id, 'preprocess', StageStatus.COMPLETED)
                        
                        # Add to captioning queue
                        self.caption_queue.put(video_id)
                    else:
                        self.rich_console.print_stage("preprocess", video_id, "failed")
                        self._update_video_status(video_id, 'preprocess', StageStatus.FAILED)
                        
                except Exception as e:
                    self.rich_console.print_stage("preprocess", video_id, f"error: {str(e)}")
                    self._update_video_status(video_id, 'preprocess', StageStatus.FAILED)
                
                # Mark task as done
                self.preprocess_queue.task_done()
                
            except Exception as e:
                self.rich_console.print_error(f"Preprocess worker error: {str(e)}")
    
    def _caption_worker(self):
        """Worker function for generating captions."""
        while not self.shutdown_event.is_set():
            try:
                # Get next video_id from queue with timeout
                try:
                    video_id = self.caption_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Skip if already captioned
                if self._is_video_captioned(video_id):
                    self.rich_console.print_info(f"[{video_id}] Captions already generated, marking as complete")
                    self._update_video_status(video_id, 'caption', StageStatus.COMPLETED)
                    self.caption_queue.task_done()
                    continue
                
                # Make sure caption generator is initialized
                try:
                    caption_gen = self.caption_generator  # This will initialize if needed
                except Exception as e:
                    self.rich_console.print_error(f"Could not initialize caption generator: {e}")
                    self._update_video_status(video_id, 'caption', StageStatus.FAILED)
                    self.caption_queue.task_done()
                    continue
                
                # Update status
                self._update_video_status(video_id, 'caption', StageStatus.PROCESSING)
                
                # Add a delay to ensure filesystem changes are visible
                time.sleep(0.5)
                
                # Ensure audio file exists
                audio_path = os.path.join(self.preprocessor.audio_dir, f"{video_id}.{self.output_format}")
                if not os.path.exists(audio_path):
                    self.rich_console.print_error(f"[{video_id}] Audio file not found at {audio_path}, cannot generate captions")
                    self._update_video_status(video_id, 'caption', StageStatus.FAILED)
                    self.caption_queue.task_done()
                    continue
                
                # Generate captions
                self.rich_console.print_stage("caption", video_id, "starting caption generation")
                try:
                    caption_path = caption_gen.generate_captions(video_id)
                    
                    if caption_path and os.path.exists(caption_path):
                        self.rich_console.print_stage("caption", video_id, f"completed: {os.path.basename(caption_path)}")
                        self._update_video_status(video_id, 'caption', StageStatus.COMPLETED)
                    else:
                        self.rich_console.print_stage("caption", video_id, "failed")
                        self._update_video_status(video_id, 'caption', StageStatus.FAILED)
                        
                except Exception as e:
                    self.rich_console.print_stage("caption", video_id, f"error: {str(e)}")
                    self._update_video_status(video_id, 'caption', StageStatus.FAILED)
                
                # Mark task as done
                self.caption_queue.task_done()
                
            except Exception as e:
                self.rich_console.print_error(f"Caption worker error: {str(e)}")
    
    def _start_workers(self):
        """Start worker threads for all pipeline stages."""
        self.worker_threads = []
        
        # Start download workers
        for _ in range(self.download_workers):
            thread = threading.Thread(target=self._download_worker)
            thread.daemon = True
            thread.start()
            self.worker_threads.append(thread)
        
        # Start preprocess workers
        for _ in range(self.preprocess_workers):
            thread = threading.Thread(target=self._preprocess_worker)
            thread.daemon = True
            thread.start()
            self.worker_threads.append(thread)
        
        # Start caption workers
        for _ in range(self.caption_workers):
            thread = threading.Thread(target=self._caption_worker)
            thread.daemon = True
            thread.start()
            self.worker_threads.append(thread)
        
        self.rich_console.print_success(f"Started {self.download_workers} download workers, "
                   f"{self.preprocess_workers} preprocessing workers, and "
                   f"{self.caption_workers} captioning workers")
        self.rich_console.print_info("Pipeline is now active and processing videos...")
        self.rich_console.print_info("")
    
    def _is_video_downloaded(self, video_id: str) -> bool:
        """Check if a video has already been downloaded."""
        # Check metadata for downloaded status
        with self.status_lock:
            if video_id in self.video_statuses:
                return self.video_statuses[video_id]['download'] == StageStatus.COMPLETED
        
        # If not in our tracking, check with the downloader directly
        return self.downloader._is_video_downloaded(video_id)
    
    def _get_video_path(self, video_id: str) -> Optional[str]:
        """Get the file path for a downloaded video."""
        # First check with the downloader (this will handle offline video registration)
        video_path = self.downloader._get_video_path(video_id)
        if video_path and os.path.exists(video_path):
            return video_path
        
        # Fallback to standard location
        standard_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        if os.path.exists(standard_path):
            # Register this offline video with the downloader
            self.downloader._register_offline_video(video_id, standard_path)
            return standard_path
        
        return None
    
    def _is_audio_extracted(self, video_id: str) -> bool:
        """Check if audio has already been extracted for a video."""
        # Check our tracking status
        with self.status_lock:
            if video_id in self.video_statuses:
                return self.video_statuses[video_id]['preprocess'] == StageStatus.COMPLETED
        
        # Also check for the actual audio file
        audio_path = os.path.join(AUDIO_DIR, f"{video_id}.{self.output_format}")
        return os.path.exists(audio_path)
    
    def _is_video_captioned(self, video_id: str) -> bool:
        """Check if captions have already been generated for a video."""
        # Check our tracking status
        with self.status_lock:
            if video_id in self.video_statuses:
                return self.video_statuses[video_id]['caption'] == StageStatus.COMPLETED
        
        # Also check for the actual caption file
        caption_path = os.path.join(self.caption_generator.captions_dir, f"{video_id}.json")
        if os.path.exists(caption_path):
            return True
            
        # Check for partial files that might indicate a failed previous attempt
        partial_path = os.path.join(self.caption_generator.captions_dir, f"{video_id}_partial.json")
        if os.path.exists(partial_path):
            # Remove stale partial file and treat as not captioned
            if safe_remove_file(partial_path):
                self.rich_console.print_info(f"Removed stale partial caption file for {video_id}")
            return False
        
        return False
    
    def _count_pending_items(self) -> Tuple[int, int, int]:
        """Count pending items in each stage (that aren't already counted in status)."""
        pending_downloads = 0
        pending_preprocess = 0
        pending_captions = 0
        
        with self.status_lock:
            for video_id, status in self.video_statuses.items():
                # Only count as pending if it's in the queue but not yet processing
                if status['download'] == StageStatus.PENDING:
                    pending_downloads += 1
                if status['preprocess'] == StageStatus.PENDING and status['download'] == StageStatus.COMPLETED:
                    pending_preprocess += 1 
                if status['caption'] == StageStatus.PENDING and status['preprocess'] == StageStatus.COMPLETED:
                    pending_captions += 1
                    
                # Also count videos that are currently being processed
                if status['download'] == StageStatus.PROCESSING:
                    pending_downloads += 1
                if status['preprocess'] == StageStatus.PROCESSING:
                    pending_preprocess += 1
                if status['caption'] == StageStatus.PROCESSING:
                    pending_captions += 1
        
        return (pending_downloads, pending_preprocess, pending_captions)
    
    def _are_all_videos_processed(self) -> bool:
        """Check if all videos have been fully processed (completed or failed)."""
        with self.status_lock:
            for video_id, status in self.video_statuses.items():
                # Check if any stage is still pending or processing
                if (status['download'] in [StageStatus.PENDING, StageStatus.PROCESSING] or
                    status['preprocess'] in [StageStatus.PENDING, StageStatus.PROCESSING] or
                    status['caption'] in [StageStatus.PENDING, StageStatus.PROCESSING]):
                    return False
            
            # If we got here, all videos are either completed or failed
            return True
    
    def _print_startup_info(self, video_ids: List[str]):
        """Print pipeline configuration and existing processing status."""
        # Pipeline configuration
        self.rich_console.print_info("Pipeline Configuration:")
        self.rich_console.print_info(f"  - Download workers: {self.download_workers}")
        self.rich_console.print_info(f"  - Preprocessing workers: {self.preprocess_workers}")
        self.rich_console.print_info(f"  - Caption workers: {self.caption_workers}")
        self.rich_console.print_info(f"  - Audio output format: {self.output_format}")
        self.rich_console.print_info(f"  - Whisper model: {self.whisper_model_size} ({self.whisper_compute_type})")
        self.rich_console.print_info(f"  - Device: {self.device}")
        if self.max_videos:
            self.rich_console.print_info(f"  - Maximum videos: {self.max_videos}")

        # Existing processing status
        already_downloaded = sum(1 for v in video_ids if self._is_video_downloaded(v))
        already_preprocessed = sum(1 for v in video_ids if self._is_audio_extracted(v))
        already_captioned = sum(1 for v in video_ids if self._is_video_captioned(v))

        if already_downloaded > 0 or already_preprocessed > 0 or already_captioned > 0:
            self.rich_console.print_info("Existing Progress:")
            if already_downloaded > 0:
                self.rich_console.print_info(f"  - {already_downloaded} videos already downloaded")
            if already_preprocessed > 0:
                self.rich_console.print_info(f"  - {already_preprocessed} videos already preprocessed")
            if already_captioned > 0:
                self.rich_console.print_info(f"  - {already_captioned} videos already captioned")

        remaining = len(video_ids) - already_captioned
        if remaining < len(video_ids):
            self.rich_console.print_info(f"  - {remaining} videos require processing")
        self.rich_console.print_info("")

    def _cleanup_main_pipeline_models(self):
        """Clean up models used in the main pipeline (Whisper + CLIP)."""
        try:
            components_to_cleanup = {}
            
            # Add caption generator if initialized
            if hasattr(self, '_caption_generator') and self._caption_generator is not None:
                components_to_cleanup['caption_generator'] = self._caption_generator
            
            # Clean up synchronously since we're at the end of the main pipeline
            if components_to_cleanup:
                self.rich_console.print_info("Cleaning up main pipeline models (Whisper + CLIP)...")
                model_cleanup_manager.cleanup_all_models(components_to_cleanup, async_cleanup=False)
                self.rich_console.print_success("Main pipeline model cleanup completed")
            else:
                self.rich_console.print_info("No main pipeline models to clean up")
                
        except Exception as e:
            self.rich_console.print_warning(f"Warning during main pipeline model cleanup: {e}")
    
    def cleanup_and_prepare_for_next_stage(self, next_stage_init_func, *args, **kwargs):
        """
        Clean up main pipeline models in parallel with next stage initialization.
        
        This method enables overlapping cleanup with next stage setup for better performance.
        """
        try:
            components_to_cleanup = {}
            
            # Add caption generator if initialized  
            if hasattr(self, '_caption_generator') and self._caption_generator is not None:
                components_to_cleanup['caption_generator'] = self._caption_generator
            
            if components_to_cleanup:
                # Run cleanup in parallel with next stage initialization
                return model_cleanup_manager.parallel_cleanup_with_next_stage_init(
                    components_to_cleanup, next_stage_init_func, args, kwargs
                )
            else:
                # No cleanup needed, just run next stage
                return next_stage_init_func(*args, **kwargs)
                
        except Exception as e:
            self.rich_console.print_error(f"Error during parallel cleanup: {e}")
            # Fall back to running next stage without cleanup
            return next_stage_init_func(*args, **kwargs)
