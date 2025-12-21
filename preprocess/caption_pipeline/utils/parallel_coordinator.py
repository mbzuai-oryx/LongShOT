"""
Parallel Processing Coordinator for Video and Audio Descriptions

This module provides utilities to coordinate parallel processing between video descriptions
and audio descriptions at the segment level. As soon as a video segment completes processing,
its corresponding audio segment is queued for Audio Flamingo 3 processing.
"""

import os
import json
import time
import queue
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

from caption_pipeline.utils.rich_console import get_console

logger = logging.getLogger(__name__)
rich_console = get_console()


@dataclass
class SegmentProcessingTask:
    """Represents a segment processing task for audio descriptions."""
    video_id: str
    video_path: str
    segment_index: int
    segment_data: Dict
    priority: int = 0  # Higher numbers = higher priority


class SegmentLevelParallelCoordinator:
    """
    Coordinates parallel processing between video descriptions and audio descriptions.
    
    This coordinator enables maximum parallelism by:
    1. Starting Audio Flamingo 3 initialization as soon as the first video description starts
    2. Processing audio segments immediately when their corresponding video segments complete
    3. Managing resource allocation between video and audio processing
    4. Handling cleanup and error recovery
    """
    
    def __init__(self, audio_descriptor_config: Dict, max_concurrent_audio_tasks: int = 4):
        """
        Initialize the parallel coordinator.
        
        Args:
            audio_descriptor_config: Configuration for AudioDescriptor initialization
            max_concurrent_audio_tasks: Maximum concurrent audio processing tasks
        """
        self.audio_descriptor_config = audio_descriptor_config
        self.max_concurrent_audio_tasks = max_concurrent_audio_tasks
        
        # Processing queues and tracking
        self.audio_processing_queue = queue.PriorityQueue()
        self.completed_segments = {}  # video_id -> list of completed segments
        self.processing_lock = threading.RLock()
        
        # Audio descriptor and processing threads
        self.audio_descriptor = None
        self.audio_processing_executor = None
        self.audio_init_thread = None
        self.audio_init_event = threading.Event()
        
        # Status tracking
        self.active_audio_tasks = {}  # task_id -> future
        self.task_counter = 0
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.stats = {
            'video_segments_completed': 0,
            'audio_segments_queued': 0,
            'audio_segments_completed': 0,
            'audio_segments_failed': 0
        }
        self.stats_lock = threading.RLock()
    
    def start_parallel_processing(self, video_ids: Optional[List[str]] = None) -> bool:
        """
        Start the parallel processing coordinator.
        
        Args:
            video_ids: List of video IDs that will be processed
            
        Returns:
            True if initialization was successful
        """
        try:
            rich_console.print_info("Starting segment-level parallel processing coordinator...")
            
            # Start audio descriptor initialization in background
            self.audio_init_thread = threading.Thread(
                target=self._initialize_audio_descriptor,
                name="AudioDescriptorInit"
            )
            self.audio_init_thread.daemon = True
            self.audio_init_thread.start()
            
            # Start audio processing executor
            self.audio_processing_executor = ThreadPoolExecutor(
                max_workers=self.max_concurrent_audio_tasks,
                thread_name_prefix="AudioProcessor"
            )
            
            # Start audio processing worker
            self.audio_worker_thread = threading.Thread(
                target=self._audio_processing_worker,
                name="AudioWorker"
            )
            self.audio_worker_thread.daemon = True
            self.audio_worker_thread.start()
            
            rich_console.print_success("Parallel processing coordinator started successfully")
            return True
            
        except Exception as e:
            rich_console.print_error(f"Failed to start parallel processing coordinator: {e}")
            return False
    
    def on_video_segment_completed(self, video_id: str, video_path: str, segment_data: Dict) -> None:
        """
        Callback function called when a video segment completes processing.
        
        This queues the corresponding audio segment for processing.
        """
        try:
            with self.stats_lock:
                self.stats['video_segments_completed'] += 1
            
            # Create audio processing task
            with self.processing_lock:
                self.task_counter += 1
                task = SegmentProcessingTask(
                    video_id=video_id,
                    video_path=video_path,
                    segment_index=segment_data.get('segment_index', 0),
                    segment_data=segment_data,
                    priority=0  # Could be adjusted based on segment importance
                )
                
                # Queue for audio processing (priority queue uses (priority, task))
                self.audio_processing_queue.put((task.priority, self.task_counter, task))
                
                with self.stats_lock:
                    self.stats['audio_segments_queued'] += 1
                
                rich_console.print_info(f"Queued audio segment for {video_id} segment {task.segment_index}")
        
        except Exception as e:
            rich_console.print_error(f"Error queuing audio segment for {video_id}: {e}")
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for all audio processing to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dictionary with processing statistics
        """
        start_time = time.time()
        
        try:
            # Wait for audio processing queue to be empty
            while not self.audio_processing_queue.empty():
                if timeout and (time.time() - start_time) > timeout:
                    rich_console.print_warning("Timeout waiting for audio processing completion")
                    break
                time.sleep(1)
            
            # Wait for active tasks to complete
            if self.active_audio_tasks:
                rich_console.print_info(f"Waiting for {len(self.active_audio_tasks)} active audio tasks to complete...")
                
                for task_id, future in list(self.active_audio_tasks.items()):
                    try:
                        remaining_time = None
                        if timeout:
                            remaining_time = timeout - (time.time() - start_time)
                            if remaining_time <= 0:
                                break
                        
                        future.result(timeout=remaining_time)
                    except Exception as e:
                        rich_console.print_warning(f"Audio task {task_id} completed with error: {e}")
                    finally:
                        self.active_audio_tasks.pop(task_id, None)
            
            # Final statistics
            with self.stats_lock:
                final_stats = self.stats.copy()
            
            rich_console.print_success("Parallel audio processing completed")
            rich_console.print_info(f"  - Video segments processed: {final_stats['video_segments_completed']}")
            rich_console.print_info(f"  - Audio segments queued: {final_stats['audio_segments_queued']}")
            rich_console.print_info(f"  - Audio segments completed: {final_stats['audio_segments_completed']}")
            if final_stats['audio_segments_failed'] > 0:
                rich_console.print_warning(f"  - Audio segments failed: {final_stats['audio_segments_failed']}")
            
            return final_stats
            
        except Exception as e:
            rich_console.print_error(f"Error waiting for completion: {e}")
            return self.stats.copy()
    
    def shutdown(self) -> None:
        """Shutdown the parallel processing coordinator and clean up resources."""
        try:
            rich_console.print_info("Shutting down parallel processing coordinator...")
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Stop audio processing
            if self.audio_processing_executor:
                self.audio_processing_executor.shutdown(wait=True, timeout=30)
            
            # Wait for threads to complete
            if self.audio_init_thread and self.audio_init_thread.is_alive():
                self.audio_init_thread.join(timeout=10)
            
            if hasattr(self, 'audio_worker_thread') and self.audio_worker_thread.is_alive():
                self.audio_worker_thread.join(timeout=10)
            
            # Cleanup audio descriptor
            if self.audio_descriptor:
                try:
                    self.audio_descriptor.cleanup()
                except Exception as e:
                    rich_console.print_warning(f"Warning during audio descriptor cleanup: {e}")
                self.audio_descriptor = None
            
            rich_console.print_success("Parallel processing coordinator shutdown completed")
            
        except Exception as e:
            rich_console.print_warning(f"Warning during coordinator shutdown: {e}")
    
    def _initialize_audio_descriptor(self) -> None:
        """Initialize Audio Flamingo 3 in background thread."""
        try:
            rich_console.print_info("Initializing Audio Flamingo 3 in background...")
            
            # Import here to avoid circular imports
            from caption_pipeline.pipeline.audio_descriptor import AudioDescriptor
            
            self.audio_descriptor = AudioDescriptor(**self.audio_descriptor_config)
            self.audio_init_event.set()
            
            rich_console.print_success("Audio Flamingo 3 initialization completed")
            
        except Exception as e:
            rich_console.print_error(f"Failed to initialize Audio Flamingo 3: {e}")
            self.audio_init_event.set()  # Set event anyway to unblock other threads
    
    def _audio_processing_worker(self) -> None:
        """Worker thread that processes audio segments from the queue."""
        rich_console.print_info("Audio processing worker started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get next task from queue (timeout to allow shutdown check)
                try:
                    _, task_id, task = self.audio_processing_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Wait for audio descriptor to be initialized
                if not self.audio_init_event.wait(timeout=300):  # 5 minute timeout
                    rich_console.print_error("Audio descriptor initialization timeout")
                    continue
                
                if not self.audio_descriptor:
                    rich_console.print_error("Audio descriptor not available")
                    continue
                
                # Submit audio processing task
                future = self.audio_processing_executor.submit(
                    self._process_audio_segment, task
                )
                
                # Track active task
                with self.processing_lock:
                    self.active_audio_tasks[task_id] = future
                
                # Handle completion in background
                def on_completion(task_id=task_id):
                    try:
                        result = future.result()
                        if result:
                            with self.stats_lock:
                                self.stats['audio_segments_completed'] += 1
                        else:
                            with self.stats_lock:
                                self.stats['audio_segments_failed'] += 1
                    except Exception as e:
                        rich_console.print_error(f"Audio processing task {task_id} failed: {e}")
                        with self.stats_lock:
                            self.stats['audio_segments_failed'] += 1
                    finally:
                        with self.processing_lock:
                            self.active_audio_tasks.pop(task_id, None)
                
                # Schedule completion handler
                threading.Thread(target=on_completion, daemon=True).start()
                
            except Exception as e:
                rich_console.print_error(f"Error in audio processing worker: {e}")
                continue
        
        rich_console.print_info("Audio processing worker stopped")
    
    def _process_audio_segment(self, task: SegmentProcessingTask) -> bool:
        """Process a single audio segment with Audio Flamingo 3."""
        try:
            # Wait for audio descriptor to be initialized
            if not self.audio_init_event.wait(timeout=300):
                rich_console.print_error("Audio descriptor initialization timeout")
                return False
            
            if not self.audio_descriptor:
                rich_console.print_error("Audio descriptor not available")
                return False
            
            # Process single segment using the existing audio descriptor
            # We don't create temp files anymore - use the descriptor's pipeline method directly
            result = self.audio_descriptor.generate_audio_descriptions_pipeline(task.video_id)
            
            if result:
                rich_console.print_info(f"Completed audio processing for {task.video_id} segment {task.segment_index}")
                return True
            else:
                rich_console.print_warning(f"Audio processing failed for {task.video_id} segment {task.segment_index}")
                return False
                
        except Exception as e:
            rich_console.print_error(f"Error processing audio segment for {task.video_id}: {e}")
            return False

# Global coordinator instance
_global_coordinator: Optional[SegmentLevelParallelCoordinator] = None


def create_parallel_coordinator(audio_descriptor_config: Dict, 
                              max_concurrent_audio_tasks: int = 4) -> SegmentLevelParallelCoordinator:
    """Create and return a new parallel processing coordinator."""
    global _global_coordinator
    
    if _global_coordinator:
        _global_coordinator.shutdown()
    
    _global_coordinator = SegmentLevelParallelCoordinator(
        audio_descriptor_config, max_concurrent_audio_tasks
    )
    return _global_coordinator


def get_parallel_coordinator() -> Optional[SegmentLevelParallelCoordinator]:
    """Get the current global parallel coordinator."""
    return _global_coordinator


def shutdown_parallel_coordinator() -> None:
    """Shutdown the global parallel coordinator."""
    global _global_coordinator
    
    if _global_coordinator:
        _global_coordinator.shutdown()
        _global_coordinator = None