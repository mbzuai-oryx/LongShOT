"""
Simple console interface for the video caption pipeline.
Provides clean line-by-line progress updates without complex layouts.
"""

import time
import threading
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class StageStats:
    """Statistics for a pipeline stage."""
    total: int = 0
    completed: int = 0
    failed: int = 0
    active: int = 0
    
    @property
    def progress_percentage(self) -> float:
        """Return completion percentage (0.0 to 100.0)."""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100

@dataclass
class VideoStatus:
    """Status information for a single video."""
    video_id: str
    stage: str = "pending"
    start_time: Optional[float] = None
    error: Optional[str] = None

class SimpleConsole:
    """Simple console manager for the video caption pipeline."""
    
    def __init__(self):
        self.lock = threading.RLock()
        
        # Stage tracking
        self.stage_stats: Dict[str, StageStats] = {
            "download": StageStats(),
            "preprocess": StageStats(), 
            "caption": StageStats(),
            "video_descriptions": StageStats(),
            "metadata": StageStats()
        }
        
        # Video status tracking
        self.video_statuses: Dict[str, VideoStatus] = {}
        
        # Pipeline stats
        self.pipeline_start_time = time.time()
        self.total_videos = 0
        self.stages_to_show = []
        
        # Stage display names
        self.stage_names = {
            "download": "📥 Downloading videos",
            "preprocess": "🎵 Extracting audio", 
            "caption": "📝 Generating captions",
            "video_descriptions": "👁️  Creating video descriptions",
            "metadata": "📊 Processing metadata"
        }
        
        # Last progress update to avoid spam
        self.last_progress_update = {}
        
    def start_pipeline(self, total_videos: int, stages: List[str]):
        """Initialize the pipeline with total video count and stages."""
        with self.lock:
            self.total_videos = total_videos
            self.pipeline_start_time = time.time()
            self.stages_to_show = stages
            
            # Initialize stage statistics
            for stage in stages:
                if stage in self.stage_stats:
                    self.stage_stats[stage].total = total_videos
                    
            # Print initial message
            stage_list = ", ".join([self.stage_names.get(s, s) for s in stages])
            print(f"\n🎬 Starting video caption pipeline")
            print(f"   Videos to process: {total_videos}")
            print(f"   Pipeline stages: {stage_list}")
            print(f"   Started at: {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 80)
    
    def update_video_status(self, video_id: str, stage: str, status: str, error: Optional[str] = None):
        """Update status for a specific video."""
        with self.lock:
            if video_id not in self.video_statuses:
                self.video_statuses[video_id] = VideoStatus(video_id)
            
            video_status = self.video_statuses[video_id]
            old_stage = video_status.stage
            
            # Update video status
            if status == "processing" and video_status.start_time is None:
                video_status.start_time = time.time()
                
            video_status.stage = stage if status == "processing" else status
            video_status.error = error
            
            # Update stage statistics
            if stage in self.stage_stats:
                stats = self.stage_stats[stage]
                
                # Remove from old counts
                if old_stage == "processing" and stage in self.stage_stats:
                    self.stage_stats[stage].active = max(0, self.stage_stats[stage].active - 1)
                
                # Add to new counts
                if status == "completed":
                    stats.completed += 1
                    stats.active = max(0, stats.active - 1)
                    self._print_video_completion(video_id, stage)
                    
                elif status == "failed":
                    stats.failed += 1
                    stats.active = max(0, stats.active - 1)
                    self._print_video_failure(video_id, stage, error)
                    
                elif status == "processing":
                    stats.active += 1
                    self._print_video_start(video_id, stage)
                
                # Print stage progress periodically
                self._print_stage_progress(stage)
    
    def _print_video_start(self, video_id: str, stage: str):
        """Print when a video starts processing in a stage."""
        stage_name = self.stage_names.get(stage, stage)
        # Truncate video ID for cleaner display
        display_id = video_id[:12] + "..." if len(video_id) > 15 else video_id
        print(f"   🔄 {stage_name}: {display_id}")
    
    def _print_video_completion(self, video_id: str, stage: str):
        """Print when a video completes a stage."""
        stage_name = self.stage_names.get(stage, stage)
        display_id = video_id[:12] + "..." if len(video_id) > 15 else video_id
        print(f"   ✅ {stage_name}: {display_id} completed")
    
    def _print_video_failure(self, video_id: str, stage: str, error: Optional[str]):
        """Print when a video fails in a stage."""
        stage_name = self.stage_names.get(stage, stage)
        display_id = video_id[:12] + "..." if len(video_id) > 15 else video_id
        error_msg = f" - {error[:50]}..." if error and len(error) > 50 else f" - {error}" if error else ""
        print(f"   ❌ {stage_name}: {display_id} failed{error_msg}")
    
    def _print_stage_progress(self, stage: str):
        """Print stage progress periodically to avoid spam."""
        current_time = time.time()
        if stage not in self.last_progress_update:
            self.last_progress_update[stage] = 0
            
        # Only print progress every 3 seconds to avoid spam
        if current_time - self.last_progress_update[stage] < 3:
            return
            
        self.last_progress_update[stage] = current_time
        
        if stage in self.stage_stats and stage in self.stages_to_show:
            stats = self.stage_stats[stage]
            if stats.total > 0:
                percentage = stats.progress_percentage
                stage_name = self.stage_names.get(stage, stage)
                
                # Create a simple progress bar
                bar_width = 30
                filled = int((percentage / 100) * bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)
                
                print(f"📊 {stage_name}: [{bar}] {stats.completed}/{stats.total} ({percentage:.1f}%)")
    
    def add_stage(self, stage: str, total_videos: int):
        """Add a new stage to the console."""
        with self.lock:
            if stage not in self.stage_stats:
                self.stage_stats[stage] = StageStats()
            
            self.stage_stats[stage].total = total_videos
            
            if stage not in self.stages_to_show:
                self.stages_to_show.append(stage)
                stage_name = self.stage_names.get(stage, stage)
                print(f"\n🔄 Starting {stage_name} for {total_videos} videos")
    
    def log_message(self, level: str, message: str, stage: Optional[str] = None):
        """Log a message (for important events only)."""
        if level in ["ERROR", "WARNING"]:
            timestamp = datetime.now().strftime("%H:%M:%S")
            icon = "❌" if level == "ERROR" else "⚠️"
            stage_prefix = f"[{stage}] " if stage else ""
            print(f"{icon} [{timestamp}] {stage_prefix}{message}")
    
    def finish_stage(self, stage: str):
        """Mark a stage as completely finished."""
        with self.lock:
            if stage in self.stage_stats:
                stats = self.stage_stats[stage]
                stage_name = self.stage_names.get(stage, stage)
                success_rate = (stats.completed / stats.total * 100) if stats.total > 0 else 0
                status_icon = "✅" if success_rate > 90 else "⚠️" if success_rate > 70 else "❌"
                print(f"{status_icon} {stage_name} complete: {stats.completed}/{stats.total} ({success_rate:.1f}%) successful")
    
    def stop_pipeline(self, show_summary: bool = True):
        """Stop the pipeline and optionally show final summary."""
        if show_summary:
            self._show_final_summary()
    
    def _show_final_summary(self):
        """Display final pipeline summary."""
        elapsed = time.time() - self.pipeline_start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        print("\n" + "=" * 80)
        print("🏁 PIPELINE SUMMARY")
        print("=" * 80)
        
        total_processed = 0
        total_successful = 0
        total_failed = 0
        
        for stage in self.stages_to_show:
            if stage in self.stage_stats:
                stats = self.stage_stats[stage]
                if stats.total > 0:
                    stage_name = self.stage_names.get(stage, stage)
                    success_rate = (stats.completed / stats.total * 100) if stats.total > 0 else 0
                    
                    status_icon = "✅" if success_rate > 90 else "⚠️" if success_rate > 70 else "❌"
                    print(f"{status_icon} {stage_name}: {stats.completed}/{stats.total} ({success_rate:.1f}%)")
                    
                    total_processed += stats.total
                    total_successful += stats.completed
                    total_failed += stats.failed
        
        print("-" * 80)
        overall_success = (total_successful / total_processed * 100) if total_processed > 0 else 0
        overall_icon = "✅" if overall_success > 90 else "⚠️" if overall_success > 70 else "❌"
        print(f"{overall_icon} OVERALL: {total_successful}/{total_processed} ({overall_success:.1f}%) successful")
        print(f"⏱️  Total time: {elapsed_str}")
        print(f"🕒 Finished at: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)

# Global console instance
simple_console: Optional[SimpleConsole] = None

def get_console() -> SimpleConsole:
    """Get or create the global simple console instance."""
    global simple_console
    if simple_console is None:
        simple_console = SimpleConsole()
    return simple_console

def update_video_status(video_id: str, stage: str, status: str, error: Optional[str] = None):
    """Update video status in the console."""
    console = get_console()
    console.update_video_status(video_id, stage, status, error)

def log_message(level: str, message: str, stage: Optional[str] = None):
    """Log a message to the console."""
    console = get_console()
    console.log_message(level, message, stage)

def add_stage(stage: str, total_videos: int):
    """Add a new stage to the console after initialization."""
    console = get_console()
    console.add_stage(stage, total_videos)

def pipeline_console(total_videos: int, stages: List[str], show_summary: bool = True):
    """Context manager for simple pipeline console."""
    console = get_console()
    console.start_pipeline(total_videos, stages)
    
    class PipelineContext:
        def __init__(self, console_instance):
            self.console = console_instance
            
        def __enter__(self):
            return self.console
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.console.stop_pipeline(show_summary=show_summary)
    
    return PipelineContext(console)