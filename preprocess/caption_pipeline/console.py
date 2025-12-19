"""
Rich console interface for the video caption pipeline.
Provides beautiful progress bars, status displays, and organized output.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import contextmanager

from rich.console import Console
from rich.progress import (
    Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn, 
    TimeElapsedColumn, MofNCompleteColumn, SpinnerColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich import box

@dataclass
class StageStats:
    """Statistics for a pipeline stage."""
    total: int = 0
    completed: int = 0
    failed: int = 0
    active: int = 0
    pending: int = 0
    
    @property
    def progress_ratio(self) -> float:
        """Return completion ratio (0.0 to 1.0)."""
        if self.total == 0:
            return 0.0
        return self.completed / self.total

@dataclass
class VideoStatus:
    """Status information for a single video."""
    video_id: str
    stage: str = "pending"
    start_time: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Return processing duration in seconds."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    @property
    def duration_str(self) -> str:
        """Return formatted duration string."""
        duration = self.duration
        if duration < 60:
            return f"{duration:.1f}s"
        elif duration < 3600:
            return f"{duration//60:.0f}m {duration%60:.0f}s"
        else:
            return f"{duration//3600:.0f}h {(duration%3600)//60:.0f}m"

class RichConsole:
    """Rich console manager for the video caption pipeline."""
    
    def __init__(self):
        self.console = Console()
        self.lock = threading.RLock()
        
        # Progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console
        )
        
        # Stage progress tracking
        self.stage_tasks: Dict[str, TaskID] = {}
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
        
        # Live display
        self.live: Optional[Live] = None
        self.layout = Layout()
        
        # Setup layout structure
        self._setup_layout()
        
    def _setup_layout(self):
        """Setup the rich layout structure."""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=8)
        )
        
        self.layout["main"].split_row(
            Layout(name="progress", ratio=2),
            Layout(name="status", ratio=1)
        )
    
    def start_pipeline(self, total_videos: int, stages: List[str]):
        """Initialize the pipeline with total video count and stages."""
        with self.lock:
            self.total_videos = total_videos
            self.pipeline_start_time = time.time()
            
            # Initialize stage statistics
            for stage in stages:
                if stage in self.stage_stats:
                    self.stage_stats[stage].total = total_videos
                    
            # Create progress tasks for each stage
            stage_names = {
                "download": "🔽 Downloading",
                "preprocess": "🎵 Audio Extraction", 
                "caption": "📝 Caption Generation",
                "video_descriptions": "👁️ Video Descriptions",
                "metadata": "📊 Metadata Generation"
            }
            
            for stage in stages:
                if stage in stage_names:
                    task_id = self.progress.add_task(
                        stage_names[stage],
                        total=total_videos
                    )
                    self.stage_tasks[stage] = task_id
            
            # Start live display  
            self._initial_layout_update()
            self.live = Live(
                self.layout,
                console=self.console,
                refresh_per_second=0.5,  # Slower refresh to reduce flicker
                screen=True,  # Use full screen for cleaner display
                auto_refresh=False  # Manual refresh for better control
            )
            self.live.start()
    
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
                    # Update progress bar
                    if stage in self.stage_tasks:
                        self.progress.update(self.stage_tasks[stage], completed=stats.completed)
                        
                elif status == "failed":
                    stats.failed += 1
                    stats.active = max(0, stats.active - 1)
                    # Update progress bar
                    if stage in self.stage_tasks:
                        self.progress.update(self.stage_tasks[stage], completed=stats.completed + stats.failed)
                        
                elif status == "processing":
                    stats.active += 1
                    
                # Update pending count
                stats.pending = max(0, stats.total - stats.completed - stats.failed - stats.active)
                
            # Trigger layout update
            self._update_layout_content()
    
    def _initial_layout_update(self):
        """Initialize the layout content once."""
        self._update_layout_content()
        
    def _update_layout_content(self):
        """Update the layout content."""
        try:
            # Header
            self.layout["header"].update(Panel(
                self._create_header(),
                title="🎬 Video Caption Pipeline",
                border_style="bright_blue",
                padding=(0, 1)
            ))
            
            # Progress area
            self.layout["progress"].update(Panel(
                self.progress,
                title="📊 Pipeline Progress",
                border_style="bright_green",
                padding=(0, 1)
            ))
            
            # Status area  
            self.layout["status"].update(Panel(
                self._create_status_table(),
                title="📈 Stage Overview",
                border_style="bright_yellow",
                padding=(0, 1)
            ))
            
            # Footer
            self.layout["footer"].update(Panel(
                self._create_active_videos_table(),
                title="⚡ Currently Processing",
                border_style="bright_cyan",
                padding=(0, 1)
            ))
            
            # Refresh the live display
            if self.live and self.live.is_started:
                self.live.refresh()
        except Exception:
            pass  # Ignore layout update errors
    
    def _create_header(self) -> Text:
        """Create the header text with pipeline summary."""
        elapsed = time.time() - self.pipeline_start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        total_completed = sum(stats.completed for stats in self.stage_stats.values())
        total_failed = sum(stats.failed for stats in self.stage_stats.values())
        total_active = sum(stats.active for stats in self.stage_stats.values())
        
        header = Text()
        header.append(f"📹 Videos: {self.total_videos} ", style="bold")
        header.append(f"✅ Completed: {total_completed} ", style="green") 
        header.append(f"❌ Failed: {total_failed} ", style="red")
        header.append(f"⚡ Active: {total_active} ", style="yellow")
        header.append(f"⏱️  Elapsed: {elapsed_str}", style="blue")
        
        return Align.center(header)
    
    def _create_status_table(self) -> Table:
        """Create status overview table with responsive sizing."""
        # Get console size for responsive layout
        try:
            console_width = self.console.size.width
        except:
            console_width = 80  # fallback
        
        # Calculate column widths based on console size
        if console_width < 80:
            # Narrow console - compact layout
            stage_width = 10
            progress_width = 12
            status_width = 20
            details_width = 8
        elif console_width < 120:
            # Medium console - standard layout
            stage_width = 14
            progress_width = 16
            status_width = 25
            details_width = 10
        else:
            # Wide console - expanded layout
            stage_width = 16
            progress_width = 20
            status_width = 30
            details_width = 12
        
        table = Table(show_header=True, header_style="bold bright_magenta", box=box.ROUNDED)
        table.add_column("Stage", style="bright_cyan", width=stage_width)
        table.add_column("Progress", justify="center", width=progress_width)
        table.add_column("Status", justify="left", width=status_width)
        table.add_column("Details", justify="right", width=details_width)
        
        stage_names = {
            "download": "Download",
            "preprocess": "Audio Extract", 
            "caption": "Caption Gen",
            "video_descriptions": "Video Desc",
            "metadata": "Metadata"
        }
        
        stage_icons = {
            "download": "🔽",
            "preprocess": "🎵", 
            "caption": "📝",
            "video_descriptions": "👁️",
            "metadata": "📊"
        }
        
        for stage, stats in self.stage_stats.items():
            if stats.total == 0:
                continue
                
            # Progress bar with responsive width
            progress_ratio = stats.progress_ratio
            bar_width = max(8, progress_width - 6)  # Responsive bar width
            filled = int(progress_ratio * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            progress_text = f"[bright_blue]{bar}[/bright_blue] {progress_ratio*100:.0f}%"
            
            # Status text with better descriptions
            status_parts = []
            if stats.completed > 0:
                status_parts.append(f"[bright_green]✅ {stats.completed} done[/bright_green]")
            if stats.active > 0:
                status_parts.append(f"[bright_yellow]⚡ {stats.active} active[/bright_yellow]")
            if stats.failed > 0:
                status_parts.append(f"[bright_red]❌ {stats.failed} failed[/bright_red]")
            if stats.pending > 0:
                status_parts.append(f"[dim]⏳ {stats.pending} waiting[/dim]")
            
            if not status_parts:
                if stats.total > 0 and stats.completed == stats.total:
                    status_text = "[bright_green]✅ Complete[/bright_green]"
                else:
                    status_text = "[dim]⏸️  Idle[/dim]"
            else:
                status_text = " • ".join(status_parts)
            
            # Details column
            if stats.total > 0:
                details = f"{stats.completed}/{stats.total}"
            else:
                details = "0/0"
            
            # Add row
            stage_display = f"{stage_icons.get(stage, '📋')} {stage_names.get(stage, stage.title())}"
            table.add_row(stage_display, progress_text, status_text, details)
        
        # Add empty row if no stages
        if not any(stats.total > 0 for stats in self.stage_stats.values()):
            table.add_row("[dim]No active stages[/dim]", "", "", "")
        
        return table
    
    def _create_active_videos_table(self) -> Table:
        """Create table showing currently active videos with responsive sizing."""
        # Get console size for responsive layout
        try:
            console_width = self.console.size.width
        except:
            console_width = 80  # fallback
        
        # Calculate column widths based on console size
        if console_width < 80:
            # Narrow console - compact layout
            video_width = 12
            stage_width = 10
            duration_width = 8
            status_width = 25
            max_videos = 4
        elif console_width < 120:
            # Medium console - standard layout
            video_width = 16
            stage_width = 14
            duration_width = 10
            status_width = 30
            max_videos = 6
        else:
            # Wide console - expanded layout
            video_width = 20
            stage_width = 16
            duration_width = 12
            status_width = 40
            max_videos = 8
        
        table = Table(show_header=True, header_style="bold bright_cyan", box=box.ROUNDED)
        table.add_column("Video ID", style="bright_yellow", width=video_width)
        table.add_column("Current Stage", style="bright_green", width=stage_width)
        table.add_column("Duration", justify="right", width=duration_width)
        table.add_column("Status", width=status_width)
        
        # Get active videos (currently processing)
        active_videos = []
        for video_id, status in self.video_statuses.items():
            if (status.stage == "processing" or 
                (status.stage in ["download", "preprocess", "caption", "video_descriptions", "metadata"] 
                 and status.start_time is not None)):
                active_videos.append((video_id, status))
        
        # Sort by duration (longest running first)
        active_videos.sort(key=lambda x: x[1].duration, reverse=True)
        
        # Show active videos based on console size
        for i, (video_id, status) in enumerate(active_videos[:max_videos]):
            # Truncate video ID based on available width
            max_id_length = video_width - 3
            display_id = video_id[:max_id_length] + "..." if len(video_id) > video_width else video_id
            
            # Stage with icon
            stage_icons = {
                "download": "🔽",
                "preprocess": "🎵",
                "caption": "📝", 
                "video_descriptions": "👁️",
                "metadata": "📊"
            }
            
            stage_names = {
                "download": "Downloading",
                "preprocess": "Audio Extract",
                "caption": "Captioning", 
                "video_descriptions": "Describing",
                "metadata": "Metadata Gen"
            }
            
            stage_display = f"{stage_icons.get(status.stage, '📋')} {stage_names.get(status.stage, status.stage.title())}"
            
            # Status message with more detail, truncated based on available width
            if status.error:
                max_error_length = status_width - 15  # Account for prefix
                error_text = status.error[:max_error_length] + "..." if len(status.error) > max_error_length else status.error
                status_msg = f"[bright_red]❌ Error: {error_text}[/bright_red]"
            else:
                # Add progress indication based on stage
                if status.stage == "download":
                    status_msg = "[bright_blue]📥 Downloading from source...[/bright_blue]"
                elif status.stage == "preprocess":
                    status_msg = "[bright_magenta]🔄 Extracting audio tracks...[/bright_magenta]"
                elif status.stage == "caption":
                    status_msg = "[bright_green]🎙️  Generating captions...[/bright_green]"
                elif status.stage == "video_descriptions":
                    status_msg = "[bright_cyan]👁️  Analyzing video content...[/bright_cyan]"
                elif status.stage == "metadata":
                    status_msg = "[bright_yellow]📊 Processing metadata...[/bright_yellow]"
                else:
                    status_msg = "[yellow]⚡ Processing...[/yellow]"
            
            table.add_row(
                display_id,
                stage_display, 
                f"[dim]{status.duration_str}[/dim]",
                status_msg
            )
        
        if not active_videos:
            table.add_row(
                "[dim]🏁 All videos completed[/dim]", 
                "[dim]No active stages[/dim]", 
                "[dim]—[/dim]", 
                "[dim]Pipeline idle or finished[/dim]"
            )
            
        return table
    
    def log_message(self, level: str, message: str, stage: Optional[str] = None):
        """Log a message (for important events only)."""
        # Only show critical messages to avoid cluttering the display
        if level in ["ERROR", "WARNING"]:
            timestamp = datetime.now().strftime("%H:%M:%S")
            with self.lock:
                # Print above the live display
                if level == "ERROR":
                    self.console.print(f"[red][{timestamp}] {stage or 'PIPELINE'}: {message}[/red]")
                elif level == "WARNING":
                    self.console.print(f"[yellow][{timestamp}] {stage or 'PIPELINE'}: {message}[/yellow]")
    
    def finish_stage(self, stage: str):
        """Mark a stage as completely finished."""
        with self.lock:
            if stage in self.stage_tasks:
                stats = self.stage_stats[stage]
                self.progress.update(
                    self.stage_tasks[stage], 
                    completed=stats.completed + stats.failed,
                    total=stats.total
                )
    
    def stop_pipeline(self, show_summary: bool = True):
        """Stop the pipeline and optionally show final summary."""
        with self.lock:
            if self.live:
                self.live.stop()
                
            # Show final summary only if requested
            if show_summary:
                self._show_final_summary()
    
    def _show_final_summary(self):
        """Display final pipeline summary."""
        elapsed = time.time() - self.pipeline_start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        # Create summary table
        summary_table = Table(title="🏁 Pipeline Summary", box=box.DOUBLE_EDGE)
        summary_table.add_column("Stage", style="cyan")
        summary_table.add_column("Total", justify="center")
        summary_table.add_column("Completed", justify="center", style="green")
        summary_table.add_column("Failed", justify="center", style="red")  
        summary_table.add_column("Success Rate", justify="center")
        
        total_all_completed = 0
        total_all_failed = 0
        
        for stage, stats in self.stage_stats.items():
            if stats.total == 0:
                continue
                
            success_rate = (stats.completed / stats.total * 100) if stats.total > 0 else 0
            success_style = "green" if success_rate > 90 else "yellow" if success_rate > 70 else "red"
            
            summary_table.add_row(
                stage.replace("_", " ").title(),
                str(stats.total),
                str(stats.completed),
                str(stats.failed),
                f"[{success_style}]{success_rate:.1f}%[/{success_style}]"
            )
            
            total_all_completed += stats.completed
            total_all_failed += stats.failed
        
        # Overall stats
        total_all = total_all_completed + total_all_failed
        overall_success = (total_all_completed / total_all * 100) if total_all > 0 else 0
        overall_style = "green" if overall_success > 90 else "yellow" if overall_success > 70 else "red"
        
        summary_table.add_row(
            "[bold]OVERALL[/bold]",
            f"[bold]{total_all}[/bold]",
            f"[bold green]{total_all_completed}[/bold green]",
            f"[bold red]{total_all_failed}[/bold red]", 
            f"[bold {overall_style}]{overall_success:.1f}%[/bold {overall_style}]"
        )
        
        # Display summary
        self.console.print()
        self.console.print(summary_table)
        self.console.print()
        self.console.print(f"[bold blue]⏱️  Total execution time: {elapsed_str}[/bold blue]")
        self.console.print()

# Global console instance
rich_console: Optional[RichConsole] = None

def get_console() -> RichConsole:
    """Get or create the global rich console instance."""
    global rich_console
    if rich_console is None:
        rich_console = RichConsole()
    return rich_console

@contextmanager
def pipeline_console(total_videos: int, stages: List[str], show_summary: bool = True):
    """Context manager for pipeline console."""
    console = get_console()
    console.start_pipeline(total_videos, stages)
    try:
        yield console
    finally:
        console.stop_pipeline(show_summary=show_summary)

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
    with console.lock:
        if stage not in console.stage_stats:
            console.stage_stats[stage] = StageStats(total=total_videos)
        else:
            console.stage_stats[stage].total = total_videos
        
        # Create progress task for the new stage
        stage_names = {
            "download": "🔽 Downloading",
            "preprocess": "🎵 Audio Extraction", 
            "caption": "📝 Caption Generation",
            "video_descriptions": "👁️ Video Descriptions",
            "metadata": "📊 Metadata Generation"
        }
        
        if stage in stage_names and stage not in console.stage_tasks:
            task_id = console.progress.add_task(
                stage_names[stage],
                total=total_videos
            )
            console.stage_tasks[stage] = task_id
        
        console._update_layout_content()