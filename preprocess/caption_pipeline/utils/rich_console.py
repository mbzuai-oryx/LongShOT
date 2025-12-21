"""
Rich console utilities for the caption pipeline.
This module provides consistent, beautiful console output using the rich library.
"""

import os
import time
import threading
from datetime import timedelta
from typing import Dict, List, Optional, Any, Union
from rich.console import Console
from rich.progress import (
    Progress, TaskID, BarColumn, TimeRemainingColumn, 
    SpinnerColumn, MofNCompleteColumn, TextColumn,
    ProgressColumn, Text
)
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text as RichText
from rich import box
from rich.align import Align
from rich.columns import Columns


class RateColumn(ProgressColumn):
    """A custom column to show processing rate."""
    
    def render(self, task) -> Text:
        """Show speed in tasks per second."""
        speed = task.speed
        if speed is None:
            return Text("--", style="progress.data.speed")
        return Text(f"{speed:.1f}/s", style="progress.data.speed")


class StatusColumn(ProgressColumn):
    """A custom column to show current status."""
    
    def render(self, task) -> Text:
        """Show current status from task fields."""
        status = task.fields.get("status", "")
        if status:
            return Text(status, style="dim")
        return Text("", style="dim")


class PipelineConsole:
    """Central console manager for the caption pipeline with rich formatting."""
    
    def __init__(self):
        """Initialize the console with rich formatting."""
        self.console = Console(width=120, stderr=False)
        self.error_console = Console(stderr=True)
        
        # Progress tracking
        self.stage_progress = None
        self.stage_tasks: Dict[str, TaskID] = {}
        self.live = None
        self.stats = {
            'total_videos': 0,
            'stage_counts': {'download': 0, 'preprocess': 0, 'caption': 0}
        }
        self._lock = threading.RLock()
        self._last_update = 0
        self._update_interval = 2.0
        self._last_stage_update = {}
        
        # Info message buffer
        self._info_messages = []
        self._max_info_messages = 50
        
        # Color schemes
        self.colors = {
            'success': 'green',
            'error': 'red',
            'warning': 'yellow',
            'info': 'blue',
            'progress': 'cyan',
            'stage': 'magenta',
            'video_id': 'bright_blue',
            'metric': 'white',
            'header': 'bold white'
        }
        
        # Display state
        self._display_active = False
        self._suppress_stage_messages = False
        
        # Setup cleaner logging
        self._setup_clean_logging()
    
    def _setup_clean_logging(self):
        """Setup cleaner logging to reduce verbosity when rich console is active."""
        import logging
        
        # Reduce verbosity for specific loggers
        verbose_loggers = [
            'httpx',
            'urllib3',
            'requests'
        ]
        
        for logger_name in verbose_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
    
    def print_header(self, title: str, subtitle: str = None):
        """Print a formatted header."""
        header_text = f"[{self.colors['header']}]{title}[/]"
        if subtitle:
            header_text += f"\n[dim]{subtitle}[/]"
        
        panel = Panel(
            header_text,
            box=box.DOUBLE,
            padding=(1, 2),
            style=self.colors['info']
        )
        self.console.print(panel)
        self.console.print()
    
    def _print_message(self, message: str, prefix: str, color: str, use_error_console: bool = False):
        """Internal helper for consistent message printing."""
        timestamp = time.strftime("%H:%M:%S")
        formatted = f"[dim]{timestamp}[/] [{self.colors[color]}]{prefix}[/] {message}"

        if self._display_active and self.live:
            if use_error_console:
                self.error_console.print(formatted)
            self._add_info_message(formatted)
            self.update_live_display()
        else:
            console = self.error_console if use_error_console else self.console
            console.print(formatted)

    def print_info(self, message: str, prefix: str = "INFO"):
        """Print an info message with formatting."""
        self._print_message(message, prefix, 'info')

    def print_success(self, message: str, prefix: str = "[OK]"):
        """Print a success message with formatting."""
        self._print_message(message, prefix, 'success')

    def print_error(self, message: str, prefix: str = "[ERROR]"):
        """Print an error message with formatting."""
        self._print_message(message, prefix, 'error', use_error_console=True)

    def print_warning(self, message: str, prefix: str = "!"):
        """Print a warning message with formatting."""
        self._print_message(message, prefix, 'warning')
    
    def print_stage(self, stage: str, video_id: str, action: str):
        """Print a stage-specific message."""
        # Only print important stage messages if not suppressed
        if not self._suppress_stage_messages and action.startswith(('starting', 'completed', 'failed', 'error')):
            timestamp = time.strftime("%H:%M:%S")
            message = (
                f"[dim]{timestamp}[/] "
                f"[{self.colors['stage']}]{stage.upper()}[/] "
                f"[{self.colors['video_id']}]{video_id}[/] "
                f"{action}"
            )
            
            # Add to info messages buffer if display is active
            if self._display_active and self.live:
                self._add_info_message(message)
                self.update_live_display()
            else:
                self.console.print(message)
    
    def print_important_info(self, message: str, prefix: str = "INFO"):
        """Print an important info message that should always be visible."""
        timestamp = time.strftime("%H:%M:%S")
        if prefix:
            formatted_message = f"[dim]{timestamp}[/] [{self.colors['info']}]{prefix}[/] {message}"
        else:
            formatted_message = f"[dim]{timestamp}[/] {message}"
        
        # Always print important messages directly
        self.console.print(formatted_message)
        
        # Also add to info buffer if live display is active
        if self._display_active and self.live:
            self._add_info_message(formatted_message)
    
    def _add_info_message(self, message: str):
        """Add a message to the info buffer."""
        with self._lock:
            self._info_messages.append(message)
            # Keep only the last N messages
            if len(self._info_messages) > self._max_info_messages:
                self._info_messages.pop(0)
    
    def start_pipeline_progress(self, total_videos: int, stages: List[str]):
        """Start the stage progress tracking only."""
        with self._lock:
            self.stats['total_videos'] = total_videos
            self._display_active = True
            self._suppress_stage_messages = True
            self.stage_progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(bar_width=40),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                StatusColumn(),
                console=self.console,
                refresh_per_second=0.2,
                transient=False
            )
            for stage in stages:
                self.stage_tasks[stage] = self.stage_progress.add_task(
                    f"[{self.colors['stage']}]{stage.title()}[/]",
                    total=total_videos,
                    status="waiting"
                )
                self.stats['stage_counts'][stage] = 0
    
    def update_stage_progress(self, stage: str, advance: int = 1, status: str = None):
        """Update progress for a specific stage."""
        with self._lock:
            # Throttle updates per stage
            current_time = time.time()
            stage_key = f"stage_{stage}"
            if stage_key in self._last_stage_update:
                if current_time - self._last_stage_update[stage_key] < self._update_interval:
                    return
            self._last_stage_update[stage_key] = current_time
            
            # Update stage progress
            if self.stage_progress and stage in self.stage_tasks:
                self.stats['stage_counts'][stage] += advance
                
                # Calculate completion percentage for better status display
                task = self.stage_progress.tasks[self.stage_tasks[stage]]
                completed = task.completed + advance
                total = task.total or 1
                completion_pct = (completed / total) * 100
                
                if status:
                    display_status = status
                elif completion_pct >= 100:
                    display_status = "completed"
                elif completed > 0:
                    display_status = f"processing ({completion_pct:.0f}%)"
                else:
                    display_status = "waiting"
                
                self.stage_progress.update(self.stage_tasks[stage], advance=advance, status=display_status)
                
                # Trigger live display update
                if self._display_active and self._should_update():
                    self.update_live_display()
    
    def update_stage_active_counts(self, stage_counts: Dict[str, int]):
        """Update active processing counts for stages."""
        with self._lock:
            if self.stage_progress:
                for stage, count in stage_counts.items():
                    if stage in self.stage_tasks:
                        task = self.stage_progress.tasks[self.stage_tasks[stage]]
                        completed = task.completed
                        total = task.total or 1
                        
                        if completed >= total:
                            status_text = "completed"
                        elif count > 0:
                            status_text = f"{count} active"
                        elif completed > 0:
                            status_text = f"processing ({(completed/total)*100:.0f}%)"
                        else:
                            status_text = "waiting"
                        
                        self.stage_progress.update(self.stage_tasks[stage], status=status_text)
                
                # Trigger live display update if significant change
                if self._display_active and any(count > 0 for count in stage_counts.values()):
                    if self._should_update():
                        self.update_live_display()

    def update_pipeline_progress(self, advance: int = 1, status: str = None):
        """No-op for backward compatibility."""
        pass

    def start_live_display(self):
        """Start the live display for real-time progress updates (stage progress only)."""
        if not self.stage_progress:
            return
        try:
            self.live = Live(self.stage_progress, console=self.console, refresh_per_second=0.2, vertical_overflow="visible")
            self.live.start()
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not start live display: {e}[/]")
            self._display_active = False
    
    def update_live_display(self):
        """Update the live display with current info (stage progress only)."""
        if self.live and self.stage_progress:
            self.live.update(self.stage_progress)

    def _should_update(self) -> bool:
        """Check if enough time has passed since last update."""
        current_time = time.time()
        if current_time - self._last_update >= self._update_interval:
            self._last_update = current_time
            return True
        return False
    
    def update_stats(self, completed: int = None, failed: int = None):
        """Update pipeline statistics."""
        with self._lock:
            if completed is not None:
                self.stats['completed'] = completed
            if failed is not None:
                self.stats['failed'] = failed
    
    def stop_live_display(self):
        """Stop the live display."""
        if self.live:
            self.live.stop()
            self.live = None
    
    def finish_pipeline_progress(self):
        """Finish and clean up pipeline progress."""
        with self._lock:
            self._display_active = False
            self._suppress_stage_messages = False
            if self.stage_progress:
                for stage, task_id in self.stage_tasks.items():
                    if not self.stage_progress.tasks[task_id].finished:
                        completed = self.stats['stage_counts'].get(stage, 0)
                        total = self.stats['total_videos']
                        if completed >= total:
                            self.stage_progress.update(task_id, status="completed")
                        else:
                            self.stage_progress.update(task_id, status=f"{completed}/{total} completed")
            self.stop_live_display()
            self.stage_progress = None
            self.stage_tasks.clear()
    
    def print_pipeline_summary(self, execution_times: Dict[str, float]):
        """Print a comprehensive pipeline execution summary."""
        self.console.print()
        
        # Create summary table
        table = Table(title="Pipeline Execution Summary", box=box.ROUNDED)
        table.add_column("Component", style=self.colors['stage'])
        table.add_column("Execution Time", style=self.colors['metric'])
        table.add_column("Videos/Min", style=self.colors['info'])
        
        # Use full_pipeline time if available, otherwise sum components
        full_pipeline_time = execution_times.get('full_pipeline')
        
        for component, exec_time in execution_times.items():
            # Calculate rate if we have video count
            rate = ""
            if self.stats['total_videos'] > 0 and exec_time > 0:
                videos_per_min = (self.stats['total_videos'] / exec_time) * 60
                rate = f"{videos_per_min:.1f}"
            
            table.add_row(
                component.replace('_', ' ').title(),
                str(timedelta(seconds=int(exec_time))),
                rate
            )
        
        # Add total row - use full pipeline time to avoid double counting
        if full_pipeline_time:
            total_display_time = full_pipeline_time
        else:
            # Fallback: sum all components if no full_pipeline time
            total_display_time = sum(execution_times.values())
            
        table.add_row(
            "[bold]Total Pipeline Time[/]",
            f"[bold]{timedelta(seconds=int(total_display_time))}[/]",
            ""
        )
        
        self.console.print(table)
        self.console.print()
    
    def print_video_status_table(self, video_statuses: Dict[str, Dict]):
        """Print a detailed status table for all videos."""
        if not video_statuses:
            return
        
        # Temporarily stop live display to print the table cleanly
        was_live = self.live is not None
        if was_live:
            self.stop_live_display()
        
        table = Table(title="Video Processing Status", box=box.SIMPLE)
        table.add_column("Video ID", style=self.colors['video_id'])
        table.add_column("Download", justify="center")
        table.add_column("Preprocess", justify="center")
        table.add_column("Caption", justify="center")
        table.add_column("Overall", style="bold")
        
        for video_id, status in video_statuses.items():
            # Map status to icons and colors
            status_map = {
                'pending': ('[dim]...[/]', 'dim'),
                'processing': ('[yellow]>>>[/]', 'yellow'),
                'completed': ('[green][OK][/]', 'green'),
                'failed': ('[red][X][/]', 'red')
            }

            download_icon, _ = status_map.get(status.get('download', 'pending'), ('?', 'dim'))
            preprocess_icon, _ = status_map.get(status.get('preprocess', 'pending'), ('?', 'dim'))
            caption_icon, _ = status_map.get(status.get('caption', 'pending'), ('?', 'dim'))

            overall_status = status.get('overall', 'pending')
            overall_icon, overall_color = status_map.get(overall_status, ('?', 'dim'))
            
            table.add_row(
                video_id[:12] + "..." if len(video_id) > 15 else video_id,
                download_icon,
                preprocess_icon,
                caption_icon,
                f"[{overall_color}]{overall_status.title()}[/]"
            )
        
        self.console.print(table)
        
        # Restart live display if it was active
        if was_live and self._display_active:
            self.start_live_display()
    
    def create_video_description_progress(self, video_id: str, total_segments: int) -> Progress:
        """Create a progress bar for video description generation."""
        progress = Progress(
            TextColumn(f"[{self.colors['video_id']}]{video_id}[/]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=0.5
        )
        
        task_id = progress.add_task(
            "Processing segments",
            total=total_segments
        )
        
        return progress, task_id
    
    def create_metadata_progress(self, total_videos: int) -> Progress:
        """Create a progress bar for metadata generation."""
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=0.5
        )
        
        task_id = progress.add_task(
            f"[{self.colors['progress']}]Generating metadata[/]",
            total=total_videos
        )
        
        return progress, task_id
    
    def create_multimodal_understanding_progress(self, total_videos: int) -> tuple[Progress, TaskID]:
        """Create a progress bar for multimodal understanding generation."""
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("|"),
            StatusColumn(),
            console=self.console,
            refresh_per_second=0.5,
            transient=True  # Make progress bar disappear after completion
        )

        task_id = progress.add_task(
            f"[{self.colors['progress']}]Multimodal Understanding[/]",
            total=total_videos,
            status=""
        )
        
        return progress, task_id
    
    def create_video_segment_progress(self, video_id: str, total_segments: int) -> tuple[Progress, TaskID]:
        """Create a progress bar for processing segments within a video."""
        progress = Progress(
            TextColumn(f"[{self.colors['video_id']}]{video_id}[/]"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("segments"),
            TextColumn("|"),
            RateColumn(),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=1
        )
        
        task_id = progress.add_task(
            "Processing",
            total=total_segments
        )
        
        return progress, task_id
    
    def create_multimodal_alignment_progress(self, operation: str, total_tasks: int) -> tuple[Progress, TaskID]:
        """Create a progress bar for multimodal understanding alignment."""
        progress = Progress(
            TextColumn(f"[{self.colors['progress']}]Multimodal {operation}[/]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=0.5
        )
        
        task_id = progress.add_task(
            "Aligning segments",
            total=total_tasks
        )
        
        return progress, task_id
    
    def create_consolidation_progress(self, total_videos: int) -> tuple[Progress, TaskID]:
        """Create a progress bar for final consolidation."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Final Consolidation"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("videos"),
            TextColumn("|"),
            RateColumn(),
            TextColumn("|"),
            StatusColumn(),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=2
        )
        
        task_id = progress.add_task(
            "Consolidating videos...",
            total=total_videos
        )
        
        return progress, task_id
    
    def create_key_events_progress(self, operation: str, total_tasks: int) -> tuple[Progress, TaskID]:
        """Create a progress bar for key events generation."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[bold magenta]{operation} Generation"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("segments"),
            TextColumn("|"),
            RateColumn(),
            TextColumn("|"),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=2
        )

        task_id = progress.add_task(
            f"Generating {operation.lower()}...",
            total=total_tasks
        )
        
        return progress, task_id

    def print_component_header(self, component: str, action: str):
        """Print a header for a specific component."""
        header = f"[{self.colors['stage']}]{component}[/] - {action}"
        self.console.print(f"\n{header}")
        self.console.print("─" * 60)
    
    def print_completion_message(self, component: str, stats: Dict[str, Any]):
        """Print a completion message with statistics."""
        success_count = stats.get('successful', 0)
        total_count = stats.get('total', 0)
        failed_count = total_count - success_count if isinstance(success_count, int) else 0
        duration = stats.get('duration', 0)
        
        # Create completion status message
        if failed_count == 0:
            status = f"[{self.colors['success']}][OK] {component} completed - {total_count} items processed in {timedelta(seconds=int(duration))}[/]"
        else:
            status = f"[{self.colors['warning']}]! {component} completed - {success_count}/{total_count} successful in {timedelta(seconds=int(duration))}[/]"
        
        # Use important info to make sure completion messages are always visible
        self.print_important_info(status, "")
        self.console.print()


# Global console instance
console = PipelineConsole()


def get_console() -> PipelineConsole:
    """Get the global console instance."""
    return console