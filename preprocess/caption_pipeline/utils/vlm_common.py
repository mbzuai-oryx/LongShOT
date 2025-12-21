"""
Common utilities for VLM (Vision Language Model) stages.

This module provides shared functionality used across video_descriptor, audio_descriptor,
video_description_aligner, metadata_generator, and related VLM pipeline stages.
"""

import os
import json
import shutil
from typing import Dict, List, Optional, Any

# Import rich console for logging
from caption_pipeline.utils.rich_console import get_console

rich_console = get_console()


# =============================================================================
# vLLM Client Management
# =============================================================================

def create_vllm_client(api_base: str):
    """Create an OpenAI client for vLLM server communication.

    Args:
        api_base: Base URL for the vLLM server API

    Returns:
        OpenAI client instance
    """
    from openai import OpenAI
    return OpenAI(api_key="EMPTY", base_url=api_base)


def test_vllm_connection(model_name: str, api_base: str, console=None, raise_on_error: bool = True, timeout: int = 30) -> bool:
    """Test connection to vLLM server.

    Args:
        model_name: Name of the model to test
        api_base: Base URL for the vLLM server
        console: Optional console for logging (uses default if None)
        raise_on_error: Whether to raise exception on connection failure
        timeout: Connection timeout in seconds (default: 30)

    Returns:
        True if connection successful, False otherwise
    """
    if console is None:
        console = rich_console

    try:
        from openai import OpenAI
        client = OpenAI(api_key="EMPTY", base_url=api_base, timeout=timeout)
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        console.print_info(f"vLLM server connection successful at {api_base}")
        return True
    except Exception as e:
        console.print_error(f"Failed to connect to vLLM server: {e}")
        if raise_on_error:
            raise
        return False


# =============================================================================
# JSON File Operations
# =============================================================================

def load_json_file(file_path: str, console=None) -> Optional[Dict]:
    """Load JSON file with standard error handling.

    Args:
        file_path: Path to JSON file
        console: Optional console for logging

    Returns:
        Loaded dictionary or None if failed
    """
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        if console:
            console.print_error(f"Error loading {file_path}: {e}")
        return None


def save_json_file(file_path: str, data: Dict, atomic: bool = True, console=None) -> bool:
    """Save JSON file, optionally with atomic write (temp file + move).

    Args:
        file_path: Path to save JSON file
        data: Dictionary to save
        atomic: Whether to use atomic write (temp file + move)
        console: Optional console for logging

    Returns:
        True if successful, False otherwise
    """
    try:
        if atomic:
            temp_file = file_path + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            shutil.move(temp_file, file_path)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        if console:
            console.print_error(f"Error saving {file_path}: {e}")
        # Clean up temp file if it exists
        if atomic:
            temp_file = file_path + '.tmp'
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        return False


def load_json_with_fallback(primary_path: str, fallback_path: str, console=None) -> Optional[Dict]:
    """Load JSON from primary path, fall back to secondary if needed.

    Args:
        primary_path: Primary file path to try first
        fallback_path: Fallback file path if primary doesn't exist
        console: Optional console for logging

    Returns:
        Loaded dictionary or None if both failed
    """
    if os.path.exists(primary_path):
        return load_json_file(primary_path, console)
    if os.path.exists(fallback_path):
        return load_json_file(fallback_path, console)
    return None


# =============================================================================
# Segment Utilities
# =============================================================================

# Invalid description strings to check against
INVALID_DESCRIPTIONS = {
    "No audio description available",
    "No description available",
    "No frames could be extracted from this video segment",
    "No multimodal understanding available",
    "",
    None
}


def has_valid_description(segment: Dict, field: str = 'description') -> bool:
    """Check if segment has valid description.

    Args:
        segment: Segment dictionary
        field: Field name to check (default: 'description')

    Returns:
        True if description is valid, False otherwise
    """
    desc = segment.get(field)
    return desc and desc not in INVALID_DESCRIPTIONS


def create_segment_result(segment: Dict, description: str,
                         description_field: str = 'description',
                         processing_method: str = '', **extras) -> Dict:
    """Create standardized segment result dictionary.

    Args:
        segment: Original segment dictionary
        description: Generated description
        description_field: Field name for description (default: 'description')
        processing_method: Processing method used
        **extras: Additional fields to include

    Returns:
        Segment result dictionary
    """
    result = {
        'start': segment.get('start', 0),
        'end': segment.get('end', 0),
        'segment_type': segment.get('segment_type', 'audio'),
        'processing_method': processing_method,
        description_field: description
    }
    result.update(extras)
    return result


# =============================================================================
# File Cleanup Utilities
# =============================================================================

def safe_remove_file(file_path: str) -> bool:
    """Safely remove a file with error handling.

    Args:
        file_path: Path to file to remove

    Returns:
        True if successfully removed or didn't exist, False on error
    """
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            return True
        except Exception:
            return False
    return True


def cleanup_temp_file(file_path: str) -> None:
    """Clean up a temporary file (silent, no return value).

    Args:
        file_path: Path to temp file to remove
    """
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass


# =============================================================================
# Progress Utilities
# =============================================================================

def calculate_progress_interval(total: int, target_updates: int = 10, minimum: int = 1) -> int:
    """Calculate progress update interval for reporting.

    Args:
        total: Total number of items
        target_updates: Target number of progress updates (default: 10)
        minimum: Minimum interval value (default: 1)

    Returns:
        Progress interval
    """
    return max(minimum, total // target_updates)


# =============================================================================
# Error Logging Helper
# =============================================================================

class ThrottledErrorLogger:
    """Log errors with suppression after threshold.

    This class helps reduce log spam by suppressing repeated errors
    after a configurable threshold has been reached.
    """

    def __init__(self, threshold: int = 3, console=None):
        """Initialize throttled error logger.

        Args:
            threshold: Number of errors to log before suppression
            console: Console for logging (uses default if None)
        """
        self.threshold = threshold
        self.console = console or rich_console
        self.error_count = 0
        self._suppression_logged = False

    def log(self, message: str) -> None:
        """Log an error message with throttling.

        Args:
            message: Error message to log
        """
        self.error_count += 1
        if self.error_count <= self.threshold:
            self.console.print_error(message)
        elif not self._suppression_logged:
            self.console.print_warning("Further errors will be suppressed...")
            self._suppression_logged = True

    def reset(self) -> None:
        """Reset the error counter."""
        self.error_count = 0
        self._suppression_logged = False

    @property
    def total_errors(self) -> int:
        """Return total error count."""
        return self.error_count


# =============================================================================
# Frame Count Calculation (for video_descriptor)
# =============================================================================

# Frame count tiers based on duration
FRAME_COUNT_TIERS = [
    (0.5, 1),
    (2.0, 4),
    (5.0, 8),
    (10.0, 16),
    (20.0, 24),
    (float('inf'), 32)
]


def calculate_optimal_frame_count(duration: float, max_frames: int) -> int:
    """Calculate optimal number of frames based on duration.

    Uses a tiered approach targeting 1-2fps with adaptive scaling.

    Args:
        duration: Segment duration in seconds
        max_frames: Maximum allowed frames

    Returns:
        Optimal frame count for the segment
    """
    for threshold, count in FRAME_COUNT_TIERS:
        if duration <= threshold:
            return min(count, max_frames)
    return min(32, max_frames)


# =============================================================================
# Aligned File Loading
# =============================================================================

def load_aligned_json(base_path: str, suffixes: List[str] = None, console=None) -> Optional[Dict]:
    """Load JSON file, trying suffixes in order (aligned first by default).

    This handles the common pattern of preferring aligned files over originals:
    - Try {base_path}_aligned.json first
    - Fall back to {base_path}.json

    Args:
        base_path: Base path without extension (e.g., '/path/to/video_descriptions')
        suffixes: List of suffixes to try in order (default: ['_aligned', ''])
        console: Optional console for logging

    Returns:
        Loaded dictionary or None if all attempts failed

    Example:
        data = load_aligned_json('/data/video_id_descriptions')
        # Tries: /data/video_id_descriptions_aligned.json
        # Then:  /data/video_id_descriptions.json
    """
    if suffixes is None:
        suffixes = ['_aligned', '']

    for suffix in suffixes:
        path = f"{base_path}{suffix}.json"
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                if console:
                    console.print_warning(f"Failed to load {path}: {e}")
                continue
    return None


# =============================================================================
# Concurrent Processing Utilities
# =============================================================================

def collect_concurrent_results(future_to_index: Dict, progress=None, task_id=None,
                               error_value=None, console=None) -> List:
    """Collect results from concurrent futures with optional progress tracking.

    This eliminates the common duplicated pattern of if/else progress tracking
    in concurrent processing across multiple pipeline files.

    Args:
        future_to_index: Dict mapping futures to their result indices
        progress: Optional progress bar object (Rich Progress)
        task_id: Optional task ID for progress updates
        error_value: Value to use for failed results (default: None)
        console: Optional console for error logging

    Returns:
        List of results in index order

    Example:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(fn, arg): i for i, arg in enumerate(args)}
            results = collect_concurrent_results(futures, progress, task_id)
    """
    from concurrent.futures import as_completed

    results = [error_value] * len(future_to_index)

    for future in as_completed(future_to_index):
        index = future_to_index[future]
        try:
            results[index] = future.result()
        except Exception as e:
            if console:
                console.print_error(f"Error processing item {index}: {e}")
            results[index] = error_value

        if progress is not None and task_id is not None:
            progress.update(task_id, advance=1)

    return results


def format_context_segments(segments: List[Dict], desc_field: str = 'description') -> str:
    """Format previous segments into context string for alignment prompts.

    Replaces string concatenation in loops with join pattern.

    Args:
        segments: List of segment dictionaries
        desc_field: Field name containing description

    Returns:
        Formatted context string with segments separated by double newlines
    """
    lines = []
    for i, seg in enumerate(segments):
        segment_num = seg.get('segment_index', i)
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        desc = seg.get(desc_field, '')
        lines.append(f"Segment {segment_num} ({start:.1f}s - {end:.1f}s): {desc}")
    return "\n\n".join(lines)
