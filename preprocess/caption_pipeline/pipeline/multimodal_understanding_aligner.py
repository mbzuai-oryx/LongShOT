"""
Multimodal Understanding Aligner module for temporal alignment of multimodal segment descriptions.

This module performs a second pass on already-generated multimodal understanding descriptions to align
consecutive segments temporally by providing previous segment context. This approach enhances the 
temporal continuity and narrative flow across multimodal segments.
"""

import os
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
import sys

# Import project configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import LLM_MODEL, LLM_SERVER_URL, MULTIMODAL_UNDERSTANDING_DIR

# Import rich console utilities
from caption_pipeline.utils.rich_console import get_console

# Import common VLM utilities
from caption_pipeline.utils.vlm_common import (
    create_vllm_client, has_valid_description, collect_concurrent_results
)

# Set up logging
logger = logging.getLogger(__name__)
rich_console = get_console()

# Comprehensive multimodal alignment prompt with integrated temporal guidance
MULTIMODAL_ALIGNMENT_PROMPT = """
Create a comprehensive multimodal description that integrates visual, audio, and contextual elements with smooth temporal alignment to the previous segment.

PRIMARY OBJECTIVE: Synthesize current multimodal content into a coherent description that flows naturally from previous context while highlighting meaningful developments across all modalities.

MULTIMODAL INTEGRATION FRAMEWORK:
1. Analyze visual-audio relationships and how they've evolved from previous context
2. Identify cross-modal interactions and temporal changes
3. Create unified description that captures multimodal continuity and progression
4. Maintain narrative coherence across different sensory modalities

TEMPORAL ALIGNMENT STRUCTURE:
- Target 100-150 words for comprehensive multimodal coverage
- Begin with primary modal continuation or transition from previous segment
- Integrate visual, audio, and contextual elements cohesively
- Describe cross-modal relationships (how visual and audio elements interact)
- Include temporal progression indicators across modalities
- Conclude with prominent new developments or changes

MULTIMODAL QUALITY INDICATORS:
- Seamless integration of visual and audio elements
- Clear temporal progression from previous context
- Cross-modal relationships and interactions described
- Unified narrative that captures the complete sensory experience
- Natural flow connecting different modality descriptions

Previous Multimodal Context: {previous_multimodal_understanding}
Current Multimodal Content: {current_multimodal_understanding}

Integrated Aligned Description:
"""


class MultimodalUnderstandingAligner:
    """Aligns consecutive multimodal understanding segments for temporal continuity."""
    
    def __init__(self, model_name=LLM_MODEL, api_base=LLM_SERVER_URL, max_workers=8):
        """Initialize the multimodal understanding aligner.
        
        Args:
            model_name: Name of the language model to use for alignment
            api_base: Base URL for the vLLM server API
            max_workers: Maximum number of concurrent workers
        """
        self.model_name = model_name
        self.api_base = api_base
        self.max_workers = max_workers
        
        rich_console.print_info(f"Multimodal Understanding Aligner initialized with model: {model_name}")
        rich_console.print_info(f"Using API base: {self.api_base}")

    def align_multimodal_understanding(self, multimodal_file: str) -> Optional[str]:
        """Align multimodal understanding in a file to improve temporal continuity.
        
        Args:
            multimodal_file: Path to the multimodal understanding JSON file
            
        Returns:
            Path to the aligned multimodal understanding file, or None if failed
        """
        rich_console.print_component_header("Multimodal Understanding Alignment", 
                                           f"Aligning understanding from {os.path.basename(multimodal_file)}")
        
        start_time = time.time()
        
        # Load existing multimodal understanding
        try:
            with open(multimodal_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            rich_console.print_error(f"Error loading multimodal understanding file: {e}")
            return None
        
        segments = data.get('segments', [])
        if len(segments) < 2:
            rich_console.print_warning("Video has less than 2 segments, no alignment needed")
            return multimodal_file
        
        # Check if already aligned
        if any(seg.get('processing_method', '').endswith('_aligned') for seg in segments):
            rich_console.print_info("Multimodal understanding already aligned, skipping")
            return multimodal_file
        
        video_id = data.get('video_id', 'unknown')
        rich_console.print_info(f"Aligning {len(segments)} multimodal segments for video {video_id}")
        
        # Create alignment tasks (skip first segment, no previous context)
        alignment_tasks = []
        for i in range(1, len(segments)):
            current_segment = segments[i]
            previous_segment = segments[i - 1]

            # Only align segments that have valid multimodal understanding
            if (has_valid_description(current_segment, 'multimodal_understanding') and
                has_valid_description(previous_segment, 'multimodal_understanding')):
                alignment_tasks.append({
                    'segment_index': i,
                    'current_understanding': current_segment['multimodal_understanding'],
                    'previous_understanding': previous_segment['multimodal_understanding']
                })
        
        if not alignment_tasks:
            rich_console.print_warning("No segments found that need alignment")
            return multimodal_file
        
        rich_console.print_info(f"Created {len(alignment_tasks)} alignment tasks")
        
        # Process alignments concurrently
        aligned_understandings = self._process_alignments_concurrent(alignment_tasks)
        
        # Update segments with aligned understanding
        aligned_count = 0
        for task_index, aligned_understanding in enumerate(aligned_understandings):
            if aligned_understanding and aligned_understanding != "ALIGNMENT_ERROR":
                segment_index = alignment_tasks[task_index]['segment_index']
                segments[segment_index]['multimodal_understanding'] = aligned_understanding
                
                # Update processing method to indicate alignment
                original_method = segments[segment_index].get('processing_method', 'unknown')
                segments[segment_index]['processing_method'] = f"{original_method}_aligned"
                
                aligned_count += 1
        
        # Update metadata
        data['alignment_info'] = {
            'alignment_completed': True,
            'alignment_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'segments_aligned': aligned_count,
            'total_segments': len(segments),
            'alignment_method': 'concurrent_multimodal_temporal_alignment'
        }
        
        # Create aligned file path
        base_name = os.path.splitext(multimodal_file)[0]
        aligned_file = f"{base_name}_aligned.json"
        
        # Save aligned understanding
        try:
            with open(aligned_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            processing_time = time.time() - start_time
            rich_console.print_success(f"Alignment completed for {video_id} in {processing_time:.1f}s")
            rich_console.print_info(f"Aligned {aligned_count}/{len(alignment_tasks)} segments")
            rich_console.print_info(f"Aligned multimodal understanding saved to: {aligned_file}")
            
            return aligned_file
            
        except Exception as e:
            rich_console.print_error(f"Error saving aligned multimodal understanding: {e}")
            return None
    
    def _process_alignments_concurrent(self, alignment_tasks: List[Dict]) -> List[str]:
        """Process alignment tasks concurrently."""
        if not alignment_tasks:
            return []

        rich_console.print_info(f"Processing {len(alignment_tasks)} alignment tasks with {self.max_workers} workers")

        # Create progress tracking
        try:
            progress, task_id = rich_console.create_multimodal_alignment_progress("Alignment", len(alignment_tasks))
        except Exception:
            progress, task_id = None, None

        # Process alignments concurrently
        max_concurrent = min(len(alignment_tasks), self.max_workers)
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_index = {
                executor.submit(
                    self._align_single_understanding,
                    task['current_understanding'],
                    task['previous_understanding']
                ): i
                for i, task in enumerate(alignment_tasks)
            }

            # Use shared utility for result collection with progress
            if progress:
                with progress:
                    return collect_concurrent_results(
                        future_to_index, progress, task_id,
                        error_value="ALIGNMENT_ERROR", console=rich_console
                    )
            else:
                return collect_concurrent_results(
                    future_to_index, error_value="ALIGNMENT_ERROR", console=rich_console
                )
    
    def _align_single_understanding(self, current_understanding: str, previous_understanding: str) -> str:
        """Align a single segment's multimodal understanding with its previous segment."""
        try:
            client = create_vllm_client(self.api_base)
            
            # Create prompt for alignment
            prompt = MULTIMODAL_ALIGNMENT_PROMPT.format(
                previous_multimodal_understanding=previous_understanding,
                current_multimodal_understanding=current_understanding
            )
            
            # Make request to vLLM server
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,  # Allow for comprehensive multimodal descriptions
                temperature=0.3   # Lower temperature for more consistent alignment
            )
            
            aligned_understanding = response.choices[0].message.content.strip()
            
            # Basic validation - ensure we got a reasonable response
            if len(aligned_understanding) < 80:  # Too short, likely an error
                rich_console.print_warning("Alignment produced very short result, using original")
                return current_understanding
            
            return aligned_understanding
            
        except Exception as e:
            rich_console.print_error(f"Error in alignment request: {e}")
            return current_understanding  # Fall back to original understanding
    
    def batch_align_multimodal_understanding(self, video_ids: List[str] = None, max_videos: int = None, 
                                           multimodal_dir: str = None) -> List[str]:
        """Align multimodal understanding for multiple videos.
        
        Args:
            video_ids: List of specific video IDs to align, or None for all
            max_videos: Maximum number of videos to process
            multimodal_dir: Directory containing multimodal understanding files
            
        Returns:
            List of aligned multimodal understanding file paths
        """
        if multimodal_dir is None:
            multimodal_dir = MULTIMODAL_UNDERSTANDING_DIR
        
        rich_console.print_component_header("Batch Multimodal Understanding Alignment", 
                                           f"Processing videos in {multimodal_dir}")
        
        # Find multimodal understanding files to process
        files_to_process = []
        
        if video_ids:
            # Process specific video IDs
            for video_id in video_ids:
                multimodal_file = os.path.join(multimodal_dir, f"{video_id}_multimodal_understanding.json")
                if os.path.exists(multimodal_file):
                    files_to_process.append(multimodal_file)
                else:
                    rich_console.print_warning(f"Multimodal understanding file not found for {video_id}")
        else:
            # Find all multimodal understanding files that aren't already aligned
            for filename in os.listdir(multimodal_dir):
                if filename.endswith('_multimodal_understanding.json') and not filename.endswith('_aligned.json'):
                    multimodal_file = os.path.join(multimodal_dir, filename)
                    files_to_process.append(multimodal_file)
        
        if max_videos:
            files_to_process = files_to_process[:max_videos]
        
        rich_console.print_info(f"Found {len(files_to_process)} multimodal understanding files to align")
        
        if not files_to_process:
            rich_console.print_warning("No multimodal understanding files found to align")
            return []
        
        # Process alignments
        aligned_files = []
        failed_files = []
        
        for i, multimodal_file in enumerate(files_to_process, 1):
            video_id = os.path.basename(multimodal_file).replace('_multimodal_understanding.json', '')
            rich_console.print_info(f"Aligning multimodal understanding for video {i}/{len(files_to_process)}: {video_id}")
            
            try:
                aligned_file = self.align_multimodal_understanding(multimodal_file)
                if aligned_file:
                    aligned_files.append(aligned_file)
                    rich_console.print_success(f"Successfully aligned multimodal understanding for {video_id} ({i}/{len(files_to_process)})")
                else:
                    failed_files.append(video_id)
                    rich_console.print_error(f"Failed to align multimodal understanding for {video_id} ({i}/{len(files_to_process)})")
            except Exception as e:
                failed_files.append(video_id)
                rich_console.print_error(f"Error aligning multimodal understanding for {video_id}: {e}")
        
        # Print summary
        success_count = len(aligned_files)
        total_count = len(files_to_process)
        rich_console.print_completion_message("Multimodal Understanding Alignment", {
            'total': total_count,
            'successful': success_count,
            'duration': 0  # Duration calculated externally
        })
        
        if failed_files:
            rich_console.print_warning(f"Failed to align: {', '.join(failed_files)}")
        
        return aligned_files
