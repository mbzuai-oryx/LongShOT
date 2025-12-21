"""
Audio Description Aligner module for temporal and spatial alignment of audio environment descriptions.

This module performs a second pass on already-generated audio descriptions to align
consecutive segments temporally and spatially by providing multiple previous segments as context.
This approach enhances the temporal continuity and environmental consistency across audio segments.
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
from config import LLM_MODEL, LLM_SERVER_URL, AUDIO_DESCRIPTIONS_DIR

# Import rich console utilities
from caption_pipeline.utils.rich_console import get_console

# Import common VLM utilities
from caption_pipeline.utils.vlm_common import (
    create_vllm_client, has_valid_description,
    collect_concurrent_results, format_context_segments
)

# Set up logging
logger = logging.getLogger(__name__)
rich_console = get_console()

# Constants for alignment
ALIGNMENT_MAX_TOKENS = 500  # Tokens for comprehensive audio descriptions
ALIGNMENT_TEMPERATURE = 0.3  # Lower temperature for consistent alignment
MIN_DESCRIPTION_LENGTH = 50  # Minimum valid description length

# Improved audio alignment prompt with positive guidance for temporal coherence
AUDIO_ALIGNMENT_PROMPT = """
Create a temporally coherent audio description by analyzing the current segment in relation to the previous context.

PRIMARY OBJECTIVE: Refine the current audio description to enhance temporal continuity while highlighting genuine changes.

TEMPORAL ANALYSIS FRAMEWORK:
1. Compare current audio elements with previous context
2. Identify transitions, changes, or new audio elements
3. Create a coherent description that flows naturally from previous segments
4. Maintain consistency in audio environment description

DESCRIPTION GUIDELINES:
- Target 60-100 words for comprehensive coverage
- Focus on acoustic changes, new elements, or environmental shifts
- Describe audio transitions smoothly (e.g., "sound fades to," "new layer emerges," "environment shifts to")
- Use varied, descriptive language that captures audio nuances
- Include spatial audio cues when relevant (direction, distance, movement)

QUALITY INDICATORS:
- Smooth transition from previous audio context
- Clear identification of new or changed audio elements
- Consistent environmental description
- Natural, flowing language without repetitive phrases

Previous Context: {previous_audio_context}
Current Description: {current_audio_description}

Refined Aligned Description:
"""


class AudioDescriptionAligner:
    """Aligns consecutive audio description segments for temporal and spatial continuity."""
    
    def __init__(self, model_name=LLM_MODEL, api_base=LLM_SERVER_URL, max_workers=8, context_segments=3):
        """Initialize the audio description aligner.
        
        Args:
            model_name: Name of the language model to use for alignment
            api_base: Base URL for the vLLM server API
            max_workers: Maximum number of concurrent workers
            context_segments: Number of previous segments to use as context (default: 3)
        """
        self.model_name = model_name
        self.api_base = api_base
        self.max_workers = max_workers
        self.context_segments = context_segments
        
        rich_console.print_info(f"Audio Description Aligner initialized with model: {model_name}")
        rich_console.print_info(f"Using API base: {self.api_base}")
        rich_console.print_info(f"Context segments: {self.context_segments}")

    def align_audio_descriptions(self, audio_descriptions_file: str) -> Optional[str]:
        """Align audio descriptions in a file to improve temporal and spatial continuity.
        
        Args:
            audio_descriptions_file: Path to the audio descriptions JSON file
            
        Returns:
            Path to the aligned audio descriptions file, or None if failed
        """
        rich_console.print_component_header("Audio Description Alignment", 
                                           f"Aligning descriptions from {os.path.basename(audio_descriptions_file)}")
        
        start_time = time.time()
        
        # Load existing audio descriptions
        try:
            with open(audio_descriptions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            rich_console.print_error(f"Error loading audio descriptions file: {e}")
            return None
        
        segments = data.get('segments', [])
        if len(segments) < 2:
            rich_console.print_warning("Audio file has less than 2 segments, no alignment needed")
            return audio_descriptions_file
        
        # Check if already aligned
        if any(seg.get('processing_method', '').endswith('_aligned') for seg in segments):
            rich_console.print_info("Audio descriptions already aligned, skipping")
            return audio_descriptions_file
        
        video_id = data.get('video_id', 'unknown')
        rich_console.print_info(f"Aligning {len(segments)} audio segments for video {video_id}")
        
        # Create alignment tasks (skip first segment or segments without enough context)
        alignment_tasks = []
        for i in range(1, len(segments)):  # Start from second segment
            current_segment = segments[i]

            # Only align segments that have valid audio descriptions
            if has_valid_description(current_segment, 'audio_description'):
                # Gather context from previous segments
                context_start = max(0, i - self.context_segments)
                previous_segments = segments[context_start:i]

                # Filter previous segments to only include those with valid audio descriptions
                valid_previous_segments = [
                    seg for seg in previous_segments
                    if has_valid_description(seg, 'audio_description')
                ]

                # Only create alignment task if we have valid previous context
                if valid_previous_segments:
                    alignment_tasks.append({
                        'segment_index': i,
                        'current_description': current_segment['audio_description'],
                        'previous_segments': valid_previous_segments
                    })
        
        if not alignment_tasks:
            rich_console.print_warning("No segments found that need alignment")
            return audio_descriptions_file
        
        rich_console.print_info(f"Created {len(alignment_tasks)} alignment tasks")
        
        # Process alignments concurrently
        aligned_descriptions = self._process_alignments_concurrent(alignment_tasks)
        
        # Update segments with aligned descriptions
        aligned_count = 0
        for task_index, aligned_desc in enumerate(aligned_descriptions):
            if aligned_desc and aligned_desc != "ALIGNMENT_ERROR":
                segment_index = alignment_tasks[task_index]['segment_index']
                segments[segment_index]['audio_description'] = aligned_desc
                
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
            'context_segments_used': self.context_segments,
            'alignment_method': 'concurrent_audio_temporal_spatial_alignment'
        }
        
        # Create aligned file path
        base_name = os.path.splitext(audio_descriptions_file)[0]
        aligned_file = f"{base_name}_aligned.json"
        
        # Save aligned descriptions
        try:
            with open(aligned_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            processing_time = time.time() - start_time
            rich_console.print_success(f"Alignment completed for {video_id} in {processing_time:.1f}s")
            rich_console.print_info(f"Aligned {aligned_count}/{len(alignment_tasks)} segments")
            rich_console.print_info(f"Aligned audio descriptions saved to: {aligned_file}")
            
            return aligned_file
            
        except Exception as e:
            rich_console.print_error(f"Error saving aligned audio descriptions: {e}")
            return None
    
    def _process_alignments_concurrent(self, alignment_tasks: List[Dict]) -> List[str]:
        """Process alignment tasks concurrently."""
        if not alignment_tasks:
            return []

        rich_console.print_info(f"Processing {len(alignment_tasks)} alignment tasks with {self.max_workers} workers")

        # Create progress tracking
        try:
            progress, task_id = rich_console.create_multimodal_alignment_progress("Audio Alignment", len(alignment_tasks))
        except Exception:
            progress, task_id = None, None

        # Process alignments concurrently
        max_concurrent = min(len(alignment_tasks), self.max_workers)
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_index = {
                executor.submit(
                    self._align_single_description,
                    task['current_description'],
                    task['previous_segments']
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
    
    def _align_single_description(self, current_description: str, previous_segments: List[Dict]) -> str:
        """Align a single segment's audio description with multiple previous segments."""
        try:
            client = create_vllm_client(self.api_base)

            # Build context from previous segments using shared utility
            previous_context = format_context_segments(previous_segments, 'audio_description')

            # Create prompt for alignment
            prompt = AUDIO_ALIGNMENT_PROMPT.format(
                previous_audio_context=previous_context,
                current_audio_description=current_description
            )

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=ALIGNMENT_MAX_TOKENS,
                temperature=ALIGNMENT_TEMPERATURE
            )

            aligned_description = response.choices[0].message.content.strip()

            # Validate response length
            if len(aligned_description) < MIN_DESCRIPTION_LENGTH:
                rich_console.print_warning("Alignment produced very short result, using original")
                return current_description

            return aligned_description

        except Exception as e:
            rich_console.print_error(f"Error in alignment request: {e}")
            return current_description
    
    def batch_align_audio_descriptions(self, video_ids: List[str] = None, max_videos: int = None, 
                                     audio_descriptions_dir: str = None) -> List[str]:
        """Align audio descriptions for multiple videos.
        
        Args:
            video_ids: List of specific video IDs to align, or None for all
            max_videos: Maximum number of videos to process
            audio_descriptions_dir: Directory containing audio description files
            
        Returns:
            List of aligned audio description file paths
        """
        if audio_descriptions_dir is None:
            audio_descriptions_dir = AUDIO_DESCRIPTIONS_DIR
        
        rich_console.print_component_header("Batch Audio Description Alignment", 
                                           f"Processing videos in {audio_descriptions_dir}")
        
        # Find audio description files to process
        files_to_process = []
        
        if video_ids:
            # Process specific video IDs
            for video_id in video_ids:
                audio_file = os.path.join(audio_descriptions_dir, f"{video_id}_audio_descriptions.json")
                if os.path.exists(audio_file):
                    files_to_process.append(audio_file)
                else:
                    rich_console.print_warning(f"Audio descriptions file not found for {video_id}")
        else:
            # Find all audio description files that aren't already aligned
            for filename in os.listdir(audio_descriptions_dir):
                if filename.endswith('_audio_descriptions.json') and not filename.endswith('_aligned.json'):
                    audio_file = os.path.join(audio_descriptions_dir, filename)
                    files_to_process.append(audio_file)
        
        if max_videos:
            files_to_process = files_to_process[:max_videos]
        
        rich_console.print_info(f"Found {len(files_to_process)} audio description files to align")
        
        if not files_to_process:
            rich_console.print_warning("No audio description files found to align")
            return []
        
        # Process alignments
        aligned_files = []
        failed_files = []
        
        for i, audio_file in enumerate(files_to_process, 1):
            video_id = os.path.basename(audio_file).replace('_audio_descriptions.json', '')
            rich_console.print_info(f"Aligning audio descriptions for video {i}/{len(files_to_process)}: {video_id}")
            
            try:
                aligned_file = self.align_audio_descriptions(audio_file)
                if aligned_file:
                    aligned_files.append(aligned_file)
                    rich_console.print_success(f"Successfully aligned audio descriptions for {video_id} ({i}/{len(files_to_process)})")
                else:
                    failed_files.append(video_id)
                    rich_console.print_error(f"Failed to align audio descriptions for {video_id} ({i}/{len(files_to_process)})")
            except Exception as e:
                failed_files.append(video_id)
                rich_console.print_error(f"Error aligning audio descriptions for {video_id}: {e}")
        
        # Print summary
        success_count = len(aligned_files)
        total_count = len(files_to_process)
        rich_console.print_completion_message("Audio Description Alignment", {
            'total': total_count,
            'successful': success_count,
            'duration': 0  # Duration calculated externally
        })
        
        if failed_files:
            rich_console.print_warning(f"Failed to align: {', '.join(failed_files)}")

        return aligned_files
