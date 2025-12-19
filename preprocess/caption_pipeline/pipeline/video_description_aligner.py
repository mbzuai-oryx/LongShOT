"""
Video Description Aligner module for temporal alignment of video segment descriptions.

This module performs a second pass on already-generated video descriptions to align
consecutive segments temporally by providing previous segment context. This approach
maintains the speed of concurrent processing while adding temporal continuity.
"""

import os
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
import sys

# Import project configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import VIDEO_DESCRIPTION_MODEL, VLLM_SERVER_URL

# Import rich console utilities
from caption_pipeline.utils.rich_console import get_console

# Import required components
from openai import OpenAI

# Set up logging
logger = logging.getLogger(__name__)
rich_console = get_console()

# Enhanced visual alignment prompt with structured temporal guidance
ALIGNMENT_PROMPT = """
Create a temporally aligned visual description that maintains continuity with the previous segment while clearly capturing new visual developments.

PRIMARY OBJECTIVE: Refine the current visual description to create smooth temporal flow and highlight meaningful visual progression.

TEMPORAL ALIGNMENT PROCESS:
1. Analyze visual elements that carry over from the previous segment
2. Identify new visual actions, movements, or environmental changes
3. Describe transitions between segments naturally
4. Maintain consistent perspective and visual style

VISUAL DESCRIPTION FRAMEWORK:
- Target 80-120 words for comprehensive visual coverage
- Describe subject movements, positioning, and actions
- Note environmental changes, lighting shifts, or new visual elements  
- Use transition words that connect to previous visual context
- Include spatial relationships and directional movements
- Capture visual continuity while emphasizing new developments

EFFECTIVE TRANSITIONS:
✓ "Movement continues as..." / "Action progresses to..."
✓ "Visual focus shifts from... to..."
✓ "Scene develops with..." / "Frame reveals..."
✓ Smooth narrative flow connecting visual elements

Previous Visual Context: {previous_description}
Current Visual Content: {current_description}

Temporally Aligned Description:
"""


class VideoDescriptionAligner:
    """Aligns consecutive video segment descriptions for temporal continuity."""
    
    def __init__(self, model_name=VIDEO_DESCRIPTION_MODEL, api_base=VLLM_SERVER_URL, max_workers=8):
        """Initialize the video description aligner.
        
        Args:
            model_name: Name of the language model to use for alignment
            api_base: Base URL for the vLLM server API
            max_workers: Maximum number of concurrent workers
        """
        self.model_name = model_name
        self.api_base = api_base
        self.max_workers = max_workers
        
        rich_console.print_info(f"Video Description Aligner initialized with model: {model_name}")
        rich_console.print_info(f"Using API base: {self.api_base}")
    
    def _test_connection(self):
        """Test connection to vLLM server."""
        try:
            client = OpenAI(api_key="token-abc123", base_url=self.api_base)
            client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            rich_console.print_info(f"vLLM server connection successful at {self.api_base}")
        except Exception as e:
            rich_console.print_error(f"Failed to connect to vLLM server: {e}")
            raise
    
    def align_video_descriptions(self, video_descriptions_file: str) -> Optional[str]:
        """Align video descriptions in a file to improve temporal continuity.
        
        Args:
            video_descriptions_file: Path to the video descriptions JSON file
            
        Returns:
            Path to the aligned descriptions file, or None if failed
        """
        rich_console.print_component_header("Video Description Alignment", 
                                           f"Aligning descriptions from {os.path.basename(video_descriptions_file)}")
        
        start_time = time.time()
        
        # Load existing descriptions
        try:
            with open(video_descriptions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            rich_console.print_error(f"Error loading video descriptions file: {e}")
            return None
        
        segments = data.get('segments', [])
        if len(segments) < 2:
            rich_console.print_warning("Video has less than 2 segments, no alignment needed")
            return video_descriptions_file
        
        # Check if already aligned
        if any(seg.get('processing_method', '').endswith('_aligned') for seg in segments):
            rich_console.print_info("Video descriptions already aligned, skipping")
            return video_descriptions_file
        
        video_id = data.get('video_id', 'unknown')
        rich_console.print_info(f"Aligning {len(segments)} segments for video {video_id}")
        
        # Create alignment tasks (skip first segment, no previous context)
        alignment_tasks = []
        for i in range(1, len(segments)):
            current_segment = segments[i]
            previous_segment = segments[i - 1]
            
            # Only align segments that have visual descriptions
            if (current_segment.get('visual_description') and 
                previous_segment.get('visual_description') and
                current_segment['visual_description'] != "No frames could be extracted from this video segment" and
                previous_segment['visual_description'] != "No frames could be extracted from this video segment"):
                
                alignment_tasks.append({
                    'segment_index': i,
                    'current_description': current_segment['visual_description'],
                    'previous_description': previous_segment['visual_description'],
                    'segment_data': current_segment
                })
        
        if not alignment_tasks:
            rich_console.print_warning("No segments found that need alignment")
            return video_descriptions_file
        
        rich_console.print_info(f"Created {len(alignment_tasks)} alignment tasks")
        
        # Process alignments concurrently
        aligned_descriptions = self._process_alignments_concurrent(alignment_tasks)
        
        # Update segments with aligned descriptions
        aligned_count = 0
        for task_index, aligned_desc in enumerate(aligned_descriptions):
            if aligned_desc and aligned_desc != "ALIGNMENT_ERROR":
                segment_index = alignment_tasks[task_index]['segment_index']
                segments[segment_index]['visual_description'] = aligned_desc
                
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
            'alignment_method': 'concurrent_temporal_alignment'
        }
        
        # Create aligned file path
        base_name = os.path.splitext(video_descriptions_file)[0]
        aligned_file = f"{base_name}_aligned.json"
        
        # Save aligned descriptions
        try:
            with open(aligned_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            processing_time = time.time() - start_time
            rich_console.print_success(f"Alignment completed for {video_id} in {processing_time:.1f}s")
            rich_console.print_info(f"Aligned {aligned_count}/{len(alignment_tasks)} segments")
            rich_console.print_info(f"Aligned descriptions saved to: {aligned_file}")
            
            return aligned_file
            
        except Exception as e:
            rich_console.print_error(f"Error saving aligned descriptions: {e}")
            return None
    
    def _process_alignments_concurrent(self, alignment_tasks: List[Dict]) -> List[str]:
        """Process alignment tasks concurrently."""
        aligned_descriptions = []
        
        if not alignment_tasks:
            return aligned_descriptions
        
        rich_console.print_info(f"Processing {len(alignment_tasks)} alignment tasks with {self.max_workers} workers")
        
        # Create progress tracking
        try:
            progress, task_id = rich_console.create_video_description_progress("Alignment", len(alignment_tasks))
        except:
            progress, task_id = None, None
        
        # Process alignments concurrently
        max_concurrent = min(len(alignment_tasks), self.max_workers)
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all alignment requests
            future_to_index = {}
            for i, task in enumerate(alignment_tasks):
                future = executor.submit(self._align_single_description, 
                                       task['current_description'], 
                                       task['previous_description'])
                future_to_index[future] = i
            
            # Initialize results with None values
            aligned_descriptions = [None] * len(alignment_tasks)
            
            # Collect results with progress tracking
            completed_count = 0
            if progress:
                with progress:
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        try:
                            aligned_description = future.result()
                            aligned_descriptions[index] = aligned_description
                            completed_count += 1
                            progress.update(task_id, advance=1)
                        except Exception as e:
                            rich_console.print_error(f"Error aligning segment {index}: {e}")
                            aligned_descriptions[index] = "ALIGNMENT_ERROR"
            else:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        aligned_description = future.result()
                        aligned_descriptions[index] = aligned_description
                        completed_count += 1
                        rich_console.print_info(f"Completed alignment {completed_count}/{len(alignment_tasks)}")
                    except Exception as e:
                        rich_console.print_error(f"Error aligning segment {index}: {e}")
                        aligned_descriptions[index] = "ALIGNMENT_ERROR"
        
        return aligned_descriptions
    
    def _align_single_description(self, current_description: str, previous_description: str) -> str:
        """Align a single segment description with its previous segment."""
        try:
            client = OpenAI(api_key="token-abc123", base_url=self.api_base)
            
            # Create prompt for alignment
            prompt = f"{ALIGNMENT_PROMPT}\n\n"
            prompt += f"**Previous Segment Description:**\n{previous_description}\n\n"
            prompt += f"**Current Segment Description (to be aligned):**\n{current_description}\n\n"
            prompt += "**Refined Aligned Description:**"
            
            # Make request to vLLM server
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,  # Slightly less than original to focus on refinement
                temperature=0.3   # Lower temperature for more consistent alignment
            )
            
            aligned_description = response.choices[0].message.content.strip()
            
            # Basic validation - ensure we got a reasonable response
            if len(aligned_description) < 50:  # Too short, likely an error
                rich_console.print_warning("Alignment produced very short result, using original")
                return current_description
            
            return aligned_description
            
        except Exception as e:
            rich_console.print_error(f"Error in alignment request: {e}")
            return current_description  # Fall back to original description
    
    def batch_align_descriptions(self, video_ids: List[str] = None, max_videos: int = None, 
                                descriptions_dir: str = None) -> List[str]:
        """Align descriptions for multiple videos.
        
        Args:
            video_ids: List of specific video IDs to align, or None for all
            max_videos: Maximum number of videos to process
            descriptions_dir: Directory containing video description files
            
        Returns:
            List of aligned description file paths
        """
        if descriptions_dir is None:
            # Import here to avoid circular imports
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from config import VIDEO_DESCRIPTIONS_DIR
            descriptions_dir = VIDEO_DESCRIPTIONS_DIR
        
        rich_console.print_component_header("Batch Video Description Alignment", 
                                           f"Processing videos in {descriptions_dir}")
        
        # Find description files to process
        files_to_process = []
        
        if video_ids:
            # Process specific video IDs
            for video_id in video_ids:
                desc_file = os.path.join(descriptions_dir, f"{video_id}_descriptions.json")
                if os.path.exists(desc_file):
                    files_to_process.append(desc_file)
                else:
                    rich_console.print_warning(f"Description file not found for {video_id}")
        else:
            # Find all description files that aren't already aligned
            for filename in os.listdir(descriptions_dir):
                if filename.endswith('_descriptions.json') and not filename.endswith('_aligned.json'):
                    desc_file = os.path.join(descriptions_dir, filename)
                    files_to_process.append(desc_file)
        
        if max_videos:
            files_to_process = files_to_process[:max_videos]
        
        rich_console.print_info(f"Found {len(files_to_process)} video description files to align")
        
        if not files_to_process:
            rich_console.print_warning("No video description files found to align")
            return []
        
        # Process alignments
        aligned_files = []
        failed_files = []
        
        for i, desc_file in enumerate(files_to_process, 1):
            video_id = os.path.basename(desc_file).replace('_descriptions.json', '')
            rich_console.print_info(f"Aligning descriptions for video {i}/{len(files_to_process)}: {video_id}")
            
            try:
                aligned_file = self.align_video_descriptions(desc_file)
                if aligned_file:
                    aligned_files.append(aligned_file)
                    rich_console.print_success(f"✓ Successfully aligned descriptions for {video_id} ({i}/{len(files_to_process)})")
                else:
                    failed_files.append(video_id)
                    rich_console.print_error(f"✗ Failed to align descriptions for {video_id} ({i}/{len(files_to_process)})")
            except Exception as e:
                failed_files.append(video_id)
                rich_console.print_error(f"✗ Error aligning descriptions for {video_id}: {e}")
        
        # Print summary
        success_count = len(aligned_files)
        total_count = len(files_to_process)
        rich_console.print_completion_message("Video Description Alignment", {
            'total': total_count,
            'successful': success_count,
            'duration': 0  # Duration calculated externally
        })
        
        if failed_files:
            rich_console.print_warning(f"Failed to align: {', '.join(failed_files)}")
        
        return aligned_files


def main():
    """Main function for testing the video description aligner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Align video descriptions for temporal continuity')
    parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to align')
    parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    parser.add_argument('--model', type=str, default=VIDEO_DESCRIPTION_MODEL, help='Language model to use')
    parser.add_argument('--api-base', type=str, default=VLLM_SERVER_URL, help='vLLM server API base URL')
    parser.add_argument('--max-workers', type=int, default=8, help='Maximum number of concurrent workers')
    parser.add_argument('--descriptions-dir', type=str, help='Directory containing video description files')
    
    args = parser.parse_args()
    
    # Initialize aligner
    aligner = VideoDescriptionAligner(
        model_name=args.model,
        api_base=args.api_base,
        max_workers=args.max_workers
    )
    
    # Process alignments
    aligned_files = aligner.batch_align_descriptions(
        video_ids=args.video_ids,
        max_videos=args.max_videos,
        descriptions_dir=args.descriptions_dir
    )
    
    rich_console.print_info(f"Successfully aligned descriptions for {len(aligned_files)} videos")


if __name__ == "__main__":
    main()