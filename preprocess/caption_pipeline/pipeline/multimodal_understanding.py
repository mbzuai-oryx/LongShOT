"""
Multimodal Video Understanding module for comprehensive segment analysis.
This module combines audio transcripts, visual descriptions, and audio environment descriptions
into unified, comprehensive segment understanding using Qwen 2.5-VL via vLLM.
"""

import os
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import sys
from datetime import datetime

# Import project configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (LLM_MODEL, LLM_SERVER_URL, 
                   CAPTIONS_DIR, VIDEO_DESCRIPTIONS_DIR, AUDIO_DESCRIPTIONS_DIR, MULTIMODAL_UNDERSTANDING_DIR)

# Import rich console utilities
from caption_pipeline.utils.rich_console import get_console

# Try to import vLLM components
try:
    from openai import OpenAI
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    OpenAI = None

# Set up logging
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)
logger = logging.getLogger(__name__)
rich_console = get_console()

# Concise domain-agnostic multimodal synthesis prompt with anti-repetition rules
MULTIMODAL_UNDERSTANDING_PROMPT = """
Create a concise, integrated description synthesizing speech, visual, and audio content for video understanding benchmarks.

CRITICAL CONSTRAINTS:
- EXACTLY 100-150 words (strict limit)
- NO temporal references ("segment", "portion", "moment", "sequence")
- NO repetitive patterns or templated language
- Focus on observable content integration across modalities

FORBIDDEN PHRASES:
- "The segment shows", "This moment captures", "In this sequence"  
- "Subsequently", "Meanwhile", "Following this", "Transitions into"
- "Audio remains consistent", "Similar environment persists"

INTEGRATION REQUIREMENTS:
- Synthesize speech, visual, and audio information naturally
- Describe only explicit content from provided modalities
- Use varied, direct language without templates
- Resolve modality conflicts by prioritizing most detailed source
- Maintain factual, neutral tone without speculation

DOMAIN-AGNOSTIC APPROACH:
- Use generic terminology applicable to any video type
- Focus on actions, objects, sounds, and speech content
- Avoid genre-specific language or assumptions
- Ensure universal applicability across content types

SOURCE MATERIALS:
Speech/Dialogue: {speech_content}
Visual: {visual_description}
Audio Environment: {audio_environment}

Integrated Description (100-150 words):
"""



class MultimodalVideoUnderstanding:
    """
    Class to generate comprehensive video segment understanding by combining
    audio transcripts, visual descriptions, and audio environment descriptions.
    """
    
    def __init__(self, 
                 model_name=LLM_MODEL,
                 api_base=LLM_SERVER_URL,
                 max_workers=8):
        """
        Initialize MultimodalVideoUnderstanding processor.
        
        Args:
            model_name: Name of the language model to use
            api_base: Base URL for the vLLM server API
            max_workers: Maximum number of concurrent workers
        """
        self.model_name = model_name
        self.api_base = api_base
        self.max_workers = max_workers
        
        # Ensure output directory exists
        os.makedirs(MULTIMODAL_UNDERSTANDING_DIR, exist_ok=True)
        
        # Initialize client
        self.client = None
        if VLLM_AVAILABLE:
            try:
                self.client = OpenAI(api_key="EMPTY", base_url=api_base)
                # Test connection
                self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=1
                )
                rich_console.print_success("Connected to vLLM server for multimodal understanding")
            except Exception as e:
                rich_console.print_error(f"Failed to connect to vLLM server: {e}")
                self.client = None
        else:
            rich_console.print_error("OpenAI library not available. Cannot use vLLM.")
    
    def _load_captions_data(self, video_id: str) -> Dict:
        """Load caption data for a video."""
        captions_file = os.path.join(CAPTIONS_DIR, f"{video_id}.json")
        try:
            with open(captions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading captions for video {video_id}: {e}")
            return {}
    
    def _load_video_descriptions_data(self, video_id: str) -> Dict:
        """Load video descriptions data for a video."""
        # Try aligned version first, fall back to non-aligned
        aligned_file = os.path.join(VIDEO_DESCRIPTIONS_DIR, f"{video_id}_descriptions_aligned.json")
        non_aligned_file = os.path.join(VIDEO_DESCRIPTIONS_DIR, f"{video_id}_descriptions.json")
        
        try:
            if os.path.exists(aligned_file):
                with open(aligned_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif os.path.exists(non_aligned_file):
                logger.warning(f"Using non-aligned video descriptions for {video_id}")
                with open(non_aligned_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.error(f"No video descriptions file found for {video_id}")
                return {}
        except Exception as e:
            logger.error(f"Error loading video descriptions for video {video_id}: {e}")
            return {}
    
    def _load_audio_descriptions_data(self, video_id: str) -> Dict:
        """Load audio descriptions data for a video, preferring aligned files."""
        # First try to load aligned audio descriptions
        aligned_audio_descriptions_file = os.path.join(AUDIO_DESCRIPTIONS_DIR, f"{video_id}_audio_descriptions_aligned.json")
        audio_descriptions_file = os.path.join(AUDIO_DESCRIPTIONS_DIR, f"{video_id}_audio_descriptions.json")
        
        # Prefer aligned file if it exists
        if os.path.exists(aligned_audio_descriptions_file):
            try:
                with open(aligned_audio_descriptions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.debug(f"Loaded aligned audio descriptions for video {video_id}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load aligned audio descriptions for video {video_id}: {e}")
                # Fall back to original file
        
        # Fall back to original audio descriptions file
        try:
            with open(audio_descriptions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"Loaded original audio descriptions for video {video_id}")
                return data
        except Exception as e:
            logger.debug(f"No audio descriptions found for video {video_id}: {e}")
            return {}
    
    def _find_temporal_matches(self, captions_data: Dict, video_descriptions_data: Dict, 
                              audio_descriptions_data: Dict) -> List[Dict]:
        """
        Find and align segments across all three modalities based on temporal overlap.
        Uses video description segments as the primary temporal structure to preserve
        gap elimination logic (gaps < 5 seconds are already eliminated in video descriptions).
        """
        # Get segments from each source
        caption_segments = captions_data.get('transcript', {}).get('segments', [])
        video_segments = video_descriptions_data.get('segments', [])
        audio_segments = audio_descriptions_data.get('segments', [])
        
        # Use video description segments as the primary temporal structure
        # This preserves the gap elimination logic that was already applied
        unified_segments = []
        
        for idx, vid_seg in enumerate(video_segments):
            start_time = vid_seg.get('start', 0)
            end_time = vid_seg.get('end', 0)
            
            # Create segment data using video description timing
            segment_data = {
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'speech_content': '',
                'visual_description': '',
                'audio_environment': ''
            }
            
            # Get visual description from video segment
            if 'visual_description' in vid_seg and vid_seg['visual_description']:
                segment_data['visual_description'] = vid_seg['visual_description']
            
            # Find overlapping caption segments
            caption_texts = []
            for cap_seg in caption_segments:
                if (cap_seg.get('start', 0) < end_time and cap_seg.get('end', 0) > start_time):
                    caption_texts.append(cap_seg.get('text', '').strip())
            segment_data['speech_content'] = ' '.join(caption_texts).strip()
            
            # Find overlapping audio description segments
            audio_descriptions = []
            for aud_seg in audio_segments:
                if (aud_seg.get('start', 0) < end_time and aud_seg.get('end', 0) > start_time):
                    if 'audio_description' in aud_seg and aud_seg['audio_description']:
                        audio_descriptions.append(aud_seg['audio_description'])
            segment_data['audio_environment'] = ' '.join(audio_descriptions).strip()
            
            # Always include the segment since video descriptions are the primary structure
            # This preserves the gap elimination that was already applied
            unified_segments.append(segment_data)
        
        rich_console.print_info(f"Created {len(unified_segments)} unified segments using video description temporal structure (preserving gap elimination)")
        
        return unified_segments
    
    def _generate_multimodal_understanding(self, segment_data: Dict) -> str:
        """
        Generate comprehensive understanding for a single segment using vLLM.
        """
        if not self.client:
            return "Multimodal understanding generation failed due to server connection issues."
        
        # Prepare content for each modality
        speech_content = segment_data.get('speech_content', '') or '[No speech/dialogue]'
        visual_description = segment_data.get('visual_description', '') or '[No visual description available]'
        audio_environment = segment_data.get('audio_environment', '') or '[No audio environment description available]'
        
        # Format the prompt
        prompt = MULTIMODAL_UNDERSTANDING_PROMPT.format(
            speech_content=speech_content,
            visual_description=visual_description,
            audio_environment=audio_environment
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert video analyst specializing in comprehensive multimodal understanding."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Low temperature for consistent, detailed analysis
                max_tokens=500,   # Allow for comprehensive descriptions
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Error generating multimodal understanding: {e}")
            return f"Error generating comprehensive understanding: {str(e)}"
    
    def _process_single_segment(self, segment_data: Dict, segment_index: int) -> Dict:
        """Process a single segment to generate multimodal understanding."""
        start_time = time.time()
        
        # Generate multimodal understanding
        multimodal_understanding = self._generate_multimodal_understanding(segment_data)
        
        processing_time = time.time() - start_time
        
        return {
            'segment_index': segment_index,
            'start': segment_data['start'],
            'end': segment_data['end'],
            'duration': segment_data['duration'],
            'multimodal_understanding': multimodal_understanding,
            'source_modalities': {
                'speech_content': segment_data['speech_content'],
                'visual_description': segment_data['visual_description'],
                'audio_environment': segment_data['audio_environment']
            },
            'processing_time': processing_time,
            'processing_method': 'multimodal_vllm_integration',
            'timestamp': datetime.now().isoformat()
        }
    
    def process_video(self, video_id: str) -> bool:
        """
        Process a single video to generate comprehensive multimodal understanding.
        
        Args:
            video_id: ID of the video to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            # Load data from all modalities
            captions_data = self._load_captions_data(video_id)
            video_descriptions_data = self._load_video_descriptions_data(video_id)
            audio_descriptions_data = self._load_audio_descriptions_data(video_id)
            
            # Check if we have at least captions and video descriptions (minimum requirement)
            if not captions_data or not video_descriptions_data:
                rich_console.print_warning(f"Insufficient data for multimodal understanding: {video_id}")
                return False
            
            # Find temporal alignments and create unified segments
            unified_segments = self._find_temporal_matches(
                captions_data, video_descriptions_data, audio_descriptions_data
            )
            
            if not unified_segments:
                rich_console.print_warning(f"No aligned segments found for video {video_id}")
                return False
            
            # Only show segment progress for videos with many segments
            show_segment_progress = len(unified_segments) > 5
            
            if show_segment_progress:
                # Create progress bar for segment processing
                segment_progress, segment_task_id = rich_console.create_video_segment_progress(video_id, len(unified_segments))
            
            # Process segments concurrently with optional progress tracking
            processed_segments = []
            
            if show_segment_progress:
                with segment_progress:
                    with ThreadPoolExecutor(max_workers=min(self.max_workers, len(unified_segments))) as executor:
                        # Submit all segment processing tasks
                        future_to_segment = {
                            executor.submit(self._process_single_segment, segment, idx): idx
                            for idx, segment in enumerate(unified_segments)
                        }
                        
                        # Collect results with progress updates
                        for future in as_completed(future_to_segment):
                            try:
                                result = future.result()
                                processed_segments.append(result)
                                segment_progress.update(segment_task_id, advance=1)
                            except Exception as e:
                                segment_idx = future_to_segment[future]
                                logger.error(f"Error processing segment {segment_idx} for video {video_id}: {e}")
                                segment_progress.update(segment_task_id, advance=1)
            else:
                # Process without progress bar for small videos
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(unified_segments))) as executor:
                    # Submit all segment processing tasks
                    future_to_segment = {
                        executor.submit(self._process_single_segment, segment, idx): idx
                        for idx, segment in enumerate(unified_segments)
                    }
                    
                    # Collect results
                    for future in as_completed(future_to_segment):
                        try:
                            result = future.result()
                            processed_segments.append(result)
                        except Exception as e:
                            segment_idx = future_to_segment[future]
                            logger.error(f"Error processing segment {segment_idx} for video {video_id}: {e}")
            
            # Sort segments by index to maintain order
            processed_segments.sort(key=lambda x: x['segment_index'])
            
            # Create final output structure
            total_processing_time = time.time() - start_time
            
            output_data = {
                'video_id': video_id,
                'model_used': self.model_name,
                'api_base': self.api_base,
                'processing_method': 'comprehensive_multimodal_understanding',
                'total_segments': len(processed_segments),
                'total_processing_time': total_processing_time,
                'processing_timestamp': datetime.now().isoformat(),
                'data_sources': {
                    'captions_available': bool(captions_data),
                    'video_descriptions_available': bool(video_descriptions_data),
                    'audio_descriptions_available': bool(audio_descriptions_data)
                },
                'segments': processed_segments
            }
            
            # Save to file
            output_file = os.path.join(MULTIMODAL_UNDERSTANDING_DIR, f"{video_id}_multimodal_understanding.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Rich success message with timing info
            duration_str = f"{total_processing_time:.1f}s"
            segments_per_sec = len(processed_segments) / total_processing_time if total_processing_time > 0 else 0
            rich_console.print_success(f"✨ Generated multimodal understanding for {video_id}: {len(processed_segments)} segments in {duration_str} ({segments_per_sec:.1f} segments/s)")
            return True
            
        except Exception as e:
            rich_console.print_error(f"Error processing video {video_id}: {e}")
            logger.error(f"Error processing video {video_id}: {e}")
            return False
    
    def process_videos(self, video_ids: List[str] = None, max_videos: int = None,
                      use_concurrent: bool = True) -> Dict[str, bool]:
        """
        Process multiple videos for multimodal understanding generation.
        
        Args:
            video_ids: List of specific video IDs to process, or None for all available
            max_videos: Maximum number of videos to process
            use_concurrent: Whether to use concurrent processing
            
        Returns:
            Dictionary mapping video IDs to success status
        """
        # Find videos to process
        if video_ids is None:
            # Use the method that properly filters out already processed videos
            available_videos = self.get_videos_needing_processing()
            
            if max_videos:
                available_videos = available_videos[:max_videos]
            
            video_ids = available_videos
        
        if not video_ids:
            rich_console.print_warning("No videos found for multimodal understanding processing")
            return {}
        
        # Create rich progress bar for overall processing (simpler version)
        if len(video_ids) > 1:
            progress, task_id = rich_console.create_multimodal_understanding_progress(len(video_ids))
        else:
            progress, task_id = None, None
        
        results = {}
        
        if progress:
            with progress:
                if use_concurrent and len(video_ids) > 1:
                    # Process videos concurrently
                    with ThreadPoolExecutor(max_workers=min(self.max_workers // 2, len(video_ids))) as executor:
                        future_to_video = {
                            executor.submit(self.process_video, video_id): video_id
                            for video_id in video_ids
                        }
                        
                        for future in as_completed(future_to_video):
                            video_id = future_to_video[future]
                            try:
                                success = future.result()
                                results[video_id] = success
                                
                                # Update progress with current video status
                                status = "✓ Completed" if success else "✗ Failed"
                                progress.update(task_id, advance=1, status=f"{video_id}: {status}")
                                
                            except Exception as e:
                                rich_console.print_error(f"Error processing video {video_id}: {e}")
                                results[video_id] = False
                                progress.update(task_id, advance=1, status=f"{video_id}: ✗ Error")
                else:
                    # Process videos sequentially
                    for i, video_id in enumerate(video_ids):
                        progress.update(task_id, status=f"Processing {video_id}...")
                        success = self.process_video(video_id)
                        results[video_id] = success
                        
                        status = "✓ Completed" if success else "✗ Failed"
                        progress.update(task_id, advance=1, status=f"{video_id}: {status}")
        else:
            # Single video - no main progress bar
            for video_id in video_ids:
                success = self.process_video(video_id)
                results[video_id] = success
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        # Enhanced summary with rich output
        if successful == total:
            rich_console.print_success(f"🎯 Multimodal understanding complete: {successful}/{total} videos processed successfully")
        else:
            rich_console.print_warning(f"⚠️  Multimodal understanding complete: {successful}/{total} videos processed successfully, {total - successful} failed")
        
        return results
    
    def get_videos_needing_processing(self) -> List[str]:
        """
        Get list of videos that need multimodal understanding processing.
        """
        try:
            # Find videos with captions and video descriptions but no multimodal understanding
            captions_files = {f[:-5] for f in os.listdir(CAPTIONS_DIR) if f.endswith('.json')}
            video_desc_files = {f.replace('_descriptions_aligned.json', '') for f in os.listdir(VIDEO_DESCRIPTIONS_DIR) 
                               if f.endswith('_descriptions_aligned.json')}
            
            # Create multimodal understanding directory if it doesn't exist
            os.makedirs(MULTIMODAL_UNDERSTANDING_DIR, exist_ok=True)
            # Look for both aligned and non-aligned multimodal understanding files
            existing_multimodal = set()
            for f in os.listdir(MULTIMODAL_UNDERSTANDING_DIR):
                if f.endswith('_multimodal_understanding.json'):
                    video_id = f.replace('_multimodal_understanding.json', '')
                    existing_multimodal.add(video_id)
                elif f.endswith('_multimodal_understanding_aligned.json'):
                    video_id = f.replace('_multimodal_understanding_aligned.json', '')
                    existing_multimodal.add(video_id)
            
            # Debug logging
            rich_console.print_info(f"Found {len(captions_files)} caption files, {len(video_desc_files)} video description files")
            rich_console.print_info(f"Found {len(existing_multimodal)} existing multimodal understanding files")
            
            # Videos that have prerequisites but no multimodal understanding yet
            eligible_videos = captions_files.intersection(video_desc_files)
            videos_needing_processing = list(eligible_videos - existing_multimodal)
            
            if videos_needing_processing:
                rich_console.print_info(f"Videos needing multimodal understanding: {len(videos_needing_processing)}")
            else:
                rich_console.print_info("All eligible videos already have multimodal understanding")
            
            return videos_needing_processing
        except Exception as e:
            logger.error(f"Error scanning for videos needing processing: {e}")
            return []


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comprehensive multimodal video understanding')
    parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to process')
    parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    parser.add_argument('--max-workers', type=int, default=8, help='Maximum number of concurrent workers')
    parser.add_argument('--model', type=str, default=LLM_MODEL, help='Model to use')
    parser.add_argument('--api-base', type=str, default=LLM_SERVER_URL, help='vLLM server API base URL')
    parser.add_argument('--sequential', action='store_true', help='Process videos sequentially instead of concurrently')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = MultimodalVideoUnderstanding(
        model_name=args.model,
        api_base=args.api_base,
        max_workers=args.max_workers
    )
    
    # Process videos
    results = processor.process_videos(
        video_ids=args.video_ids,
        max_videos=args.max_videos,
        use_concurrent=not args.sequential
    )
    
    # Print summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    print(f"\nProcessing complete: {successful}/{total} videos processed successfully")


if __name__ == "__main__":
    main()
