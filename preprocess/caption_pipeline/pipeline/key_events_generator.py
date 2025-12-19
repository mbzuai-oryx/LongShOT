"""
Key Events Generator module for extracting key events from video segments.

This module analyzes visual, audio, speech, and multimodal description texts to generate
a concise list of key events that occur in each video segment. This provides a quick
understanding of the video content and helps in content summarization and analysis.
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
from config import LLM_MODEL, LLM_SERVER_URL, MULTIMODAL_UNDERSTANDING_DIR, KEY_EVENTS_DIR, VIDEO_DESCRIPTIONS_DIR, AUDIO_DESCRIPTIONS_DIR

# Import rich console utilities
from caption_pipeline.utils.rich_console import get_console

# Import required components
from openai import OpenAI

# Set up logging
logger = logging.getLogger(__name__)
rich_console = get_console()

# Domain-agnostic key events extraction prompt with action-specific guidance
KEY_EVENTS_EXTRACTION_PROMPT = """
You are an expert multimedia analyst extracting key events from video segments across any domain. Analyze all provided modality data and generate a concise list of significant events using active, precise language.

EXTRACTION REQUIREMENTS:
- Generate 3-7 key events per segment
- Each event MUST be action-specific using active verbs (5-15 words)
- List events chronologically as they occur in the segment
- Focus on meaningful actions, transitions, changes, or notable moments
- Include cross-modal events when multiple modalities align

ACTION-SPECIFIC VERB GUIDANCE:
Use precise action verbs like: moves, speaks, points, transitions, begins, stops, changes, demonstrates, explains, performs, executes, approaches, displays, announces, shifts, enters, exits, operates, adjusts, responds, reacts

DOMAIN-AGNOSTIC LANGUAGE:
- Avoid domain-specific terms (sports jargon, technical terminology, genre assumptions)
- Use generic descriptive language: "individual", "person", "speaker", "subject", "participant"
- Focus on observable actions rather than contextual interpretation
- Ensure events apply universally across video types (sports, news, tutorials, documentaries, entertainment)

LANGUAGE VARIATION:
- Vary sentence structures to avoid repetition
- Use different action verbs and descriptive patterns
- Alternate between different event description styles

OUTPUT FORMAT:
JSON array of strings only. Examples showing variation:
["Individual adjusts equipment and begins demonstration",
 "Audio transitions from quiet to musical background", 
 "Subject points toward displayed information",
 "Movement shifts from left side to center position",
 "Speaker completes explanation and pauses briefly"]

Content Data:
Visual Description: {visual_description}
Audio Environment: {audio_environment}
Speech Content: {speech_content}
Multimodal Understanding: {multimodal_understanding}

Key Events (JSON array):
"""


class KeyEventsGenerator:
    """Generates key events from video segment descriptions."""
    
    def __init__(self, model_name=LLM_MODEL, api_base=LLM_SERVER_URL, max_workers=8):
        """Initialize the key events generator.
        
        Args:
            model_name: Name of the language model to use
            api_base: Base URL for the vLLM server API
            max_workers: Maximum number of concurrent workers
        """
        self.model_name = model_name
        self.api_base = api_base
        self.max_workers = max_workers
        
        rich_console.print_info(f"Key Events Generator initialized with model: {model_name}")
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
    
    def _load_aligned_descriptions(self, video_id: str) -> Optional[Dict]:
        """Load and combine aligned descriptions from multiple sources.
        
        Args:
            video_id: The video ID to load descriptions for
            
        Returns:
            Combined data structure with all available descriptions, or None if failed
        """
        # Try to load video descriptions (aligned if available)
        video_descriptions_data = None
        video_descriptions_source = None
        
        aligned_video_file = os.path.join(VIDEO_DESCRIPTIONS_DIR, f"{video_id}_descriptions_aligned.json")
        video_file = os.path.join(VIDEO_DESCRIPTIONS_DIR, f"{video_id}_descriptions.json")
        
        if os.path.exists(aligned_video_file):
            try:
                with open(aligned_video_file, 'r', encoding='utf-8') as f:
                    video_descriptions_data = json.load(f)
                    video_descriptions_source = "aligned"
            except Exception as e:
                rich_console.print_warning(f"Error loading aligned video descriptions: {e}")
        elif os.path.exists(video_file):
            try:
                with open(video_file, 'r', encoding='utf-8') as f:
                    video_descriptions_data = json.load(f)
                    video_descriptions_source = "original"
            except Exception as e:
                rich_console.print_warning(f"Error loading video descriptions: {e}")
        
        # Try to load audio descriptions (aligned if available)
        audio_descriptions_data = None
        audio_descriptions_source = None
        
        aligned_audio_file = os.path.join(AUDIO_DESCRIPTIONS_DIR, f"{video_id}_audio_descriptions_aligned.json")
        audio_file = os.path.join(AUDIO_DESCRIPTIONS_DIR, f"{video_id}_audio_descriptions.json")
        
        if os.path.exists(aligned_audio_file):
            try:
                with open(aligned_audio_file, 'r', encoding='utf-8') as f:
                    audio_descriptions_data = json.load(f)
                    audio_descriptions_source = "aligned"
            except Exception as e:
                rich_console.print_warning(f"Error loading aligned audio descriptions: {e}")
        elif os.path.exists(audio_file):
            try:
                with open(audio_file, 'r', encoding='utf-8') as f:
                    audio_descriptions_data = json.load(f)
                    audio_descriptions_source = "original"
            except Exception as e:
                rich_console.print_warning(f"Error loading audio descriptions: {e}")
        
        # Try to load multimodal understanding (aligned if available)
        multimodal_data = None
        multimodal_source = None
        
        aligned_multimodal_file = os.path.join(MULTIMODAL_UNDERSTANDING_DIR, f"{video_id}_multimodal_understanding_aligned.json")
        multimodal_file = os.path.join(MULTIMODAL_UNDERSTANDING_DIR, f"{video_id}_multimodal_understanding.json")
        
        if os.path.exists(aligned_multimodal_file):
            try:
                with open(aligned_multimodal_file, 'r', encoding='utf-8') as f:
                    multimodal_data = json.load(f)
                    multimodal_source = "aligned"
            except Exception as e:
                rich_console.print_warning(f"Error loading aligned multimodal understanding: {e}")
        elif os.path.exists(multimodal_file):
            try:
                with open(multimodal_file, 'r', encoding='utf-8') as f:
                    multimodal_data = json.load(f)
                    multimodal_source = "original"
            except Exception as e:
                rich_console.print_warning(f"Error loading multimodal understanding: {e}")
        
        # Check if we have at least one data source
        if not any([video_descriptions_data, audio_descriptions_data, multimodal_data]):
            rich_console.print_error(f"No description data found for video {video_id}")
            return None
        
        # Use multimodal data as the base structure if available, otherwise use video descriptions
        if multimodal_data:
            base_data = multimodal_data
            base_segments = multimodal_data.get('segments', [])
        elif video_descriptions_data:
            base_data = video_descriptions_data
            base_segments = video_descriptions_data.get('segments', [])
        else:
            base_data = audio_descriptions_data
            base_segments = audio_descriptions_data.get('segments', [])
        
        # Combine data from all sources into unified segments
        combined_segments = []
        
        for i, base_segment in enumerate(base_segments):
            combined_segment = base_segment.copy()
            
            # Add video description if available
            if video_descriptions_data and i < len(video_descriptions_data.get('segments', [])):
                video_segment = video_descriptions_data['segments'][i]
                combined_segment['video_description'] = video_segment.get('description', '')
            else:
                combined_segment['video_description'] = ''
            
            # Add audio description if available
            if audio_descriptions_data and i < len(audio_descriptions_data.get('segments', [])):
                audio_segment = audio_descriptions_data['segments'][i]
                combined_segment['audio_description'] = audio_segment.get('description', '')
            else:
                combined_segment['audio_description'] = ''
            
            # Add multimodal understanding if available
            if multimodal_data and i < len(multimodal_data.get('segments', [])):
                multimodal_segment = multimodal_data['segments'][i]
                combined_segment['multimodal_understanding'] = multimodal_segment.get('multimodal_understanding', '')
                # Also extract speech content from multimodal data if available
                if 'source_modalities' in multimodal_segment:
                    combined_segment['speech_content'] = multimodal_segment['source_modalities'].get('speech_content', '')
                else:
                    combined_segment['speech_content'] = combined_segment.get('speech_content', '')
            else:
                combined_segment['multimodal_understanding'] = ''
                combined_segment['speech_content'] = combined_segment.get('speech_content', '')
            
            combined_segments.append(combined_segment)
        
        # Create combined data structure
        combined_data = {
            'video_id': video_id,
            'segments': combined_segments,
            'metadata': base_data.get('metadata', {}),
            'video_descriptions_source': video_descriptions_source,
            'audio_descriptions_source': audio_descriptions_source,
            'multimodal_understanding_source': multimodal_source
        }
        
        rich_console.print_info(f"Loaded data for {video_id}:")
        rich_console.print_info(f"  - Video descriptions: {video_descriptions_source or 'Not available'}")
        rich_console.print_info(f"  - Audio descriptions: {audio_descriptions_source or 'Not available'}")
        rich_console.print_info(f"  - Multimodal understanding: {multimodal_source or 'Not available'}")
        rich_console.print_info(f"  - Total segments: {len(combined_segments)}")
        
        return combined_data
    
    def generate_key_events_for_video(self, video_id: str) -> Optional[str]:
        """Generate key events for all segments in a video using aligned descriptions from multiple sources.
        
        Args:
            video_id: The video ID to process
            
        Returns:
            Path to the key events file, or None if failed
        """
        rich_console.print_component_header("Key Events Generation", 
                                           f"Processing video {video_id}")
        
        start_time = time.time()
        
        # Load data from multiple sources
        video_data = self._load_aligned_descriptions(video_id)
        if not video_data:
            rich_console.print_error(f"Failed to load aligned descriptions for video {video_id}")
            return None
        
        segments = video_data.get('segments', [])
        if not segments:
            rich_console.print_warning(f"No segments found for video {video_id}")
            return None
        
        # Create output directory
        os.makedirs(KEY_EVENTS_DIR, exist_ok=True)
        
        # Check if key events already exist
        key_events_file = os.path.join(KEY_EVENTS_DIR, f"{video_id}_key_events.json")
        if os.path.exists(key_events_file):
            try:
                with open(key_events_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if existing_data.get('key_events_info', {}).get('generation_completed', False):
                    rich_console.print_info("Key events already exist, skipping generation")
                    return key_events_file
            except Exception as e:
                rich_console.print_warning(f"Error reading existing key events file: {e}")
        
        rich_console.print_info(f"Generating key events for {len(segments)} segments in video {video_id}")
        
        # Create key events generation tasks
        generation_tasks = []
        for i, segment in enumerate(segments):
            # Extract descriptions from different modalities
            video_desc = segment.get('video_description', '')
            audio_desc = segment.get('audio_description', '')
            speech_content = segment.get('speech_content', '')
            multimodal_understanding = segment.get('multimodal_understanding', '')
            
            # Only process segments that have some content
            if any([video_desc, audio_desc, speech_content, multimodal_understanding]):
                generation_tasks.append({
                    'segment_index': i,
                    'visual_description': video_desc,
                    'audio_environment': audio_desc,
                    'speech_content': speech_content,
                    'multimodal_understanding': multimodal_understanding,
                    'segment_data': segment
                })
        
        if not generation_tasks:
            rich_console.print_warning("No segments found that need key events generation")
            return None
        
        rich_console.print_info(f"Created {len(generation_tasks)} key events generation tasks")
        
        # Process key events generation concurrently
        generated_events = self._process_key_events_concurrent(generation_tasks)
        
        # Update segments with generated key events
        events_count = 0
        for task_index, key_events in enumerate(generated_events):
            if key_events and key_events != "GENERATION_ERROR":
                segment_index = generation_tasks[task_index]['segment_index']
                segments[segment_index]['key_events'] = key_events
                events_count += 1
        
        # Create final data structure
        final_data = {
            'video_id': video_id,
            'segments': segments,
            'metadata': video_data.get('metadata', {}),
            'key_events_info': {
                'generation_completed': True,
                'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'segments_with_key_events': events_count,
                'total_segments': len(segments),
                'generation_method': 'concurrent_llm_extraction',
                'model_used': self.model_name,
                'sources_used': {
                    'video_descriptions': bool(video_data.get('video_descriptions_source')),
                    'audio_descriptions': bool(video_data.get('audio_descriptions_source')),
                    'multimodal_understanding': bool(video_data.get('multimodal_understanding_source'))
                }
            }
        }
        
        # Save key events data
        try:
            with open(key_events_file, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)
            
            processing_time = time.time() - start_time
            rich_console.print_success(f"Key events generation completed for {video_id} in {processing_time:.1f}s")
            rich_console.print_info(f"Generated key events for {events_count}/{len(generation_tasks)} segments")
            rich_console.print_info(f"Key events data saved to: {key_events_file}")
            
            return key_events_file
            
        except Exception as e:
            rich_console.print_error(f"Error saving key events data: {e}")
            return None
    
    def _process_key_events_concurrent(self, generation_tasks: List[Dict]) -> List[List[str]]:
        """Process key events generation tasks concurrently."""
        generated_events = []
        
        if not generation_tasks:
            return generated_events
        
        rich_console.print_info(f"Processing {len(generation_tasks)} key events tasks with {self.max_workers} workers")
        
        # Create progress tracking
        try:
            progress, task_id = rich_console.create_key_events_progress("Key Events", len(generation_tasks))
        except:
            progress, task_id = None, None
        
        # Process key events generation concurrently
        max_concurrent = min(len(generation_tasks), self.max_workers)
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all generation requests
            future_to_index = {}
            for i, task in enumerate(generation_tasks):
                future = executor.submit(self._generate_single_key_events, 
                                       task['visual_description'], 
                                       task['audio_environment'],
                                       task['speech_content'],
                                       task['multimodal_understanding'])
                future_to_index[future] = i
            
            # Initialize results with None values
            generated_events = [None] * len(generation_tasks)
            
            # Collect results with progress tracking
            completed_count = 0
            if progress:
                with progress:
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        try:
                            key_events = future.result()
                            generated_events[index] = key_events
                            completed_count += 1
                            progress.update(task_id, advance=1)
                        except Exception as e:
                            rich_console.print_error(f"Error generating key events for segment {index}: {e}")
                            generated_events[index] = "GENERATION_ERROR"
            else:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        key_events = future.result()
                        generated_events[index] = key_events
                        completed_count += 1
                        rich_console.print_info(f"Completed key events generation {completed_count}/{len(generation_tasks)}")
                    except Exception as e:
                        rich_console.print_error(f"Error generating key events for segment {index}: {e}")
                        generated_events[index] = "GENERATION_ERROR"
        
        return generated_events
    
    def _generate_single_key_events(self, visual_description: str, audio_environment: str, 
                                   speech_content: str, multimodal_understanding: str) -> List[str]:
        """Generate key events for a single segment."""
        try:
            client = OpenAI(api_key="token-abc123", base_url=self.api_base)
            
            # Create prompt for key events extraction
            prompt = KEY_EVENTS_EXTRACTION_PROMPT.format(
                visual_description=visual_description,
                audio_environment=audio_environment,
                speech_content=speech_content,
                multimodal_understanding=multimodal_understanding
            )
            
            # Make request to vLLM server
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,  # Allow for multiple key events
                temperature=0.2   # Lower temperature for more consistent extraction
            )
            
            events_text = response.choices[0].message.content.strip()
            
            # Try to parse as JSON array
            try:
                # Clean up the response to extract JSON
                if events_text.startswith('```json'):
                    events_text = events_text.replace('```json', '').replace('```', '')
                elif events_text.startswith('```'):
                    events_text = events_text.replace('```', '')
                
                events_text = events_text.strip()
                
                # Try to parse as JSON
                key_events = json.loads(events_text)
                
                # Validate that it's a list of strings
                if isinstance(key_events, list) and all(isinstance(event, str) for event in key_events):
                    # Filter out empty events and limit to reasonable length
                    key_events = [event.strip() for event in key_events if event.strip()]
                    key_events = key_events[:7]  # Limit to max 7 events
                    return key_events
                else:
                    rich_console.print_warning("Generated key events not in expected format, parsing manually")
                    return self._parse_events_manually(events_text)
                    
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract events manually
                rich_console.print_warning("Failed to parse key events as JSON, parsing manually")
                return self._parse_events_manually(events_text)
            
        except Exception as e:
            rich_console.print_error(f"Error in key events generation request: {e}")
            return []  # Return empty list on error
    
    def _parse_events_manually(self, events_text: str) -> List[str]:
        """Manually parse key events from text when JSON parsing fails."""
        try:
            # Try to extract events from common formats
            lines = events_text.split('\n')
            events = []
            
            for line in lines:
                line = line.strip()
                # Skip empty lines and headers
                if not line or line.lower().startswith(('key events', 'events:', '[')):
                    continue
                
                # Remove common prefixes
                if line.startswith(('"', "'", '-', '*', '•', '1.', '2.', '3.', '4.', '5.', '6.', '7.')):
                    # Remove prefix and quotes
                    line = line.lstrip('"-*•123456789. ').rstrip('",')
                
                # Add if it looks like an event (not too short or long)
                if 5 <= len(line) <= 100 and line not in events:
                    events.append(line)
                
                # Limit to 7 events
                if len(events) >= 7:
                    break
            
            return events[:7] if events else ["Key events extraction failed"]
            
        except Exception as e:
            rich_console.print_error(f"Error in manual parsing: {e}")
            return ["Key events extraction failed"]
    
    def batch_generate_key_events(self, video_ids: List[str] = None, max_videos: int = None) -> List[str]:
        """Generate key events for multiple videos.
        
        Args:
            video_ids: List of specific video IDs to process, or None for all
            max_videos: Maximum number of videos to process
            
        Returns:
            List of key events file paths
        """
        rich_console.print_component_header("Batch Key Events Generation", 
                                           "Processing videos from aligned descriptions")
        
        # Find video IDs to process
        videos_to_process = []
        
        if video_ids:
            # Process specific video IDs
            videos_to_process = video_ids
        else:
            # Find all videos that have description data available
            processed_videos = set()
            
            # Check multimodal understanding directory
            if os.path.exists(MULTIMODAL_UNDERSTANDING_DIR):
                for filename in os.listdir(MULTIMODAL_UNDERSTANDING_DIR):
                    if filename.endswith('_multimodal_understanding_aligned.json'):
                        video_id = filename.replace('_multimodal_understanding_aligned.json', '')
                        processed_videos.add(video_id)
                    elif filename.endswith('_multimodal_understanding.json') and not filename.endswith('_aligned.json'):
                        video_id = filename.replace('_multimodal_understanding.json', '')
                        processed_videos.add(video_id)
            
            # Check video descriptions directory
            if os.path.exists(VIDEO_DESCRIPTIONS_DIR):
                for filename in os.listdir(VIDEO_DESCRIPTIONS_DIR):
                    if filename.endswith('_descriptions_aligned.json'):
                        video_id = filename.replace('_descriptions_aligned.json', '')
                        processed_videos.add(video_id)
                    elif filename.endswith('_descriptions.json') and not filename.endswith('_aligned.json'):
                        video_id = filename.replace('_descriptions.json', '')
                        processed_videos.add(video_id)
            
            # Check audio descriptions directory
            if os.path.exists(AUDIO_DESCRIPTIONS_DIR):
                for filename in os.listdir(AUDIO_DESCRIPTIONS_DIR):
                    if filename.endswith('_audio_descriptions_aligned.json'):
                        video_id = filename.replace('_audio_descriptions_aligned.json', '')
                        processed_videos.add(video_id)
                    elif filename.endswith('_audio_descriptions.json') and not filename.endswith('_aligned.json'):
                        video_id = filename.replace('_audio_descriptions.json', '')
                        processed_videos.add(video_id)
            
            videos_to_process = list(processed_videos)
        
        if max_videos:
            videos_to_process = videos_to_process[:max_videos]
        
        rich_console.print_info(f"Found {len(videos_to_process)} videos to process")
        
        if not videos_to_process:
            rich_console.print_warning("No videos found for key events generation")
            return []
        
        # Process key events generation
        key_events_files = []
        failed_videos = []
        
        for i, video_id in enumerate(videos_to_process, 1):
            rich_console.print_info(f"Generating key events for video {i}/{len(videos_to_process)}: {video_id}")
            
            try:
                key_events_file = self.generate_key_events_for_video(video_id)
                if key_events_file:
                    key_events_files.append(key_events_file)
                    rich_console.print_success(f"✓ Successfully generated key events for {video_id} ({i}/{len(videos_to_process)})")
                else:
                    failed_videos.append(video_id)
                    rich_console.print_error(f"✗ Failed to generate key events for {video_id} ({i}/{len(videos_to_process)})")
            except Exception as e:
                failed_videos.append(video_id)
                rich_console.print_error(f"✗ Error generating key events for {video_id}: {e}")
        
        # Print summary
        success_count = len(key_events_files)
        total_count = len(videos_to_process)
        rich_console.print_completion_message("Key Events Generation", {
            'total': total_count,
            'successful': success_count,
            'duration': 0  # Duration calculated externally
        })
        
        if failed_videos:
            rich_console.print_warning(f"Failed to generate key events for: {', '.join(failed_videos)}")
        
        return key_events_files


def main():
    """Main function for testing the key events generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate key events from aligned descriptions')
    parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to process')
    parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    parser.add_argument('--model', type=str, default=LLM_MODEL, help='Language model to use')
    parser.add_argument('--api-base', type=str, default=LLM_SERVER_URL, help='vLLM server API base URL')
    parser.add_argument('--max-workers', type=int, default=8, help='Maximum number of concurrent workers')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = KeyEventsGenerator(
        model_name=args.model,
        api_base=args.api_base,
        max_workers=args.max_workers
    )
    
    # Process key events generation
    key_events_files = generator.batch_generate_key_events(
        video_ids=args.video_ids,
        max_videos=args.max_videos
    )
    
    rich_console.print_info(f"Successfully generated key events for {len(key_events_files)} videos")


if __name__ == "__main__":
    main()
