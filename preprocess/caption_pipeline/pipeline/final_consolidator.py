"""
Final Consolidator module for creating training-ready JSONL dataset files.
This module consolidates all pipeline outputs into properly formatted JSONL files
suitable for training large multimodal models and video benchmarks.
"""

import os
import json
import csv
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys

# Import project configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (CAPTIONS_DIR, VIDEO_DESCRIPTIONS_DIR, AUDIO_DESCRIPTIONS_DIR, 
                   MULTIMODAL_UNDERSTANDING_DIR, KEY_EVENTS_DIR, METADATA_DIR, FINAL_DIR)

# Import rich console utilities
from caption_pipeline.utils.rich_console import get_console

# Import common VLM utilities
from caption_pipeline.utils.vlm_common import load_aligned_json, load_json_file

# Set up logging
logger = logging.getLogger(__name__)
rich_console = get_console()


class FinalConsolidator:
    """
    Class to consolidate all pipeline outputs into training-ready JSONL files.
    Creates segment-level and video-level entries for multimodal model training.
    """
    
    def __init__(self):
        """Initialize the FinalConsolidator."""
        # Ensure output directory exists
        os.makedirs(FINAL_DIR, exist_ok=True)
        
        # Load video metadata once for efficiency
        self.video_metadata = self._load_video_metadata()
    
    def _load_video_metadata(self) -> Dict[str, Dict]:
        """Load video metadata from CSV file."""
        metadata = {}
        metadata_file = os.path.join(METADATA_DIR, "video_metadata.csv")
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    video_id = row.get('video_id', '')
                    if video_id:
                        metadata[video_id] = {
                            'title': row.get('title', ''),
                            'channel': row.get('channel', ''),
                            'duration': float(row.get('duration', 0)) if row.get('duration') else 0,
                            'view_count': int(row.get('view_count', 0)) if row.get('view_count') else 0,
                            'publish_date': row.get('publish_date', ''),
                            'description': row.get('description', ''),
                            'content_categories': row.get('content_categories', ''),
                            'original_description': row.get('original_description', ''),
                            'data_source': row.get('data_source', '')
                        }
        except Exception as e:
            logger.warning(f"Could not load video metadata: {e}")
            
        return metadata
    
    def _load_multimodal_understanding(self, video_id: str) -> Optional[Dict]:
        """Load multimodal understanding data for a video."""
        base_path = os.path.join(MULTIMODAL_UNDERSTANDING_DIR, f"{video_id}_multimodal_understanding")
        return load_aligned_json(base_path)
    
    def _load_key_events(self, video_id: str) -> Optional[Dict]:
        """Load key events data for a video from the separate key events directory."""
        return load_json_file(os.path.join(KEY_EVENTS_DIR, f"{video_id}_key_events.json"))
    
    def _load_captions_data(self, video_id: str) -> Optional[Dict]:
        """Load caption data for a video."""
        return load_json_file(os.path.join(CAPTIONS_DIR, f"{video_id}.json"))
    
    def _load_video_descriptions(self, video_id: str) -> Optional[Dict]:
        """Load video descriptions data for a video."""
        return load_json_file(os.path.join(VIDEO_DESCRIPTIONS_DIR, f"{video_id}_descriptions_aligned.json"))
    
    def _load_audio_descriptions(self, video_id: str) -> Optional[Dict]:
        """Load audio descriptions data for a video, preferring aligned files."""
        base_path = os.path.join(AUDIO_DESCRIPTIONS_DIR, f"{video_id}_audio_descriptions")
        return load_aligned_json(base_path)
    
    def _create_fallback_segments(self, captions_data: Dict, video_descriptions_data: Dict, 
                                 audio_descriptions_data: Dict) -> List[Dict]:
        """
        Create segments when multimodal understanding is not available.
        Uses caption segments as the base and adds available descriptions.
        """
        segments = []
        
        if not captions_data or 'transcript' not in captions_data:
            return segments
        
        caption_segments = captions_data.get('transcript', {}).get('segments', [])
        video_segments = video_descriptions_data.get('segments', []) if video_descriptions_data else []
        audio_segments = audio_descriptions_data.get('segments', []) if audio_descriptions_data else []
        
        for idx, cap_segment in enumerate(caption_segments):
            start_time = cap_segment.get('start', 0)
            end_time = cap_segment.get('end', 0)
            
            # Find overlapping descriptions
            visual_desc = ""
            audio_env = ""
            
            # Find overlapping video descriptions
            for vid_seg in video_segments:
                if (vid_seg.get('start', 0) < end_time and vid_seg.get('end', 0) > start_time):
                    if 'visual_description' in vid_seg:
                        visual_desc = vid_seg['visual_description']
                        break
            
            # Find overlapping audio descriptions
            for aud_seg in audio_segments:
                if (aud_seg.get('start', 0) < end_time and aud_seg.get('end', 0) > start_time):
                    if 'audio_description' in aud_seg:
                        audio_env = aud_seg['audio_description']
                        break
            
            segments.append({
                'segment_index': idx,
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'speech_content': cap_segment.get('text', ''),
                'visual_description': visual_desc,
                'audio_environment': audio_env,
                'multimodal_understanding': None,  # Not available in fallback
                'key_events': []  # Not available in fallback
            })
        
        return segments
    
    def _create_segment_entry(self, segment: Dict) -> Dict:
        """Create a segment entry for the video JSON."""
        return {
            'segment_index': segment.get('segment_index', 0),
            'temporal': {
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'duration': segment.get('duration', 0)
            },
            'multimodal_description': segment.get('multimodal_understanding', ''),
            'key_events': segment.get('key_events', []),
            'modalities': {
                'speech': segment.get('speech_content', ''),
                'visual': segment.get('visual_description', ''),
                'audio_environment': segment.get('audio_environment', '')
            }
        }
    
    def _create_video_json(self, video_id: str, segments: List[Dict], video_meta: Dict) -> Dict:
        """Create a complete video JSON structure."""
        # Parse categories
        categories = []
        if video_meta.get('content_categories'):
            categories = [cat.strip() for cat in video_meta.get('content_categories', '').split(',') if cat.strip()]

        # Calculate statistics in a single pass
        stats = {'duration': 0, 'speech': 0, 'visual': 0, 'audio': 0, 'multimodal': 0, 'key_events': 0, 'total_events': 0}
        for seg in segments:
            stats['duration'] += seg.get('duration', 0)
            if seg.get('speech_content', ''):
                stats['speech'] += 1
            if seg.get('visual_description', ''):
                stats['visual'] += 1
            if seg.get('audio_environment', ''):
                stats['audio'] += 1
            if seg.get('multimodal_understanding'):
                stats['multimodal'] += 1
            key_events = seg.get('key_events', [])
            if key_events:
                stats['key_events'] += 1
                stats['total_events'] += len(key_events)
        
        # Create segment entries
        segment_entries = [self._create_segment_entry(seg) for seg in segments]
        
        num_segments = len(segments)
        return {
            'video_id': video_id,
            'video': {
                'title': video_meta.get('title', ''),
                'channel': video_meta.get('channel', ''),
                'duration': video_meta.get('duration', 0),
                'categories': categories,
                'description': video_meta.get('description', ''),
                'publish_date': video_meta.get('publish_date', ''),
                'view_count': video_meta.get('view_count', 0),
                'data_source': video_meta.get('data_source', '')
            },
            'segments': segment_entries,
            'statistics': {
                'total_segments': num_segments,
                'total_processed_duration': stats['duration'],
                'segments_with_speech': stats['speech'],
                'segments_with_visual': stats['visual'],
                'segments_with_audio_env': stats['audio'],
                'segments_with_multimodal': stats['multimodal'],
                'segments_with_key_events': stats['key_events'],
                'total_key_events': stats['total_events'],
                'avg_key_events_per_segment': stats['total_events'] / num_segments if num_segments else 0,
                'completeness_ratio': {
                    'speech': stats['speech'] / num_segments if num_segments else 0,
                    'visual': stats['visual'] / num_segments if num_segments else 0,
                    'audio_environment': stats['audio'] / num_segments if num_segments else 0,
                    'multimodal': stats['multimodal'] / num_segments if num_segments else 0,
                    'key_events': stats['key_events'] / num_segments if num_segments else 0
                }
            },
            'meta': {
                'consolidated_at': datetime.now().isoformat(),
                'has_multimodal_understanding': stats['multimodal'] > 0,
                'has_key_events': stats['key_events'] > 0,
                'data_source_priority': 'multimodal_understanding' if stats['multimodal'] > 0 else 'individual_modalities'
            }
        }
    
    
    def consolidate_video(self, video_id: str) -> Optional[Dict]:
        """
        Consolidate all data for a single video into a JSONL file.

        Args:
            video_id: ID of the video to consolidate

        Returns:
            Dict with video JSON if successful, None otherwise
        """
        try:
            # Get video metadata
            video_meta = self.video_metadata.get(video_id, {})
            
            # Try to load multimodal understanding first (preferred)
            multimodal_data = self._load_multimodal_understanding(video_id)
            
            # Load key events data separately
            key_events_data = self._load_key_events(video_id)
            
            if multimodal_data and 'segments' in multimodal_data:
                # Use multimodal understanding data
                segments = []
                for i, segment in enumerate(multimodal_data['segments']):
                    # Get key events for this segment if available
                    key_events = []
                    if key_events_data and 'segments' in key_events_data:
                        if i < len(key_events_data['segments']):
                            key_events = key_events_data['segments'][i].get('key_events', [])
                    
                    segments.append({
                        'segment_index': segment.get('segment_index', 0),
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'duration': segment.get('duration', 0),
                        'speech_content': segment.get('source_modalities', {}).get('speech_content', ''),
                        'visual_description': segment.get('source_modalities', {}).get('visual_description', ''),
                        'audio_environment': segment.get('source_modalities', {}).get('audio_environment', ''),
                        'multimodal_understanding': segment.get('multimodal_understanding', ''),
                        'key_events': key_events
                    })
            else:
                # Fallback to individual modality files
                captions_data = self._load_captions_data(video_id)
                video_descriptions_data = self._load_video_descriptions(video_id)
                audio_descriptions_data = self._load_audio_descriptions(video_id)
                
                segments = self._create_fallback_segments(
                    captions_data, video_descriptions_data, audio_descriptions_data
                )
                
                # Add key events to fallback segments if available
                if key_events_data and 'segments' in key_events_data:
                    for i, segment in enumerate(segments):
                        if i < len(key_events_data['segments']):
                            segment['key_events'] = key_events_data['segments'][i].get('key_events', [])
            
            if not segments:
                rich_console.print_warning(f"No segments found for video {video_id}")
                return None
            
            # Create complete video JSON structure
            video_json = self._create_video_json(video_id, segments, video_meta)
            
            # Write individual JSON file in final/ directory
            output_file = os.path.join(FINAL_DIR, f"{video_id}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(video_json, f, indent=2, ensure_ascii=False)
            
            rich_console.print_success(f"Individual JSON: {video_id} -> {len(segments)} segments")
            return video_json
            
        except Exception as e:
            rich_console.print_error(f"Error consolidating video {video_id}: {e}")
            return None
    
    def consolidate_all_videos(self, video_ids: Optional[List[str]] = None, 
                              max_videos: Optional[int] = None) -> Dict[str, bool]:
        """
        Consolidate all available videos into individual JSON files and create combined JSONL.
        
        Args:
            video_ids: Specific video IDs to consolidate, or None for all available
            max_videos: Maximum number of videos to process
            
        Returns:
            Dictionary mapping video IDs to success status
        """
        # Find videos to consolidate
        if video_ids is None:
            video_ids = self._get_videos_for_consolidation()
            
        if max_videos:
            video_ids = video_ids[:max_videos]
        
        if not video_ids:
            rich_console.print_warning("No videos found for final consolidation")
            return {}
        
        rich_console.print_info(f"Starting final consolidation for {len(video_ids)} videos...")
        
        # Create progress bar for consolidation
        if len(video_ids) > 1:
            progress, task_id = rich_console.create_consolidation_progress(len(video_ids))
        else:
            progress, task_id = None, None
        
        results = {}
        video_jsons = []
        
        if progress:
            with progress:
                for video_id in video_ids:
                    progress.update(task_id, status=f"Consolidating {video_id}...")
                    video_json = self.consolidate_video(video_id)
                    success = video_json is not None
                    results[video_id] = success

                    if success:
                        video_jsons.append(video_json)
                    
                    status = "[OK] Completed" if success else "[X] Failed"
                    progress.update(task_id, advance=1, status=f"{video_id}: {status}")
        else:
            # Single video processing
            for video_id in video_ids:
                video_json = self.consolidate_video(video_id)
                success = video_json is not None
                results[video_id] = success

                if success:
                    video_jsons.append(video_json)
        
        # Create combined videos.jsonl file in dataset/ directory
        if video_jsons:
            dataset_dir = os.path.dirname(FINAL_DIR)  # Get parent directory (dataset/)
            combined_jsonl_file = os.path.join(dataset_dir, "videos.jsonl")
            
            with open(combined_jsonl_file, 'w', encoding='utf-8') as f:
                for video_json in video_jsons:
                    f.write(json.dumps(video_json, ensure_ascii=False) + '\n')
            
            rich_console.print_success(f"Combined JSONL: {len(video_jsons)} videos -> {combined_jsonl_file}")
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        if successful == total:
            rich_console.print_success(f"Final consolidation complete: {successful}/{total} videos processed successfully")
        else:
            rich_console.print_warning(f"Final consolidation complete: {successful}/{total} videos processed successfully, {total - successful} failed")
        
        return results
    
    def _get_videos_for_consolidation(self) -> List[str]:
        """Get list of videos that need consolidation."""
        available_videos = set()
        
        # Check for videos with captions (minimum requirement)
        try:
            if os.path.exists(CAPTIONS_DIR):
                caption_files = [f[:-5] for f in os.listdir(CAPTIONS_DIR) if f.endswith('.json')]
                available_videos.update(caption_files)
        except Exception as e:
            logger.error(f"Error scanning captions directory: {e}")
        
        # Filter out already consolidated videos
        try:
            if os.path.exists(FINAL_DIR):
                consolidated_files = [f[:-5] for f in os.listdir(FINAL_DIR) if f.endswith('.json')]
                available_videos = available_videos - set(consolidated_files)
        except Exception as e:
            logger.error(f"Error scanning final directory: {e}")
        
        return list(available_videos)


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Consolidate pipeline outputs into training-ready JSONL files')
    parser.add_argument('--video-ids', nargs='+', help='Specific video IDs to consolidate')
    parser.add_argument('--max-videos', type=int, help='Maximum number of videos to consolidate')
    
    args = parser.parse_args()
    
    # Initialize consolidator
    consolidator = FinalConsolidator()
    
    # Consolidate videos
    results = consolidator.consolidate_all_videos(
        video_ids=args.video_ids,
        max_videos=args.max_videos
    )
    
    # Print summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    print(f"\nConsolidation complete: {successful}/{total} videos processed successfully")


if __name__ == "__main__":
    main()