"""
Dataset analyzer for generating comprehensive statistics and summaries.
"""

import os
import json
import csv
import statistics
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from .rich_console import get_console

rich_console = get_console()

class DatasetAnalyzer:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.metadata_file = self.dataset_dir / "metadata" / "video_metadata.csv"
        self.videos_jsonl = self.dataset_dir / "videos.jsonl"
        
        # Component directories
        self.videos_dir = self.dataset_dir / "videos"
        self.audio_dir = self.dataset_dir / "audio"
        self.captions_dir = self.dataset_dir / "captions"
        self.video_descriptions_dir = self.dataset_dir / "video_descriptions"
        self.audio_descriptions_dir = self.dataset_dir / "audio_descriptions"
        self.multimodal_dir = self.dataset_dir / "multimodal_understanding"
        self.final_dir = self.dataset_dir / "final"
        
    def load_metadata(self) -> List[Dict]:
        """Load video metadata from CSV file."""
        metadata = []
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                metadata = list(reader)
        return metadata
    
    def get_file_stats(self, directory: Path, extension: str) -> Dict[str, Any]:
        """Get statistics for files in a directory."""
        if not directory.exists():
            return {"count": 0, "total_size": 0, "files": []}
        
        files = list(directory.glob(f"*{extension}"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "count": len(files),
            "total_size": total_size,
            "files": [f.name for f in files]
        }
    
    def analyze_captions(self) -> Dict[str, Any]:
        """Analyze caption files and extract statistics."""
        stats = {"total_segments": 0, "total_duration": 0, "videos": {}}
        
        if not self.captions_dir.exists():
            return stats
        
        for caption_file in self.captions_dir.glob("*.json"):
            video_id = caption_file.stem
            try:
                with open(caption_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Look for segments in the correct path - either direct or under transcript
                segments = data.get('segments', [])
                if not segments and 'transcript' in data:
                    segments = data['transcript'].get('segments', [])
                
                video_stats = {
                    "segment_count": len(segments),
                    "duration": 0,
                    "word_count": 0
                }
                
                for segment in segments:
                    if 'end' in segment and 'start' in segment:
                        video_stats["duration"] += segment['end'] - segment['start']
                    if 'text' in segment:
                        video_stats["word_count"] += len(segment['text'].split())
                
                stats["videos"][video_id] = video_stats
                stats["total_segments"] += video_stats["segment_count"]
                stats["total_duration"] += video_stats["duration"]
                
            except Exception as e:
                rich_console.print_warning(f"Error analyzing caption file {caption_file}: {e}")
        
        return stats
    
    def analyze_video_descriptions(self) -> Dict[str, Any]:
        """Analyze video description files."""
        stats = {"total_descriptions": 0, "videos": {}}
        
        if not self.video_descriptions_dir.exists():
            return stats
        
        for desc_file in self.video_descriptions_dir.glob("*_descriptions_aligned.json"):
            video_id = desc_file.name.replace('_descriptions_aligned.json', '')
            try:
                with open(desc_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                segments = data.get('segments', [])
                video_stats = {
                    "segment_count": len(segments),
                    "total_frames": sum(seg.get('frames_extracted', 0) for seg in segments),
                    "avg_frames_per_segment": 0
                }
                
                if video_stats["segment_count"] > 0:
                    video_stats["avg_frames_per_segment"] = video_stats["total_frames"] / video_stats["segment_count"]
                
                stats["videos"][video_id] = video_stats
                stats["total_descriptions"] += video_stats["segment_count"]
                
            except Exception as e:
                rich_console.print_warning(f"Error analyzing video description file {desc_file}: {e}")
        
        return stats
    
    def analyze_audio_descriptions(self) -> Dict[str, Any]:
        """Analyze audio description files, preferring aligned files."""
        stats = {"total_descriptions": 0, "videos": {}}
        
        if not self.audio_descriptions_dir.exists():
            return stats
        
        # Collect all video IDs that have audio descriptions (aligned or original)
        video_ids_processed = set()
        
        # First process aligned files
        for desc_file in self.audio_descriptions_dir.glob("*_audio_descriptions_aligned.json"):
            video_id = desc_file.name.replace('_audio_descriptions_aligned.json', '')
            video_ids_processed.add(video_id)
            try:
                with open(desc_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                segments = data.get('segments', [])
                video_stats = {
                    "segment_count": len(segments),
                    "avg_description_length": 0,
                    "alignment_status": "aligned"
                }
                
                if segments:
                    descriptions = [seg.get('audio_description', '') for seg in segments if seg.get('audio_description')]
                    if descriptions:
                        video_stats["avg_description_length"] = statistics.mean(len(desc.split()) for desc in descriptions)
                
                stats["videos"][video_id] = video_stats
                stats["total_descriptions"] += video_stats["segment_count"]
                
            except Exception as e:
                rich_console.print_warning(f"Error analyzing aligned audio description file {desc_file}: {e}")
        
        # Then process original files for videos not yet processed
        for desc_file in self.audio_descriptions_dir.glob("*_audio_descriptions.json"):
            if desc_file.name.endswith("_aligned.json"):
                continue  # Skip already processed aligned files
                
            video_id = desc_file.name.replace('_audio_descriptions.json', '')
            if video_id in video_ids_processed:
                continue  # Skip if we already processed the aligned version
                
            try:
                with open(desc_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                segments = data.get('segments', [])
                video_stats = {
                    "segment_count": len(segments),
                    "avg_description_length": 0,
                    "alignment_status": "original"
                }
                
                if segments:
                    descriptions = [seg.get('audio_description', '') for seg in segments if seg.get('audio_description')]
                    if descriptions:
                        video_stats["avg_description_length"] = statistics.mean(len(desc.split()) for desc in descriptions)
                
                stats["videos"][video_id] = video_stats
                stats["total_descriptions"] += video_stats["segment_count"]
                
            except Exception as e:
                rich_console.print_warning(f"Error analyzing audio description file {desc_file}: {e}")
        
        return stats
    
    def format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive dataset summary."""
        rich_console.print_info("🔍 Analyzing dataset components...")
        
        # Load metadata
        metadata = self.load_metadata()
        
        # Analyze different components
        video_stats = self.get_file_stats(self.videos_dir, ".mp4")
        audio_stats = self.get_file_stats(self.audio_dir, ".wav")
        caption_analysis = self.analyze_captions()
        video_desc_analysis = self.analyze_video_descriptions()
        audio_desc_analysis = self.analyze_audio_descriptions()
        multimodal_stats = self.get_file_stats(self.multimodal_dir, ".json")
        final_stats = self.get_file_stats(self.final_dir, ".json")
        
        # Calculate totals and statistics
        total_videos = len(metadata)
        completed_videos = len([m for m in metadata if m.get('status') == 'captioned'])
        failed_videos = total_videos - completed_videos
        
        # Duration statistics
        durations = []
        categories = {}
        for meta in metadata:
            if meta.get('duration'):
                try:
                    duration = float(meta['duration'])
                    durations.append(duration)
                except ValueError:
                    pass
            
            # Count categories
            cats = meta.get('content_categories', '').split(', ')
            for cat in cats:
                cat = cat.strip()
                if cat:
                    categories[cat] = categories.get(cat, 0) + 1
        
        # Calculate duration statistics
        duration_stats = {}
        if durations:
            duration_stats = {
                "total_duration_seconds": sum(durations),
                "total_duration_hours": sum(durations) / 3600,
                "average_duration_seconds": statistics.mean(durations),
                "median_duration_seconds": statistics.median(durations),
                "min_duration_seconds": min(durations),
                "max_duration_seconds": max(durations)
            }
        
        # Create comprehensive summary
        summary = {
            "dataset_overview": {
                "generation_date": datetime.now().isoformat(),
                "dataset_path": str(self.dataset_dir),
                "total_videos": total_videos,
                "completed_videos": completed_videos,
                "failed_videos": failed_videos,
                "success_rate": f"{(completed_videos/total_videos*100):.1f}%" if total_videos > 0 else "0%"
            },
            "duration_statistics": duration_stats,
            "content_categories": dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)),
            "file_statistics": {
                "videos": {
                    "count": video_stats["count"],
                    "total_size": self.format_size(video_stats["total_size"]),
                    "average_size": self.format_size(video_stats["total_size"] / max(1, video_stats["count"]))
                },
                "audio": {
                    "count": audio_stats["count"],
                    "total_size": self.format_size(audio_stats["total_size"]),
                    "average_size": self.format_size(audio_stats["total_size"] / max(1, audio_stats["count"]))
                },
                "captions": {
                    "count": len(caption_analysis["videos"]),
                    "total_segments": caption_analysis["total_segments"],
                    "total_duration_seconds": caption_analysis["total_duration"],
                    "average_segments_per_video": caption_analysis["total_segments"] / max(1, len(caption_analysis["videos"]))
                },
                "video_descriptions": {
                    "count": len(video_desc_analysis["videos"]),
                    "total_descriptions": video_desc_analysis["total_descriptions"],
                    "average_descriptions_per_video": video_desc_analysis["total_descriptions"] / max(1, len(video_desc_analysis["videos"]))
                },
                "audio_descriptions": {
                    "count": len(audio_desc_analysis["videos"]),
                    "total_descriptions": audio_desc_analysis["total_descriptions"],
                    "average_descriptions_per_video": audio_desc_analysis["total_descriptions"] / max(1, len(audio_desc_analysis["videos"]))
                },
                "multimodal_understanding": {
                    "count": multimodal_stats["count"],
                    "total_size": self.format_size(multimodal_stats["total_size"])
                },
                "final_consolidated": {
                    "count": final_stats["count"],
                    "total_size": self.format_size(final_stats["total_size"])
                }
            },
            "video_details": {}
        }
        
        # Add detailed video information
        for meta in metadata:
            video_id = meta.get('video_id', '')
            if video_id:
                video_detail = {
                    "title": meta.get('title', ''),
                    "channel": meta.get('channel', ''),
                    "duration_seconds": float(meta.get('duration', 0)) if meta.get('duration') else 0,
                    "status": meta.get('status', ''),
                    "categories": meta.get('content_categories', '').split(', ') if meta.get('content_categories') else [],
                    "view_count": int(meta.get('view_count', 0)) if meta.get('view_count') else 0,
                    "components": {
                        "video_file": video_id in [f.replace('.mp4', '') for f in video_stats["files"]],
                        "audio_file": video_id in [f.replace('.wav', '') for f in audio_stats["files"]],
                        "captions": video_id in caption_analysis["videos"],
                        "video_descriptions": video_id in video_desc_analysis["videos"],
                        "audio_descriptions": video_id in audio_desc_analysis["videos"],
                        "multimodal_understanding": (f"{video_id}_multimodal_understanding_aligned.json" in multimodal_stats["files"] or 
                                                     f"{video_id}_multimodal_understanding.json" in multimodal_stats["files"]),
                        "final_consolidated": f"{video_id}.json" in final_stats["files"]
                    }
                }
                
                # Add component-specific stats
                if video_id in caption_analysis["videos"]:
                    video_detail["caption_stats"] = caption_analysis["videos"][video_id]
                if video_id in video_desc_analysis["videos"]:
                    video_detail["video_description_stats"] = video_desc_analysis["videos"][video_id]
                if video_id in audio_desc_analysis["videos"]:
                    video_detail["audio_description_stats"] = audio_desc_analysis["videos"][video_id]
                
                summary["video_details"][video_id] = video_detail
        
        return summary
    
    def save_summary(self, summary: Dict[str, Any], output_file: str = None) -> str:
        """Save the dataset summary to a JSON file."""
        if output_file is None:
            output_file = self.dataset_dir / "dataset_summary.json"
        else:
            output_file = Path(output_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return str(output_file)
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print a formatted summary to console."""
        overview = summary["dataset_overview"]
        duration_stats = summary["duration_statistics"]
        file_stats = summary["file_statistics"]
        
        rich_console.print_component_header("Dataset Summary", "Comprehensive overview of generated dataset")
        
        # Overview
        rich_console.print_info("📊 Dataset Overview:")
        rich_console.print_info(f"  • Total videos: {overview['total_videos']}")
        rich_console.print_info(f"  • Completed videos: {overview['completed_videos']}")
        rich_console.print_info(f"  • Failed videos: {overview['failed_videos']}")
        rich_console.print_info(f"  • Success rate: {overview['success_rate']}")
        rich_console.print_info("")
        
        # Duration statistics
        if duration_stats:
            hours = duration_stats["total_duration_hours"]
            avg_min = duration_stats["average_duration_seconds"] / 60
            rich_console.print_info("⏱️ Duration Statistics:")
            rich_console.print_info(f"  • Total duration: {hours:.1f} hours ({duration_stats['total_duration_seconds']:.0f} seconds)")
            rich_console.print_info(f"  • Average video length: {avg_min:.1f} minutes")
            rich_console.print_info(f"  • Shortest video: {duration_stats['min_duration_seconds']:.0f} seconds")
            rich_console.print_info(f"  • Longest video: {duration_stats['max_duration_seconds']:.0f} seconds")
            rich_console.print_info("")
        
        # Categories
        categories = summary["content_categories"]
        if categories:
            rich_console.print_info("🏷️ Content Categories:")
            for category, count in list(categories.items())[:10]:  # Top 10 categories
                if category.strip():  # Only show non-empty categories
                    rich_console.print_info(f"  • {category}: {count} videos")
            rich_console.print_info("")
        
        # File statistics
        rich_console.print_info("📁 Component Statistics:")
        rich_console.print_info(f"  • Videos: {file_stats['videos']['count']} files ({file_stats['videos']['total_size']})")
        rich_console.print_info(f"  • Audio: {file_stats['audio']['count']} files ({file_stats['audio']['total_size']})")
        rich_console.print_info(f"  • Captions: {file_stats['captions']['count']} files ({file_stats['captions']['total_segments']} segments)")
        rich_console.print_info(f"  • Video descriptions: {file_stats['video_descriptions']['count']} files ({file_stats['video_descriptions']['total_descriptions']} descriptions)")
        rich_console.print_info(f"  • Audio descriptions: {file_stats['audio_descriptions']['count']} files ({file_stats['audio_descriptions']['total_descriptions']} descriptions)")
        rich_console.print_info(f"  • Multimodal understanding: {file_stats['multimodal_understanding']['count']} files ({file_stats['multimodal_understanding']['total_size']})")
        rich_console.print_info(f"  • Final consolidated: {file_stats['final_consolidated']['count']} files ({file_stats['final_consolidated']['total_size']})")
    
    def generate_and_save_summary(self) -> str:
        """Generate summary, save to file, and print to console."""
        summary = self.generate_summary()
        output_file = self.save_summary(summary)
        self.print_summary(summary)
        return output_file