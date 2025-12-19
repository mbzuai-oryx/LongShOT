"""
Audio Segment Extractor for Caption Pipeline

This module provides functionality to extract audio segments from video files
based on timestamp information for Audio Flamingo 3 processing.
"""

import os
import subprocess
import tempfile
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

# Set up logging
logger = logging.getLogger(__name__)

class AudioSegmentExtractor:
    """Extract audio segments from video files based on timestamps."""
    
    def __init__(self, temp_dir: Optional[str] = None, sample_rate: int = 16000):
        """
        Initialize the Audio Segment Extractor.
        
        Args:
            temp_dir: Directory for temporary files (None for system temp)
            sample_rate: Target sample rate for extracted audio
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.sample_rate = sample_rate
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info(f"AudioSegmentExtractor initialized with temp_dir={self.temp_dir}, sample_rate={sample_rate}")
    
    def extract_segment_audio(self, video_path: str, start_time: float, end_time: float, 
                             output_path: Optional[str] = None) -> str:
        """
        Extract a single audio segment from a video file.
        
        Args:
            video_path: Path to the input video file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Optional output path (creates temp file if None)
            
        Returns:
            Path to the extracted audio file
            
        Raises:
            RuntimeError: If audio extraction fails
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Generate output path if not provided
        if output_path is None:
            video_name = Path(video_path).stem
            temp_filename = f"{video_name}_{start_time:.2f}s-{end_time:.2f}s.wav"
            output_path = os.path.join(self.temp_dir, temp_filename)
        
        # Calculate duration
        duration = end_time - start_time
        if duration <= 0:
            raise ValueError(f"Invalid duration: {duration} seconds (start={start_time}, end={end_time})")
        
        # Build ffmpeg command for audio extraction
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-ar', str(self.sample_rate),
            '-ac', '1',  # Mono
            '-acodec', 'pcm_s16le',
            '-f', 'wav',
            '-y',  # Overwrite output file
            output_path
        ]
        
        try:
            # Run ffmpeg command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=60  # 60 second timeout
            )
            
            # Verify output file was created
            if not os.path.exists(output_path):
                raise RuntimeError(f"Audio extraction failed - output file not created: {output_path}")
            
            # Check if file has content
            if os.path.getsize(output_path) == 0:
                raise RuntimeError(f"Audio extraction failed - empty output file: {output_path}")
            
            logger.debug(f"Successfully extracted audio segment: {start_time:.2f}s-{end_time:.2f}s -> {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"ffmpeg failed: {e.stderr}"
            logger.error(error_msg)
            # Clean up failed output file
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise RuntimeError(error_msg)
        
        except subprocess.TimeoutExpired:
            error_msg = f"Audio extraction timed out for segment {start_time:.2f}s-{end_time:.2f}s"
            logger.error(error_msg)
            # Clean up failed output file
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise RuntimeError(error_msg)
    
    def extract_segments_batch(self, video_path: str, segments: List[Dict[str, any]], 
                              output_dir: Optional[str] = None) -> List[Dict[str, any]]:
        """
        Extract multiple audio segments from a video file.
        
        Args:
            video_path: Path to the input video file
            segments: List of segment dictionaries with 'start' and 'end' keys
            output_dir: Directory for output files (uses temp_dir if None)
            
        Returns:
            List of dictionaries with segment info and extracted audio paths
        """
        if not segments:
            return []
        
        output_dir = output_dir or self.temp_dir
        os.makedirs(output_dir, exist_ok=True)
        
        video_name = Path(video_path).stem
        extracted_segments = []
        
        logger.info(f"Extracting {len(segments)} audio segments from {video_name}")
        
        for i, segment in enumerate(segments):
            try:
                start_time = float(segment.get('start', 0))
                end_time = float(segment.get('end', start_time + 1))
                
                # Generate output filename
                output_filename = f"{video_name}_segment_{i:03d}_{start_time:.2f}s-{end_time:.2f}s.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                # Extract segment
                extracted_path = self.extract_segment_audio(video_path, start_time, end_time, output_path)
                
                # Create result dictionary
                result_segment = {
                    'segment_index': i,
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time,
                    'audio_path': extracted_path,
                    'extraction_success': True,
                    'original_segment': segment
                }
                
                extracted_segments.append(result_segment)
                
            except Exception as e:
                logger.error(f"Failed to extract segment {i} ({segment.get('start', 0):.2f}s-{segment.get('end', 0):.2f}s): {e}")
                
                # Add failed segment to results
                result_segment = {
                    'segment_index': i,
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0),
                    'duration': segment.get('end', 0) - segment.get('start', 0),
                    'audio_path': None,
                    'extraction_success': False,
                    'error': str(e),
                    'original_segment': segment
                }
                
                extracted_segments.append(result_segment)
        
        successful_extractions = sum(1 for seg in extracted_segments if seg['extraction_success'])
        logger.info(f"Successfully extracted {successful_extractions}/{len(segments)} audio segments")
        
        return extracted_segments
    
    def extract_single_segment(self, video_path: str, segment_data: Dict[str, any], 
                              output_dir: Optional[str] = None) -> Optional[Dict[str, any]]:
        """
        Extract a single audio segment from a video file (for pipeline processing).
        
        Args:
            video_path: Path to the input video file
            segment_data: Dictionary with 'start', 'end', and 'segment_index' keys
            output_dir: Directory for output files (uses temp_dir if None)
            
        Returns:
            Dictionary with segment info and extracted audio path, or None if failed
        """
        output_dir = output_dir or self.temp_dir
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            start_time = float(segment_data.get('start', 0))
            end_time = float(segment_data.get('end', start_time + 1))
            segment_index = segment_data.get('segment_index', 0)
            
            video_name = Path(video_path).stem
            
            # Generate output filename
            output_filename = f"{video_name}_segment_{segment_index:03d}_{start_time:.2f}s-{end_time:.2f}s.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # Extract segment
            extracted_path = self.extract_segment_audio(video_path, start_time, end_time, output_path)
            
            # Create result dictionary
            result_segment = {
                'segment_index': segment_index,
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'audio_path': extracted_path,
                'extraction_success': True
            }
            
            return result_segment
            
        except Exception as e:
            logger.error(f"Failed to extract single segment {segment_data.get('segment_index', 0)}: {e}")
            
            # Return failed segment info
            return {
                'segment_index': segment_data.get('segment_index', 0),
                'start': segment_data.get('start', 0),
                'end': segment_data.get('end', 0),
                'duration': segment_data.get('end', 0) - segment_data.get('start', 0),
                'audio_path': None,
                'extraction_success': False,
                'error': str(e)
            }
    
    def cleanup_extracted_files(self, extracted_segments: List[Dict[str, any]]):
        """
        Clean up extracted audio files.
        
        Args:
            extracted_segments: List of segment dictionaries from extract_segments_batch
        """
        cleaned_count = 0
        for segment in extracted_segments:
            audio_path = segment.get('audio_path')
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    cleaned_count += 1
                    logger.debug(f"Cleaned up audio file: {audio_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up audio file {audio_path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} extracted audio files")
    
    def extract_audio_segments_from_descriptions(self, video_path: str, 
                                                video_descriptions_file: str,
                                                output_dir: Optional[str] = None) -> List[Dict[str, any]]:
        """
        Extract audio segments based on video descriptions JSON file.
        
        Args:
            video_path: Path to the input video file
            video_descriptions_file: Path to video descriptions JSON file
            output_dir: Directory for output files (uses temp_dir if None)
            
        Returns:
            List of dictionaries with segment info and extracted audio paths
        """
        if not os.path.exists(video_descriptions_file):
            raise FileNotFoundError(f"Video descriptions file not found: {video_descriptions_file}")
        
        try:
            with open(video_descriptions_file, 'r', encoding='utf-8') as f:
                descriptions_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in video descriptions file: {e}")
        
        # Extract segments from descriptions data
        segments = descriptions_data.get('segments', [])
        if not segments:
            logger.warning(f"No segments found in video descriptions file: {video_descriptions_file}")
            return []
        
        logger.info(f"Found {len(segments)} segments in video descriptions file")
        
        # Extract audio for all segments
        return self.extract_segments_batch(video_path, segments, output_dir)
    
    def create_segments_manifest(self, extracted_segments: List[Dict[str, any]], 
                                output_path: str) -> str:
        """
        Create a manifest file listing all extracted audio segments.
        
        Args:
            extracted_segments: List of segment dictionaries from extract_segments_batch
            output_path: Path for the manifest file
            
        Returns:
            Path to the created manifest file
        """
        manifest_data = {
            'total_segments': len(extracted_segments),
            'successful_extractions': sum(1 for seg in extracted_segments if seg['extraction_success']),
            'failed_extractions': sum(1 for seg in extracted_segments if not seg['extraction_success']),
            'extraction_timestamp': os.path.getctime(extracted_segments[0]['audio_path']) if extracted_segments and extracted_segments[0].get('audio_path') else None,
            'segments': extracted_segments
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created segments manifest: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create segments manifest: {e}")
            raise


def main():
    """Test the AudioSegmentExtractor functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract audio segments from video files')
    parser.add_argument('--video', required=True, help='Input video file path')
    parser.add_argument('--descriptions', help='Video descriptions JSON file')
    parser.add_argument('--output-dir', help='Output directory for extracted audio')
    parser.add_argument('--start', type=float, help='Start time for single segment extraction')
    parser.add_argument('--end', type=float, help='End time for single segment extraction')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate for extracted audio')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = AudioSegmentExtractor(sample_rate=args.sample_rate)
    
    if args.descriptions:
        # Extract from video descriptions file
        extracted_segments = extractor.extract_audio_segments_from_descriptions(
            args.video, args.descriptions, args.output_dir
        )
        
        # Create manifest
        if args.output_dir:
            manifest_path = os.path.join(args.output_dir, 'segments_manifest.json')
            extractor.create_segments_manifest(extracted_segments, manifest_path)
        
        print(f"Extracted {len(extracted_segments)} segments")
        
    elif args.start is not None and args.end is not None:
        # Extract single segment
        output_path = extractor.extract_segment_audio(args.video, args.start, args.end)
        print(f"Extracted segment: {output_path}")
        
    else:
        print("Error: Either --descriptions or both --start and --end must be provided")


if __name__ == "__main__":
    main()