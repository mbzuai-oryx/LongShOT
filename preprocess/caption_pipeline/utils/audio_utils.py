"""
Utility functions for audio processing.
"""

import os
import tempfile
import logging
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment

from caption_pipeline.utils.rich_console import get_console

# Try to import ffmpeg-python, but prepare for fallback
try:
    import ffmpeg
    HAS_FFMPEG_PY = hasattr(ffmpeg, 'input')
except (ImportError, AttributeError):
    HAS_FFMPEG_PY = False

logger = logging.getLogger(__name__)
rich_console = get_console()


def convert_audio_format(audio_path: str, output_format: str = 'wav', output_path: Optional[str] = None) -> Optional[str]:
    """
    Convert audio to a different format.
    
    Args:
        audio_path: Path to the audio file
        output_format: Output format (wav, mp3, etc.)
        output_path: Path to save the converted audio, or None to use a temporary file
        
    Returns:
        Path to the converted audio file, or None if conversion failed
    """
    try:
        if output_path is None:
            # Create temporary file
            fd, output_path = tempfile.mkstemp(suffix=f'.{output_format}')
            os.close(fd)
        
        if HAS_FFMPEG_PY:
            # Convert using ffmpeg-python
            (
                ffmpeg
                .input(audio_path)
                .output(output_path)
                .run(quiet=True, overwrite_output=True)
            )
        else:
            # Fallback to subprocess
            import subprocess
            cmd = ['ffmpeg', '-i', audio_path, '-y', output_path]
            subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
    
    except Exception as e:
        rich_console.print_error(f"Error converting audio format: {e}")
        return None


def split_audio(audio_path: str, segment_length: float = 30.0, overlap: float = 1.0) -> List[Dict[str, Any]]:
    """
    Split audio into overlapping segments.
    
    Args:
        audio_path: Path to the audio file
        segment_length: Length of each segment in seconds
        overlap: Overlap between segments in seconds
        
    Returns:
        List of dictionaries with segment information (start, end, path)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(audio_path), 'segments')
        os.makedirs(output_dir, exist_ok=True)
        
        segments = []
        start = 0
        segment_count = 0
        
        while start < duration:
            end = min(start + segment_length, duration)
            
            # Calculate sample indices
            start_idx = int(start * sr)
            end_idx = int(end * sr)
            
            # Extract segment audio
            segment_audio = y[start_idx:end_idx]
            
            # Create segment filename
            segment_filename = os.path.join(
                output_dir, 
                f"{os.path.splitext(os.path.basename(audio_path))[0]}_segment_{segment_count}.wav"
            )
            
            # Save segment
            sf.write(segment_filename, segment_audio, sr)
            
            # Add segment info
            segments.append({
                'segment_id': segment_count,
                'start_time': start,
                'end_time': end,
                'path': segment_filename,
                'duration': end - start,
            })
            
            # Move to next segment with overlap
            start += segment_length - overlap
            segment_count += 1
        
        return segments
    
    except Exception as e:
        rich_console.print_error(f"Error splitting audio: {e}")
        return []


def detect_silence(audio_path: str, min_silence_len: int = 1000, silence_thresh: int = -40) -> List[Dict[str, float]]:
    """
    Detect silent portions of an audio file.
    
    Args:
        audio_path: Path to the audio file
        min_silence_len: Minimum length of silence in milliseconds
        silence_thresh: Silence threshold in dB
        
    Returns:
        List of dictionaries with silence information (start, end)
    """
    try:
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Find silent segments
        from pydub.silence import detect_silence
        silent_segments = detect_silence(
            audio, 
            min_silence_len=min_silence_len, 
            silence_thresh=silence_thresh
        )
        
        # Convert to seconds
        result = [
            {
                'start': start / 1000.0,
                'end': end / 1000.0,
                'duration': (end - start) / 1000.0
            }
            for start, end in silent_segments
        ]
        
        return result
    
    except Exception as e:
        rich_console.print_error(f"Error detecting silence: {e}")
        return []


def detect_speech_segments(audio_path: str, min_silence_len: int = 700, silence_thresh: int = -36) -> List[Dict[str, float]]:
    """
    Detect speech segments in an audio file (inverse of silence detection).
    
    Args:
        audio_path: Path to the audio file
        min_silence_len: Minimum length of silence in milliseconds
        silence_thresh: Silence threshold in dB
        
    Returns:
        List of dictionaries with speech segment information (start, end)
    """
    try:
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0  # Duration in seconds
        
        # Find silent segments
        silences = detect_silence(
            audio_path, 
            min_silence_len=min_silence_len, 
            silence_thresh=silence_thresh
        )
        
        # Convert silences to speech segments
        speech_segments = []
        prev_end = 0
        
        for silence in silences:
            silence_start = silence['start']
            
            # If there's a gap between the previous silence and this one, it's speech
            if silence_start > prev_end:
                speech_segments.append({
                    'start': prev_end,
                    'end': silence_start,
                    'duration': silence_start - prev_end
                })
            
            prev_end = silence['end']
        
        # Add final segment if needed
        if prev_end < duration:
            speech_segments.append({
                'start': prev_end,
                'end': duration,
                'duration': duration - prev_end
            })
        
        return speech_segments
    
    except Exception as e:
        rich_console.print_error(f"Error detecting speech segments: {e}")
        return []


def get_audio_features(audio_path: str) -> Dict[str, Any]:
    """
    Extract audio features from an audio file.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary containing audio features
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Basic features
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        
        # Temporal features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        
        # Energy
        energy = np.sum(y**2) / len(y)
        
        # RMS energy
        rms = librosa.feature.rms(y=y).mean()
        
        return {
            'duration': float(duration),
            'sample_rate': sr,
            'spectral_centroid': float(spectral_centroid),
            'spectral_bandwidth': float(spectral_bandwidth),
            'spectral_rolloff': float(spectral_rolloff),
            'zero_crossing_rate': float(zero_crossing_rate),
            'energy': float(energy),
            'rms': float(rms)
        }
    
    except Exception as e:
        rich_console.print_error(f"Error extracting audio features: {e}")
        return {
            'duration': 0,
            'sample_rate': 0,
            'error': str(e)
        }
