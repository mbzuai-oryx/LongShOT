"""
Movie-style caption enhancer using multimodal models.
Focuses on audio cues (music, sound effects, ambient sounds) and essential visual information
only when not obvious from the video. Follows movie caption conventions with minimal
visual descriptions and emphasis on audio accessibility.
"""

import cv2
import numpy as np
import torch
import librosa
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import re

from caption_pipeline.utils.rich_console import get_console

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    
# BLIP is not used in this implementation as detailed image captioning
# is skipped for movie-style captions where visual details should be obvious
# from watching the video. Only CLIP is needed for high-level scene classification.
BLIP_AVAILABLE = False

logger = logging.getLogger(__name__)
rich_console = get_console()

@dataclass
class AudioEvent:
    """Represents detected audio events with timestamps."""
    start_time: float
    end_time: float
    event_type: str
    confidence: float
    description: str

@dataclass
class VisualEvent:
    """Represents detected visual events with timestamps."""
    timestamp: float
    event_type: str
    confidence: float
    description: str

@dataclass
class EnhancedSegment:
    """Represents a caption segment enhanced with descriptive elements."""
    start: float
    end: float
    text: str
    audio_events: List[AudioEvent]
    visual_events: List[VisualEvent]
    enhanced_text: str

class AudioEventDetector:
    """Detects audio events like music, sound effects, speech tone."""
    
    def __init__(self):
        self.sample_rate = 16000
        self.hop_length = 512
        
    def extract_audio_features(self, audio_path: str, start_time: float, end_time: float) -> Dict:
        """Extract audio features for the given time segment."""
        try:
            # Check if segment duration is too short or invalid
            if end_time <= start_time or end_time - start_time < 0.1:
                rich_console.print_warning(f"Audio segment too short or invalid: {start_time:.2f}s-{end_time:.2f}s")
                return {}
                
            # Load audio segment
            y, sr = librosa.load(audio_path, sr=self.sample_rate, 
                               offset=start_time, duration=end_time - start_time)
            
            # Check if loaded audio is empty or too short
            if len(y) < sr * 0.1:  # Less than 0.1 seconds
                rich_console.print_warning(f"Loaded audio segment is too short or empty: {len(y)} samples")
                return {}
            
            # Extract features
            features = {
                'rms': librosa.feature.rms(y=y, hop_length=self.hop_length),
                'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr),
                'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr),
                'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13),
                'chroma': librosa.feature.chroma_stft(y=y, sr=sr),
                'tempo': librosa.feature.tempo(y=y, sr=sr),  # Updated from beat.tempo to feature.tempo
                'zero_crossing_rate': librosa.feature.zero_crossing_rate(y)
            }
            
            return features
            
        except Exception as e:
            rich_console.print_error(f"Error extracting audio features: {e}")
            return {}
    
    def detect_music(self, features: Dict) -> Tuple[bool, float]:
        """Detect if segment contains music."""
        if not features:
            return False, 0.0
            
        try:
            # Music detection heuristics
            chroma_var = np.var(features['chroma'])
            tempo = features['tempo'][0] if len(features['tempo']) > 0 else 0
            spectral_centroid_mean = np.mean(features['spectral_centroid'])
            
            # Simple music detection based on harmonic content and rhythm
            music_score = 0.0
            
            # High chroma variance indicates harmonic content
            if chroma_var > 0.5:
                music_score += 0.4
                
            # Tempo in typical music range
            if 60 <= tempo <= 180:
                music_score += 0.3
                
            # Spectral characteristics
            if 1000 <= spectral_centroid_mean <= 8000:
                music_score += 0.3
                
            return music_score > 0.6, music_score
            
        except Exception as e:
            rich_console.print_error(f"Error detecting music: {e}")
            return False, 0.0
    
    def detect_sound_effects(self, features: Dict) -> List[Tuple[str, float]]:
        """Detect movie-relevant sound effects and ambient audio."""
        if not features:
            return []
            
        effects = []
        
        try:
            rms_mean = np.mean(features['rms'])
            zcr_mean = np.mean(features['zero_crossing_rate'])
            spectral_rolloff_mean = np.mean(features['spectral_rolloff'])
            spectral_centroid_mean = np.mean(features['spectral_centroid'])
            
            # Movie-style audio cues
            
            # Sudden loud sounds (explosions, crashes, impacts)
            if rms_mean > 0.4:
                effects.append(("loud crash", 0.8))
            elif rms_mean > 0.3:
                effects.append(("impact sound", 0.7))
                
            # High frequency sounds (glass breaking, metal, alarms)
            if spectral_rolloff_mean > 10000:
                effects.append(("glass breaking", 0.7))
            elif spectral_rolloff_mean > 8000:
                effects.append(("metallic sound", 0.6))
                
            # Ambient noise and weather sounds
            if zcr_mean > 0.4:
                if spectral_centroid_mean > 5000:
                    effects.append(("wind blowing", 0.6))
                else:
                    effects.append(("background noise", 0.5))
            elif zcr_mean > 0.3:
                effects.append(("ambient sound", 0.5))
                
            # Low frequency rumble (engines, explosions distant)
            if spectral_centroid_mean < 1000 and rms_mean > 0.2:
                effects.append(("rumbling sound", 0.6))
                
        except Exception as e:
            rich_console.print_error(f"Error detecting sound effects: {e}")
            
        return effects
    
    def analyze_segment(self, audio_path: str, start_time: float, end_time: float) -> List[AudioEvent]:
        """Analyze audio segment and return detected events."""
        # Check if the audio file exists
        if not os.path.exists(audio_path):
            rich_console.print_error(f"Audio file not found: {audio_path}")
            return []
            
        # Check for valid segment times
        if start_time < 0 or end_time <= start_time:
            rich_console.print_warning(f"Invalid segment time range: {start_time:.2f}s-{end_time:.2f}s")
            return []
            
        # Extract features with error handling
        features = self.extract_audio_features(audio_path, start_time, end_time)
        if not features:
            rich_console.print_warning(f"No features extracted for segment: {start_time:.2f}s-{end_time:.2f}s")
            return []
            
        events = []
        
        try:
            # Detect music
            has_music, music_confidence = self.detect_music(features)
            if has_music:
                music_type = self._classify_music_type(features)
                events.append(AudioEvent(
                    start_time=start_time,
                    end_time=end_time,
                    event_type="music",
                    confidence=music_confidence,
                    description=f"[{music_type} music playing]"
                ))
            
            # Detect sound effects
            sound_effects = self.detect_sound_effects(features)
            for effect_type, confidence in sound_effects:
                events.append(AudioEvent(
                    start_time=start_time,
                    end_time=end_time,
                    event_type="sound_effect",
                    confidence=confidence,
                    description=f"[{effect_type}]"
                ))
        except Exception as e:
            rich_console.print_error(f"Error analyzing audio segment {start_time:.2f}s-{end_time:.2f}s: {e}")
        
        return events
    
    def _classify_music_type(self, features: Dict) -> str:
        """Classify the type of music using movie-appropriate descriptors."""
        try:
            tempo = features['tempo'][0] if len(features['tempo']) > 0 else 0
            rms_mean = np.mean(features['rms'])
            spectral_centroid_mean = np.mean(features['spectral_centroid'])
            
            # Movie-style music classifications
            if tempo > 140 and rms_mean > 0.4:
                return "intense music"
            elif tempo > 120:
                return "upbeat music"
            elif tempo < 60:
                return "slow music"
            elif rms_mean > 0.6:
                return "dramatic music"
            elif rms_mean < 0.2:
                return "soft music"
            elif spectral_centroid_mean > 4000:
                return "bright music"
            else:
                return "music"
                
        except Exception:
            return "music"

class VisualEventDetector:
    """Detects visual events and scene changes using CLIP for movie-relevant scene classification."""
    
    def __init__(self):
        self.clip_model = None
        self.clip_preprocess = None
        self._load_models()
        
    def _load_models(self):
        """Load vision models."""
        try:
            # Check if CUDA is available and set the device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if CLIP_AVAILABLE:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                rich_console.print_info(f"CLIP model loaded successfully on {self.device}")
            else:
                rich_console.print_warning("CLIP not available, visual analysis will be limited")
                
            # BLIP model loading removed - not used in movie-style caption generation
            # Visual details are considered obvious from watching the video
                
        except Exception as e:
            rich_console.print_error(f"Error loading vision models: {e}")
    
    def extract_key_frames(self, video_path: str, timestamps: List[float]) -> List[Tuple[float, np.ndarray]]:
        """Extract frames at specific timestamps."""
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                rich_console.print_error(f"Could not open video file: {video_path}")
                return frames
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                rich_console.print_warning(f"Invalid FPS value: {fps}, using default of 25")
                fps = 25
                
            for timestamp in timestamps:
                frame_number = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    # Convert from BGR to RGB color space
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Validate frame data to ensure it's a proper image
                    if frame.size == 0 or len(frame.shape) != 3:
                        rich_console.print_warning(f"Invalid frame at timestamp {timestamp}")
                        continue
                        
                    frames.append((timestamp, frame))
                else:
                    rich_console.print_warning(f"Failed to read frame at timestamp {timestamp}")
                    
            cap.release()
            
        except Exception as e:
            rich_console.print_error(f"Error extracting frames: {e}")
            
        return frames
    
    def analyze_scene_content(self, frame: np.ndarray) -> List[str]:
        """Analyze frame content using CLIP for movie-relevant visual cues only."""
        if not CLIP_AVAILABLE or self.clip_model is None:
            return []
            
        try:
            # Focus on movie-relevant visual elements, not detailed scene descriptions
            # These are visual cues that would be important for accessibility
            movie_relevant_elements = [
                "explosion", "fire", "smoke", "car chase", "fight scene", 
                "crowd scene", "night scene", "rain", "storm", "water",
                "multiple people", "single person", "empty room"
            ]
            
            # Convert frame to PIL Image for CLIP preprocessing
            from PIL import Image
            if isinstance(frame, np.ndarray):
                pil_image = Image.fromarray(frame.astype('uint8'))
            else:
                rich_console.print_warning(f"Unexpected frame type: {type(frame)}")
                return []
                
            # Get the device of the CLIP model
            device = next(self.clip_model.parameters()).device
                
            # Preprocess frame and ensure it's on the same device as the model
            image = self.clip_preprocess(pil_image).unsqueeze(0).to(device)
            text_tokens = clip.tokenize(movie_relevant_elements).to(device)
            
            # Get similarities
            with torch.no_grad():
                logits_per_image, logits_per_text = self.clip_model(image, text_tokens)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            # Return only high-confidence predictions that are movie-relevant
            detected_elements = []
            for i, prob in enumerate(probs):
                if prob > 0.4:  # Higher threshold for more selective detection
                    detected_elements.append(movie_relevant_elements[i])
                    
            return detected_elements
            
        except Exception as e:
            rich_console.print_error(f"Error analyzing scene content: {e}")
            return []
    
    # BLIP image captioning method removed - not used in movie-style captions
    # Visual details should be obvious from watching the video
    
    def analyze_segment(self, video_path: str, start_time: float, end_time: float) -> List[VisualEvent]:
        """Analyze video segment for movie-relevant visual events only."""
        events = []
        
        # Extract frames at key points in the segment
        mid_time = (start_time + end_time) / 2
        timestamps = [mid_time]  # Only analyze middle frame to reduce noise
        frames = self.extract_key_frames(video_path, timestamps)
        
        for timestamp, frame in frames:
            # Analyze scene content for movie-relevant elements only
            scene_elements = self.analyze_scene_content(frame)
            for element in scene_elements:
                # Only add truly significant visual events
                if element in ["explosion", "fire", "smoke", "car chase", "fight scene", "crowd scene", "storm"]:
                    events.append(VisualEvent(
                        timestamp=timestamp,
                        event_type="visual_effect",
                        confidence=0.8,
                        description=f"[{element}]"
                    ))
        
        # Image captioning is not used for movie-style captions
        # Visual details should be obvious from watching the video
        
        return events

class MovieCaptionEnhancer:
    """Main class for enhancing captions with movie-style descriptive elements."""
    
    def __init__(self, enable_audio_analysis: bool = True, enable_visual_analysis: bool = True):
        self.enable_audio_analysis = enable_audio_analysis
        self.enable_visual_analysis = enable_visual_analysis
        
        self.audio_detector = AudioEventDetector() if enable_audio_analysis else None
        self.visual_detector = VisualEventDetector() if enable_visual_analysis else None
        
        rich_console.print_info(f"MovieCaptionEnhancer initialized - Audio: {enable_audio_analysis}, Visual: {enable_visual_analysis}")
    
    def enhance_segments(self, segments: List[Dict], video_path: str, audio_path: Optional[str] = None) -> List[EnhancedSegment]:
        """Enhance caption segments with descriptive elements."""
        enhanced_segments = []
        
        # Use video audio if no separate audio path provided
        if audio_path is None:
            audio_path = video_path
        
        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            
            # Analyze audio events
            audio_events = []
            if self.audio_detector:
                try:
                    audio_events = self.audio_detector.analyze_segment(audio_path, start_time, end_time)
                except Exception as e:
                    rich_console.print_error(f"Error analyzing audio for segment {start_time}-{end_time}: {e}")
            
            # Analyze visual events
            visual_events = []
            if self.visual_detector:
                try:
                    visual_events = self.visual_detector.analyze_segment(video_path, start_time, end_time)
                except Exception as e:
                    rich_console.print_error(f"Error analyzing video for segment {start_time}-{end_time}: {e}")
            
            # Generate enhanced text
            enhanced_text = self._generate_enhanced_text(text, audio_events, visual_events)
            
            enhanced_segments.append(EnhancedSegment(
                start=start_time,
                end=end_time,
                text=text,
                audio_events=audio_events,
                visual_events=visual_events,
                enhanced_text=enhanced_text
            ))
        
        return enhanced_segments
    
    def _generate_enhanced_text(self, original_text: str, audio_events: List[AudioEvent], visual_events: List[VisualEvent]) -> str:
        """Generate movie-style enhanced caption text focusing on audio cues and essential visual information."""
        # Movie-style captions prioritize audio cues and minimal visual descriptions
        audio_enhancements = []
        essential_visual_enhancements = []
        
        # Add audio descriptions - these are essential for movie captions
        for event in audio_events:
            if event.confidence > 0.5:
                audio_enhancements.append(event.description)
        
        # Add only essential visual information (not redundant scene descriptions)
        # Filter out overly descriptive or obvious visual elements
        essential_visual_types = {
            "explosion", "fire", "smoke", "car chase", "fight scene", 
            "crowd scene", "storm", "rain", "water"
        }
        
        seen_descriptions = set()
        for event in visual_events:
            if (event.confidence > 0.7 and 
                event.description not in seen_descriptions and
                any(essential_type in event.description.lower() for essential_type in essential_visual_types)):
                
                # Only add if it's truly essential visual information
                essential_visual_enhancements.append(event.description)
                seen_descriptions.add(event.description)
        
        # Combine enhancements with original text
        all_enhancements = audio_enhancements + essential_visual_enhancements
        
        if all_enhancements:
            enhancement_text = " ".join(all_enhancements)
            if original_text.strip():
                # For movie-style captions, place audio cues before dialogue
                combined_text = f"{enhancement_text} {original_text}"
            else:
                combined_text = enhancement_text
        else:
            combined_text = original_text
        
        # Apply movie-style formatting
        return self._format_movie_caption(combined_text)
    
    def process_caption_file(self, caption_segments: List[Dict], video_path: str, audio_path: Optional[str] = None) -> List[Dict]:
        """Process a list of caption segments and return movie-style enhanced versions."""
        # First detect potential speaker changes for movie-style dialogue
        segments_with_speakers = self._detect_speaker_changes(caption_segments)
        
        # Then enhance with audio/visual cues
        enhanced_segments = self.enhance_segments(segments_with_speakers, video_path, audio_path)
        
        # Convert back to standard format
        result = []
        for segment in enhanced_segments:
            result.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.enhanced_text
            })
        
        return result

    def _format_movie_caption(self, text: str) -> str:
        """Format caption text to follow movie caption style conventions."""
        if not text:
            return text
            
        # Remove redundant descriptors and consolidate similar ones
        text = re.sub(r'\[dialogue scene\]\s*', '', text)  # Remove redundant dialogue markers
        text = re.sub(r'\[indoor scene\]\s*\[outdoor scene\]\s*', '', text)  # Remove conflicting scene markers
        text = re.sub(r'\[outdoor scene\]\s*\[indoor scene\]\s*', '', text)  # Remove conflicting scene markers
        text = re.sub(r'\[close-up shot\]\s*\[wide shot\]\s*', '', text)  # Remove conflicting shot markers
        
        # Remove overly detailed visual descriptions that are obvious from video
        text = re.sub(r'\[a \w+.*?\]', '', text)  # Remove "a person/man/woman..." descriptions
        text = re.sub(r'\[the \w+.*?\]', '', text)  # Remove "the person/man/woman..." descriptions
        
        # Clean up multiple spaces and brackets
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'\[\s*\]', '', text)  # Remove empty brackets
        text = re.sub(r'\s+\[', ' [', text)  # Ensure space before brackets
        text = re.sub(r'\]\s+', '] ', text)  # Ensure single space after brackets
        
        return text.strip()

    def _detect_speaker_changes(self, segments: List[Dict]) -> List[Dict]:
        """Detect potential speaker changes in dialogue for movie-style speaker identification."""
        if len(segments) < 2:
            return segments
            
        enhanced_segments = []
        current_speaker = 1
        
        for i, segment in enumerate(segments):
            text = segment.get('text', '').strip()
            
            # Skip if no actual speech content
            if not text or len(text) < 3:
                enhanced_segments.append(segment)
                continue
                
            # Detect potential speaker change based on pause duration
            if i > 0:
                prev_segment = segments[i-1]
                pause_duration = segment['start'] - prev_segment['end']
                
                # If there's a significant pause (>2 seconds), it might be a speaker change
                if pause_duration > 2.0:
                    current_speaker = 2 if current_speaker == 1 else 1
            
            # Add speaker identification for dialogue-heavy content
            # Only add if the text seems to be direct speech
            if any(marker in text.lower() for marker in ['؟', '!', '،', 'how', 'what', 'who', 'كيف', 'ماذا', 'من']):
                # This appears to be dialogue
                if current_speaker == 2 and i > 0:
                    segment['text'] = f"(Speaker 2) {text}"
                # Don't label Speaker 1 to keep it clean
            
            enhanced_segments.append(segment)
            
        return enhanced_segments
    
    def _handle_non_speech_segments(self, audio_events: List[AudioEvent]) -> List[str]:
        """Handle segments with no speech but important audio cues (movie-style)."""
        descriptions = []
        
        # For non-speech segments, focus entirely on audio description
        for event in audio_events:
            if event.confidence > 0.6:  # Higher threshold for non-speech
                # Clean up the description for movie-style format
                desc = event.description.strip('[]')
                if desc not in descriptions:  # Avoid duplicates
                    descriptions.append(f"[{desc}]")
        
        return descriptions
