"""
Audio processing module for extracting transcriptions from video files
using faster-whisper with timestamp segmentation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import os
import ffmpeg
from faster_whisper import WhisperModel, BatchedInferencePipeline
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """Represents a transcribed audio segment with timing information."""
    text: str
    start_time: float
    end_time: float
    language: str
    language_probability: float


class AudioProcessor:
    """
    Processes video files to extract audio transcriptions with timestamps
    using faster-whisper.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        cpu_threads: int = 4
    ):
        """
        Initialize the AudioProcessor.

        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large-v3")
            device: Device to run on ("cpu", "cuda", "auto")
            compute_type: Computation type ("int8", "float16", "float32", "auto")
            cpu_threads: Number of threads for CPU processing (OMP_NUM_THREADS)
        """
        self.model_size = model_size
        self.requested_device = device
        self.cpu_threads = cpu_threads
        self.device = self._determine_and_validate_device(device)
        self.compute_type = self._determine_compute_type(compute_type)
        self.batched_model = None

        # Set CPU threads for optimal performance
        if self.device == "cpu":
            os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
            logger.info(f"Set OMP_NUM_THREADS to {cpu_threads} for CPU processing")

    def _determine_and_validate_device(self, device: str) -> str:
        """Determine and validate the best device to use with proper CUDA testing."""
        if device == "auto":
            return self._auto_detect_device()
        elif device == "cuda":
            if self._validate_cuda_device():
                return "cuda"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        else:
            return device

    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                if self._validate_cuda_device():
                    logger.info("CUDA detected and validated, using GPU acceleration")
                    return "cuda"
                else:
                    logger.warning("CUDA available but validation failed, using CPU")
                    return "cpu"
            else:
                logger.info("CUDA not available, using CPU")
                return "cpu"
        except ImportError:
            logger.info("PyTorch not available, using CPU")
            return "cpu"

    def _validate_cuda_device(self) -> bool:
        """Validate that CUDA is actually functional for faster-whisper."""
        try:
            import torch
            if not torch.cuda.is_available():
                return False

            # Test basic CUDA operations
            device_count = torch.cuda.device_count()
            if device_count == 0:
                return False

            # Test memory allocation on first visible device
            test_tensor = torch.randn(10, 10, device='cuda')
            del test_tensor
            torch.cuda.empty_cache()

            logger.info(f"CUDA validation successful - {device_count} GPU(s) detected")
            return True

        except Exception as e:
            logger.warning(f"CUDA validation failed: {e}")
            return False

    def _determine_compute_type(self, compute_type: str) -> str:
        """Determine the best compute type based on device."""
        if compute_type == "auto":
            if self.device == "cuda":
                return "float16"
            else:
                return "int8"
        return compute_type

    def _get_batched_model(self) -> BatchedInferencePipeline:
        """Get or create the batched inference pipeline with CUDA fallback for faster processing."""
        if self.batched_model is None:
            logger.info(f"Loading Whisper batched model: {self.model_size} on {self.device}")

            # Try to load the model with current device settings
            try:
                whisper_model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type
                )
                self.batched_model = BatchedInferencePipeline(model=whisper_model)
                logger.info(f"Whisper batched model loaded successfully on {self.device}")

            except Exception as e:
                # If CUDA was requested but failed, try fallback to CPU
                if self.device == "cuda":
                    logger.warning(f"Failed to load faster-whisper on CUDA: {e}")
                    logger.info("Attempting fallback to CPU...")

                    try:
                        # Update device settings for CPU fallback
                        self.device = "cpu"
                        self.compute_type = "int8"  # Optimal for CPU

                        # Set CPU threads if not already set
                        if "OMP_NUM_THREADS" not in os.environ:
                            os.environ["OMP_NUM_THREADS"] = str(self.cpu_threads)

                        # Try loading on CPU
                        whisper_model = WhisperModel(
                            self.model_size,
                            device=self.device,
                            compute_type=self.compute_type
                        )
                        self.batched_model = BatchedInferencePipeline(model=whisper_model)
                        logger.info(f"Whisper batched model successfully loaded on CPU fallback")

                    except Exception as cpu_error:
                        logger.error(f"Failed to load faster-whisper on CPU fallback: {cpu_error}")
                        raise RuntimeError(f"Unable to load faster-whisper on any device. CUDA error: {e}, CPU error: {cpu_error}")
                else:
                    # Re-raise if not a CUDA fallback scenario
                    logger.error(f"Failed to load faster-whisper on {self.device}: {e}")
                    raise

        return self.batched_model

    def extract_audio_from_video(self, video_path: str) -> str:
        """
        Extract audio from video file to a temporary WAV file.

        Args:
            video_path: Path to the video file

        Returns:
            Path to the extracted audio file
        """
        # Create temporary file for audio
        temp_fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)

        # Extract audio using ffmpeg
        (
            ffmpeg
            .input(video_path)
            .output(temp_audio_path, acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        logger.info(f"Audio extracted to: {temp_audio_path}")
        return temp_audio_path

    def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        word_timestamps: bool = False
    ) -> Tuple[List[AudioSegment], Dict[str, Any]]:
        """
        Transcribe audio file to text with segment timestamps using optimized batch processing.

        Args:
            audio_path: Path to the audio file
            language: Language code (None for auto-detection)
            word_timestamps: Whether to include word-level timestamps

        Returns:
            Tuple of (segments, transcription_info)
        """
        # Use batch processing for consistency and optimization
        results = self.transcribe_audio_batch(
            [audio_path],
            language=language,
            word_timestamps=word_timestamps,
            batch_size=1
        )
        return results[0]

    @staticmethod
    def _has_audio_stream(file_path: str) -> bool:
        """Check if a media file contains an audio stream."""
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-select_streams", "a",
                 "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                 file_path],
                capture_output=True, text=True, timeout=10,
            )
            return "audio" in result.stdout
        except Exception:
            return True  # assume audio exists if probe fails

    def _collect_segments(
        self, segments_gen, info
    ) -> Tuple[List[AudioSegment], Dict[str, Any]]:
        audio_segments = []
        for segment in segments_gen:
            audio_segments.append(AudioSegment(
                text=segment.text.strip(),
                start_time=segment.start,
                end_time=segment.end,
                language=info.language,
                language_probability=info.language_probability,
            ))
        return audio_segments, {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "num_segments": len(audio_segments),
        }

    def _transcribe_single(
        self,
        batched_model,
        audio_path: str,
        language: Optional[str],
        word_timestamps: bool,
        batch_size: int,
    ) -> Tuple[List[AudioSegment], Dict[str, Any]]:
        """Transcribe one file with automatic fallback.

        Skips files without an audio stream. For files with audio,
        faster-whisper's BatchedInferencePipeline can raise IndexError
        when CTranslate2 returns empty decode results for certain audio
        chunks. On failure, falls back to the non-batched WhisperModel.
        """
        if not self._has_audio_stream(audio_path):
            logger.info(f"No audio stream in {audio_path} — skipping transcription.")
            return [], {
                "language": language or "unknown",
                "language_probability": 0.0,
                "duration": 0,
                "num_segments": 0,
            }

        try:
            segments_gen, info = batched_model.transcribe(
                audio_path,
                language=language,
                word_timestamps=word_timestamps,
                vad_filter=True,
                beam_size=1,
                batch_size=batch_size,
            )
            return self._collect_segments(segments_gen, info)
        except (IndexError, RuntimeError) as e:
            logger.warning(
                f"Batched transcription failed for {audio_path}: {e}. "
                "Falling back to sequential transcription..."
            )

        try:
            segments_gen, info = batched_model.model.transcribe(
                audio_path,
                language=language or "en",
                word_timestamps=word_timestamps,
                vad_filter=True,
                beam_size=1,
            )
            return self._collect_segments(segments_gen, info)
        except (IndexError, RuntimeError) as e:
            logger.warning(
                f"Sequential transcription also failed for {audio_path}: {e}. "
                "Using empty segments."
            )

        return [], {
            "language": language or "unknown",
            "language_probability": 0.0,
            "duration": 0,
            "num_segments": 0,
        }

    def transcribe_audio_batch(
        self,
        audio_paths: List[str],
        language: Optional[str] = None,
        word_timestamps: bool = False,
        batch_size: int = 16,
        cpu_batch_size: Optional[int] = None
    ) -> List[Tuple[List[AudioSegment], Dict[str, Any]]]:
        """
        Transcribe multiple audio files using BatchedInferencePipeline for improved performance.

        Args:
            audio_paths: List of paths to audio files
            language: Language code (None for auto-detection)
            word_timestamps: Whether to include word-level timestamps
            batch_size: Batch size for BatchedInferencePipeline processing (CUDA)
            cpu_batch_size: Batch size for CPU processing (uses batch_size if None)

        Returns:
            List of tuples (segments, transcription_info) for each audio file
        """
        batched_model = self._get_batched_model()

        # Determine appropriate batch size based on device
        if self.device == "cpu" and cpu_batch_size is not None:
            effective_batch_size = cpu_batch_size
            logger.info(f"Using CPU batch size: {effective_batch_size}")
        else:
            effective_batch_size = batch_size
            logger.info(f"Using {self.device.upper()} batch size: {effective_batch_size}")

        logger.info(f"Batch transcribing {len(audio_paths)} audio files on {self.device}")

        results = []

        for audio_path in audio_paths:
            logger.info(f"Transcribing: {audio_path}")

            audio_segments, transcription_info = self._transcribe_single(
                batched_model, audio_path, language, word_timestamps,
                effective_batch_size,
            )

            results.append((audio_segments, transcription_info))

        logger.info(f"Batch transcription completed: {len(results)} files processed")
        return results

    def process_video_audio(
        self,
        video_path: str,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        cleanup_temp: bool = True
    ) -> Tuple[List[AudioSegment], Dict[str, Any]]:
        """
        Extract audio from video and transcribe it using optimized batch processing.

        Args:
            video_path: Path to the video file
            language: Language code (None for auto-detection)
            word_timestamps: Whether to include word-level timestamps
            cleanup_temp: Whether to clean up temporary audio file

        Returns:
            Tuple of (segments, transcription_info)
        """
        # Use batch processing for consistency and optimization
        results = self.process_video_audio_batch(
            [video_path],
            language=language,
            word_timestamps=word_timestamps,
            cleanup_temp=cleanup_temp,
            batch_size=1
        )
        return results[0]

    def process_video_audio_batch(
        self,
        video_paths: List[str],
        language: Optional[str] = None,
        word_timestamps: bool = False,
        cleanup_temp: bool = True,
        batch_size: int = 8,
        cpu_batch_size: Optional[int] = None
    ) -> List[Tuple[List[AudioSegment], Dict[str, Any]]]:
        """
        Transcribe audio from video files directly (faster-whisper reads
        video files natively via its internal ffmpeg, no temp WAV needed).
        """
        # Pass video paths directly to transcribe — faster-whisper handles
        # audio extraction internally, avoiding redundant disk I/O.
        return self.transcribe_audio_batch(
            video_paths,
            language=language,
            word_timestamps=word_timestamps,
            batch_size=batch_size,
            cpu_batch_size=cpu_batch_size
        )

