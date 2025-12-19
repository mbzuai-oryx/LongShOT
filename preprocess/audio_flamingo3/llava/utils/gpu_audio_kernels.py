# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

"""
Custom GPU kernels for ultra-fast audio processing
These kernels provide significant speedup over standard PyTorch operations
"""

import torch
import torch.nn.functional as F
import torchaudio
import math
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
    logger.info("🚀 Triton GPU kernels available")
except ImportError:
    TRITON_AVAILABLE = False
    logger.warning("❌ Triton not available, using PyTorch fallbacks")

if TRITON_AVAILABLE:
    @triton.jit
    def audio_normalize_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for audio normalization"""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load audio data
        audio = tl.load(input_ptr + offsets, mask=mask)
        
        # Find absolute maximum
        abs_audio = tl.abs(audio)
        max_val = tl.max(abs_audio)
        
        # Normalize
        normalized = tl.where(max_val > 1e-8, audio / max_val, audio)
        
        # Store result
        tl.store(output_ptr + offsets, normalized, mask=mask)

    @triton.jit
    def audio_resample_kernel(
        input_ptr,
        output_ptr,
        input_length,
        output_length,
        scale_factor: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for audio resampling"""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < output_length
        
        # Calculate source indices with linear interpolation
        src_indices = offsets * scale_factor
        src_indices_int = src_indices.to(tl.int32)
        src_indices_frac = src_indices - src_indices_int
        
        # Bounds checking
        valid_mask = mask & (src_indices_int < input_length - 1)
        
        # Linear interpolation
        sample1 = tl.load(input_ptr + src_indices_int, mask=valid_mask)
        sample2 = tl.load(input_ptr + src_indices_int + 1, mask=valid_mask)
        
        interpolated = sample1 + src_indices_frac * (sample2 - sample1)
        
        # Store result
        tl.store(output_ptr + offsets, interpolated, mask=mask)

    @triton.jit
    def audio_windowing_kernel(
        input_ptr,
        window_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for applying window function"""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load audio and window
        audio = tl.load(input_ptr + offsets, mask=mask)
        window = tl.load(window_ptr + offsets, mask=mask)
        
        # Apply window
        windowed = audio * window
        
        # Store result
        tl.store(output_ptr + offsets, windowed, mask=mask)

class GPUAudioKernels:
    """High-performance GPU kernels for audio processing"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.use_triton = TRITON_AVAILABLE and device.type == 'cuda'
        
        if self.use_triton:
            logger.info("Using Triton GPU kernels for maximum performance")
        else:
            logger.info("Using PyTorch fallback implementations")
    
    @torch.compile
    def normalize_audio_gpu(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """GPU-optimized audio normalization"""
        if self.use_triton and audio_tensor.is_cuda:
            return self._triton_normalize(audio_tensor)
        else:
            return self._pytorch_normalize(audio_tensor)
    
    def _triton_normalize(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Triton-based normalization kernel"""
        output = torch.empty_like(audio_tensor)
        n_elements = audio_tensor.numel()
        
        # Calculate grid size
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        # Launch kernel
        audio_normalize_kernel[grid](
            audio_tensor, output, n_elements, BLOCK_SIZE
        )
        
        return output
    
    @torch.compile
    def _pytorch_normalize(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """PyTorch fallback normalization"""
        abs_max = torch.max(torch.abs(audio_tensor))
        return torch.where(abs_max > 1e-8, audio_tensor / abs_max, audio_tensor)
    
    @torch.compile
    def resample_audio_gpu(self, audio_tensor: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """GPU-optimized audio resampling"""
        if orig_sr == target_sr:
            return audio_tensor
        
        if self.use_triton and audio_tensor.is_cuda:
            return self._triton_resample(audio_tensor, orig_sr, target_sr)
        else:
            return self._pytorch_resample(audio_tensor, orig_sr, target_sr)
    
    def _triton_resample(self, audio_tensor: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """Triton-based resampling kernel"""
        scale_factor = orig_sr / target_sr
        output_length = int(audio_tensor.size(-1) / scale_factor)
        output = torch.empty(
            (*audio_tensor.shape[:-1], output_length),
            dtype=audio_tensor.dtype,
            device=audio_tensor.device
        )
        
        # Process each channel separately for simplicity
        for batch_idx in range(audio_tensor.size(0) if audio_tensor.dim() > 1 else 1):
            if audio_tensor.dim() > 1:
                input_data = audio_tensor[batch_idx]
                output_data = output[batch_idx]
            else:
                input_data = audio_tensor
                output_data = output
            
            # Calculate grid size
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(output_length, BLOCK_SIZE),)
            
            # Launch kernel
            audio_resample_kernel[grid](
                input_data, output_data, input_data.size(-1), 
                output_length, scale_factor, BLOCK_SIZE
            )
        
        return output
    
    @torch.compile  
    def _pytorch_resample(self, audio_tensor: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """PyTorch fallback resampling"""
        return torchaudio.functional.resample(audio_tensor, orig_sr, target_sr)
    
    @torch.compile
    def apply_window_gpu(self, audio_tensor: torch.Tensor, window_type: str = 'hann') -> torch.Tensor:
        """GPU-optimized windowing"""
        window_length = audio_tensor.size(-1)
        
        # Create window on GPU
        if window_type == 'hann':
            window = torch.hann_window(window_length, device=self.device, dtype=audio_tensor.dtype)
        elif window_type == 'hamming':
            window = torch.hamming_window(window_length, device=self.device, dtype=audio_tensor.dtype)
        else:
            window = torch.ones(window_length, device=self.device, dtype=audio_tensor.dtype)
        
        if self.use_triton and audio_tensor.is_cuda:
            return self._triton_windowing(audio_tensor, window)
        else:
            return audio_tensor * window
    
    def _triton_windowing(self, audio_tensor: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        """Triton-based windowing kernel"""
        output = torch.empty_like(audio_tensor)
        n_elements = audio_tensor.numel()
        
        # Calculate grid size
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        # Launch kernel
        audio_windowing_kernel[grid](
            audio_tensor, window, output, n_elements, BLOCK_SIZE
        )
        
        return output
    
    @torch.compile
    def chunked_stft_gpu(self, audio_tensor: torch.Tensor, 
                        n_fft: int = 1024, hop_length: int = 512,
                        window: str = 'hann') -> torch.Tensor:
        """GPU-optimized chunked STFT for large audio files"""
        # Use torch.stft with GPU optimization
        return torch.stft(
            audio_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft, device=self.device),
            return_complex=True
        )
    
    @torch.compile
    def batch_mel_spectrogram_gpu(self, audio_batch: torch.Tensor,
                                 sample_rate: int = 16000,
                                 n_mels: int = 80,
                                 n_fft: int = 1024) -> torch.Tensor:
        """GPU-optimized batch mel spectrogram computation"""
        # Create mel filter bank on GPU
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            power=2.0
        ).to(self.device)
        
        # Batch process
        return mel_transform(audio_batch)
    
    def benchmark_kernels(self, audio_tensor: torch.Tensor, iterations: int = 100):
        """Benchmark kernel performance"""
        logger.info("🏁 Benchmarking GPU audio kernels...")
        
        # Warm up
        for _ in range(10):
            _ = self.normalize_audio_gpu(audio_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Benchmark normalization
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for _ in range(iterations):
            _ = self.normalize_audio_gpu(audio_tensor)
        end_time.record()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            normalize_time = start_time.elapsed_time(end_time) / iterations
            logger.info(f"Normalization: {normalize_time:.3f}ms per call")
        
        # Benchmark resampling
        start_time.record()
        for _ in range(iterations):
            _ = self.resample_audio_gpu(audio_tensor, 44100, 16000)
        end_time.record()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            resample_time = start_time.elapsed_time(end_time) / iterations
            logger.info(f"Resampling: {resample_time:.3f}ms per call")
        
        # Benchmark windowing
        start_time.record()
        for _ in range(iterations):
            _ = self.apply_window_gpu(audio_tensor)
        end_time.record()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            window_time = start_time.elapsed_time(end_time) / iterations
            logger.info(f"Windowing: {window_time:.3f}ms per call")
        
        logger.info("✅ Kernel benchmarking completed")

# Factory function for easy import
def create_gpu_audio_kernels(device: Optional[torch.device] = None) -> GPUAudioKernels:
    """Create GPU audio kernels instance"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    return GPUAudioKernels(device)

# Performance testing function
def test_kernel_performance():
    """Test kernel performance with dummy data"""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping kernel tests")
        return
    
    device = torch.device("cuda:0")
    kernels = create_gpu_audio_kernels(device)
    
    # Create test audio (10 seconds at 16kHz)
    test_audio = torch.randn(1, 160000, device=device, dtype=torch.float32)
    
    logger.info("Testing GPU audio kernels...")
    kernels.benchmark_kernels(test_audio)

if __name__ == "__main__":
    test_kernel_performance()