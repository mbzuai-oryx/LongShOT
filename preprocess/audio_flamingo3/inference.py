#!/usr/bin/env python3
"""
Audio Flamingo 3 - Simplified Inference System

Streamlined version removing over-engineering while maintaining core functionality:
- Simple audio preprocessing with intelligent defaults
- Multi-GPU processing with round-robin distribution  
- Essential optimizations without complexity
- Reliable batch processing
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import tempfile
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from io import BytesIO
import contextlib
import warnings

import torch
import numpy as np
import soundfile as sf
import librosa

from config import AUDIO_FLAMINGO_MODEL_PATH

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logging from third-party libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("deepspeed").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("xgrammar").setLevel(logging.ERROR)
logging.getLogger("tokenizers").setLevel(logging.ERROR)
logging.getLogger("safetensors").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch.fx").setLevel(logging.ERROR)
logging.getLogger("deepspeed.runtime.zero.stage_1_and_2").setLevel(logging.ERROR)
logging.getLogger("deepspeed.runtime.zero.partition_parameters").setLevel(logging.ERROR)
logging.getLogger("deepspeed.runtime.activation_checkpointing.checkpointing").setLevel(logging.ERROR)
logging.getLogger("deepspeed.runtime.engine").setLevel(logging.ERROR)
logging.getLogger("deepspeed.runtime.zero.stage3").setLevel(logging.ERROR)

# Set environment variables to suppress progress bars and debug output
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# Add environment variable to suppress debug prints
os.environ["LLAVA_DEBUG"] = "0"

# Suppress progress bars and model loading messages
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*model_max_length.*")
warnings.filterwarnings("ignore", message=".*Downloading.*")
warnings.filterwarnings("ignore", message=".*loading.*")

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr during model inference."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

@dataclass
class AudioTask:
    """Audio processing task"""
    audio_path: str
    text_prompt: str
    preprocessed_data: Optional[bytes] = None
    sample_rate: Optional[int] = None

class SimpleAudioPreprocessor:
    """Simple audio preprocessing with intelligent defaults"""
    
    def __init__(self):
        self.memory_cache = {}
        
    def should_optimize(self, audio_path: str) -> Tuple[bool, int]:
        """Simple heuristic to determine if optimization is beneficial"""
        try:
            info = sf.info(audio_path)
            duration = info.duration
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            
            # Use 12kHz for large files or long duration, 16kHz otherwise
            if file_size_mb > 10 or duration > 30 or info.samplerate > 24000:
                return True, 12000
            else:
                return False, 16000
                
        except Exception:
            return False, 16000
    
    def preprocess_audio(self, audio_path: str) -> Tuple[Optional[bytes], int]:
        """Process audio with simple optimization"""
        try:
            # Check cache
            cache_key = f"{audio_path}_{os.path.getmtime(audio_path)}"
            if cache_key in self.memory_cache:
                return self.memory_cache[cache_key]
            
            should_opt, target_sr = self.should_optimize(audio_path)
            
            if not should_opt:
                return None, 16000
            
            # Load and process audio
            data, orig_sr = sf.read(audio_path)
            
            # Convert to mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Resample if needed
            if orig_sr != target_sr:
                data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
            
            # Convert to bytes
            buffer = BytesIO()
            sf.write(buffer, data, target_sr, format='WAV')
            audio_bytes = buffer.getvalue()
            buffer.close()
            
            # Cache result
            self.memory_cache[cache_key] = (audio_bytes, target_sr)
            
            return audio_bytes, target_sr
            
        except Exception as e:
            logger.warning(f"Preprocessing failed for {audio_path}: {e}")
            return None, 16000

class GPUWorker(mp.Process):
    """Simple GPU worker process"""
    
    def __init__(self, worker_id: int, gpu_id: int, model_path: str, 
                 task_queue: mp.Queue, result_queue: mp.Queue, 
                 shutdown_event: mp.Event):
        super().__init__()
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.shutdown_event = shutdown_event
        self.model = None
        self.preprocessor = None
        self.daemon = True
    
    def run(self):
        """Main worker loop"""
        try:
            # Setup environment for this worker
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # Handle GPU device mapping properly
            if torch.cuda.is_available():
                # Get the number of available GPUs
                num_gpus = torch.cuda.device_count()
                
                # Map logical GPU ID to physical device
                if self.gpu_id < num_gpus:
                    device_id = self.gpu_id
                else:
                    # Fallback to GPU 0 if the requested GPU ID is out of range
                    device_id = 0
                    logger.warning(f"GPU {self.gpu_id} not available, using GPU {device_id}")
                
                # Set the device for this worker
                torch.cuda.set_device(device_id)
                actual_device = torch.cuda.current_device()
                
                # Verify the device is correctly set
                if actual_device != device_id:
                    raise RuntimeError(f"Failed to set GPU device {device_id}, current device is {actual_device}")
            else:
                raise RuntimeError("CUDA is not available")
            
            # Setup logging with reduced verbosity
            worker_logger = logging.getLogger(f"gpu_worker_{self.gpu_id}")
            worker_logger.setLevel(logging.WARNING)  # Reduce worker log verbosity
            
            # Load model
            self._load_model(worker_logger)
            
            # Initialize preprocessor
            self.preprocessor = SimpleAudioPreprocessor()
            
            # Worker is ready
            
            # Process tasks
            while True:
                # Check for shutdown signal
                if self.shutdown_event.is_set():
                    break
                try:
                    task = self.task_queue.get(timeout=2.0)
                    
                    if task is None:  # Shutdown signal
                        break
                    
                    result = self._process_task(task, worker_logger)
                    self.result_queue.put(result)
                    
                except Exception as e:
                    if "Empty" not in str(e) and not self.task_queue.empty():
                        worker_logger.error(f"Task processing error: {e}")
                        
        except Exception as e:
            logger.error(f"GPU {self.gpu_id} worker error: {e}")
        finally:
            # Cleanup
            if self.model is not None:
                del self.model
            torch.cuda.empty_cache()
    
    def _load_model(self, worker_logger):
        """Load model with optimal settings"""
        try:
            # Get the current device that was set in run()
            current_device = torch.cuda.current_device()
            
            # Optimized memory and performance settings
            torch.cuda.set_per_process_memory_fraction(0.75, current_device)  # Reduced from 0.85 to leave more room
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Better performance, less determinism
            torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable memory efficient optimizations
            torch.cuda.empty_cache()  # Start clean
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)  # Flash attention if available
            
            # Suppress individual worker loading messages for cleaner output
            pass
            
            # Load model
            # sys.path.insert(0, str(Path(__file__).parent.parent))
            import llava
            
            # Use current device instead of self.gpu_id
            self.model = llava.load(self.model_path, devices=[current_device])
            
            self.model.eval()
            self.model = self.model.to(f'cuda:{current_device}', non_blocking=True)
            
            # Disable gradients for inference
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Compile for optimization
            try:
                compile_start = time.perf_counter()
                self.model = torch.compile(self.model, mode="reduce-overhead")
                compile_time = time.perf_counter() - compile_start
                worker_logger.info(f"GPU {self.gpu_id}: Model compiled in {compile_time:.2f}s")
            except Exception as e:
                worker_logger.warning(f"GPU {self.gpu_id}: Compilation failed: {e}")
            
            # Model loaded - logging handled at system level
            pass
            
            # Warmup
            self._warmup_model(worker_logger)
            
            # Memory info logged at system level to reduce redundancy
            pass
            
        except Exception as e:
            worker_logger.error(f"GPU {self.gpu_id}: Model loading failed: {e}")
            raise
    
    def _warmup_model(self, worker_logger):
        """Aggressive warmup with batch inference to trigger compilation"""
        try:
            # Create multiple dummy audio files for batch warmup
            dummy_files = []
            for i in range(3):  # Warmup with small batch
                dummy_audio = np.random.normal(0, 0.001, 1600 + i*100).astype(np.float32)  # Slightly different lengths
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    sf.write(tmp.name, dummy_audio, 16000)
                    dummy_files.append(tmp.name)
            
            try:
                from llava.media import Sound
                
                # Single inference warmup (basic)
                sound = Sound(dummy_files[0])
                prompt = [sound, "warmup"]
                
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                    with torch.inference_mode():
                        with suppress_stdout_stderr():
                            _ = self.model.generate_content(prompt)
                    torch.cuda.synchronize()
                
                # Batch inference warmup (triggers compilation optimization)
                for dummy_path in dummy_files:
                    sound = Sound(dummy_path)
                    prompt = [sound, "batch warmup"]
                    
                    with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                        with torch.inference_mode():
                            with suppress_stdout_stderr():
                                _ = self.model.generate_content(prompt)
                        torch.cuda.synchronize()
                
                # Force CUDA cache optimization and memory preallocation
                torch.cuda.empty_cache()
                
                # Preallocate some GPU memory to avoid fragmentation during actual inference
                try:
                    dummy_tensor = torch.zeros(1024, 1024, device=f'cuda:{torch.cuda.current_device()}', dtype=torch.float16)
                    del dummy_tensor
                    torch.cuda.synchronize()
                except Exception:
                    pass  # Memory preallocation failed, continue anyway
                
            finally:
                # Cleanup all dummy files
                for dummy_path in dummy_files:
                    try:
                        os.unlink(dummy_path)
                    except Exception:
                        pass
                    
        except Exception as e:
            worker_logger.warning(f"GPU {self.gpu_id}: Warmup failed: {e}")
    
    def _process_task(self, task: dict, worker_logger) -> dict:
        """Process inference task"""
        audio_path = task['audio_path']
        text_prompt = task['text_prompt']
        
        start_time = time.perf_counter()
        
        try:
            # Preprocess audio if needed
            preprocessed_data, sample_rate = self.preprocessor.preprocess_audio(audio_path)
            
            if preprocessed_data is not None:
                # Use preprocessed data
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp.write(preprocessed_data)
                    optimized_path = tmp.name
                should_cleanup = True
            else:
                # Use original file
                optimized_path = audio_path
                should_cleanup = False
            
            # Create prompt
            from llava.media import Sound
            sound = Sound(optimized_path)
            prompt = [sound, text_prompt]
            
            # Run inference with stdout/stderr suppression
            inference_start = time.perf_counter()
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                with torch.inference_mode():
                    with suppress_stdout_stderr():  # Suppress debug prints during inference
                        response = self.model.generate_content(prompt)
                torch.cuda.synchronize()
            
            inference_time = time.perf_counter() - inference_start
            total_time = time.perf_counter() - start_time
            
            result = {
                'audio_path': audio_path,
                'response': response,
                'success': True,
                'inference_time': inference_time,
                'total_time': total_time,
                'gpu_id': torch.cuda.current_device(),
                'sample_rate': sample_rate
            }
            
            # Cleanup
            if should_cleanup:
                try:
                    os.unlink(optimized_path)
                except Exception:
                    pass
            
            return result
            
        except Exception as e:
            total_time = time.perf_counter() - start_time
            worker_logger.error(f"GPU {self.gpu_id} inference failed for {audio_path}: {e}")
            return {
                'audio_path': audio_path,
                'success': False,
                'error': str(e),
                'gpu_id': torch.cuda.current_device(),
                'total_time': total_time
            }

class SimpleInferenceSystem:
    """Simplified inference system with persistent model loading"""
    
    def __init__(self, model_path: str, num_gpus: int = None, persistent_model: bool = True):
        self.model_path = model_path
        self.persistent_model = persistent_model
        
        # Respect CUDA_VISIBLE_DEVICES if set
        available_gpus = torch.cuda.device_count()
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            if visible_devices:
                available_gpus = len([d.strip() for d in visible_devices.split(',') if d.strip()])
        
        self.num_gpus = min(num_gpus or available_gpus, available_gpus)
        self.workers = []
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self._workers_started = False  # Track worker state
        self._shutdown_requested = mp.Event()  # Signal for early shutdown
        
        # Model persistence for single inference mode
        self._persistent_model = None
        self._model_loaded = False
        self._model_lock = threading.RLock()  # Add thread safety
        
        logger.info(f"Initializing simplified system with {self.num_gpus} GPUs (persistent_model={persistent_model})")
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            logger.info(f"Using CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            logger.info(f"PyTorch sees {available_gpus} GPUs")
    
    def start_workers(self):
        """Start GPU worker processes if not already started"""
        if self._workers_started and len(self.workers) > 0:
            # Check if existing workers are still alive
            alive_workers = [w for w in self.workers if w.is_alive()]
            if len(alive_workers) == self.num_gpus:
                logger.debug(f"Workers already running ({len(alive_workers)} active)")
                return
            else:
                logger.warning(f"Some workers died ({len(alive_workers)}/{self.num_gpus} alive), restarting all")
                self.shutdown_workers()
        
        logger.info(f"🚀 Starting {self.num_gpus} GPU workers...")
        
        # Clear any existing workers
        self.workers.clear()
        
        # Start all workers in parallel
        worker_pids = []
        startup_start = time.time()
        
        for gpu_id in range(self.num_gpus):
            worker = GPUWorker(gpu_id, gpu_id, self.model_path, 
                             self.task_queue, self.result_queue,
                             self._shutdown_requested)
            worker.start()
            self.workers.append(worker)
            worker_pids.append(str(worker.pid))
            logger.info(f"✓ Started GPU {gpu_id} worker (PID: {worker.pid})")
        
        total_startup_time = time.time() - startup_start
        self._workers_started = True
        logger.info(f"✅ All {self.num_gpus} GPU workers started in {total_startup_time:.1f}s (PIDs: {', '.join(worker_pids)})")
    
    def batch_inference(self, audio_paths: List[str], text_prompt: str, auto_shutdown: bool = True) -> List[Dict[str, Any]]:
        """Process batch of audio files"""
        
        # Always use workers for batch processing, even for single files
        # This prevents OOM issues when single_inference tries to load models
        # while workers already have models loaded
        
        # Start workers
        self.start_workers()
        
        try:
            
            # Add tasks to queue
            for audio_path in audio_paths:
                task = {
                    'audio_path': audio_path,
                    'text_prompt': text_prompt
                }
                self.task_queue.put(task)
            
            # Collect results
            results = []
            completed_tasks = 0
            
            for _ in range(len(audio_paths)):
                try:
                    result = self.result_queue.get(timeout=300)
                    results.append(result)
                    completed_tasks += 1
                        
                except Exception as e:
                    logger.error(f"Failed to get result: {e}")
                    results.append({
                        'success': False,
                        'error': 'Timeout or processing error',
                        'audio_path': 'unknown'
                    })
            
            return results
            
        finally:
            # Only shutdown if auto_shutdown is True (default behavior for backward compatibility)
            if auto_shutdown:
                self._shutdown_requested.set()
                self.shutdown_workers()
    
    def single_inference(self, audio_path: str, text_prompt: str) -> Dict[str, Any]:
        """Single file inference with persistent model support"""
        start_time = time.perf_counter()
        should_cleanup = False
        optimized_path = None
        
        try:
            # If workers are already running, use them instead of loading a new model
            # This prevents OOM errors when single_inference is called while workers exist
            if hasattr(self, '_workers_started') and self._workers_started and self.workers:
                logger.debug(f"Using existing workers for single inference instead of loading new model")
                return self.batch_inference([audio_path], text_prompt, auto_shutdown=False)[0]
            
            # Load model (will use persistent model if enabled)
            model = self._load_single_model()
            
            # Preprocess audio
            preprocessor = SimpleAudioPreprocessor()
            preprocessed_data, sample_rate = preprocessor.preprocess_audio(audio_path)
            
            if preprocessed_data is not None:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp.write(preprocessed_data)
                    optimized_path = tmp.name
                should_cleanup = True
            else:
                optimized_path = audio_path
                should_cleanup = False
            
            # Create prompt and run inference
            from llava.media import Sound
            sound = Sound(optimized_path)
            prompt = [sound, text_prompt]
            
            inference_start = time.perf_counter()
            
            # Handle CUDA OOM specifically
            try:
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                    with torch.inference_mode():
                        with suppress_stdout_stderr():  # Suppress debug prints during inference
                            response = model.generate_content(prompt)
                    torch.cuda.synchronize()
                
                inference_time = time.perf_counter() - inference_start
                total_time = time.perf_counter() - start_time
                
                result = {
                    'audio_path': audio_path,
                    'response': response,
                    'success': True,
                    'inference_time': inference_time,
                    'total_time': total_time,
                    'sample_rate': sample_rate
                }
                
                return result
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    # Clear cache and return error without reloading model
                    torch.cuda.empty_cache()
                    logger.error(f"CUDA OOM for {audio_path}: {e}")
                    return {
                        'audio_path': audio_path,
                        'success': False,
                        'error': f"CUDA out of memory: {str(e)}",
                        'total_time': time.perf_counter() - start_time
                    }
                else:
                    raise  # Re-raise non-OOM CUDA errors
            
        except Exception as e:
            logger.error(f"Single inference failed for {audio_path}: {e}")
            return {
                'audio_path': audio_path,
                'success': False,
                'error': str(e),
                'total_time': time.perf_counter() - start_time
            }
            
        finally:
            # Cleanup temporary files
            if should_cleanup and optimized_path and os.path.exists(optimized_path):
                try:
                    os.unlink(optimized_path)
                except Exception:
                    pass
    
    def cleanup_persistent_model(self):
        """Clean up persistent model to free GPU memory"""
        with self._model_lock:
            if self.persistent_model and self._persistent_model is not None:
                del self._persistent_model
                self._persistent_model = None
                self._model_loaded = False
                torch.cuda.empty_cache()
                logger.info("Persistent model cleaned up and GPU memory freed")
    
    def _check_gpu_memory(self, gpu_id: int = 0, required_gb: float = 2.0) -> bool:
        """Check if GPU has enough free memory for inference."""
        if not torch.cuda.is_available():
            return False
        
        try:
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(gpu_id) / 1024**3
            free_memory = total_memory - allocated_memory
            
            logger.debug(f"GPU {gpu_id} memory: {allocated_memory:.2f}GB used, {free_memory:.2f}GB free of {total_memory:.2f}GB total")
            
            return free_memory >= required_gb
        except Exception as e:
            logger.warning(f"Could not check GPU memory: {e}")
            return True  # Assume it's OK if we can't check
    
    def _load_single_model(self):
        """Load model for single inference with persistence support"""
        with self._model_lock:
            # Check if model is already loaded and cached
            if self.persistent_model and self._model_loaded and self._persistent_model is not None:
                try:
                    # Verify the model is still valid by checking its device
                    device = next(self._persistent_model.parameters()).device
                    logger.debug(f"Using cached persistent model on {device}")
                    return self._persistent_model
                except Exception as e:
                    logger.warning(f"Cached model is invalid: {e}. Reloading...")
                    self._persistent_model = None
                    self._model_loaded = False
            
            # Check if workers are running - if so, we shouldn't load a new model
            if hasattr(self, '_workers_started') and self._workers_started and self.workers:
                logger.warning("Workers are running but single model load requested - this may cause OOM. Consider using workers instead.")
            
            # Use first available GPU (respects CUDA_VISIBLE_DEVICES)
            gpu_id = 0
            logger.info(f"Loading model on GPU {gpu_id} (persistent={self.persistent_model})...")
            
            try:
                # Clear any existing CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Setup environment with more conservative memory settings
                torch.cuda.set_per_process_memory_fraction(0.7, gpu_id)  # Reduced from 0.8 to be even more conservative
                torch.backends.cudnn.benchmark = True
                torch.set_grad_enabled(False)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Load model
                import llava
                
                model = llava.load(self.model_path, devices=[gpu_id])
                model.eval()
                model = model.to(f'cuda:{gpu_id}', non_blocking=True)
                
                # Disable gradients for inference
                for param in model.parameters():
                    param.requires_grad = False
                
                # Skip compilation for now to reduce memory usage
                # try:
                #     model = torch.compile(model, mode="max-autotune")
                #     logger.info("Model compiled successfully")
                # except Exception as e:
                #     logger.warning(f"Model compilation failed: {e}")
                
                # Cache the model if persistence is enabled
                if self.persistent_model:
                    self._persistent_model = model
                    self._model_loaded = True
                    logger.info("Model loaded and cached for persistent use")
                
                # Log memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    logger.info(f"GPU {gpu_id} memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                
                logger.info("Model loaded successfully")
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                # Clear cache on failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._persistent_model = None
                self._model_loaded = False
                raise
    
    def shutdown_workers(self):
        """Shutdown all workers with intelligent handling of incomplete initialization"""
        if not self._workers_started:
            logger.debug("No workers to shut down")
            return
            
        logger.info("Shutting down workers...")
        
        # Signal shutdown to prevent new workers from continuing initialization
        self._shutdown_requested.set()
        
        # Send shutdown signals to task queue
        for _ in range(len(self.workers)):
            self.task_queue.put(None)
        
        # Wait for workers to finish, with different timeouts for different states
        active_workers = []
        initializing_workers = []
        
        for worker in self.workers:
            if worker.is_alive():
                # Check if worker is still initializing (hasn't signaled ready yet)
                # Give initializing workers less time since they might be stuck in model loading
                timeout = 5 if not hasattr(worker, '_model_loaded') else 10
                worker.join(timeout=timeout)
                
                if worker.is_alive():
                    if not hasattr(worker, '_model_loaded'):
                        initializing_workers.append(worker)
                    else:
                        active_workers.append(worker)
        
        # Force terminate workers that didn't shutdown gracefully
        terminated_count = 0
        for worker in initializing_workers + active_workers:
            if worker.is_alive():
                logger.warning(f"Force terminating GPU {worker.gpu_id} worker")
                worker.terminate()
                worker.join()
                terminated_count += 1
        
        if terminated_count > 0:
            logger.info(f"Force terminated {terminated_count} workers (likely still initializing)")
        
        # Reset state
        self.workers.clear()
        self._workers_started = False
        self._shutdown_requested.clear()  # Reset for next use
        
        # Queue cleanup handled by OS
        
        logger.info("All workers shut down")

    def manual_shutdown_workers(self):
        """Manually shutdown workers without resetting state flags"""
        if not self._workers_started or not self.workers:
            return
        
        logger.info("Manually shutting down workers...")
        
        # Signal shutdown to prevent new workers from continuing initialization
        self._shutdown_requested.set()
        
        # Send shutdown signals to task queue
        for _ in range(len(self.workers)):
            try:
                self.task_queue.put(None, timeout=1.0)
            except:
                pass
        
        # Wait for workers to finish with shorter timeout
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=5.0)
                if worker.is_alive():
                    logger.warning(f"Force terminating worker {worker.pid}")
                    worker.terminate()
                    worker.join(timeout=2.0)
        
        # Reset state
        self.workers.clear()
        self._workers_started = False
        self._shutdown_requested.clear()
        
        logger.info("Manual worker shutdown completed")

def get_audio_files(directory: str) -> List[str]:
    """Get audio files from directory"""
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    files = []
    
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if Path(filename).suffix.lower() in audio_extensions:
                files.append(os.path.join(root, filename))
    
    return sorted(files)

def main():
    parser = argparse.ArgumentParser(description="Audio Flamingo 3 - Simplified Inference")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--audio", type=str, help="Single audio file")
    mode_group.add_argument("--dir", type=str, help="Directory with audio files")
    
    # Common arguments
    parser.add_argument("--text", type=str, default="Please describe the audio in detail", help="Text prompt")
    parser.add_argument("--output", type=str, help="Output JSON file (required for batch)")
    parser.add_argument("--gpus", type=int, help="Number of GPUs")
    parser.add_argument("--model-path", type=str, default=AUDIO_FLAMINGO_MODEL_PATH, help="Model path")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    # Single file mode
    if args.audio:
        if not os.path.exists(args.audio):
            print(f"❌ Audio file not found: {args.audio}")
            sys.exit(1)
        
        if not args.quiet:
            print(f"🎵 Processing: {Path(args.audio).name}")
        
        system = SimpleInferenceSystem(args.model_path)
        result = system.single_inference(args.audio, args.text)
        
        if result['success']:
            if args.quiet:
                print(result['response'])
            else:
                print(f"\n{'='*60}")
                print("🎯 RESPONSE:")
                print(f"{'='*60}")
                print(result['response'])
                print(f"{'='*60}")
                print(f"⏱️  Processing time: {result['total_time']:.3f}s")
                print(f"   • Inference: {result['inference_time']:.3f}s")
        else:
            print(f"❌ Failed: {result['error']}")
            sys.exit(1)
    
    # Batch processing mode
    else:
        if not args.output:
            print("❌ --output required for batch processing")
            sys.exit(1)
        
        if not os.path.exists(args.dir):
            print(f"❌ Directory not found: {args.dir}")
            sys.exit(1)
        
        audio_files = get_audio_files(args.dir)
        if not audio_files:
            print("❌ No audio files found")
            sys.exit(1)
        
        num_gpus = args.gpus or torch.cuda.device_count()
        
        # Respect CUDA_VISIBLE_DEVICES if set
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            if visible_devices:
                available_gpus = len([d.strip() for d in visible_devices.split(',') if d.strip()])
                num_gpus = min(num_gpus, available_gpus)
        
        if not args.quiet:
            print("🚀 SIMPLIFIED BATCH PROCESSING")
            print(f"{'='*60}")
            print(f"📁 Audio files: {len(audio_files):,}")
            print(f"💻 GPUs: {num_gpus}")
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                print(f"🔧 CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            print(f"{'='*60}")
        
        # Set environment
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Process batch
        system = SimpleInferenceSystem(args.model_path, num_gpus)
        start_time = time.time()
        results = system.batch_inference(audio_files, args.text)
        total_time = time.time() - start_time
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Statistics
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        if not args.quiet:
            if successful:
                avg_inference = sum(r['inference_time'] for r in successful) / len(successful)
                throughput = len(successful) / total_time * 60
                
                print("\n🎉 Batch processing completed!")
                print(f"📊 Successful: {len(successful)}/{len(results)} files")
                print(f"⏱️  Average inference: {avg_inference:.3f}s per file")
                print(f"🚀 Total time: {total_time/60:.1f} minutes")
                print(f"🎯 Throughput: {throughput:.1f} files/minute")
                print(f"💾 Results saved: {args.output}")
            
            if failed:
                print(f"❌ Failed: {len(failed)} files")
        else:
            print(f"Processed {len(successful)}/{len(results)} files in {total_time/60:.1f}min")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)