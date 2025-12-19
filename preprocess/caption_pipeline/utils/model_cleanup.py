"""
Model Resource Cleanup Utilities

This module provides utilities to clean up GPU memory and resources from models
used in the caption pipeline. It ensures proper cleanup between pipeline stages
and enables parallel cleanup during next stage initialization.

Pipeline stages and their models:
1. Main Pipeline (orchestrator): Whisper + CLIP models
2. Video Descriptions: vLLM (excluded from cleanup as requested)
3. Audio Descriptions: Audio Flamingo 3 (separate stage)
4. Other stages: vLLM (excluded from cleanup)
"""

import gc
import threading
import time
from typing import Optional, List, Callable, Any
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import importlib.util
    CLIP_AVAILABLE = importlib.util.find_spec("clip") is not None
except ImportError:
    CLIP_AVAILABLE = False

from caption_pipeline.utils.rich_console import get_console

logger = logging.getLogger(__name__)
rich_console = get_console()


class ModelCleanupManager:
    """
    Manages cleanup of model resources with parallel execution support.
    
    This class handles cleanup of various models used in the pipeline:
    - Whisper models (faster-whisper)
    - CLIP models 
    - Audio Flamingo 3 models
    - General PyTorch models
    
    It supports both synchronous and asynchronous cleanup to enable
    parallel cleanup during next stage initialization.
    """
    
    def __init__(self):
        self.cleanup_thread_pool = []
        self.active_cleanups = set()
        self.cleanup_lock = threading.RLock()
    
    def cleanup_whisper_model(self, caption_generator, async_cleanup: bool = False) -> Optional[threading.Thread]:
        """
        Clean up Whisper model resources.
        
        Args:
            caption_generator: CaptionGenerator or EnhancedCaptionGenerator instance
            async_cleanup: Whether to run cleanup in background thread
            
        Returns:
            Threading.Thread if async_cleanup=True, None otherwise
        """
        def _cleanup_whisper():
            try:
                rich_console.print_info("🧹 Cleaning up Whisper model resources...")
                
                # Clear Whisper models
                if hasattr(caption_generator, 'model'):
                    del caption_generator.model
                    caption_generator.model = None
                
                if hasattr(caption_generator, 'base_model'):
                    del caption_generator.base_model  
                    caption_generator.base_model = None
                
                # Clear enhancer if present
                if hasattr(caption_generator, 'enhancer') and caption_generator.enhancer:
                    if hasattr(caption_generator.enhancer, 'visual_detector'):
                        self._cleanup_clip_model(caption_generator.enhancer.visual_detector)
                    del caption_generator.enhancer
                    caption_generator.enhancer = None
                
                # Force GPU memory cleanup
                self._force_gpu_cleanup()
                
                rich_console.print_success("✓ Whisper model cleanup completed")
                
            except Exception as e:
                rich_console.print_warning(f"Warning during Whisper cleanup: {e}")
            finally:
                with self.cleanup_lock:
                    self.active_cleanups.discard('whisper')
        
        if async_cleanup:
            with self.cleanup_lock:
                self.active_cleanups.add('whisper')
            thread = threading.Thread(target=_cleanup_whisper, name="WhisperCleanup")
            thread.daemon = True
            thread.start()
            self.cleanup_thread_pool.append(thread)
            return thread
        else:
            _cleanup_whisper()
            return None
    
    def cleanup_clip_model(self, visual_detector_or_enhancer, async_cleanup: bool = False) -> Optional[threading.Thread]:
        """
        Clean up CLIP model resources.
        
        Args:
            visual_detector_or_enhancer: VisualEventDetector or MovieCaptionEnhancer instance
            async_cleanup: Whether to run cleanup in background thread
            
        Returns:
            Threading.Thread if async_cleanup=True, None otherwise
        """
        def _cleanup_clip():
            try:
                rich_console.print_info("🧹 Cleaning up CLIP model resources...")
                
                self._cleanup_clip_model(visual_detector_or_enhancer)
                
                # Force GPU memory cleanup
                self._force_gpu_cleanup()
                
                rich_console.print_success("✓ CLIP model cleanup completed")
                
            except Exception as e:
                rich_console.print_warning(f"Warning during CLIP cleanup: {e}")
            finally:
                with self.cleanup_lock:
                    self.active_cleanups.discard('clip')
        
        if async_cleanup:
            with self.cleanup_lock:
                self.active_cleanups.add('clip')
            thread = threading.Thread(target=_cleanup_clip, name="CLIPCleanup")
            thread.daemon = True
            thread.start() 
            self.cleanup_thread_pool.append(thread)
            return thread
        else:
            _cleanup_clip()
            return None
    
    def cleanup_audio_flamingo_model(self, audio_descriptor, async_cleanup: bool = False) -> Optional[threading.Thread]:
        """
        Clean up Audio Flamingo 3 model resources.
        
        Args:
            audio_descriptor: AudioDescriptor instance
            async_cleanup: Whether to run cleanup in background thread
            
        Returns:
            Threading.Thread if async_cleanup=True, None otherwise
        """
        def _cleanup_audio_flamingo():
            try:
                rich_console.print_info("🧹 Cleaning up Audio Flamingo 3 model resources...")
                
                # Use the built-in cleanup method
                if hasattr(audio_descriptor, 'cleanup'):
                    audio_descriptor.cleanup()
                elif hasattr(audio_descriptor, 'inference_system') and audio_descriptor.inference_system:
                    if hasattr(audio_descriptor.inference_system, 'cleanup_persistent_model'):
                        audio_descriptor.inference_system.cleanup_persistent_model()
                    if hasattr(audio_descriptor.inference_system, 'shutdown_workers'):
                        audio_descriptor.inference_system.shutdown_workers()
                    audio_descriptor.inference_system = None
                
                # Force GPU memory cleanup
                self._force_gpu_cleanup()
                
                rich_console.print_success("✓ Audio Flamingo 3 model cleanup completed")
                
            except Exception as e:
                rich_console.print_warning(f"Warning during Audio Flamingo cleanup: {e}")
            finally:
                with self.cleanup_lock:
                    self.active_cleanups.discard('audio_flamingo')
        
        if async_cleanup:
            with self.cleanup_lock:
                self.active_cleanups.add('audio_flamingo')
            thread = threading.Thread(target=_cleanup_audio_flamingo, name="AudioFlamingoCleanup")
            thread.daemon = True
            thread.start()
            self.cleanup_thread_pool.append(thread)
            return thread
        else:
            _cleanup_audio_flamingo()
            return None
    
    def cleanup_all_models(self, components: dict, async_cleanup: bool = False) -> List[threading.Thread]:
        """
        Clean up all model resources from multiple components.
        
        Args:
            components: Dictionary of component_name -> component_instance
            async_cleanup: Whether to run cleanups in background threads
            
        Returns:
            List of cleanup threads if async_cleanup=True, empty list otherwise
        """
        cleanup_threads = []
        
        for component_name, component in components.items():
            if component is None:
                continue
                
            try:
                # Whisper models
                if hasattr(component, 'model') or hasattr(component, 'base_model'):
                    thread = self.cleanup_whisper_model(component, async_cleanup)
                    if thread:
                        cleanup_threads.append(thread)
                
                # CLIP models  
                elif hasattr(component, 'clip_model') or hasattr(component, 'visual_detector'):
                    thread = self.cleanup_clip_model(component, async_cleanup)
                    if thread:
                        cleanup_threads.append(thread)
                
                # Audio Flamingo models
                elif hasattr(component, 'inference_system') or component_name == 'audio_descriptor':
                    thread = self.cleanup_audio_flamingo_model(component, async_cleanup)
                    if thread:
                        cleanup_threads.append(thread)
                
                # Generic PyTorch model cleanup
                else:
                    thread = self._cleanup_generic_model(component, component_name, async_cleanup)
                    if thread:
                        cleanup_threads.append(thread)
                        
            except Exception as e:
                rich_console.print_warning(f"Error cleaning up {component_name}: {e}")
        
        return cleanup_threads
    
    def wait_for_cleanup_completion(self, timeout: float = 30.0) -> bool:
        """
        Wait for all active cleanup operations to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all cleanups completed, False if timeout occurred
        """
        start_time = time.time()
        
        # Wait for active cleanup threads
        for thread in self.cleanup_thread_pool[:]:  # Copy to avoid modification during iteration
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                rich_console.print_warning("Cleanup timeout reached")
                return False
                
            thread.join(timeout=remaining_time)
            if thread.is_alive():
                rich_console.print_warning(f"Cleanup thread {thread.name} did not complete in time")
                return False
            else:
                self.cleanup_thread_pool.remove(thread)
        
        # Double-check that no active cleanups remain
        with self.cleanup_lock:
            if self.active_cleanups:
                rich_console.print_warning(f"Some cleanups still active: {self.active_cleanups}")
                return False
        
        rich_console.print_success("✓ All model cleanup operations completed")
        return True
    
    def parallel_cleanup_with_next_stage_init(self, components_to_cleanup: dict, 
                                            next_stage_init_func: Callable, 
                                            next_stage_args: tuple = (),
                                            next_stage_kwargs: dict = None) -> Any:
        """
        Run model cleanup in parallel with next stage initialization.
        
        This enables overlapping cleanup of current stage models with
        initialization of next stage models for better pipeline efficiency.
        
        Args:
            components_to_cleanup: Dictionary of components to clean up
            next_stage_init_func: Function to initialize next stage
            next_stage_args: Arguments for next stage function
            next_stage_kwargs: Keyword arguments for next stage function
            
        Returns:
            Result from next_stage_init_func
        """
        if next_stage_kwargs is None:
            next_stage_kwargs = {}
        
        rich_console.print_info("🔄 Starting parallel cleanup with next stage initialization...")
        
        # Start cleanup in background
        self.cleanup_all_models(components_to_cleanup, async_cleanup=True)
        
        try:
            # Initialize next stage while cleanup happens in background
            rich_console.print_info("🚀 Initializing next stage while cleanup runs in background...")
            result = next_stage_init_func(*next_stage_args, **next_stage_kwargs)
            
            # Wait for cleanup to complete (with reasonable timeout)
            cleanup_completed = self.wait_for_cleanup_completion(timeout=60.0)
            
            if not cleanup_completed:
                rich_console.print_warning("⚠️  Some cleanup operations may still be running")
            else:
                rich_console.print_success("✓ Parallel cleanup and next stage initialization completed successfully")
            
            return result
            
        except Exception as e:
            rich_console.print_error(f"Error during next stage initialization: {e}")
            # Still wait for cleanup even if next stage failed
            self.wait_for_cleanup_completion(timeout=30.0)
            raise
    
    def _cleanup_clip_model(self, component):
        """Internal method to clean up CLIP model from component."""
        # Handle VisualEventDetector
        if hasattr(component, 'clip_model'):
            del component.clip_model
            component.clip_model = None
        
        if hasattr(component, 'clip_preprocess'):
            del component.clip_preprocess
            component.clip_preprocess = None
        
        # Handle MovieCaptionEnhancer with VisualEventDetector
        if hasattr(component, 'visual_detector') and component.visual_detector:
            if hasattr(component.visual_detector, 'clip_model'):
                del component.visual_detector.clip_model
                component.visual_detector.clip_model = None
            if hasattr(component.visual_detector, 'clip_preprocess'):
                del component.visual_detector.clip_preprocess
                component.visual_detector.clip_preprocess = None
    
    def _cleanup_generic_model(self, component, component_name: str, async_cleanup: bool = False) -> Optional[threading.Thread]:
        """Clean up generic PyTorch models."""
        def _cleanup():
            try:
                rich_console.print_info(f"🧹 Cleaning up {component_name} model resources...")
                
                # Look for common PyTorch model attributes
                model_attrs = ['model', 'net', 'network', 'encoder', 'decoder']
                cleaned_any = False
                
                for attr in model_attrs:
                    if hasattr(component, attr):
                        model = getattr(component, attr)
                        if model is not None:
                            if TORCH_AVAILABLE and hasattr(model, 'parameters'):
                                # This looks like a PyTorch model
                                del model
                                setattr(component, attr, None)
                                cleaned_any = True
                
                if cleaned_any:
                    self._force_gpu_cleanup()
                    rich_console.print_success(f"✓ {component_name} model cleanup completed")
                
            except Exception as e:
                rich_console.print_warning(f"Warning during {component_name} cleanup: {e}")
            finally:
                with self.cleanup_lock:
                    self.active_cleanups.discard(f'generic_{component_name}')
        
        if async_cleanup:
            with self.cleanup_lock:
                self.active_cleanups.add(f'generic_{component_name}')
            thread = threading.Thread(target=_cleanup, name=f"{component_name}Cleanup")
            thread.daemon = True
            thread.start()
            self.cleanup_thread_pool.append(thread)
            return thread
        else:
            _cleanup()
            return None
    
    def _force_gpu_cleanup(self):
        """Force GPU memory cleanup."""
        if not TORCH_AVAILABLE:
            return
            
        try:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Error during GPU cleanup: {e}")


# Global instance for easy access
model_cleanup_manager = ModelCleanupManager()


def cleanup_whisper_model(caption_generator, async_cleanup: bool = False) -> Optional[threading.Thread]:
    """Convenience function to clean up Whisper model."""
    return model_cleanup_manager.cleanup_whisper_model(caption_generator, async_cleanup)


def cleanup_clip_model(visual_detector_or_enhancer, async_cleanup: bool = False) -> Optional[threading.Thread]:
    """Convenience function to clean up CLIP model."""
    return model_cleanup_manager.cleanup_clip_model(visual_detector_or_enhancer, async_cleanup)


def cleanup_audio_flamingo_model(audio_descriptor, async_cleanup: bool = False) -> Optional[threading.Thread]:
    """Convenience function to clean up Audio Flamingo 3 model."""
    return model_cleanup_manager.cleanup_audio_flamingo_model(audio_descriptor, async_cleanup)


def cleanup_all_models(components: dict, async_cleanup: bool = False) -> List[threading.Thread]:
    """Convenience function to clean up all models."""
    return model_cleanup_manager.cleanup_all_models(components, async_cleanup)


def parallel_cleanup_with_next_stage_init(components_to_cleanup: dict,
                                        next_stage_init_func: Callable,
                                        next_stage_args: tuple = (),
                                        next_stage_kwargs: dict = None) -> Any:
    """Convenience function for parallel cleanup with next stage initialization."""
    return model_cleanup_manager.parallel_cleanup_with_next_stage_init(
        components_to_cleanup, next_stage_init_func, next_stage_args, next_stage_kwargs
    )