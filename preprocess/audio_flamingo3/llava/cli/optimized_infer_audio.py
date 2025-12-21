# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Heavily optimized inference script for audio-flamingo
# Supports batch processing, parallel preprocessing, and efficient GPU utilization
# Designed to scale to thousands of audio files

import argparse
import asyncio
import csv
import gc
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import as_completed

import numpy as np
import psutil
import soundfile as sf
import tempfile
import torch
import torch.multiprocessing as mp
from tqdm.asyncio import tqdm
from pydantic import BaseModel
from scipy import signal

# Import original modules
from config import QWEN_AUDIO_MODEL
import llava
from llava.media import Sound
from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat
from llava.utils.media import _load_sound_mask

# Configure multiprocessing
mp.set_start_method('spawn', force=True)

# Force single GPU usage to avoid device mismatch issues
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AudioProcessingConfig:
    """Configuration for audio processing optimization"""
    batch_size: int = 16
    num_workers: int = min(mp.cpu_count(), 16)
    max_audio_length: float = 600.0  # 10 minutes max
    sample_rate: int = 16000
    window_length: float = 30.0
    window_overlap: float = 0.0
    max_num_window: int = 20
    use_half_precision: bool = True
    prefetch_factor: int = 2
    speed_factor: float = 2  # Speed up audio by this factor
    
    def __post_init__(self):
        # Adjust batch size based on available GPU memory if CUDA is available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            # Rough estimation: 1GB allows batch size of 8
            suggested_batch_size = min(max(int(gpu_memory / (1024**3) * 8), 4), 64)
            self.batch_size = min(self.batch_size, suggested_batch_size)
            logger.info(f"Adjusted batch size to {self.batch_size} based on GPU memory")
        
        # Log speed optimization
        if self.speed_factor != 1.0:
            logger.info(f"Audio speed optimization enabled: {self.speed_factor}x speed")

class AudioProcessor:
    """Handles parallel audio preprocessing using multiprocessing"""
    
    def __init__(self, config: AudioProcessingConfig):
        self.config = config
        self.executor = None
    
    def __enter__(self):
        self.executor = ProcessPoolExecutor(max_workers=self.config.num_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    @staticmethod
    def speed_up_audio(audio_data: np.ndarray, speed_factor: float, sample_rate: int) -> np.ndarray:
        """Speed up audio by a given factor using resampling"""
        if speed_factor == 1.0:
            return audio_data
        
        try:
            from scipy import signal
            # Calculate new sample rate for speed change
            new_length = int(len(audio_data) / speed_factor)
            # Use scipy's resample for high-quality speed change
            speeded_audio = signal.resample(audio_data, new_length)
            return speeded_audio.astype(np.float32)
        except ImportError:
            # Fallback to simple downsampling if scipy not available
            logger.warning("scipy not available, using simple downsampling for speed optimization")
            step = int(speed_factor)
            return audio_data[::step]
    
    @staticmethod
    def process_single_audio(audio_path: str, config: AudioProcessingConfig) -> Optional[Dict]:
        """Process a single audio file - runs in worker process"""
        try:
            # Import required modules in worker process
            from llava.utils.media import _load_sound_mask
            from transformers import AutoFeatureExtractor
            import tempfile
            import os
            
            # Load feature extractor (cached in worker)
            if not hasattr(AudioProcessor.process_single_audio, 'wav_processor'):
                AudioProcessor.process_single_audio.wav_processor = AutoFeatureExtractor.from_pretrained(
                    QWEN_AUDIO_MODEL, trust_remote_code=True
                )
            
            # If speed optimization is enabled, preprocess audio first
            if config.speed_factor != 1.0:
                # Load original audio
                import soundfile as sf
                audio_data, orig_sample_rate = sf.read(audio_path)
                
                # Speed up the audio
                speeded_audio = AudioProcessor.speed_up_audio(
                    audio_data, config.speed_factor, orig_sample_rate
                )
                
                # Create temporary file with speeded audio
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    sf.write(tmp_file.name, speeded_audio, orig_sample_rate)
                    temp_audio_path = tmp_file.name
                
                try:
                    # Process the speeded audio
                    sound_outputs, audio_feature_masks, audio_embed_masks = _load_sound_mask(
                        temp_audio_path,
                        sample_rate=config.sample_rate,
                        window_length=config.window_length,
                        window_overlap=config.window_overlap,
                        max_num_window=config.max_num_window
                    )
                finally:
                    # Clean up temporary file
                    os.unlink(temp_audio_path)
            else:
                # Process audio normally
                sound_outputs, audio_feature_masks, audio_embed_masks = _load_sound_mask(
                    audio_path,
                    sample_rate=config.sample_rate,
                    window_length=config.window_length,
                    window_overlap=config.window_overlap,
                    max_num_window=config.max_num_window
                )
            
            return {
                'audio_path': audio_path,
                'sound_outputs': sound_outputs,
                'audio_feature_masks': audio_feature_masks,
                'audio_embed_masks': audio_embed_masks,
                'success': True,
                'speed_factor': config.speed_factor
            }
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            return {
                'audio_path': audio_path,
                'error': str(e),
                'success': False,
                'speed_factor': config.speed_factor
            }
    
    async def process_batch_async(self, audio_paths: List[str]) -> List[Dict]:
        """Process a batch of audio files asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Submit tasks to process pool
        tasks = []
        for audio_path in audio_paths:
            task = loop.run_in_executor(
                self.executor,
                self.process_single_audio,
                audio_path,
                self.config
            )
            tasks.append(task)
        
        # Wait for all tasks to complete with progress bar
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing audio"):
            result = await coro
            results.append(result)
        
        return results

class ModelManager:
    """Manages model loading and batch inference"""
    
    def __init__(self, model_base: str, enable_warmup: bool = True):
        self.model_base = model_base
        self.enable_warmup = enable_warmup
        self.model = None
        # Use cuda:0 specifically to avoid multi-GPU issues
        self.device = torch.device("cuda:0")
        
    def load_model(self):
        """Load model once and keep in memory"""
        if self.model is not None:
            return
        
        logger.info(f"Loading model from {self.model_base}")
        start_time = time.time()
        
        # Load base model
        model_path = "../../../MODELS/audio-flamingo-3"
        self.model = llava.load(model_path)
        
        
        # Move to GPU and set to eval mode
        self.model.eval()
        if torch.cuda.is_available():
            # Force model to use cuda:0 to avoid device mismatch
            torch.cuda.set_device(0)
            self.model = self.model.to(self.device)
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Warm up the model for better inference performance
        if self.enable_warmup:
            self.warm_up()
        else:
            logger.info("Model warm-up skipped (--no-warmup flag used)")
    
    def warm_up(self):
        """Warm up the model with a dummy inference to optimize subsequent calls"""
        logger.info("Warming up model for optimal performance...")
        warm_up_start = time.time()
        
        try:
            # Create a dummy audio file path (we'll use a minimal sound object)
            import tempfile
            import numpy as np
            import soundfile as sf
            
            # Create a short dummy audio file
            sample_rate = 16000
            duration = 1.0  # 1 second
            dummy_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, dummy_audio, sample_rate)
                dummy_sound = Sound(tmp_file.name)
                
                # Dummy prompt
                dummy_prompt = [dummy_sound, "Describe this audio."]
                
                # Perform warm-up inference
                if torch.cuda.is_available():
                    torch.cuda.set_device(0)
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    _ = self.model.generate_content(dummy_prompt)
                
                # Clean up temporary file
                os.unlink(tmp_file.name)
            
            warm_up_time = time.time() - warm_up_start
            logger.info(f"Model warm-up completed in {warm_up_time:.2f} seconds")
            
            # Clear cache after warm-up
            self.clear_gpu_cache()
            
        except Exception as e:
            logger.warning(f"Model warm-up failed (this may impact first inference speed): {str(e)}")
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    @torch.no_grad()
    def batch_inference(self, batch_data: List[Dict], text_prompt: str, response_format: Optional[ResponseFormat] = None) -> List[Dict]:
        """Perform batch inference on preprocessed audio data"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        
        for item in batch_data:
            if not item['success']:
                result_item = {
                    'audio_path': item['audio_path'],
                    'response': f"Error: {item.get('error', 'Unknown error')}",
                    'inference_time': 0.0,
                    'success': False
                }
                results.append(result_item)
                continue
            
            try:
                # Create Sound object from preprocessed data
                sound = Sound(item['audio_path'])
                
                # Prepare prompt
                prompt = [sound, text_prompt]
                
                # Generate response with device synchronization and timing
                if torch.cuda.is_available():
                    torch.cuda.set_device(0)  # Ensure we're on the right device
                
                inference_start = time.time()
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    response = self.model.generate_content(prompt, response_format=response_format)
                inference_time = time.time() - inference_start
                
                result_item = {
                    'audio_path': item['audio_path'],
                    'response': response,
                    'inference_time': inference_time,
                    'success': True
                }
                results.append(result_item)
                
                # Print individual result
                speed_info = f" (Speed: {item.get('speed_factor', 2.0)}x)" if item.get('speed_factor', 2.0) != 1.0 else ""
                print(f"{item['audio_path']}{speed_info}: {response}")
                print(f"   Inference time: {inference_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Inference error for {item['audio_path']}: {str(e)}")
                result_item = {
                    'audio_path': item['audio_path'],
                    'response': f"Inference error: {str(e)}",
                    'inference_time': 0.0,
                    'success': False
                }
                results.append(result_item)
        
        return results

class OptimizedInferenceEngine:
    """Main engine that orchestrates the optimized inference pipeline"""
    
    def __init__(self, config: AudioProcessingConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.stats = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_time': 0,
            'avg_time_per_file': 0,
            'total_inference_time': 0,
            'avg_inference_time': 0
        }
    
    async def process_files(self, 
                          audio_paths: List[str], 
                          text_prompt: str,
                          response_format: Optional[ResponseFormat] = None,
                          output_file: Optional[str] = None) -> List[Dict]:
        """Process multiple audio files with optimized pipeline"""
        
        start_time = time.time()
        self.stats['total_files'] = len(audio_paths)
        
        # Load model once
        self.model_manager.load_model()
        
        # Process in batches
        all_results = []
        batch_size = self.config.batch_size
        
        with AudioProcessor(self.config) as processor:
            for i in tqdm(range(0, len(audio_paths), batch_size), desc="Processing batches"):
                batch_paths = audio_paths[i:i + batch_size]
                
                # Step 1: Parallel audio preprocessing
                logger.info(f"Preprocessing batch {i//batch_size + 1}")
                batch_data = await processor.process_batch_async(batch_paths)
                
                # Step 2: Batch inference
                logger.info(f"Running inference on batch {i//batch_size + 1}")
                batch_results = self.model_manager.batch_inference(
                    batch_data, text_prompt, response_format
                )
                
                # Combine results and collect inference time stats
                for result_item in batch_results:
                    item_result = {
                        'audio_path': result_item['audio_path'],
                        'result': result_item['response'],
                        'success': result_item['success'],
                        'inference_time': result_item['inference_time']
                    }
                    all_results.append(item_result)
                    
                    if result_item['success']:
                        self.stats['successful_files'] += 1
                        self.stats['total_inference_time'] += result_item['inference_time']
                    else:
                        self.stats['failed_files'] += 1
                
                # Clear GPU cache after each batch
                self.model_manager.clear_gpu_cache()
        
        # Save results if output file specified
        if output_file:
            await self.save_results(all_results, output_file)
        
        # Update stats
        self.stats['total_time'] = time.time() - start_time
        self.stats['avg_time_per_file'] = self.stats['total_time'] / len(audio_paths)
        
        # Calculate average inference time (only for successful files)
        if self.stats['successful_files'] > 0:
            self.stats['avg_inference_time'] = self.stats['total_inference_time'] / self.stats['successful_files']
        
        return all_results
    
    async def save_results(self, results: List[Dict], output_file: str):
        """Save results to file asynchronously"""
        try:
            if output_file.endswith('.json'):
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            elif output_file.endswith('.csv'):
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['audio_path', 'result', 'success', 'inference_time'])
                    writer.writeheader()
                    writer.writerows(results)
            else:
                # Default to JSON
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                    
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def print_stats(self):
        """Print processing statistics"""
        print("\n" + "="*50)
        print("PROCESSING STATISTICS")
        print("="*50)
        print(f"Total files processed: {self.stats['total_files']}")
        print(f"Successful: {self.stats['successful_files']}")
        print(f"Failed: {self.stats['failed_files']}")
        print(f"Success rate: {self.stats['successful_files']/self.stats['total_files']*100:.1f}%")
        print(f"Total processing time: {self.stats['total_time']:.2f} seconds")
        print(f"Average time per file: {self.stats['avg_time_per_file']:.3f} seconds")
        print(f"Total inference time: {self.stats['total_inference_time']:.2f} seconds")
        print(f"Average inference time: {self.stats['avg_inference_time']:.3f} seconds")
        print(f"Files per second: {self.stats['total_files']/self.stats['total_time']:.2f}")
        if self.config.speed_factor != 1.0:
            print(f"Audio speed optimization: {self.config.speed_factor}x (saves ~{((self.config.speed_factor-1)/self.config.speed_factor)*100:.1f}% time)")
        print("="*50)

def get_audio_files_from_directory(directory: str) -> List[str]:
    """Get all audio files from directory recursively"""
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    audio_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))
    
    return sorted(audio_files)

def load_audio_list_from_csv(csv_file: str) -> List[str]:
    """Load audio file paths from CSV"""
    audio_files = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Assume first column contains audio paths
            audio_path = list(row.values())[0]
            if os.path.exists(audio_path):
                audio_files.append(audio_path)
            else:
                logger.warning(f"Audio file not found: {audio_path}")
    return audio_files

async def main():
    parser = argparse.ArgumentParser(description="Optimized Audio Flamingo Inference")
    
    # Model arguments
    parser.add_argument("--model-base", "-m", type=str, required=True,
                      help="Model base name or path")
    
    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio-file", type=str,
                           help="Single audio file to process")
    input_group.add_argument("--audio-dir", type=str,
                           help="Directory containing audio files")
    input_group.add_argument("--audio-list", type=str,
                           help="Text file with audio file paths (one per line)")
    input_group.add_argument("--input-csv", type=str,
                           help="CSV file with audio file paths")
    
    # Prompt arguments
    parser.add_argument("--text", type=str, 
                      default="Please describe the audio in detail",
                      help="Text prompt for inference")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str,
                      help="Directory to save results")
    parser.add_argument("--output-file", type=str,
                      help="Output file for results (JSON or CSV)")
    
    # JSON mode arguments
    parser.add_argument("--json-mode", action="store_true",
                      help="Enable JSON output mode")
    parser.add_argument("--json-schema", type=str,
                      help="JSON schema file path")
    
    # Performance arguments
    parser.add_argument("--batch-size", type=int, default=16,
                      help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=min(mp.cpu_count(), 16),
                      help="Number of worker processes for audio preprocessing")
    parser.add_argument("--max-audio-length", type=float, default=600.0,
                      help="Maximum audio length in seconds")
    parser.add_argument("--speed-factor", type=float, default=20.0,
                      help="Speed up audio by this factor (default: 1.5x)")
    parser.add_argument("--no-warmup", action="store_true",
                      help="Skip model warm-up (may result in slower first inference)")
    
    args = parser.parse_args()
    
    # Setup configuration
    config = AudioProcessingConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_audio_length=args.max_audio_length,
        speed_factor=args.speed_factor
    )
    
    # Setup response format
    response_format = None
    if args.json_mode:
        if args.json_schema:
            from llava.cli.infer_audio import get_schema_from_python_path
            schema_str = get_schema_from_python_path(args.json_schema)
            response_format = ResponseFormat(
                type="json_schema", 
                json_schema=JsonSchemaResponseFormat(schema=schema_str)
            )
        else:
            response_format = ResponseFormat(type="json_object")
    
    # Get audio files
    audio_files = []
    if args.audio_file:
        if os.path.exists(args.audio_file):
            audio_files = [args.audio_file]
        else:
            logger.error(f"Audio file not found: {args.audio_file}")
            sys.exit(1)
    elif args.audio_dir:
        audio_files = get_audio_files_from_directory(args.audio_dir)
    elif args.audio_list:
        with open(args.audio_list, 'r') as f:
            audio_files = [line.strip() for line in f if line.strip() and os.path.exists(line.strip())]
    elif args.input_csv:
        audio_files = load_audio_list_from_csv(args.input_csv)
    
    if not audio_files:
        logger.error("No audio files found to process")
        sys.exit(1)
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    # Setup model manager
    model_manager = ModelManager(
        model_base=args.model_base,
        enable_warmup=not args.no_warmup
    )
    
    # Setup inference engine
    engine = OptimizedInferenceEngine(config, model_manager)
    
    # Process files
    try:
        results = await engine.process_files(
            audio_paths=audio_files,
            text_prompt=args.text,
            response_format=response_format,
            output_file=args.output_file
        )
        
        # Print results for single file (backward compatibility)
        if args.audio_file and len(results) == 1:
            print("\n" + "="*60)
            print("SINGLE FILE RESULT")
            print("="*60)
            print(results[0]['result'])
            print("="*60)
        
        # Print statistics
        engine.print_stats()
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Handle Windows compatibility
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())