#!/usr/bin/env python3
"""
Base stage class providing common functionality for all pipeline stages.
"""

import asyncio
import json
import time
import fcntl
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List

import aiohttp
from tqdm import tqdm

from utils import load_schema, load_prompt, save_jsonl, load_jsonl


class BaseStage(ABC):
    """Base class for all pipeline stages."""
    
    def __init__(self, session: aiohttp.ClientSession, base_url: str, model_name: str, max_concurrent: int, config: Dict[str, Any]):
        self.session = session
        self.base_url = base_url
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.config = config
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Output directory
        self.output_dir = Path(config['processing']['output_directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Incremental saving settings
        self.enable_incremental_save = config.get('processing', {}).get('incremental_save', True)
        self.save_batch_size = config.get('processing', {}).get('save_batch_size', 10)
        self._save_lock = asyncio.Lock()
    
    async def call_llm(self, prompt: str, schema: Dict = None, guidance_response_format: str = None, temperature: float = None) -> Dict:
        """Call the vLLM API with proper error handling and retries."""
        max_retries = 3
        base_delay = 1
        
        # Use timeout from config (reduced to avoid hanging)
        timeout_seconds = min(self.config.get("timeout", 300), 300)  # Max 5 minutes
        
        for attempt in range(max_retries):
            try:
                # Use semaphore to control concurrency
                async with self.semaphore:
                    # Prepare the request
                    messages = [{"role": "user", "content": prompt}]
                    
                    # Use guidance_response_format for vLLM compatibility
                    request_data = {
                        "model": self.config["model"]["name"],
                        "messages": messages,
                        "max_tokens": self.config.get("model", {}).get("max_tokens", 40000),
                        "temperature": temperature if temperature is not None else self.config.get("model", {}).get("temperature", 0.7),
                        "guided_json": schema if schema else None
                    }

                    # Use the shared session instead of creating a new one
                    try:
                        async with self.session.post(
                            f"{self.config['model']['base_url']}/chat/completions",
                            json=request_data,
                            headers={"Content-Type": "application/json"},
                            # timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                if "choices" in result and len(result["choices"]) > 0:
                                    content = result["choices"][0]["message"]["content"]
                                    try:
                                        parsed_content = json.loads(content)
                                        return {"success": True, "data": parsed_content}
                                    except json.JSONDecodeError:
                                        return {"success": True, "data": {"content": content}}
                                else:
                                    return {"success": False, "error": "No choices in response"}
                            elif response.status == 400:
                                error_text = await response.text()
                                if "image" in error_text.lower() and attempt < max_retries - 1:
                                    print(f"⚠️ Image-related error (attempt {attempt + 1}): {error_text}")
                                    await asyncio.sleep(base_delay * (2 ** attempt))
                                    continue
                                else:
                                    return {"success": False, "error": f"HTTP 400: {error_text}"}
                            else:
                                error_text = await response.text()
                                return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
                    except asyncio.TimeoutError:
                        if attempt < max_retries - 1:
                            print(f"⚠️ Timeout on attempt {attempt + 1}, retrying...")
                            await asyncio.sleep(base_delay * (2 ** attempt))
                            continue
                        else:
                            return {"success": False, "error": f"Request timed out after {timeout_seconds}s"}
            
            except (aiohttp.ClientError, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ All {max_retries} attempts failed. Last error: {e}")
                    return {"success": False, "error": f"Max retries exceeded: {str(e)}"}
        
        return {"success": False, "error": "Unexpected error in call_llm"}
    
    async def save_result_incrementally(self, result: Dict[str, Any], output_file: Path) -> None:
        """Save a single result incrementally to avoid data loss."""
        if not self.enable_incremental_save:
            return
            
        async with self._save_lock:
            try:
                # Ensure directory exists
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Use atomic write with temporary file
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(
                    mode='w', 
                    encoding='utf-8', 
                    dir=output_file.parent, 
                    delete=False,
                    suffix='.tmp'
                ) as temp_file:
                    # Write result as JSON line
                    json.dump(result, temp_file, ensure_ascii=False)
                    temp_file.write('\n')
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                    temp_filename = temp_file.name
                
                # Atomic move to append to the target file
                with open(output_file, 'a', encoding='utf-8') as target_file:
                    with open(temp_filename, 'r', encoding='utf-8') as temp_read:
                        target_file.write(temp_read.read())
                        target_file.flush()
                        os.fsync(target_file.fileno())
                
                # Clean up temp file
                os.unlink(temp_filename)
                
                # print(f"💾 Saved result for {result.get('video_id', 'unknown')}")
                
            except Exception as e:
                print(f"⚠️ Warning: Failed to save incremental result: {e}")
                # Try to clean up temp file if it exists
                try:
                    if 'temp_filename' in locals():
                        os.unlink(temp_filename)
                except:
                    pass
    
    async def load_existing_results(self, output_file: Path) -> List[Dict[str, Any]]:
        """Load existing results to resume processing."""
        if output_file.exists():
            try:
                existing_results = load_jsonl(output_file)
                print(f"📂 Found {len(existing_results)} existing results, resuming from checkpoint")
                return existing_results
            except Exception as e:
                print(f"⚠️ Warning: Could not load existing results: {e}")
        return []
    
    def get_processed_keys(self, existing_results: List[Dict[str, Any]], key_field: str = "video_id") -> set:
        """Get set of already processed keys to avoid reprocessing."""
        return {result.get(key_field, "") for result in existing_results}
    
    @abstractmethod
    async def process(self, *args, **kwargs) -> None:
        """Main processing method to be implemented by each stage."""
        pass
    
    def get_output_file(self, filename: str) -> Path:
        """Get output file path."""
        return self.output_dir / filename
