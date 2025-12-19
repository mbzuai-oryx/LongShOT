#!/usr/bin/env python3
"""
Video analysis stage for generating scenarios from video metadata.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from tqdm import tqdm

from stages.base import BaseStage
from utils import load_schema, load_prompt, save_jsonl


class AnalysisStage(BaseStage):
    """Stage for analyzing videos and generating evaluation scenarios."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load schema and prompt
        self.analysis_schema = load_schema("analysis_schema.json")
        self.analysis_prompt_template = load_prompt("analysis_prompt.txt")
        
        # Output file
        self.scenarios_file = self.get_output_file("scenarios.jsonl")
    
    async def process(self, video_files: List[Path]) -> None:
        """Process video analysis stage."""
        print(f"🔍 Preparing analysis for {len(video_files)} videos...")
        
        # Load existing results for resume capability
        existing_results = await self.load_existing_results(self.scenarios_file)
        processed_video_ids = self.get_processed_keys(existing_results, "video_id")
        
        # Filter out already processed videos
        remaining_files = []
        for video_file in video_files:
            # Get video_id from file content or filename
            try:
                with open(video_file, 'r', encoding='utf-8') as f:
                    video_data = json.load(f)
                video_id = video_data.get("video_id", video_file.stem)
            except:
                video_id = video_file.stem
            
            if video_id not in processed_video_ids:
                remaining_files.append(video_file)
        
        if not remaining_files:
            print("✅ All videos already processed!")
            return
        
        print(f"📊 Processing {len(remaining_files)} remaining videos (skipped {len(video_files) - len(remaining_files)} already completed)")
        
        # Apply debug sample limit if specified
        debug_samples = self.config.get('debug_samples')
        if debug_samples and len(remaining_files) > debug_samples:
            print(f"🐛 Debug mode: limiting to {debug_samples} videos (from {len(remaining_files)})")
            remaining_files = remaining_files[:debug_samples]
        
        # Prepare tasks
        tasks = []
        for i, video_file in enumerate(remaining_files):
            task = self._analyze_single_video(i, video_file)
            tasks.append(task)
        
        # Execute all tasks with progress tracking
        print(f"🚀 Processing {len(tasks)} videos...")
        start_time = time.time()
        
        results = []
        successful_count = 0
        failed_count = 0
        
        with tqdm(total=len(tasks), desc="Analyzing videos", unit="video") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                
                # Save successful results incrementally
                if result.get("success", False):
                    successful_count += 1
                    await self.save_result_incrementally(result, self.scenarios_file)
                else:
                    failed_count += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    "Success": successful_count,
                    "Failed": failed_count
                })
        
        processing_time = time.time() - start_time
        
        # Final consolidation - merge with existing results and save final file
        all_results = existing_results + [r for r in results if r.get("success", False)]
        save_jsonl(all_results, self.scenarios_file)
        
        total_scenarios = sum(len(r.get('scenarios', [])) for r in all_results)
        print(f"\n✅ Analysis complete: {successful_count}/{len(results)} successful (total: {len(all_results)} videos)")
        print(f"📊 Total scenarios generated: {total_scenarios}")
        print(f"⚡ Processing rate: {len(results)/processing_time:.1f} videos/second")
    
    async def _analyze_single_video(self, video_index: int, video_file: Path) -> Dict[str, Any]:
        """Analyze a single video file."""
        try:
            # Load video data
            with open(video_file, 'r', encoding='utf-8') as f:
                video_data = json.load(f)
            
            # Create analysis prompt
            prompt = self.analysis_prompt_template.replace("{video_metadata}", 
                                                          json.dumps(video_data, indent=2, ensure_ascii=False))
            
            # Call LLM
            response = await self.call_llm(prompt, self.analysis_schema, temperature=0.1)
            
            if response["success"]:
                scenarios = response["data"].get("scenarios", [])
                video_id_clean = video_data.get("video_id", video_file.stem)
                
                # Add hierarchical IDs for tree structure
                for i, scenario in enumerate(scenarios):
                    scenario_id = f"{video_id_clean}_s{i+1}"
                    scenario["scenario_id"] = scenario_id
                    
                    # Add task IDs within each scenario
                    for j, task in enumerate(scenario.get("tasks", [])):
                        task_id = f"{scenario_id}_t{j+1}"
                        scenario["tasks"][j] = {
                            "task_id": task_id,
                            "task": task
                        }
                
                return {
                    "video_id": video_id_clean,
                    "scenarios": scenarios,
                    "success": True
                }
            else:
                return {
                    "video_id": video_file.stem,
                    "scenarios": [],
                    "success": False,
                    "error": response["error"]
                }
                
        except Exception as e:
            return {
                "video_id": video_file.stem,
                "scenarios": [],
                "success": False,
                "error": str(e)
            }
    
    def get_results_file(self) -> Path:
        """Get the output file path for this stage."""
        return self.scenarios_file
