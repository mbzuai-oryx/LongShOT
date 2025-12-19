#!/usr/bin/env python3
"""
Question types generation stage for mapping scenario-task combinations to appropriate question types.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm

from stages.base import BaseStage
from utils import load_schema, load_prompt, save_jsonl, load_jsonl


class QuestionTypesStage(BaseStage):
    """Stage for generating question type mappings from evaluation scenarios."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load schema and prompt
        self.question_types_schema = load_schema("question_types_schema.json")
        self.question_types_prompt_template = load_prompt("question_types_prompt.txt")
        
        # Input and output files
        self.scenarios_file = self.get_output_file("scenarios.jsonl")
        self.question_types_file = self.get_output_file("question_types.jsonl")
        
        # Video metadata directory
        self.video_metadata_dir = Path(self.config['processing']['input_directory'])
    
    async def process(self) -> None:
        """Process question types generation stage."""
        # Load analysis results
        analysis_results = load_jsonl(self.scenarios_file)
        if not analysis_results:
            print("❌ No analysis results found")
            return
        
        print(f"🎯 Processing question type generation for {len(analysis_results)} videos...")
        
        # Load existing results for resume capability
        existing_results = await self.load_existing_results(self.question_types_file)
        processed_scenarios = {r.get("scenario_id", "") for r in existing_results}
        
        # Prepare tasks (one per scenario for all tasks together)
        tasks = []
        
        # Check debug sample limit
        debug_samples = self.config.get('debug_samples')
        tasks_created = 0
        
        for video_result in analysis_results:
            video_id = video_result.get("video_id", "")
            scenarios = video_result.get("scenarios", [])
            
            for scenario_data in scenarios:
                if debug_samples and tasks_created >= debug_samples:
                    break
                    
                scenario_id = scenario_data.get("scenario_id", "")
                
                # Create a task for the entire scenario with all its tasks
                if scenario_id not in processed_scenarios:
                    task_item = self._generate_question_types_for_scenario(
                        video_id, scenario_data
                    )
                    tasks.append(task_item)
                    tasks_created += 1
                    
                    if debug_samples and tasks_created >= debug_samples:
                        break
            
            if debug_samples and tasks_created >= debug_samples:
                break
        
        if not tasks:
            print("✅ All scenario-task pairs already processed!")
            return
        
        # Show debug info if applicable
        if debug_samples:
            print(f"🐛 Debug mode: created {len(tasks)} scenario tasks (limited to {debug_samples})")
        
        # Execute tasks
        print(f"🚀 Processing {len(tasks)} remaining scenarios (skipped {len(processed_scenarios)} already completed)...")
        start_time = time.time()
        
        results = []
        successful_count = 0
        failed_count = 0
        
        # Create incremental save file for individual scenario-task pairs
        incremental_file = self.get_output_file("question_types_incremental.jsonl")
        
        with tqdm(total=len(tasks), desc="Generating question types", unit="scenario") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                
                # Save individual results incrementally
                if result.get("success", False):
                    successful_count += 1
                    await self.save_result_incrementally(result, incremental_file)
                else:
                    failed_count += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    "Success": successful_count,
                    "Failed": failed_count
                })
        
        # Save results directly without complex grouping
        all_results = existing_results + [r for r in results if r.get("success", False)]
        save_jsonl(all_results, self.question_types_file)
        
        processing_time = time.time() - start_time
        total_question_types = sum(
            len(task.get("question_types", [])) 
            for result in all_results 
            for task in result.get("tasks", [])
        )
        print(f"\n✅ Question types generation complete: {successful_count}/{len(results)} successful")
        print(f"📊 Total question types generated: {total_question_types}")  
        print(f"⚡ Processing rate: {len(tasks)/processing_time:.1f} scenarios/second")
    
    def _load_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """Load original video metadata from input directory."""
        try:
            video_file = self.video_metadata_dir / f"{video_id}.json"
            
            if video_file.exists():
                with open(video_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {"video_id": video_id}
        except Exception as e:
            print(f"⚠️ Warning: Failed to load video metadata for {video_id}: {e}")
            return {"video_id": video_id}

    async def _generate_question_types_for_scenario(
        self, video_id: str, 
        scenario_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate question types for all tasks in a scenario."""
        try:
            scenario = scenario_data.get("scenario", "")
            scenario_id = scenario_data.get("scenario_id", "")
            tasks = scenario_data.get("tasks", [])
            
            # Extract task strings for LLM prompt
            task_strings = [task.get("task", "") if isinstance(task, dict) else task for task in tasks]
            
            # Load full video metadata
            video_metadata = self._load_video_metadata(video_id)
            video_metadata_str = json.dumps(video_metadata, indent=2, ensure_ascii=False)
            
            scenario_with_tasks_str = json.dumps({
                "scenario": scenario,
                "tasks": task_strings
            }, indent=2, ensure_ascii=False)
            
            prompt = self.question_types_prompt_template.replace("{video_metadata}", video_metadata_str)
            prompt = prompt.replace("{scenario_with_tasks}", scenario_with_tasks_str)
            
            # Call LLM
            response = await self.call_llm(prompt, self.question_types_schema, temperature=0.1)
            
            if response["success"]:
                question_type_data = response["data"]
                llm_tasks = question_type_data.get("tasks", [])
                
                # Merge LLM response with original task data and add question type IDs
                enriched_tasks = []
                for original_task in tasks:
                    # Get original task info
                    if isinstance(original_task, dict):
                        task_id = original_task.get("task_id", "")
                        task_name = original_task.get("task", "")
                    else:
                        task_id = ""
                        task_name = original_task
                    
                    # Find matching task in LLM response
                    llm_task_data = None
                    for llm_task in llm_tasks:
                        if llm_task.get("task") == task_name:
                            llm_task_data = llm_task
                            break
                    
                    question_types = llm_task_data.get("question_types", []) if llm_task_data else []
                    
                    # Add question type IDs  
                    enriched_question_types = []
                    for qt_idx, qt in enumerate(question_types):
                        qt_id = f"{task_id}_qt{qt_idx+1}"
                        enriched_question_types.append({
                            "question_type_id": qt_id,
                            "question_type": qt
                        })
                    
                    enriched_tasks.append({
                        "task_id": task_id,
                        "task": task_name,
                        "question_types": enriched_question_types
                    })
                
                return {
                    "video_id": video_id,
                    "scenario": question_type_data.get("scenario", scenario),
                    "scenario_id": scenario_id,
                    "tasks": enriched_tasks,
                    "success": True
                }
            else:
                # Create fallback structure with IDs
                fallback_tasks = []
                for original_task in tasks:
                    if isinstance(original_task, dict):
                        task_id = original_task.get("task_id", "")
                        task_name = original_task.get("task", "")
                    else:
                        task_id = ""
                        task_name = original_task
                    
                    fallback_tasks.append({
                        "task_id": task_id,
                        "task": task_name,
                        "question_types": []
                    })
                
                return {
                    "video_id": video_id,
                    "scenario": scenario,
                    "scenario_id": scenario_id,
                    "tasks": fallback_tasks,
                    "success": False,
                    "error": response["error"]
                }
                
        except Exception as e:
            # Create fallback structure with IDs for exception case
            scenario = scenario_data.get("scenario", "")
            scenario_id = scenario_data.get("scenario_id", "")
            tasks = scenario_data.get("tasks", [])
            
            fallback_tasks = []
            for original_task in tasks:
                if isinstance(original_task, dict):
                    task_id = original_task.get("task_id", "")
                    task_name = original_task.get("task", "")
                else:
                    task_id = ""
                    task_name = original_task
                
                fallback_tasks.append({
                    "task_id": task_id,
                    "task": task_name,
                    "question_types": []
                })
            
            return {
                "video_id": video_id,
                "scenario": scenario,
                "scenario_id": scenario_id,
                "tasks": fallback_tasks,
                "success": False,
                "error": str(e)
            }
    
    
    def get_results_file(self):
        """Get the output file path for this stage."""
        return self.question_types_file