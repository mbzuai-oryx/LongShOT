#!/usr/bin/env python3
"""
Question generation stage for creating questions from analysis and question types data.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm

from stages.base import BaseStage
from utils import load_schema, load_prompt, save_jsonl, load_jsonl


class QuestionStage(BaseStage):
    """Stage for generating questions from analysis and question types data."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load schema and prompt
        self.question_schema = load_schema("question_schema.json")
        self.question_prompt_template = load_prompt("question_prompt.txt")
        
        # Load question types catalog
        base_dir = Path(__file__).parent.parent
        with open(base_dir / "question_types_catalog.json", 'r', encoding='utf-8') as f:
            self.question_catalog = json.load(f)
        
        # Input and output files
        self.question_types_file = self.get_output_file("question_types.jsonl")
        self.questions_file = self.get_output_file("questions.jsonl")
        
        # Video metadata directory
        self.video_metadata_dir = Path(self.config['processing']['input_directory'])
    
    async def process(self) -> None:
        """Process question generation stage."""
        # Load question types results
        question_types_results = load_jsonl(self.question_types_file)
        if not question_types_results:
            print("❌ No question types results found")
            return
        
        print(f"🎯 Processing question generation for {len(question_types_results)} scenario-task combinations...")
        
        # Load existing results for resume capability
        existing_results = await self.load_existing_results(self.questions_file)
        processed_task_ids = {r.get("task_id", "") for r in existing_results}
        
        # Prepare tasks (one per scenario-task combination)
        tasks = []
        
        # Check debug sample limit
        debug_samples = self.config.get('debug_samples')
        tasks_created = 0
        
        for qt_result in question_types_results:
            scenario_id = qt_result.get("scenario_id", "")
            scenario_tasks = qt_result.get("tasks", [])
            
            for task_data in scenario_tasks:
                if debug_samples and tasks_created >= debug_samples:
                    break
                    
                task_id = task_data.get("task_id", "")
                
                if task_id not in processed_task_ids:
                    task = self._generate_questions_for_scenario_task(
                        qt_result, task_data
                    )
                    tasks.append(task)
                    tasks_created += 1
                    
            if debug_samples and tasks_created >= debug_samples:
                break
        
        if not tasks:
            print("✅ All scenario-task combinations already processed!")
            return
        
        # Show debug info if applicable
        if debug_samples:
            print(f"🐛 Debug mode: created {len(tasks)} scenario tasks (limited to {debug_samples})")
        
        # Execute tasks
        print(f"🚀 Processing {len(tasks)} remaining scenario-task combinations (skipped {len(processed_task_ids)} already completed)...")
        start_time = time.time()
        
        results = []
        successful_count = 0
        failed_count = 0
        
        # Create incremental save file for individual scenario-task combinations
        incremental_file = self.get_output_file("questions_incremental.jsonl")
        
        with tqdm(total=len(tasks), desc="Generating questions", unit="scenario-task") as pbar:
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
        save_jsonl(all_results, self.questions_file)
        
        processing_time = time.time() - start_time
        total_questions = sum(
            len(r.get("questions", {}).get("single_turn_questions", []))
            + sum(len(seq) for seq in r.get("questions", {}).get("multi_turn_questions", []))
            for r in all_results
        )
        print(f"\n✅ Question generation complete: {successful_count}/{len(results)} successful")
        print(f"📊 Total questions generated: {total_questions}")
        print(f"⚡ Processing rate: {len(tasks)/processing_time:.1f} scenario-tasks/second")
    
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

    async def _generate_questions_for_scenario_task(
        self, qt_result: Dict[str, Any], task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate questions for a single scenario-task combination."""
        try:
            video_id = qt_result.get("video_id", "")
            scenario = qt_result.get("scenario", "")
            scenario_id = qt_result.get("scenario_id", "")
            task_id = task_data.get("task_id", "")
            task = task_data.get("task", "")
            question_types = task_data.get("question_types", [])
            
            # Load full video metadata
            video_metadata = self._load_video_metadata(video_id)
            video_metadata_str = json.dumps(video_metadata, indent=2, ensure_ascii=False)
            
            scenario_context = json.dumps({
                "scenario": scenario,
                "task": task
            }, indent=2, ensure_ascii=False)
            
            # Create enriched question types with full catalog information
            enriched_question_types = self._create_enriched_question_types(question_types)
            enriched_question_types_str = json.dumps(enriched_question_types, indent=2, ensure_ascii=False)
            
            prompt = self.question_prompt_template.replace("{video_metadata}", video_metadata_str)
            prompt = prompt.replace("{scenario}", scenario_context)
            prompt = prompt.replace("{enriched_question_types}", enriched_question_types_str)
            
            # Call LLM
            response = await self.call_llm(prompt, self.question_schema, temperature=0.7)
            
            if response["success"]:
                questions_data = response["data"]
                
                # Add internal tracking IDs to the questions
                enriched_questions = self._add_internal_tracking_ids(
                    questions_data, video_id, scenario_id, task_id
                )
                
                return {
                    "video_id": video_id,
                    "scenario": scenario,
                    "scenario_id": scenario_id,
                    "task": task,
                    "task_id": task_id,
                    "question_types": question_types,
                    "questions": enriched_questions,
                    "success": True
                }
            else:
                return {
                    "video_id": video_id,
                    "scenario": scenario,
                    "scenario_id": scenario_id,
                    "task": task,
                    "task_id": task_id,
                    "question_types": question_types,
                    "questions": {"single_turn_questions": [], "multi_turn_questions": []},
                    "success": False,
                    "error": response["error"]
                }
                
        except Exception as e:
            video_id = qt_result.get("video_id", "")
            scenario = qt_result.get("scenario", "")
            scenario_id = qt_result.get("scenario_id", "")
            
            return {
                "video_id": video_id,
                "scenario": scenario,
                "scenario_id": scenario_id,
                "task": task_data.get("task", ""),
                "task_id": task_data.get("task_id", ""),
                "question_types": task_data.get("question_types", []),
                "questions": {"single_turn_questions": [], "multi_turn_questions": []},
                "success": False,
                "error": str(e)
            }
    
    def _create_enriched_question_types(self, question_types) -> list:
        """Create enriched question types by merging scenario-specific types with catalog definitions."""
        enriched_types = []
        
        for qt in question_types:
            qt_name = qt.get("question_type", "").strip()
            
            # Find matching definition in the catalog
            catalog_definition = self._find_catalog_definition(qt_name)
            
            # Create enriched question type entry
            enriched_qt = {
                "question_type": qt_name
            }
            
            # Add catalog information if found
            if catalog_definition:
                enriched_qt.update({
                    "description": catalog_definition.get("description", ""),
                    "example": catalog_definition.get("example", ""),
                })
            
            enriched_types.append(enriched_qt)
        
        return enriched_types
    
    def _find_catalog_definition(self, question_type_name: str) -> Dict[str, Any]:
        """Find the catalog definition for a given question type name."""
        if not question_type_name:
            return {}
        
        # Search through all categories in the catalog
        for category, types_list in self.question_catalog.get("question_types", {}).items():
            for qt_def in types_list:
                qt_id = qt_def.get("id", "")
                qt_name = qt_def.get("name", "")
                qt_description = qt_def.get("description", "")
                
                # Check if this matches the question type we're looking for
                if (qt_id == question_type_name or 
                    qt_name == question_type_name or
                    question_type_name.lower() in qt_description.lower() or
                    qt_id.lower() in question_type_name.lower()):
                    
                    # Return the definition with category info
                    definition = qt_def.copy()
                    definition["category"] = category
                    return definition
        
        return {}
    
    def _add_internal_tracking_ids(
        self, questions_data: Dict[str, Any], 
        video_id: str, scenario_id: str, task_id: str
    ) -> Dict[str, Any]:
        """Add internal tracking IDs to questions for post-processing."""
        enriched_data = questions_data.copy()
        
        # Add IDs to single-turn questions
        for i, question in enumerate(enriched_data.get("single_turn_questions", [])):
            question_id = f"{task_id}_q{i+1}"
            question["question_id"] = question_id
            question["video_id"] = video_id
            question["scenario_id"] = scenario_id
            question["task_id"] = task_id
        
        # Add IDs to multi-turn questions
        for seq_i, question_sequence in enumerate(enriched_data.get("multi_turn_questions", [])):
            for turn_i, question in enumerate(question_sequence):
                question_id = f"{task_id}_mq{seq_i+1}_t{turn_i+1}"
                question["question_id"] = question_id
                question["video_id"] = video_id
                question["scenario_id"] = scenario_id
                question["task_id"] = task_id
        
        return enriched_data
    
    
    def get_results_file(self):
        """Get the output file path for this stage."""
        return self.questions_file
