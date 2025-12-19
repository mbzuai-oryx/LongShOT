#!/usr/bin/env python3
"""
Criteria generation stage for creating evaluation criteria for Q&A pairs.
"""

import asyncio
import json
import time
from typing import Dict, Any

from tqdm import tqdm

from stages.base import BaseStage
from utils import load_schema, load_prompt, save_jsonl, load_jsonl


class CriteriaStage(BaseStage):
    """Stage for generating evaluation criteria for Q&A pairs."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load schema and prompt
        self.criteria_schema = load_schema("criteria_schema.json")
        self.criteria_prompt_template = load_prompt("criteria_prompt.txt")
        
        # Input and output files
        self.answers_file = self.get_output_file("answers.jsonl")
        self.criteria_file = self.get_output_file("criteria.jsonl")
    
    async def process(self) -> None:
        """Process criteria generation stage with API-level progress tracking."""
        # Load answer results
        answer_results = load_jsonl(self.answers_file)
        if not answer_results:
            print("❌ No answer results found")
            return
        
        print(f"📋 Processing criteria generation for {len(answer_results)} scenario-task combinations...")
        
        # Load existing results for resume capability
        existing_results = await self.load_existing_results(self.criteria_file)
        processed_task_ids = {r.get("task_id", "") for r in existing_results}
        
        # Prepare tasks and PRE-CALCULATE TOTAL API REQUESTS
        tasks = []
        total_api_requests = 0
        
        # Check debug sample limit
        debug_samples = self.config.get('debug_samples')
        tasks_created = 0
        
        for answer_result in answer_results:
            if debug_samples and tasks_created >= debug_samples:
                break
                
            task_id = answer_result.get("task_id", "")
            
            if task_id not in processed_task_ids:
                # Count expected API requests (one per successful Q&A pair)
                answers = answer_result.get("answers", {})
                
                # Count single-turn Q&A pairs that have successful answers
                single_turn_qa_count = len([
                    qa for qa in answers.get("single_turn_answers", [])
                    if qa.get("status") == "success"
                ])
                
                # Count multi-turn Q&A pairs that have successful answers
                multi_turn_qa_count = 0
                for sequence in answers.get("multi_turn_answers", []):
                    multi_turn_qa_count += len([
                        qa for qa in sequence
                        if qa.get("status") == "success"
                    ])
                
                total_api_requests += single_turn_qa_count + multi_turn_qa_count
                
                task = self._generate_criteria_for_scenario_task(answer_result)
                tasks.append(task)
                tasks_created += 1
        
        if not tasks:
            print("✅ All scenario-task combinations already processed!")
            return
        
        # Show debug info if applicable
        if debug_samples:
            print(f"🐛 Debug mode: created {len(tasks)} scenario-task combinations (limited to {debug_samples})")
        
        # Execute tasks with API-level progress tracking
        print(f"🚀 Processing {len(tasks)} remaining scenario-task combinations (skipped {len(processed_task_ids)} already completed)...")
        print(f"📊 Expected API requests: {total_api_requests}")
        start_time = time.time()
        
        results = []
        successful_count = 0
        failed_count = 0
        
        # Create incremental save file for individual scenario-task combinations
        incremental_file = self.get_output_file("criteria_incremental.jsonl")
        
        # Use shared progress tracking across all async tasks
        self._shared_pbar = tqdm(total=total_api_requests, desc="Generating criteria", unit="API request")
        
        try:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                
                # Save individual results incrementally
                if result.get("success", False):
                    successful_count += 1
                    await self.save_result_incrementally(result, incremental_file)
                else:
                    failed_count += 1
                
                # Progress updates are handled inside _generate_criteria_for_scenario_task
                # Just update the postfix information here
                self._shared_pbar.set_postfix({
                    "Tasks": f"{successful_count + failed_count}/{len(tasks)}",
                    "Success": successful_count,
                    "Failed": failed_count
                })
        finally:
            self._shared_pbar.close()
        
        # Save results directly without complex grouping
        all_results = existing_results + [r for r in results if r.get("success", False)]
        save_jsonl(all_results, self.criteria_file)
        
        processing_time = time.time() - start_time
        total_criteria = sum(
            len(criteria["criteria"]) 
            for r in all_results 
            for criteria in r.get("single_turn_criteria", []) + 
                           [item for sublist in r.get("multi_turn_criteria", []) for item in sublist]
        )
        print(f"\n✅ Criteria generation complete: {successful_count}/{len(results)} successful")
        print(f"📊 Total criteria generated: {total_criteria}")
        print(f"⚡ Processing rate: {len(tasks)/processing_time:.1f} scenario-tasks/second")
    
    async def _generate_criteria_for_scenario_task(
        self, answer_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate criteria for all Q&A pairs in a scenario-task combination."""
        try:
            video_id = answer_result.get("video_id", "")
            scenario = answer_result.get("scenario", "")
            scenario_id = answer_result.get("scenario_id", "")
            task = answer_result.get("task", "")
            task_id = answer_result.get("task_id", "")
            question_types = answer_result.get("question_types", [])
            answers = answer_result.get("answers", {})
            
            # Process single-turn Q&A pairs
            single_turn_criteria = []
            for qa_pair in answers.get("single_turn_answers", []):
                if qa_pair.get("status") == "success":
                    # Create criteria prompt for this Q&A pair
                    qa_context = {
                        "question": qa_pair.get("question", ""),
                        "answer": qa_pair.get("answer", ""),
                    }
                    qa_pair_str = json.dumps(qa_context, indent=2, ensure_ascii=False)
                    prompt = self.criteria_prompt_template.replace("{Question_answer_pair}", qa_pair_str)
                    
                    # Call LLM for criteria generation with progress update
                    response = await self.call_llm(prompt, self.criteria_schema, temperature=0.2)
                    if hasattr(self, '_shared_pbar'):
                        self._shared_pbar.update(1)
                    
                    if response["success"]:
                        criteria_data = response["data"]
                        processed_criteria = self._process_criteria_with_weights(criteria_data)
                        
                        # Remove redundant metadata (video_id, scenario_id, task_id already at top level)
                        qa_clean = {k: v for k, v in qa_pair.items() 
                                  if k not in ["video_id", "scenario_id", "task_id"]}
                        
                        single_turn_criteria.append({
                            **qa_clean,
                            "criteria": processed_criteria,
                            "status": "success"
                        })
                    else:
                        qa_clean = {k: v for k, v in qa_pair.items() 
                                  if k not in ["video_id", "scenario_id", "task_id"]}
                        single_turn_criteria.append({
                            **qa_clean,
                            "criteria": [],
                            "status": "failed",
                            "error": response.get("error", "No criteria generated")
                        })
                else:
                    # Skip failed Q&A pairs
                    qa_clean = {k: v for k, v in qa_pair.items() 
                              if k not in ["video_id", "scenario_id", "task_id"]}
                    single_turn_criteria.append({
                        **qa_clean,
                        "criteria": [],
                        "status": "skipped",
                        "error": "Q&A pair failed in answers stage"
                    })
            
            # Process multi-turn Q&A pairs
            multi_turn_criteria = []
            for sequence in answers.get("multi_turn_answers", []):
                sequence_criteria = []
                for qa_pair in sequence:
                    if qa_pair.get("status") == "success":
                        # Create criteria prompt for this Q&A pair
                        qa_context = {
                            "question": qa_pair.get("question", ""),
                            "answer": qa_pair.get("answer", ""),
                            "scenario": scenario,
                            "task": task,
                            "question_types": question_types
                        }
                        qa_pair_str = json.dumps(qa_context, indent=2, ensure_ascii=False)
                        prompt = self.criteria_prompt_template.replace("{Question_answer_pair}", qa_pair_str)
                        
                        # Call LLM for criteria generation with progress update
                        response = await self.call_llm(prompt, self.criteria_schema, temperature=0.2)
                        if hasattr(self, '_shared_pbar'):
                            self._shared_pbar.update(1)
                        
                        if response["success"]:
                            criteria_data = response["data"]
                            processed_criteria = self._process_criteria_with_weights(criteria_data)
                            
                            qa_clean = {k: v for k, v in qa_pair.items() 
                                      if k not in ["video_id", "scenario_id", "task_id"]}
                            sequence_criteria.append({
                                **qa_clean,
                                "criteria": processed_criteria,
                                "status": "success"
                            })
                        else:
                            qa_clean = {k: v for k, v in qa_pair.items() 
                                      if k not in ["video_id", "scenario_id", "task_id"]}
                            sequence_criteria.append({
                                **qa_clean,
                                "criteria": [],
                                "status": "failed",
                                "error": response.get("error", "No criteria generated")
                            })
                    else:
                        # Skip failed Q&A pairs
                        qa_clean = {k: v for k, v in qa_pair.items() 
                                  if k not in ["video_id", "scenario_id", "task_id"]}
                        sequence_criteria.append({
                            **qa_clean,
                            "criteria": [],
                            "status": "skipped",
                            "error": "Q&A pair failed in answers stage"
                        })
                
                multi_turn_criteria.append(sequence_criteria)
            
            return {
                "video_id": video_id,
                "scenario": scenario,
                "scenario_id": scenario_id,
                "task": task,
                "task_id": task_id,
                "single_turn_criteria": single_turn_criteria,
                "multi_turn_criteria": multi_turn_criteria,
                "success": True
            }
            
        except Exception as e:
            return {
                "video_id": answer_result.get("video_id", ""),
                "scenario": answer_result.get("scenario", ""),
                "scenario_id": answer_result.get("scenario_id", ""),
                "task": answer_result.get("task", ""),
                "task_id": answer_result.get("task_id", ""),
                "single_turn_criteria": [],
                "multi_turn_criteria": [],
                "success": False,
                "error": str(e)
            }
    


    def _process_criteria_with_weights(self, criteria_data):
        """Add weights and is_penalty flags based on category."""
        processed_criteria = []
        
        # Weight mapping by category
        category_weights = {
            "high_priority": 5,
            "medium_priority": 3,
            "low_priority": 1,
            "penalty": -5  # Default penalty weight
        }
        
        for criterion in criteria_data:
            category = criterion.get("category", "medium_priority")
            is_penalty = category == "penalty"
            weight = category_weights.get(category, 3)
            
            processed_criteria.append({
                "name": criterion["name"],
                "description": criterion["description"],
                "category": category,
                "weight": weight,
                "is_penalty": is_penalty
            })
        
        return processed_criteria
    
    def get_results_file(self):
        """Get the output file path for this stage."""
        return self.criteria_file
