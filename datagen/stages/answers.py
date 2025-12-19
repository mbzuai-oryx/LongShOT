#!/usr/bin/env python3
"""
Answer generation stage for creating reference answers to questions.
"""

import asyncio
import json
import time
from typing import Dict, Any

from tqdm import tqdm

from stages.base import BaseStage
from utils import load_schema, load_prompt, save_jsonl, load_jsonl


class AnswerStage(BaseStage):
    """Stage for generating reference answers to questions."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load schema and prompt
        self.answer_schema = load_schema("answer_schema.json")
        self.answer_prompt_template = load_prompt("answer_prompt.txt")
        
        # Input and output files
        self.questions_file = self.get_output_file("questions.jsonl")
        self.answers_file = self.get_output_file("answers.jsonl")
    
    async def process(self) -> None:
        """Process answer generation stage with API-level progress tracking."""
        # Load question results
        question_results = load_jsonl(self.questions_file)
        if not question_results:
            print("❌ No question results found")
            return
        
        print(f"💬 Processing answer generation for {len(question_results)} scenario-task combinations...")
        
        # Load existing results for resume capability
        existing_results = await self.load_existing_results(self.answers_file)
        processed_task_ids = {r.get("task_id", "") for r in existing_results}
        
        # Prepare tasks and PRE-CALCULATE TOTAL API REQUESTS
        tasks = []
        total_api_requests = 0
        
        # Check debug sample limit
        debug_samples = self.config.get('debug_samples')
        tasks_created = 0
        
        for question_result in question_results:
            if debug_samples and tasks_created >= debug_samples:
                break
                
            task_id = question_result.get("task_id", "")
            
            if task_id not in processed_task_ids:
                # Count expected API requests for this task
                questions = question_result.get("questions", {})
                
                # Count single-turn questions (1 API request each)
                single_turn_count = len(questions.get("single_turn_questions", []))
                
                # Count multi-turn turns (1 API request per turn)
                multi_turn_count = 0
                for sequence in questions.get("multi_turn_questions", []):
                    multi_turn_count += len(sequence)
                
                total_api_requests += single_turn_count + multi_turn_count
                
                task = self._generate_answers_for_scenario_task(question_result)
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
        incremental_file = self.get_output_file("answers_incremental.jsonl")
        
        # Use shared progress tracking across all async tasks
        self._shared_pbar = tqdm(total=total_api_requests, desc="Generating answers", unit="API request")
        
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
                
                # Progress updates are handled inside _generate_answers_for_scenario_task
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
        save_jsonl(all_results, self.answers_file)
        
        processing_time = time.time() - start_time
        total_answers = sum(
            len(r.get("answers", {}).get("single_turn_answers", []))
            + sum(len(seq) for seq in r.get("answers", {}).get("multi_turn_answers", []))
            for r in all_results
        )
        print(f"\n✅ Answer generation complete: {successful_count}/{len(results)} successful")
        print(f"📊 Total answers generated: {total_answers}")
        print(f"⚡ Processing rate: {len(tasks)/processing_time:.1f} scenario-tasks/second")
    
    def _load_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """Load original video metadata from input directory."""
        try:
            from pathlib import Path
            video_metadata_dir = Path(self.config['processing']['input_directory'])
            video_file = video_metadata_dir / f"{video_id}.json"
            
            if video_file.exists():
                with open(video_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {"video_id": video_id}
        except Exception as e:
            print(f"⚠️ Warning: Failed to load video metadata for {video_id}: {e}")
            return {"video_id": video_id}
    
    async def _generate_answers_for_scenario_task(
        self, question_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate answers for all questions in a scenario-task combination with progress updates."""
        try:
            video_id = question_result.get("video_id", "")
            scenario = question_result.get("scenario", "")
            scenario_id = question_result.get("scenario_id", "")
            task = question_result.get("task", "")
            task_id = question_result.get("task_id", "")
            question_types = question_result.get("question_types", [])
            questions = question_result.get("questions", {})
            
            # Load full video metadata
            video_metadata = self._load_video_metadata(video_id)
            video_metadata_str = json.dumps(video_metadata, indent=2, ensure_ascii=False)
            
            # Remove question_type_id from question_types and flatten it for prompt clarity and flat
            question_types = [qt.get("question_type", "") for qt in question_types]
            
            # Create task context
            task_context_str = json.dumps({
                "task": task,
                "question_types": question_types
            }, indent=2, ensure_ascii=False)
            
            # Process single-turn questions
            single_turn_answers = []
            for question in questions.get("single_turn_questions", []):
                # Create question context
                question_context_str = json.dumps({
                    "difficulty": question.get("difficulty", 3),
                    "modalities": question.get("modalities", []),
                    "key_segments": question.get("key_segments", [])
                }, indent=2, ensure_ascii=False)
                
                conversation = [{"role": "user", "content": question.get("question", "")}]
                conversation_str = json.dumps(conversation, indent=2, ensure_ascii=False)
                
                prompt = self.answer_prompt_template.replace("{video_metadata}", video_metadata_str)
                prompt = prompt.replace("{scenario}", scenario)
                prompt = prompt.replace("{task_context}", task_context_str)
                prompt = prompt.replace("{question_context}", question_context_str)
                prompt = prompt.replace("{conversation}", conversation_str)
                
                # Call LLM with progress update
                response = await self.call_llm(prompt, self.answer_schema, temperature=0.2)
                if hasattr(self, '_shared_pbar'):
                    self._shared_pbar.update(1)
                
                if response["success"] and response["data"]:
                    answer_data = response["data"][0] if response["data"] else {}
                    single_turn_answers.append({
                        **question,  # Include all question metadata
                        "answer": answer_data.get("answer", ""),
                        "status": "success"
                    })
                else:
                    single_turn_answers.append({
                        **question,
                        "answer": "",
                        "status": "failed",
                        "error": response.get("error", "No answer generated")
                    })
            
            # Process multi-turn questions
            multi_turn_answers = []
            for sequence in questions.get("multi_turn_questions", []):
                # Build conversation progressively - TRUE multi-turn conversation
                conversation_history = []
                sequence_qa = []
                
                # Get sequence context from first question
                first_question = sequence[0] if sequence else {}
                sequence_id = first_question.get("sequence_id", "")
                key_segments = first_question.get("key_segments", [])
                
                # Process each question in the sequence one by one
                for turn_idx, question_data in enumerate(sequence):
                    # Add current user question to conversation history
                    conversation_history.append({
                        "role": "user", 
                        "content": question_data.get("question", "")
                    })
                    
                    # Create question context for current turn
                    question_context_str = json.dumps({
                        "key_segments": key_segments,
                        "modalities": question_data.get("modalities", []),
                        "difficulty": question_data.get("difficulty", 3),
                    }, indent=2, ensure_ascii=False)
                    
                    # Create conversation string with full history
                    conversation_str = json.dumps(conversation_history, indent=2, ensure_ascii=False)
                    
                    # Build prompt for current turn
                    prompt = self.answer_prompt_template.replace("{video_metadata}", video_metadata_str)
                    prompt = prompt.replace("{scenario}", scenario)
                    prompt = prompt.replace("{task_context}", task_context_str)
                    prompt = prompt.replace("{question_context}", question_context_str)
                    prompt = prompt.replace("{conversation}", conversation_str)
                    
                    # Call LLM for current question with progress update
                    response = await self.call_llm(prompt, self.answer_schema, temperature=0.2)
                    if hasattr(self, '_shared_pbar'):
                        self._shared_pbar.update(1)
                    
                    if response["success"] and response["data"]:
                        answer_data = response["data"][0] if response["data"] else {}
                        answer_text = answer_data.get("answer", "")
                        
                        # Add assistant response to conversation history for next turn
                        conversation_history.append({
                            "role": "assistant",
                            "content": answer_text
                        })
                        
                        # Store successful Q&A pair
                        sequence_qa.append({
                            **question_data,  # Include all question metadata
                            "answer": answer_text,
                            "status": "success"
                        })
                    else:
                        # Failed - store error and stop the conversation
                        sequence_qa.append({
                            **question_data,
                            "answer": "",
                            "status": "failed",
                            "error": response.get("error", "No answer generated")
                        })
                        # Break the conversation flow on failure
                        break
                
                multi_turn_answers.append(sequence_qa)
            
            return {
                "video_id": video_id,
                "scenario": scenario,
                "scenario_id": scenario_id,
                "task": task,
                "task_id": task_id,
                "question_types": question_types,
                "answers": {
                    "single_turn_answers": single_turn_answers,
                    "multi_turn_answers": multi_turn_answers
                },
                "success": True
            }
            
        except Exception as e:
            return {
                "video_id": question_result.get("video_id", ""),
                "scenario": question_result.get("scenario", ""),
                "scenario_id": question_result.get("scenario_id", ""),
                "task": question_result.get("task", ""),
                "task_id": question_result.get("task_id", ""),
                "question_types": question_result.get("question_types", []),
                "answers": {"single_turn_answers": [], "multi_turn_answers": []},
                "success": False,
                "error": str(e)
            }
    
    
    def get_results_file(self):
        """Get the output file path for this stage."""
        return self.answers_file
