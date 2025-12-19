#!/usr/bin/env python3
"""
Dataset finalization stage for creating individual training samples.
"""

from typing import Dict, Any, List
import json
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

from stages.base import BaseStage
from utils import save_jsonl, load_jsonl


class FinalizationStage(BaseStage):
    """Stage for finalizing dataset into individual training samples."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Input and output files (scenarios.jsonl no longer needed)
        self.questions_file = self.get_output_file("questions.jsonl")
        self.answers_file = self.get_output_file("answers.jsonl")
        self.criteria_file = self.get_output_file("criteria.jsonl")
        self.final_dataset_file = self.get_output_file("final_dataset.jsonl")
        self.clean_dataset_file = self.get_output_file("clean_dataset.jsonl")
        self.analysis_file = self.get_output_file("dataset_analysis.json")
        
        # Video metadata directory - resolve relative path from project root
        input_dir = self.config['processing']['input_directory']
        if not input_dir.startswith('/'):
            # Convert relative path to absolute from project root
            self.video_metadata_dir = Path(__file__).parent.parent / input_dir.lstrip('./')
        else:
            self.video_metadata_dir = Path(input_dir)
        
        # Extract source folder name for samples
        self.source_videos = self.video_metadata_dir.name
    
    async def process(self) -> None:
        """Finalize dataset into individual samples."""
        print("🎯 Starting dataset finalization...")
        
        # Load all data
        questions_data = load_jsonl(self.questions_file)
        answers_data = load_jsonl(self.answers_file)
        criteria_data = load_jsonl(self.criteria_file)
        
        if not all([questions_data, answers_data, criteria_data]):
            print("❌ Missing data files for finalization")
            return
        
        # Create individual samples
        samples = self._create_individual_samples(questions_data, answers_data, criteria_data)
        
        # Save final dataset with IDs
        save_jsonl(samples, self.final_dataset_file)
        
        # Create clean samples without internal IDs
        clean_samples = self._create_clean_samples(samples)
        save_jsonl(clean_samples, self.clean_dataset_file)
        
        # Generate and save comprehensive analysis
        analysis = self._generate_dataset_analysis(samples, clean_samples)
        with open(self.analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Print summary
        single_turn_count = sum(1 for s in samples if s['sample_type'] == 'single_turn')
        multi_turn_count = sum(1 for s in samples if s['sample_type'] == 'multi_turn')
        total_criteria = sum(s.get('criteria_count', 0) for s in samples)
        unique_tasks = len(set(s.get('task', '') for s in samples))
        
        print(f"\n✅ Dataset finalization completed successfully!")
        print(f"📊 Final dataset statistics:")
        print(f"   • Total samples: {len(samples)}")
        print(f"   • Single-turn samples: {single_turn_count}")
        print(f"   • Multi-turn samples: {multi_turn_count}")
        print(f"   • Total criteria: {total_criteria}")
        print(f"   • Average criteria per sample: {total_criteria/len(samples):.1f}")
        print(f"   • Unique videos: {len(set(s.get('video_id') for s in samples))}")
        print(f"   • Unique tasks: {unique_tasks}")
        print(f"📄 Dataset with IDs saved to: {self.final_dataset_file}")
        print(f"🧹 Clean dataset saved to: {self.clean_dataset_file}")
        print(f"📊 Analysis report saved to: {self.analysis_file}")
    
    def _create_individual_samples(self, questions_data: List[Dict],
                                  answers_data: List[Dict], criteria_data: List[Dict]) -> List[Dict[str, Any]]:
        """Create individual samples from pipeline data."""
        samples = []
        sample_counter = 0
        
        # Create mappings by task_id for direct matching
        answers_map = {item.get('task_id', ''): item for item in answers_data}
        criteria_map = {item.get('task_id', ''): item for item in criteria_data}
        
        print(f"📊 Processing {len(questions_data)} scenario-task combinations for finalization...")
        
        # Process each scenario-task combination
        for questions_record in questions_data:
            task_id = questions_record.get('task_id', '')
            
            # Get corresponding answers and criteria by task_id
            answers_record = answers_map.get(task_id, {})
            criteria_record = criteria_map.get(task_id, {})
            
            if not answers_record or not criteria_record:
                print(f"⚠️ Missing answers or criteria for task_id: {task_id}")
                continue
            
            # Load original video metadata
            video_id = questions_record.get('video_id', '')
            video_metadata = self._load_video_metadata(video_id)
            
            # Extract common info with video metadata (fields are nested under 'video' key)
            video_info = video_metadata.get('video', {})
            base_info = {
                'video_id': video_id,
                'source_videos': self.source_videos,
                'title': video_metadata.get('title', ''),
                'duration': video_metadata.get('duration', 0),
                'categories': video_metadata.get('categories', [])
            }
            
            scenario = questions_record.get('scenario', '')
            scenario_id = questions_record.get('scenario_id', '')
            task = questions_record.get('task', '')
            
            # Extract data from the new flat structure
            questions = questions_record.get('questions', {})
            answers_data = answers_record.get('answers', {})
            single_turn_answers = answers_data.get('single_turn_answers', [])
            multi_turn_answers = answers_data.get('multi_turn_answers', [])
            single_turn_criteria = criteria_record.get('single_turn_criteria', [])
            multi_turn_criteria = criteria_record.get('multi_turn_criteria', [])
            
            # Process single-turn questions - use ID-based matching for correctness
            single_turn_questions = questions.get('single_turn_questions', [])
            
            # Create ID-based lookup maps for answers and criteria
            answers_by_id = {item.get('question_id', ''): item for item in single_turn_answers}
            criteria_by_id = {item.get('question_id', ''): item for item in single_turn_criteria}
            
            for question_obj in single_turn_questions:
                question_id = question_obj.get('question_id', '') if isinstance(question_obj, dict) else ''
                
                # Find matching answer and criteria by question_id
                answer_data_match = answers_by_id.get(question_id)
                criteria_data_match = criteria_by_id.get(question_id)
                
                if not answer_data_match or not criteria_data_match:
                    print(f"⚠️ Missing answer/criteria for question_id: {question_id}")
                    continue
                
                # Skip if answer failed
                if answer_data_match.get('status') != 'success':
                    continue
                
                # Extract question text and metadata
                if isinstance(question_obj, dict):
                    question_text = question_obj.get('question', '')
                    question_metadata = {
                        "difficulty": question_obj.get('difficulty'),
                        "modalities": question_obj.get('modalities', []),
                       
                        "key_segments": question_obj.get('key_segments', []),
                      
                    }
                else:
                    # Handle legacy string format
                    question_text = question_obj
                    question_metadata = {}
                
                # Create user message with metadata
                user_message = {
                    "role": "user",
                    "content": question_text,
                    "difficulty": question_metadata.get('difficulty') or answer_data_match.get('difficulty'),
                    "modalities": question_metadata.get('modalities') or answer_data_match.get('modalities', []),
                    "key_segments": question_metadata.get('key_segments') or answer_data_match.get('key_segments', []),
                }
                
                conversation = [
                    user_message,
                    {
                        "role": "assistant",
                        "content": answer_data_match.get('answer', ''),
                        "criteria": criteria_data_match.get('criteria', [])
                    }
                ]
                
                sample = {
                    **base_info,
                    'sample_id': f"sample_{sample_counter}",
                    'sample_type': 'single_turn',
                    'scenario': scenario,
                    'scenario_id': scenario_id,
                    'task': task,
                    'task_id': task_id,
                    'conversations': conversation,
                    'criteria_count': len(criteria_data_match.get('criteria', []))
                }
                samples.append(sample)
                sample_counter += 1
                
            # Process multi-turn questions - use ID-based matching within sequences
            multi_turn_questions = questions.get('multi_turn_questions', [])
            
            for seq_idx, question_sequence in enumerate(multi_turn_questions):
                # Skip if no matching answer sequence at same position (sequences should align by position)
                if seq_idx >= len(multi_turn_answers) or seq_idx >= len(multi_turn_criteria):
                    print(f"⚠️ Missing multi-turn answer/criteria sequence at position {seq_idx} for task_id: {task_id}")
                    continue
                    
                answer_sequence = multi_turn_answers[seq_idx]
                criteria_sequence = multi_turn_criteria[seq_idx]
                
                # Create ID-based lookup for answers and criteria within this sequence
                seq_answers_by_id = {item.get('question_id', ''): item for item in answer_sequence}
                seq_criteria_by_id = {item.get('question_id', ''): item for item in criteria_sequence}
                
                conversation = []
                total_criteria = []
                
                # Build the conversation with ID-based matching within each sequence
                for question_obj in question_sequence:
                    question_id = question_obj.get('question_id', '') if isinstance(question_obj, dict) else ''
                    
                    # Find matching answer and criteria by question_id
                    answer_data_match = seq_answers_by_id.get(question_id)
                    criteria_data_match = seq_criteria_by_id.get(question_id)
                    
                    if not answer_data_match or not criteria_data_match:
                        print(f"⚠️ Missing answer/criteria for multi-turn question_id: {question_id}")
                        continue
                    # Extract question text and metadata from question object
                    if isinstance(question_obj, dict):
                        question_text = question_obj.get('question', '')
                        question_metadata = {
                            "difficulty": question_obj.get('difficulty'),
                            "modalities": question_obj.get('modalities', []),
                            "key_segments": question_obj.get('key_segments', []),
                            "conversation_role": question_obj.get('conversation_role'),
                        }
                    else:
                        # Handle legacy string format
                        question_text = question_obj
                        question_metadata = {}
                    
                    # Use original question metadata, fall back to answer metadata if missing
                    user_message = {
                        "role": "user", 
                        "content": question_text,
                        "difficulty": question_metadata.get('difficulty') or answer_data_match.get('difficulty'),
                        "modalities": question_metadata.get('modalities') or answer_data_match.get('modalities', []),
                        "key_segments": question_metadata.get('key_segments') or answer_data_match.get('key_segments', []),
                        "conversation_role": question_metadata.get('conversation_role'),
                    }
                    
                    conversation.append(user_message)
                    
                    # Get criteria for this turn using ID-based matching
                    turn_criteria = criteria_data_match.get('criteria', [])
                    
                    # Add assistant message with answer and criteria
                    conversation.append({
                        "role": "assistant", 
                        "content": answer_data_match.get('answer', ''),
                        "criteria": turn_criteria
                    })
                    
                    total_criteria.extend(turn_criteria)
                    
                sample = {
                        **base_info,
                        'sample_id': f"sample_{sample_counter}",
                        'sample_type': 'multi_turn',
                        'scenario': scenario,
                        'scenario_id': scenario_id,
                        'task': task,
                        'task_id': task_id,
                        'conversations': conversation,
                        'criteria_count': len(total_criteria)
                    }
                samples.append(sample)
                sample_counter += 1
        
        return samples
    
    def _load_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """Load original video metadata from input directory."""
        try:
            video_file = self.video_metadata_dir / f"{video_id}.json"
            if video_file.exists():
                with open(video_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"⚠️ Video metadata file not found: {video_file}")
                return {}
        except Exception as e:
            print(f"⚠️ Error loading video metadata for {video_id}: {e}")
            return {}
    
    def _create_clean_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create clean samples without internal IDs except video_id."""
        clean_samples = []
        
        for sample in samples:
            clean_sample = {
                'video_id': sample['video_id'],
                'source_videos': sample.get('source_videos', ''),
                'title': sample.get('title', ''),
                'duration': sample.get('duration', 0),
                'categories': sample.get('categories', []),
                'sample_type': sample['sample_type'],
                'scenario': sample['scenario'],
                'task': sample['task'],
                'conversations': self._clean_conversations(sample.get('conversations', []))
            }
            clean_samples.append(clean_sample)
        
        return clean_samples
    
    def _clean_conversations(self, conversations: List[Dict]) -> List[Dict]:
        """Clean conversation data by removing metadata from user messages and keeping only essential fields."""
        clean_conversations = []
        
        for msg in conversations:
            if msg['role'] == 'user':
                # Keep only essential fields for user messages
                clean_msg = {
                    'role': 'user',
                    'content': msg['content']
                }
            else:  # assistant
                # Keep content and criteria for assistant messages
                clean_msg = {
                    'role': 'assistant',
                    'content': msg['content'],
                    'criteria': msg.get('criteria', [])
                }
            
            clean_conversations.append(clean_msg)
        
        return clean_conversations
    
    def _generate_dataset_analysis(self, samples: List[Dict], clean_samples: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive dataset analysis."""
        
        # Basic statistics
        total_samples = len(samples)
        single_turn_samples = [s for s in samples if s['sample_type'] == 'single_turn']
        multi_turn_samples = [s for s in samples if s['sample_type'] == 'multi_turn']
        
        # Video statistics
        unique_videos = set(s['video_id'] for s in samples)
        video_durations = [s['duration'] for s in samples if s.get('duration')]
        
        # Category analysis
        all_categories = []
        for sample in samples:
            all_categories.extend(sample.get('categories', []))
        category_counts = Counter(all_categories)
        
        # Task analysis  
        task_counts = Counter(s['task'] for s in samples)
        scenario_counts = Counter(s['scenario'] for s in samples)
        
        # Criteria analysis
        total_criteria = sum(s.get('criteria_count', 0) for s in samples)
        criteria_per_sample = [s.get('criteria_count', 0) for s in samples]
        
        # Conversation analysis
        conversation_lengths = []
        for sample in samples:
            conv = sample.get('conversations', [])
            conversation_lengths.append(len(conv))
        
        # Multi-turn specific analysis
        multi_turn_lengths = []
        for sample in multi_turn_samples:
            conv = sample.get('conversations', [])
            # Count user messages (turns)
            user_turns = len([msg for msg in conv if msg['role'] == 'user'])
            multi_turn_lengths.append(user_turns)
        
        # Question metadata analysis (from user messages in conversations)
        difficulties = []
        modalities = []
        
        for sample in samples:
            for msg in sample.get('conversations', []):
                if msg['role'] == 'user':
                    if msg.get('difficulty'):
                        difficulties.append(msg['difficulty'])
                    if msg.get('modalities'):
                        modalities.extend(msg['modalities'])
        
        analysis = {
            "dataset_info": {
                "generation_timestamp": datetime.now().isoformat(),
                "total_samples": total_samples,
                "single_turn_samples": len(single_turn_samples),
                "multi_turn_samples": len(multi_turn_samples),
                "file_info": {
                    "dataset_with_ids": str(self.final_dataset_file),
                    "clean_dataset": str(self.clean_dataset_file),
                    "analysis_file": str(self.analysis_file)
                }
            },
            
            "video_statistics": {
                "unique_videos": len(unique_videos),
                "video_ids": sorted(list(unique_videos)),
                "duration_stats": {
                    "min_duration": min(video_durations) if video_durations else 0,
                    "max_duration": max(video_durations) if video_durations else 0,
                    "avg_duration": sum(video_durations) / len(video_durations) if video_durations else 0,
                    "total_duration": sum(video_durations) if video_durations else 0
                }
            },
            
            "content_distribution": {
                "categories": dict(category_counts.most_common()),
                "tasks": dict(task_counts.most_common()),
                "scenarios": dict(scenario_counts.most_common())
            },
            
            "conversation_analysis": {
                "conversation_length_stats": {
                    "min_length": min(conversation_lengths) if conversation_lengths else 0,
                    "max_length": max(conversation_lengths) if conversation_lengths else 0,
                    "avg_length": sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0
                },
                "multi_turn_analysis": {
                    "avg_turns": sum(multi_turn_lengths) / len(multi_turn_lengths) if multi_turn_lengths else 0,
                    "min_turns": min(multi_turn_lengths) if multi_turn_lengths else 0,
                    "max_turns": max(multi_turn_lengths) if multi_turn_lengths else 0,
                    "turn_distribution": dict(Counter(multi_turn_lengths))
                }
            },
            
            "criteria_analysis": {
                "total_criteria": total_criteria,
                "avg_criteria_per_sample": total_criteria / total_samples if total_samples else 0,
                "criteria_distribution": {
                    "min_criteria": min(criteria_per_sample) if criteria_per_sample else 0,
                    "max_criteria": max(criteria_per_sample) if criteria_per_sample else 0,
                    "criteria_counts": dict(Counter(criteria_per_sample))
                }
            },
            
            "question_metadata_analysis": {
                "difficulty_distribution": dict(Counter(difficulties)),
                "modality_distribution": dict(Counter(modalities)),
            },
            
            "quality_metrics": {
                "samples_with_criteria": len([s for s in samples if s.get('criteria_count', 0) > 0]),
                "samples_without_criteria": len([s for s in samples if s.get('criteria_count', 0) == 0]),
                "avg_criteria_coverage": (len([s for s in samples if s.get('criteria_count', 0) > 0]) / total_samples * 100) if total_samples else 0
            }
        }
        
        return analysis
    
    def get_results_file(self):
        """Get the output file path for this stage."""
        return self.final_dataset_file
