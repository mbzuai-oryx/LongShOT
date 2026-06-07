# LongShOT Dataset Generation Pipeline

This directory contains the **LongShOTBench** dataset generation pipeline that transforms preprocessed video metadata into a comprehensive benchmark dataset with questions, answers, and evaluation criteria.

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Pipeline Stages](#pipeline-stages)
- [Usage](#usage)
- [Output Files](#output-files)
- [Next Steps](#next-steps)

## Overview

The dataset generation pipeline is a multi-stage async processing system that:

1. **Analyzes** video metadata to identify evaluation scenarios
2. **Generates** appropriate question types for each scenario
3. **Creates** diverse single-turn and multi-turn questions
4. **Produces** reference answers using video metadata
5. **Develops** graded evaluation criteria (rubrics) for each Q&A pair
6. **Finalizes** the dataset into individual samples ready for evaluation

The pipeline uses a Large Language Model (LLM) to generate high-quality, diverse questions and evaluation materials across multiple reasoning types (visual, audio, speech, temporal, causal, etc.).

## Pipeline Architecture

```
Video Metadata (JSON)
    ↓
[Stage 1: Analysis] → scenarios.jsonl
    ↓
[Stage 2: Question Types] → question_types.jsonl
    ↓
[Stage 3: Questions] → questions.jsonl
    ↓
[Stage 4: Answers] → answers.jsonl
    ↓
[Stage 5: Criteria] → criteria.jsonl
    ↓
[Stage 6: Finalization] → final_dataset.jsonl + clean_dataset.jsonl
```

Each stage:
- Loads results from previous stages
- Processes data asynchronously with concurrency control
- Saves results incrementally (resume-capable)
- Tracks progress with detailed statistics

## Prerequisites

### 1. Completed Video Preprocessing

You must have completed the preprocessing pipeline (see `preprocess/README.md`). The required input is:

```
preprocess/dataset/final/
├── video1.json
├── video2.json
└── ...
```

Each JSON file should contain:
- Video metadata (title, duration, description, etc.)
- Video descriptions (scene-by-scene captions)
- Audio descriptions (sound events and music)
- Speech transcriptions (with timestamps)
- Key events timeline
- Multimodal understanding analysis

### 2. vLLM Server Running

Start a vLLM server with an instruction-tuned LLM:

```bash
# Start the vLLM server (in a separate terminal)
./vllm_start.sh
```

The default configuration expects:
- **Server URL**: `http://localhost:8001/v1`
- **Model**: `Qwen/Qwen3-30B-A3B-Instruct-2507`

> **Note**: The vLLM server must support structured output generation (via `guided_json` parameter).

### 3. Python Environment

Ensure you have the LongShOT conda environment activated:

```bash
conda activate longshot
```

## Configuration

Edit `config.yaml` to customize the pipeline:

### Key Configuration Options

```yaml
vllm:
  base_url: "http://localhost:8001/v1"  # vLLM server endpoint
  model: "Qwen/Qwen3-30B-A3B-Instruct-2507"  # Model name
  max_tokens: 40000  # Max tokens per generation
  temperature: 0.7  # Sampling temperature
  use_guided_json: true  # Enable structured output

processing:
  max_concurrent_requests: 10  # Concurrent API calls
  input_directory: "../preprocess/dataset/final"  # Input path
  output_directory: "./results"  # Output path
  file_pattern: "*.json"  # Input file pattern
  incremental_save: true  # Save progress incrementally
  save_batch_size: 1  # Save frequency
  continue_on_error: true  # Keep processing on errors
  max_errors: 10  # Max errors before abort

timeout: 900  # Request timeout (15 min for thinking models)
```

### Performance Tuning

- **`max_concurrent_requests`**: Higher = faster but more memory
  - Recommended: 10-20 for 30B models, 5-10 for larger models
- **`temperature`**: Controls generation diversity
  - 0.7 = balanced (recommended)
  - 0.3 = more focused
  - 0.9 = more diverse
- **`timeout`**: Increase for thinking/reasoning models

## Pipeline Stages

### Stage 1: Scenario Analysis

**Purpose**: Analyze video metadata to identify diverse evaluation scenarios.

**Input**: Video metadata JSON files from preprocessing

**Process**:
1. Loads video descriptions, audio, speech, and key events
2. Uses LLM to identify distinct evaluation scenarios
3. Each scenario targets specific reasoning capabilities (visual, audio, temporal, etc.)
4. Assigns modalities required for each scenario

**Output**: `results/scenarios.jsonl`

**Example scenario**:
```json
{
  "scenario_id": "video1_scenario_1",
  "video_id": "video1",
  "scenario_description": "Evaluate understanding of cooking process...",
  "tasks": ["counting", "temporal_sequencing", "spatial_reasoning"],
  "modalities": ["video", "audio"],
  "key_events_referenced": [0, 1, 2]
}
```

### Stage 2: Question Type Generation

**Purpose**: Map each scenario-task combination to appropriate question types.

**Input**: 
- `scenarios.jsonl` (from Stage 1)
- `question_types_catalog.json` (predefined question type taxonomy)

**Process**:
1. For each scenario and its tasks, selects 2-3 appropriate question types
2. Considers scenario context, required modalities, and task characteristics
3. Uses question type catalog with 60+ predefined types across categories:
   - Factual retrieval (direct facts, counting, text extraction)
   - Temporal (timestamps, durations, sequences)
   - Spatial (locations, movements, relationships)
   - Causal (cause-effect, predictions, reasoning)
   - Comparative (differences, similarities, changes)
   - Abstract (summarization, themes, intent)
   - Multi-modal (cross-modal reasoning)

**Output**: `results/question_types.jsonl`

**Example**:
```json
{
  "scenario_id": "video1_scenario_1",
  "tasks": [
    {
      "task_id": "video1_scenario_1_counting",
      "task": "counting",
      "question_types": [
        {
          "id": "numerical_extraction",
          "name": "Numerical Extraction",
          "rationale": "Appropriate for counting ingredients..."
        }
      ]
    }
  ]
}
```

### Stage 3: Question Generation

**Purpose**: Generate diverse, high-quality questions for each question type.

**Input**: 
- `question_types.jsonl` (from Stage 2)
- Video metadata (for context-aware generation)

**Process**:
1. For each question type, generates 2-3 questions
2. Creates both single-turn questions and multi-turn sequences
3. **Single-turn**: Standalone questions answerable from video
4. **Multi-turn**: Conversational sequences (2-4 turns) with follow-ups
5. Questions reference specific video timestamps, events, and modalities
6. Ensures diversity in phrasing, complexity, and focus

**Output**: `results/questions.jsonl`

**Example**:
```json
{
  "task_id": "video1_scenario_1_counting",
  "questions": {
    "single_turn_questions": [
      {
        "id": "q1",
        "question": "How many ingredients are added to the bowl?",
        "question_type": "numerical_extraction",
        "modalities": ["video"],
        "temporal_reference": "00:45-02:30"
      }
    ],
    "multi_turn_questions": [
      [
        {
          "id": "mt1_turn1",
          "turn_number": 1,
          "question": "What happens at 1:30?",
          "question_type": "temporal_event"
        },
        {
          "id": "mt1_turn2",
          "turn_number": 2,
          "question": "How long does this process take?",
          "question_type": "duration_calculation"
        }
      ]
    ]
  ]
}
```

### Stage 4: Answer Generation

**Purpose**: Generate comprehensive reference answers for all questions.

**Input**: 
- `questions.jsonl` (from Stage 3)
- Video metadata (source of ground truth)

**Process**:
1. For each question, generates a detailed reference answer
2. Answers derived from video metadata (descriptions, audio, speech, events)
3. Includes:
   - **answer**: Direct answer text
   - **explanation**: Reasoning and supporting details
   - **evidence_timestamps**: Relevant video timestamps
   - **modalities_used**: Which modalities were needed
   - **confidence**: Answer confidence (high/medium/low)
4. Multi-turn answers maintain conversation history context

**Output**: `results/answers.jsonl`

**Example**:
```json
{
  "task_id": "video1_scenario_1_counting",
  "answers": {
    "single_turn_answers": [
      {
        "question_id": "q1",
        "answer": "Five ingredients are added to the bowl",
        "explanation": "The video shows...",
        "evidence_timestamps": ["00:45", "01:15", "01:45", "02:10", "02:25"],
        "modalities_used": ["video"],
        "confidence": "high",
        "status": "success"
      }
    ],
    "multi_turn_answers": [...]
  ]
}
```

### Stage 5: Criteria Generation

**Purpose**: Generate graded evaluation rubrics for each Q&A pair.

**Input**: 
- `answers.jsonl` (from Stage 4)
- Video metadata (for criteria grounding)

**Process**:
1. Creates detailed evaluation criteria for each Q&A pair
2. Criteria are **graded** (excellent/good/acceptable/poor)
3. Each grade level specifies:
   - Required components for that grade
   - Common errors or omissions
   - Specific details that distinguish quality levels
4. Enables interpretable, fine-grained evaluation

**Output**: `results/criteria.jsonl`

**Example**:
```json
{
  "task_id": "video1_scenario_1_counting",
  "criteria": {
    "single_turn_criteria": [
      {
        "question_id": "q1",
        "criteria": {
          "excellent": {
            "score": 10,
            "description": "Correctly counts all 5 ingredients with timestamp references"
          },
          "good": {
            "score": 7,
            "description": "Correct count of 5 but missing some timestamps"
          },
          "acceptable": {
            "score": 5,
            "description": "Count within 1 (4 or 6 ingredients)"
          },
          "poor": {
            "score": 2,
            "description": "Significantly incorrect count or no attempt"
          }
        },
        "status": "success"
      }
    ],
    "multi_turn_criteria": [...]
  ]
}
```

### Stage 6: Dataset Finalization

**Purpose**: Assemble all components into individual training samples.

**Input**: 
- `questions.jsonl`, `answers.jsonl`, `criteria.jsonl`
- Video metadata (for final enrichment)

**Process**:
1. Matches questions, answers, and criteria by IDs
2. Creates individual samples (one per question)
3. Enriches with video metadata (title, duration, modalities)
4. Adds sample IDs and cross-references
5. Generates two versions:
   - **final_dataset.jsonl**: With all internal IDs for tracking
   - **clean_dataset.jsonl**: User-facing without internal IDs
6. Creates comprehensive dataset analysis and statistics

**Output**: 
- `results/final_dataset.jsonl` (with IDs)
- `results/clean_dataset.jsonl` (clean)
- `results/dataset_analysis.json` (statistics)

**Example sample**:
```json
{
  "sample_id": "sample_0001",
  "sample_type": "single_turn",
  "video_id": "video1",
  "video_title": "How to Make Pasta Carbonara",
  "video_duration": "08:45",
  "scenario": "Evaluate understanding of cooking process...",
  "task": "counting",
  "question": "How many ingredients are added to the bowl?",
  "question_type": "numerical_extraction",
  "answer": "Five ingredients are added to the bowl",
  "explanation": "The video shows...",
  "evidence_timestamps": ["00:45", "01:15", "01:45", "02:10", "02:25"],
  "modalities": ["video"],
  "criteria": {
    "excellent": {...},
    "good": {...},
    "acceptable": {...},
    "poor": {...}
  },
  "criteria_count": 4
}
```

## Usage

### Basic Usage (All Stages)

Run the complete pipeline from start to finish:

```bash
python main.py
```

This will:
1. Process all videos in the input directory
2. Execute all 6 stages sequentially
3. Save results incrementally (resume-capable)
4. Generate final dataset and analysis

### Run Specific Stage

Run individual stages for debugging or re-generation:

```bash
# Run only scenario analysis
python main.py --stage analysis

# Run only question generation
python main.py --stage questions

# Run from answers onwards
python main.py --stage answers
```

**Available stages**: `analysis`, `question_types`, `questions`, `answers`, `criteria`, `finalize`

### Sample Limit (Testing)

Process only the first N videos for testing:

```bash
# Process only 5 videos
python main.py --sample 5
```

### Debug Mode

Limit samples per stage for faster debugging:

```bash
# Process only 3 samples per stage
python main.py --debug-samples 3

# Combine with sample limit
python main.py --sample 5 --debug-samples 2
```

This creates a small dataset quickly for testing the full pipeline.

### Custom Configuration

Use a different configuration file:

```bash
python main.py --config custom_config.yaml
```

### Resume After Interruption

The pipeline automatically resumes from where it stopped:

```bash
# If interrupted during Stage 3
python main.py  # Automatically skips completed stages 1-2
```

Each stage saves results incrementally and tracks completed items.

## Output Files

All output files are saved to `results/` directory:

### Intermediate Files

- **`scenarios.jsonl`**: Evaluation scenarios (Stage 1)
- **`question_types.jsonl`**: Question type mappings (Stage 2)
- **`questions.jsonl`**: Generated questions (Stage 3)
- **`answers.jsonl`**: Reference answers (Stage 4)
- **`criteria.jsonl`**: Evaluation criteria (Stage 5)

### Incremental Files (Optional)

- **`*_incremental.jsonl`**: Real-time progress saves
  - Used for debugging and monitoring
  - Can be deleted after successful completion

### Final Datasets

- **`final_dataset.jsonl`**: Complete dataset with all IDs
  - Contains internal IDs for tracking and debugging
  - Use for analysis and pipeline development

- **`clean_dataset.jsonl`**: User-facing dataset
  - No internal IDs or processing metadata
  - Ready for distribution and use in evaluations

### Analysis

- **`dataset_analysis.json`**: Comprehensive statistics
  - Sample counts by type, task, video
  - Question type distribution
  - Modality coverage
  - Average criteria per sample
  - Video statistics

## Next Steps

After generating the dataset:

1. **Upload to Hugging Face** (see main README.md):
```python
from datasets import Dataset
dataset = Dataset.from_json("results/clean_dataset.jsonl")
dataset.push_to_hub("anonymous/longshot-bench", config_name="postvalid")
```

2. **Run Evaluation** (see `../eval/README.md`):
```bash
cd ../eval
bash generate.sh  # Generate model responses
bash eval.sh      # Evaluate responses
```