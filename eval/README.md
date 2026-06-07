# LongShOT Evaluation Framework

This directory contains the evaluation framework for benchmarking Vision-Language Models (VLMs) on the LongShOT dataset. The framework supports model response generation, automated evaluation, and scoring with optional agentic tool use capabilities.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
  - [Main Configuration Files](#main-configuration-files)
  - [Task Configuration](#task-configuration)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Step-by-Step Workflow](#step-by-step-workflow)
- [Output Structure](#output-structure)

## Overview

```
eval.py          # Main evaluation script
├── Generation   # Generate model responses to questions
├── Evaluation   # Score responses against rubrics
└── Scoring      # Calculate final accuracy metrics

utils.py         # Core utilities (dataset loading, API calls, server management)
scoring.py       # Score calculation and aggregation
```

## Prerequisites

1. **Environment Setup**: Ensure the conda environment is activated:
   ```bash
   conda activate longshot
   ```
3. **Dataset Access**: The framework loads datasets from Hugging Face. Ensure you have:
   - Hugging Face authentication configured
   - Access to the required datasets (anonymous/longshot-bench)

4. **Video Files**: Videos must be available at the path specified in configuration:
   ```yaml
   paths:
     video_path: "../preprocess/dataset/videos"
   ```

## Configuration

### Main Configuration Files

#### `config.yaml` - Evaluation Configuration

```yaml
server:
  base_url: "http://localhost:8002/v1/"  # vLLM server endpoint
  port: "8002"                            # Server port
  api_key: "DUMMY_KEY"                    # API key (not used for local)

generation:
  max_tokens: 4096                        # Max tokens for model response
  temperature: 0.1                        # Sampling temperature
  top_p: 1.0                              # Nucleus sampling parameter
  timeout: 1800                           # Request timeout (seconds)

evaluation:
  max_tokens: 4096                        # Max tokens for evaluation
  temperature: 0.1
  top_p: 1.0
  timeout: 60

vllm_candidate:
  max_model_len: ""                       # Leave empty for auto
  max_num_seqs: "256"                     # Batch size
  trust_remote_code: false                # Set true for custom models
  allowed_local_media_path: "/"           # Allow video file access
  enable_prefix_caching: true             # Speed up repeated prefixes
  additional_args: []                     # Extra vLLM arguments

vllm_evaluation:
  max_model_len: "32768"                  # Context length for evaluator
  max_num_seqs: "256"
  trust_remote_code: true
  enable_prefix_caching: true

paths:
  logs_dir: "logs"                        # Server log directory
  video_path: "../preprocess/dataset/videos"  # Video files location

system:
  system_prompt: ""                       # Optional system prompt
  dataset_limit: 0                        # 0=unlimited, N=first N samples
  server_check_timeout: 5                 # Server readiness check timeout
```

### Task Configuration

#### `tasks.yaml` - Dataset Task Definitions

```yaml
longshot_bench:
  postvalid_v1:
    path: anonymous/longshot-bench
    name: postvalid_v1
    split: test
```

More custom task sets can be added following the same format

## Usage

### Quick Start

#### Generate Responses
```bash
cd eval
bash generate.sh
```

#### Evaluate and Score
```bash
bash eval.sh
```

### Step-by-Step Workflow

#### Step 1: Start vLLM Server (Optional)

If not using `--external-server` flag, the script will auto-manage servers. To manually start:

```bash
# Start server for candidate model generation
./vllm_start.sh
```

**vLLM Server Parameters**:
- `--tensor-parallel-size`: Number of GPUs for model parallelism
- `--max-model-len`: Maximum context length
- `--gpu-memory-utilization`: GPU memory fraction (0.0-1.0)
- `--enable-prefix-caching`: Cache common prefixes for speed
- `--allowed-local-media-path`: Path to video files

#### Step 2: Generate Model Responses

```bash
python eval.py \
    --model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --tasks postvalid_v1 \
    --num_workers 1 \
    --generate true \
    --output_dir results_postvalid \
    --tensor_parallel_size 4 \
    --config-file config.yaml \
    --external-server
```

**Arguments**:
- `--model`: HuggingFace model identifier
- `--tasks`: Task name(s) from `tasks.yaml` or "all"
- `--num_workers`: Parallel workers (1 for sequential)
- `--generate`: Enable response generation phase
- `--output_dir`: Directory for results
- `--tensor_parallel_size`: GPU parallelism level
- `--config-file`: Configuration file to use
- `--external-server`: Use existing server (don't auto-start/stop)

**Output**: Generates `{model_name}.jsonl` with model responses.

#### Step 3: Evaluate Responses

```bash
python eval.py \
    --model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --eval_model "Qwen/Qwen3-14B" \
    --tasks postvalid_v1 \
    --num_workers 1024 \
    --evaluate true \
    --output_dir results_postvalid \
    --config-file config.yaml \
    --external-server
```

**Key Differences**:
- `--eval_model`: Model to use for evaluation (usually a strong LLM)
- `--evaluate`: Enable evaluation phase
- `--num_workers`: Higher for evaluation (text-only processing)

**Output**: Generates `{model_name}_eval.jsonl` with scored responses.

#### Step 4: Calculate Final Scores

```bash
python eval.py \
    --model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --tasks postvalid_v1 \
    --score true \
    --output_dir results_postvalid \
    --config-file config.yaml
```

**Output**: 
- `{model_name}_score.txt`: Human-readable accuracy report
- `{model_name}_timing.json`: Performance timing data

## Output Structure

```
results_postvalid/
└── Qwen_2_5_VL_7B_Instruct/
    ├── Qwen_2_5_VL_7B_Instruct.jsonl              # Generated responses
    ├── Qwen_2_5_VL_7B_Instruct_eval.jsonl         # Evaluated responses
    ├── Qwen_2_5_VL_7B_Instruct_score.txt          # Final scores
    └── Qwen_2_5_VL_7B_Instruct_timing.json        # Timing metrics

logs/
├── candidate_server_*.log                          # Candidate model logs
└── evaluation_server_*.log                         # Evaluation model logs
```
