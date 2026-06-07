#!/bin/bash

# Usage: ./generate.sh [num_models]
# If num_models is not provided, all models will be processed

export CUDA_VISIBLE_DEVICES=0,1,2,3

export HF_HUB_READ_TIMEOUT=600
export HF_HUB_CONNECTION_TIMEOUT=1200
export OMP_NUM_THREADS=48

# NCCL tuning for PCIe-only (no NVLink) topology
export NCCL_P2P_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NVLS_ENABLE=0
export NCCL_BUFFSIZE=16777216

export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

TASK_NAME=postvalid_v2
OUTPUT_DIR=results_postvalid
NUM_WORKERS=8

# Accept parameter for number of models to run (default: all)
NUM_MODELS=${1:-}

MODELS=(
# --- VLM models ---
# "model_name:tensor_parallel_size"

# --- Agent ---
# "video-agent:1:llm=gemma-4-31B-it,vlm=gemma-4-31B-it"
)

# Limit models if parameter provided
if [[ -n "$NUM_MODELS" ]]; then
    MODELS=("${MODELS[@]:0:$NUM_MODELS}")
fi

# Set number of models to process
NUM_MODELS=${1:-${#MODELS[@]}}

for model_config in "${MODELS[@]:0:$NUM_MODELS}"; do
    # Format: "model_name:tensor_size[:key=val,key=val,...]"
    IFS=':' read -r model_name tensor_size agent_meta <<< "$model_config"

    EXTRA_ARGS=""
    if [[ "$model_name" == *"video-agent"* ]]; then
        EXTRA_ARGS="--external-server --port 8012"
        # Parse agent metadata (llm=X,vlm=Y)
        if [[ -n "$agent_meta" ]]; then
            IFS=',' read -ra META_PAIRS <<< "$agent_meta"
            for pair in "${META_PAIRS[@]}"; do
                key="${pair%%=*}"
                val="${pair#*=}"
                EXTRA_ARGS="$EXTRA_ARGS --agent-${key} ${val}"
            done
        fi
    fi

    python3 eval.py \
        --model "$model_name" \
        --tasks $TASK_NAME \
        --num_workers $NUM_WORKERS \
        --generate true \
        --output_dir $OUTPUT_DIR \
        --tensor_parallel_size $tensor_size \
        --config-file config.yaml \
        $EXTRA_ARGS
done
