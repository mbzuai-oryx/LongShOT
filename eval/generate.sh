#!/bin/bash

# Usage: ./generate.sh [num_models]
# If num_models is not provided, all models will be processed

export HF_HUB_READ_TIMEOUT=600
export HF_HUB_CONNECTION_TIMEOUT=1200
export OMP_NUM_THREADS=5
export VLLM_USE_TRITON_FLASH_ATTN=0

TASK_NAME=postvalid_v1
OUTPUT_DIR=results_postvalid
NUM_WORKERS=1

# Accept parameter for number of models to run (default: all)
NUM_MODELS=${1:-}

MODELS=(
# "Qwen/Qwen2_5_Omni_7B"
# "Qwen/Qwen2.5-VL-7B-Instruct:4"
"Qwen/Qwen3-VL-8B-Instruct":4
)

# Limit models if parameter provided
if [[ -n "$NUM_MODELS" ]]; then
    MODELS=("${MODELS[@]:0:$NUM_MODELS}")
fi

# Set number of models to process
NUM_MODELS=${1:-${#MODELS[@]}}

for model_config in "${MODELS[@]:0:$NUM_MODELS}"; do
    model_name="${model_config%:*}"
    tensor_size="${model_config#*:}"
    
    python3 eval.py \
        --model "$model_name" \
        --tasks $TASK_NAME \
        --num_workers $NUM_WORKERS \
        --generate true \
        --output_dir $OUTPUT_DIR \
        --tensor_parallel_size $tensor_size \
        --config-file config.yaml
done