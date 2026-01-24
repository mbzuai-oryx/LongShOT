#!/bin/bash

# Usage: ./eval.sh [num_models]
# If num_models is not provided, all models will be processed
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export HF_HUB_READ_TIMEOUT=600
export HF_HUB_CONNECTION_TIMEOUT=1200
export OMP_NUM_THREADS=5
export VLLM_USE_TRITON_FLASH_ATTN=0

TASK_NAME=postvalid_v1
OUTPUT_DIR=results_postvalid
NUM_WORKERS=256
TENSORS=8

# Accept parameter for number of models to run (default: all)
NUM_MODELS=${1:-}

MODELS=(
# "llava-hf/llava-onevision-qwen2-7b-ov-hf"
# "llava-hf/LLaVA-NeXT-Video-7B-hf"
# "Qwen/Qwen2.5-VL-7B-Instruct"
# "Qwen/Qwen2.5-Omni-7B"
# "OpenGVLab/InternVL3_5-8B"
# "gemini-2.5-flash"
# "Qwen/Qwen3-VL-8B-Instruct"
# "zai-org/GLM-4.6V-Flash"
"video-agent"
)

# Set number of models to process
NUM_MODELS=${1:-${#MODELS[@]}}


for model in "${MODELS[@]:0:$NUM_MODELS}"; do
    python3 eval.py \
        --model "$model" \
        --tasks $TASK_NAME \
        --num_workers $NUM_WORKERS \
        --evaluate true \
        --score true \
        --output_dir $OUTPUT_DIR \
        --tensor_parallel_size $TENSORS \
        --config-file config.yaml \
        ${EVAL_MODEL:+--eval_model "$EVAL_MODEL"}
done