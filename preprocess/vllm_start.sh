#!/bin/bash
# vLLM Server Startup Script for VLM and LLM models
# This script starts the vLLM server optimized for 8x80GB A100 GPUs
# 
# Usage:
#   ./vllm_start.sh vlm  # Start VLM model (Qwen2.5-VL-32B-Instruct-AWQ)
#   ./vllm_start.sh llm  # Start LLM model (Qwen3-30B-A3B-Instruct-2507)

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Enable vLLM optimizations
export VLLM_MARLIN_USE_ATOMIC_ADD=1
# export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_TRITON_FLASH_ATTN=1

# Check command line argument
MODEL_TYPE=${1:-"vlm"}

if [ "$MODEL_TYPE" = "vlm" ]; then
    # VLM Model Configuration
    MODEL_NAME="Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
    # MODEL_NAME="OpenGVLab/InternVL3_5-14B-Instruct"
    TENSOR_PARALLEL_SIZE=4
    GPU_MEMORY_UTIL=0.75
    MAX_MODEL_LEN=65536
    MAX_NUM_SEQS=32
    MAX_BATCHED_TOKENS=32768
    # QUANTIZATION="awq_marlin"
    PORT=8000
    
    echo "=== Starting VLM Model: $MODEL_NAME ==="
    echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
    echo "GPU Memory Utilization: $GPU_MEMORY_UTIL"
    echo "Max Model Length: $MAX_MODEL_LEN"
    echo "Port: $PORT"

elif [ "$MODEL_TYPE" = "llm" ]; then
    # LLM Model Configuration  
    MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
    TENSOR_PARALLEL_SIZE=2
    GPU_MEMORY_UTIL=0.75
    MAX_MODEL_LEN=65536
    MAX_NUM_SEQS=64
    MAX_BATCHED_TOKENS=65536
    QUANTIZATION=""
    PORT=8001
    
    echo "=== Starting LLM Model: $MODEL_NAME ==="
    echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
    echo "GPU Memory Utilization: $GPU_MEMORY_UTIL"
    echo "Max Model Length: $MAX_MODEL_LEN"
    echo "Port: $PORT"

else
    echo "Usage: $0 [vlm|llm]"
    echo "  vlm - Start VLM model (Qwen2.5-VL-32B-Instruct-AWQ) on port 8010"
    echo "  llm - Start LLM model (Qwen3-30B-A3B-Instruct-2507) on port 8011"
    exit 1
fi

echo "Available GPUs: $(nvidia-smi -L | wc -l)"

# Start the vLLM server with optimized configuration
CMD_ARGS=(
    --model "$MODEL_NAME"
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE
    --gpu-memory-utilization $GPU_MEMORY_UTIL
    --max-num-batched-tokens $MAX_BATCHED_TOKENS
    --max-num-seqs $MAX_NUM_SEQS
    --max-model-len $MAX_MODEL_LEN
    --dtype float16
    --disable-log-stats
    --served-model-name "$MODEL_NAME"
    --trust-remote-code
    --port $PORT
    --enable-prefix-caching
    --enable-chunked-prefill
    --swap-space 16
)

# Add quantization if specified
if [ -n "$QUANTIZATION" ]; then
    CMD_ARGS+=(--quantization "$QUANTIZATION")
fi

echo "Starting vLLM server with command:"
echo "python -m vllm.entrypoints.openai.api_server ${CMD_ARGS[*]}"
echo ""

python -m vllm.entrypoints.openai.api_server "${CMD_ARGS[@]}"
