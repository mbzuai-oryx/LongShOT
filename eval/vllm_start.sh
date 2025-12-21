#!/bin/bash
# vLLM Server Startup Script for Qwen 2.5-VL
# This script starts the vLLM server for video description generation
# 
# GPU Layout:
# - GPU 0: CLIP/Whisper (~40%) + vLLM shard (~60%)
# - GPU 1,2,3: vLLM shards with high utilization
# - Total: 4-way tensor parallelism across all GPUs

# export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_USE_V1=1

echo "Starting vLLM server on all available GPUs... "
echo "Available GPUs for vLLM: $CUDA_VISIBLE_DEVICES"

# Set memory fraction for GPU 0 to account for CLIP/Whisper usage

# export to enable vllm marlin kerlin optimizations
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

MODEL_NAME="Qwen/Qwen3-14B"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 64 \
  --max-model-len 90000 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 131072 \
  --disable-log-stats \
  --enable-prefix-caching \
  --served-model-name "$MODEL_NAME" \
  --trust-remote-code \
  --allowed-local-media-path "/" \
  --port 8002

