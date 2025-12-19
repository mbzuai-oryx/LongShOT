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

# Path to your local model directory (change this to your actual model path)
MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
# MODEL_NAME="Qwen/QwQ-32B"
# MODEL_NAME="Qwen/Qwen3-30B-A3B-Thinking-2507"

# python -m vllm.entrypoints.openai.api_server \
  # --model "$MODEL_NAME" \
  # --tensor-parallel-size 4 \
  # --gpu-memory-utilization 0.95 \
  # --max-num-batched-tokens 65536 \
  # --max-num-seqs 2 \
  # --max-model-len 262144 \
  # --enable-chunked-prefill \
  # --task generate \
  # --dtype float16 \
  # --enforce-eager \
  # --disable-log-stats \
  # --enable-expert-parallel \
  # --served-model-name "$MODEL_NAME" \
  # --trust-remote-code \
  # --port 8000

# python -m vllm.entrypoints.openai.api_server \
#   --model "$MODEL_NAME" \
#   --tensor-parallel-size 8 \
#   --gpu-memory-utilization 0.95 \
#   --max-num-seqs 256 \
#   --max-model-len 1010000 \
#   --enable-chunked-prefill \
#   --max-num-batched-tokens 131072 \
#   --task generate \
#   --dtype float16 \
#   --enforce-eager \
#   --disable-log-stats \
#   --enable-expert-parallel \
#   --served-model-name "$MODEL_NAME" \
#   --trust-remote-code \
#   --port 8001


# python -m vllm.entrypoints.openai.api_server \
#   --model "$MODEL_NAME" \
#   --tensor-parallel-size 8 \
#   --gpu-memory-utilization 0.95 \
#   --max-num-seqs 256 \
#   --max-model-len 229376 \
#   --enable-chunked-prefill \
#   --max-num-batched-tokens 131072 \
#   --task generate \
#   --dtype float16 \
#   --enforce-eager \
#   --disable-log-stats \
#   --served-model-name "$MODEL_NAME" \
#   --trust-remote-code \
#   --port 8001

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \
  --max-model-len 266536 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 65536 \
  --task generate \
  --enforce-eager \
  --disable-log-stats \
  --served-model-name "$MODEL_NAME" \
  --trust-remote-code \
  --port 8010

