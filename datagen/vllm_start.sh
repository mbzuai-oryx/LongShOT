#!/bin/bash

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

vllm serve \
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

