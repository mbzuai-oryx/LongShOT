#!/bin/bash
# vLLM Server Startup Script for Qwen3-14B (Evaluation Judge)
# Text-only model — no vision encoder overhead

# export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Starting vLLM eval server on all available GPUs..."
echo "Available GPUs for vLLM: $CUDA_VISIBLE_DEVICES"

MODEL_NAME="Qwen/Qwen3-14B"
# MODEL_NAME="google/gemma-4-31B-it"

export OMP_NUM_THREADS=48
export FLASHINFER_DISABLE_VERSION_CHECK=1

# NCCL tuning for PCIe-only (no NVLink) topology
export NCCL_P2P_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NVLS_ENABLE=0
export NCCL_BUFFSIZE=16777216

# export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve \
  --model "$MODEL_NAME" \
  --tensor-parallel-size 2 \
  --data-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \
  --max-model-len 32768 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 65536 \
  --enable-prefix-caching \
  --async-scheduling \
  --compilation-config '{"mode": "VLLM_COMPILE"}' \
  --performance-mode throughput \
  --no-disable-cascade-attn \
  --prefix-caching-hash-algo xxhash \
  --served-model-name "$MODEL_NAME" \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":0, "video":0}' \
  --api-server-count 4 \
  --language-model-only \
  --port 8002
