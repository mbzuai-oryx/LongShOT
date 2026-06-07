#!/bin/bash
# vLLM Server Startup Script for VLM (Gemma-4-31B-it)
# Multimodal (no spec decode) on GPUs 2-3, port 8011

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_MEDIA_LOADING_THREAD_COUNT=32
export VLLM_IMAGE_FETCH_TIMEOUT=60

MODEL_NAME="google/gemma-4-31B-it"

echo "Starting VLM server (${MODEL_NAME}) on GPU ${CUDA_VISIBLE_DEVICES}, port 8011..."

vllm serve "$MODEL_NAME" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 256 \
  --max-model-len 131072 \
  --max-num-batched-tokens 65536 \
  --enable-chunked-prefill \
  --served-model-name "$MODEL_NAME" \
  --trust-remote-code \
  --allowed-local-media-path "/" \
  --reasoning-parser gemma4 \
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --mm-encoder-tp-mode data \
  --media-io-kwargs '{"video": {"backend": "pyav"}}' \
  --port 8011
