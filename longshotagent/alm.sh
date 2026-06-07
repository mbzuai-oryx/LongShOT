#!/bin/bash
# vLLM Server Startup Script for ALM (audio-flamingo-3)
# Starts the audio language model on port 8013

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

MODEL_NAME="nvidia/audio-flamingo-3-hf"

echo "Starting ALM server (${MODEL_NAME}) on GPU ${CUDA_VISIBLE_DEVICES}, port 8013..."

vllm serve "$MODEL_NAME" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.5 \
  --max-num-seqs 64 \
  --served-model-name "$MODEL_NAME" \
  --trust-remote-code \
  --allowed-local-media-path "/" \
  --limit-mm-per-prompt '{"audio": 1}' \
  --enable-prefix-caching \
  --port 8013
