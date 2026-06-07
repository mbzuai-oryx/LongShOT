#!/bin/bash
# vLLM Embedding Server for visual (SigLIP)
# Serves visual query embeddings on port 8018

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL_NAME="google/siglip-base-patch16-512"

echo "Starting visual embedding server (${MODEL_NAME}) on GPU ${CUDA_VISIBLE_DEVICES}, port 8018..."

vllm serve "$MODEL_NAME" \
  --runner pooling \
  --convert embed \
  --gpu-memory-utilization 0.2 \
  --max-num-seqs 256 \
  --served-model-name "$MODEL_NAME" \
  --trust-remote-code \
  --port 8018
