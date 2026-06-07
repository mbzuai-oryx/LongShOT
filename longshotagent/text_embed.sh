#!/bin/bash
# vLLM Embedding Server for text (all-MiniLM-L6-v2)
# Serves audio/text query embeddings on port 8014

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

echo "Starting text embedding server (${MODEL_NAME}) on GPU ${CUDA_VISIBLE_DEVICES}, port 8014..."

vllm serve "$MODEL_NAME" \
  --runner pooling \
  --gpu-memory-utilization 0.2 \
  --max-num-seqs 256 \
  --served-model-name "$MODEL_NAME" \
  --port 8014
