#!/bin/bash
# vLLM Server Startup Script for LLM (Gemma-4-31B-it) with speculative decoding
# Orchestrator (text-only) on GPUs 0-1, port 8010

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_V1=1

MODEL_NAME="google/gemma-4-31B-it"

echo "Starting LLM server (${MODEL_NAME}) on GPU ${CUDA_VISIBLE_DEVICES}, port 8010..."

vllm serve "$MODEL_NAME" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 256 \
  --max-model-len 131072 \
  --max-num-batched-tokens 65536 \
  --enable-chunked-prefill \
  --served-model-name "$MODEL_NAME" \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template ./templates/tool_chat_template.jinja \
  --reasoning-parser gemma4 \
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --language-model-only \
  --speculative-config '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":2}' \
  --port 8010
