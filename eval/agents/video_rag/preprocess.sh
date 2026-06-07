#!/bin/bash
# Pre-warm the Video-RAG ASR cache across all available GPUs.
#
# Whisper-large per video is the dominant per-sample cost on first hit.
# Running this once before `./agents.sh` cuts per-sample time from ~36s
# to roughly the VLM-only cost (~10-15s).

set -e

VIDEO_DIR=${1:-./data/videos}
GPUS=${2:-0,1,2,3,4,5,6,7}
WORKERS_PER_GPU=${3:-1}
BATCH_SIZE=${4:-16}

cd "$(dirname "$0")"

python preprocess.py \
    --video_dir "$VIDEO_DIR" \
    --gpus "$GPUS" \
    --workers_per_gpu "$WORKERS_PER_GPU" \
    --batch_size "$BATCH_SIZE" \
    --model_size large-v3 \
    --compute_type float16 \
    --language en
