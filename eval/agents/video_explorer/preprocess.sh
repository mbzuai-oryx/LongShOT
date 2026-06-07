#!/bin/bash
# Preprocess videos for VideoExplorer agent
#
# Usage: ./preprocess.sh [video_dir] [output_dir]
#
# Defaults to paths from config.yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Default paths (override via arguments)
VIDEO_DIR="${1:-./data/videos}"
OUTPUT_DIR="${2:-./agents/video_explorer/data}"

# Configuration
CLIP_DURATION=10
NUM_WORKERS=64  # DataLoader workers for video decoding
GPU=7
BATCH_SIZE=128  # Large batch to keep GPU busy

echo "========================================"
echo "VideoExplorer Preprocessing"
echo "========================================"
echo "Video dir: $VIDEO_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Clip duration: ${CLIP_DURATION}s"
echo "Workers: $NUM_WORKERS"
echo "GPU: $GPU"
echo "========================================"

cd "$EVAL_DIR"

python agents/video_explorer/preprocess.py \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --clip_duration $CLIP_DURATION \
    --num_workers $NUM_WORKERS \
    --gpu $GPU \
    --batch_size $BATCH_SIZE

echo "========================================"
echo "Preprocessing complete!"
echo "========================================"
