#!/bin/bash

# Usage: ./agents.sh [num_agents]
# If num_agents is not provided, all agents will be processed

export CUDA_VISIBLE_DEVICES=4,5,6,7

export HF_HUB_READ_TIMEOUT=600
export HF_HUB_CONNECTION_TIMEOUT=1200
export OMP_NUM_THREADS=48
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL tuning for PCIe-only (no NVLink) topology
export NCCL_P2P_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NVLS_ENABLE=0
export NCCL_BUFFSIZE=16777216

# Synchronous CUDA so device-side asserts surface on the real kernel instead of
# the next `.to(device)` sync. Remove once the LanguageBind retriever assert is
# diagnosed — this hurts throughput.
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1

TASK_NAME=postvalid_v2
OUTPUT_DIR=results_postvalid

# Accept parameter for number of agents to run (default: all)
NUM_AGENTS=${1:-}

# Format: "agent_name"
# Agents manage their own multi-model servers internally
AGENTS=(
# "video_explorer"  # VideoExplorer (RUC-NLPIR) 
# "video_rag"       # Video-RAG (Leon1207)
# "videomind"       # VideoMind (yeliudev) 
# "vgent"           # Vgent (xiaoqian-shen) 
# "llovi"         # Coming soon
)

# Limit agents if parameter provided
if [[ -n "$NUM_AGENTS" ]]; then
    AGENTS=("${AGENTS[@]:0:$NUM_AGENTS}")
fi

NUM_AGENTS=${1:-${#AGENTS[@]}}

mkdir -p agents/logs

for agent in "${AGENTS[@]:0:$NUM_AGENTS}"; do
    echo "========================================"
    echo "Running agent: $agent"
    echo "========================================"

    RUN_LOG="agents/logs/run_${agent}_$(date +%Y%m%d_%H%M%S).log"
    echo "Run log: $RUN_LOG"

    # Pick the right conda env per agent
    case "$agent" in
        videomind) PYTHON="python" ;;
        vgent)     PYTHON="python" ;;
        *)         PYTHON="python3" ;;
    esac

    # tee stdout+stderr so the first CUDA assert survives past tmux scrollback
    $PYTHON -u eval.py \
        --model "video-agent-${agent}" \
        --tasks $TASK_NAME \
        --generate true \
        --evaluate false \
        --score false \
        --output_dir $OUTPUT_DIR \
        --config-file config.yaml 2>&1 | tee "$RUN_LOG"
done
