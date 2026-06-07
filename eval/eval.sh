#!/bin/bash

# Usage: ./eval.sh [num_models]
# If num_models is not provided, all models will be processed

export HF_HUB_READ_TIMEOUT=600
export HF_HUB_CONNECTION_TIMEOUT=1200
export OMP_NUM_THREADS=48

# NCCL tuning for PCIe-only (no NVLink) topology
export NCCL_P2P_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NVLS_ENABLE=0
export NCCL_BUFFSIZE=16777216

TASK_NAME=postvalid_v2
OUTPUT_DIR=results_postvalid
NUM_WORKERS=128
TENSORS=8

EVAL_MODEL="Qwen/Qwen3-14B"
EVAL_TAG="qwen3_14b"

# Accept parameter for number of models to run (default: all)
NUM_MODELS=${1:-}

MODELS=(
# --- Tier 1: Native Video + Audio (Full Modality Match) ---
# "Qwen/Qwen3-Omni-30B-A3B-Instruct"
# "Qwen/Qwen2.5-Omni-7B"
# "openbmb/MiniCPM-o-4_5"
# "openbmb/MiniCPM-o-2_6"
# "Qwen/Qwen3-Omni-30B-A3B-Thinking"
# "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
# "Qwen/Qwen2.5-Omni-3B"
# "RUC-NLPIR/OmniAtlas-Qwen2.5-7B"
# "antgroup/HumanSense_Omni_Reasoning"
# "xiaomi/mimo-v2-omni"

# --- Tier 2: Large Video VLMs (50B+) ---
# "Qwen/Qwen3-VL-235B-A22B-Instruct"
# "Qwen/Qwen3-VL-235B-A22B-Thinking"
# "OpenGVLab/InternVL3_5-241B-A28B"
# "Qwen/Qwen3.5-122B-A10B"
# "zai-org/GLM-4.5V"
# "Qwen/Qwen2.5-VL-72B-Instruct"
# "OpenGVLab/InternVL3-78B"
# "OpenGVLab/InternVL2_5-78B"

# --- Tier 3: Mid-Size Video VLMs (10B-50B) ---
# "Qwen/Qwen3.5-27B"
# "Qwen/Qwen3.5-35B-A3B"
# "Qwen/Qwen3-VL-32B-Instruct"
# "Qwen/Qwen3-VL-32B-Thinking"
# "Qwen/Qwen3-VL-30B-A3B-Instruct"
# "Qwen/Qwen3-VL-30B-A3B-Thinking"
# "Qwen/Qwen2.5-VL-32B-Instruct"
# "OpenGVLab/InternVL3_5-38B"
# "OpenGVLab/InternVL3-38B"
# "OpenGVLab/InternVL2_5-38B"
# "OpenGVLab/InternVL3-14B"
# "google/gemma-3-27b-it"
# "AIDC-AI/Ovis2.6-30B-A3B"
# "moonshotai/Kimi-VL-A3B-Thinking"
# "moonshotai/Kimi-VL-A3B-Instruct"
# "AIDC-AI/Ovis2-16B"
# "google/gemma-3-12b-it"
# "stepfun-ai/Step3-VL-10B"
# "zai-org/GLM-4.6V-Flash"
# "zai-org/GLM-4.1V-9B-Thinking"
# "baidu/ERNIE-4.5-VL-28B-A3B-PT"
# "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
# "microsoft/Phi-4-reasoning-vision-15B"

# --- Tier 4: Small Video/Vision Models (5B-10B) ---
# "Qwen/Qwen3.5-9B"
# "Qwen/Qwen3-VL-8B-Instruct"
# "Qwen/Qwen3-VL-8B-Thinking"
# "Qwen/Qwen2.5-VL-7B-Instruct"
# "OpenGVLab/InternVL3_5-8B"
# "OpenGVLab/InternVL3-8B"
# "Kwai-Keye/Keye-VL-1_5-8B"
# "allenai/Molmo2-8B"
# "XiaomiMiMo/MiMo-VL-7B-RL-2508"
# "AIDC-AI/Ovis2.5-9B"
# "AIDC-AI/Ovis2-8B"
# "llava-hf/llava-onevision-qwen2-7b-ov-hf"
# "openbmb/MiniCPM-V-4_5"
# "internlm/Intern-S1-mini"
# "longvideotool/LongVT-RL"
# "google/gemma-4-31B-it"
# "google/gemma-4-26B-A4B-it"
# "Vision-CAIR/Tempo-6B"
# "Video-R1/Video-R1-7B"
# "OneThink/OneThinker-8B"
# "TIGER-Lab/VideoScore2"

# --- Tier 5: Compact Models (4B-5B) ---
# "Qwen/Qwen3.5-4B"
# "Qwen/Qwen3-VL-4B-Instruct"
# "Qwen/Qwen3-VL-4B-Thinking"
# "google/gemma-3-4b-it"
# "OpenGVLab/InternVL3_5-4B-Instruct"
# "AIDC-AI/Ovis2-4B"
# "Qwen/Qwen3.5-2B"
# "Qwen/Qwen3-VL-2B-Instruct"
# "Qwen/Qwen2.5-VL-3B-Instruct"

# --- Tier 6: Audio-Only LLMs ---
# "Qwen/Qwen2-Audio-7B-Instruct"
# "mistralai/Voxtral-Small-24B-2507"
# "mistralai/Voxtral-Mini-3B-2507"
# "nvidia/audio-flamingo-3-hf"
# "FunAudioLLM/Fun-Audio-Chat-8B"
)

# Set number of models to process
NUM_MODELS=${1:-${#MODELS[@]}}


for entry in "${MODELS[@]:0:$NUM_MODELS}"; do
    # Support @alias syntax: "model_name@alias"
    if [[ "$entry" == *"@"* ]]; then
        model="${entry%%@*}"
        alias="${entry##*@}"
        ALIAS_ARG="--alias $alias"
    else
        model="$entry"
        ALIAS_ARG=""
    fi
    python3 eval.py \
        --model "$model" \
        --tasks $TASK_NAME \
        --num_workers $NUM_WORKERS \
        --evaluate true \
        --score true \
        --eval_model "$EVAL_MODEL" \
        --eval_tag "$EVAL_TAG" \
        --output_dir $OUTPUT_DIR \
        --tensor_parallel_size $TENSORS \
        --config-file config.yaml \
        --external-server \
        $ALIAS_ARG
done
