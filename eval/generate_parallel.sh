#!/bin/bash

# Parallel model generation using GPU scheduler.
# Runs multiple models concurrently, packing them onto available GPUs.
#
# Usage: ./generate_parallel.sh
#
# Models are defined below with "name:num_gpus" format.
# The scheduler allocates GPUs from CUDA_VISIBLE_DEVICES (NVIDIA)
# or HIP_VISIBLE_DEVICES (AMD/ROCm) and assigns unique ports
# (starting from 8100) to each concurrent vLLM instance.

export CUDA_VISIBLE_DEVICES=0,1,2,3

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
NUM_WORKERS=64
NO_STAGGER=true

# Uncomment models to run. The scheduler will pack them onto GPUs concurrently.
# Format: "model_name:num_gpus[:omni][:hf][:audio][:mount][:f{N}][@alias]"
#
# Suffix reference:
#   :omni   — enable audio extraction from video
#   :hf     — use HuggingFace transformers backend instead of vLLM
#   :audio  — audio-only LLM (no video, sends .wav only)
#   :mount  — stream weights via hf-mount (requires $HF_TOKEN)
#   :f{N}   — max video frames (e.g. :f256)
#   @alias  — custom output directory name

MODELS=(
# --- Tier 1: Native Video + Audio ---
# "Qwen/Qwen3-Omni-30B-A3B-Instruct:1:omni"
# "Qwen/Qwen2.5-Omni-7B:1:omni"
# "openbmb/MiniCPM-o-4_5:1:omni:hf"
# "openbmb/MiniCPM-o-2_6:2:omni:hf"
# "Qwen/Qwen3-Omni-30B-A3B-Thinking:2:omni"
# "Qwen/Qwen2.5-Omni-3B:2:omni"
# "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16:4:omni"
# "RUC-NLPIR/OmniAtlas-Qwen2.5-7B:1:omni"
# "antgroup/HumanSense_Omni_Reasoning:1:omni"

# --- Tier 2: Large Video VLMs (50B+) ---
# "Qwen/Qwen3-VL-235B-A22B-Instruct:8"
# "Qwen/Qwen3-VL-235B-A22B-Thinking:8"
# "OpenGVLab/InternVL3_5-241B-A28B:8"
# "Qwen/Qwen3.5-122B-A10B:8"
# "zai-org/GLM-4.5V:4"
# "Qwen/Qwen2.5-VL-72B-Instruct:4"
# "OpenGVLab/InternVL3-78B:4"
# "OpenGVLab/InternVL2_5-78B:4"

# --- Tier 3: Mid-Size Video VLMs (10B-50B) ---
# "Qwen/Qwen3.5-27B:2"
# "Qwen/Qwen3.5-35B-A3B:2"
# "Qwen/Qwen3-VL-32B-Instruct:2"
# "Qwen/Qwen3-VL-32B-Thinking:2"
# "Qwen/Qwen3-VL-30B-A3B-Instruct:2"
# "Qwen/Qwen3-VL-30B-A3B-Thinking:2"
# "Qwen/Qwen2.5-VL-32B-Instruct:2"
# "OpenGVLab/InternVL3_5-38B:2"
# "OpenGVLab/InternVL3-38B:2"
# "OpenGVLab/InternVL2_5-38B:2"
# "OpenGVLab/InternVL3-14B:2"
# "google/gemma-4-31B-it:2:f128"
# "google/gemma-4-26B-A4B-it:2:f128"
# "google/gemma-3-27b-it:4:f128"
# "AIDC-AI/Ovis2.6-30B-A3B:2"
# "moonshotai/Kimi-VL-A3B-Thinking:2:f64"
# "moonshotai/Kimi-VL-A3B-Instruct:2:f64"
# "baidu/ERNIE-4.5-VL-28B-A3B-PT:1"
# "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16:2"
# "microsoft/Phi-4-reasoning-vision-15B:2:f8"

# --- Tier 4: Small Video/Vision Models (5B-10B) ---
# "Qwen/Qwen3.5-9B:1"
# "Qwen/Qwen3-VL-8B-Instruct:1"
# "Qwen/Qwen3-VL-8B-Thinking:1"
# "Qwen/Qwen2.5-VL-7B-Instruct:1"
# "OpenGVLab/InternVL3_5-8B:2"
# "OpenGVLab/InternVL3-8B:2"
# "Kwai-Keye/Keye-VL-1_5-8B:2"
# "allenai/Molmo2-8B:2"
# "openbmb/MiniCPM-V-4_5:1"
# "longvideotool/LongVT-RL:1"
# "Video-R1/Video-R1-7B:1"
# "OneThink/OneThinker-8B:1"
# "TIGER-Lab/VideoScore2:1"

# --- Tier 5: Compact Models (4B-5B) ---
# "Qwen/Qwen3.5-4B:1"
# "Qwen/Qwen3-VL-4B-Instruct:1"
# "Qwen/Qwen3-VL-4B-Thinking:1"
# "google/gemma-3-4b-it:2:f128"
# "OpenGVLab/InternVL3_5-4B-Instruct:1"
# "Qwen/Qwen2.5-VL-3B-Instruct:1"

# --- Tier 6: Audio-Only LLMs ---
# "Qwen/Qwen2-Audio-7B-Instruct:1:audio"
# "nvidia/audio-flamingo-3-hf:1:audio"
# "FunAudioLLM/Fun-Audio-Chat-8B:4:audio"
# "mistralai/Voxtral-Small-24B-2507:2:audio"
# "mistralai/Voxtral-Mini-3B-2507:1:audio"
)

# Filter out comments and empty lines, build args
MODEL_ARGS=()
for m in "${MODELS[@]}"; do
    MODEL_ARGS+=("$m")
done

if [ ${#MODEL_ARGS[@]} -eq 0 ]; then
    echo "No models uncommented. Edit this script to select models to run."
    exit 1
fi

python3 scheduler.py \
    --models "${MODEL_ARGS[@]}" \
    --tasks $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --num_workers $NUM_WORKERS \
    --config-file config.yaml \
    ${NO_STAGGER:+$([ "$NO_STAGGER" = "true" ] && echo "--no-stagger")}
