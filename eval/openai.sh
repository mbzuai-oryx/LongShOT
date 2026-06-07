#!/bin/bash
# OpenAI API evaluation for LongShOT.
#
# Extra commands:
#   ./openai.sh check        # Check batch status
#   ./openai.sh resume       # Download batch results

set -euo pipefail
[ -f .env ] && set -a && source .env && set +a

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY not set"; exit 1
fi

# ── What to run ───────────────────────────────────────────────────────────

GENERATE=false
EVALUATE=true

# ── Settings ──────────────────────────────────────────────────────────────

MODEL="gpt-5-mini-2025-08-07"
MODE="batch"                       # realtime | batch (50% off, 24h)
TASK_NAME="postvalid_v2"
OUTPUT_DIR="results_postvalid"
MAX_FRAMES=128
DETAIL="low"                       # low (85 tok/img) | high
NUM_WORKERS=16
LIMIT=0                           # 0 = unlimited

# ── Models to evaluate ────────────────────────────────────────────────────

CANDIDATE_MODELS=(
# --- Tier 1: Native Video + Audio (Full Modality Match) ---
# "Qwen3_Omni_30B_A3B_Instruct"
# "Qwen3_Omni_30B_A3B_Thinking"
# "Qwen2_5_Omni_7B"
# "MiniCPM_o_4_5"
# "MiniCPM_o_2_6"
# "Ming_flash_omni_2_0"
# "Phi_4_multimodal_instruct"
# "Nemotron_3_Nano_Omni_30B_A3B_Reasoning_BF16"
# "omnivinci"
# "Baichuan_Omni_1d5"
# "Qwen2_5_Omni_3B"
# "OmniAtlas_Qwen2_5_7B"
# "HumanSense_Omni_Reasoning"

# --- Tier 2: Large Video VLMs (50B+) ---
# "Qwen3_VL_235B_A22B_Instruct"
# "Qwen3_VL_235B_A22B_Thinking"
# "InternVL3_5_241B_A28B"
# "Qwen3_5_122B_A10B"
# "GLM_4_5V"
# "Qwen2_5_VL_72B_Instruct"
# "InternVL3_78B"
# "InternVL2_5_78B"
# "Kimi_K2_6"

# --- Tier 3: Mid-Size Video VLMs (10B-50B) ---
# "Qwen3_5_27B"
# "Qwen3_5_35B_A3B"
# "Qwen3_VL_32B_Instruct"
# "Qwen3_VL_32B_Thinking"
# "Qwen3_VL_30B_A3B_Instruct"
# "Qwen3_VL_30B_A3B_Thinking"
# "Qwen2_5_VL_32B_Instruct"
# "InternVL3_5_38B"
# "InternVL3_38B"
# "InternVL2_5_38B"
# "InternVL3_14B"
# "gemma_3_27b_f32"
# "Ovis2_6_30B_A3B"
# "Kimi_VL_A3B_Thinking"
# "Kimi_VL_A3B_Instruct"
# "Ovis2_16B"
# "gemma_3_12b_f32"
# "Step3_VL_10B"
# "GLM_4_6V_Flash"
# "GLM_4_1V_9B_Thinking"
# "ERNIE_4_5_VL_28B_A3B_PT"
# "Qwen3_6_35B_A3B_v2"
# "NVIDIA_Nemotron_Nano_12B_v2_VL_BF16"
# "Qwen3_6_27B"
# "AURA"

# --- Tier 4: Small Video/Vision Models (5B-10B) ---
# "Qwen3_5_9B"
# "Qwen3_VL_8B_Instruct"
# "Qwen3_VL_8B_Thinking"
# "Qwen2_5_VL_7B_Instruct"
# "InternVL3_5_8B"
# "InternVL3_8B"
# "Keye_VL_1_5_8B"
# "Molmo2_8B"
# "MiMo_VL_7B_RL_2508"
# "Ovis2_5_9B"
# "Ovis2_8B"
# "llava_onevision_qwen2_7b_ov_hf"
# "MiniCPM_V_4_5"
# "Intern_S1_mini"
# "LongVT_RL"
# "gemma_4_31B_f256"
# "gemma_4_26B_A4B_it"
# "Tempo_6B"
# "Video_R1_7B"
# "Video_R2"
# "OneThinker_8B"
# "Qwen2_5_VL_7B_VRAG"
# "VideoRFT"
# "RePlan_Qwen2_5_VL_7B"
# "SenseNova_SI_1_3_InternVL3_8B"

# --- Tier 5: Compact Models (4B-5B) ---
# "Qwen3_5_4B"
# "Qwen3_VL_4B_Instruct"
# "Qwen3_VL_4B_Thinking"
# "gemma_3_4b_f32"
# "gemma_4_E4B_f256"
# "gemma_4_E2B_f256"
# "InternVL3_5_4B_Instruct"
# "Ovis2_4B"
# "Qwen3_5_2B"
# "Qwen3_VL_2B_Instruct"
# "Qwen3_5_0_8B"
# "SpaceThinker_Qwen2_5VL_3B"

# --- HF-backend models ---
# "VideoLLaMA3_7B"
# "video_SALMONN2_plus_7B_full"

# --- Tier 6: Audio-Only LLMs ---
# "Qwen2_Audio_7B_Instruct"
# "Kimi_Audio_7B_Instruct"
# "Voxtral_Small_24B_2507"
# "Voxtral_Mini_3B_2507"
# "Step_Audio_R1_1"
# "Step_Audio_2_mini"
# "audio_flamingo_3_hf"
# "Fun_Audio_Chat_8B"

# --- API-backed external models ---
# "gemini_3_1_pro_preview"

# --- Models missing gpt_5_mini judge ---
# "EchoInk_R1_7B"
# "Monet_7B"
# "Phi_4_reasoning_vision_15B"
# "Qwen2_5_VL_3B_Instruct"
# "R1_Onevision_7B"
# "R_4B"
# "VideoScore2"
# "Voxtral_Small_24B_2507"

# --- Models missing all judges ---
# "DeepEyes_7B"
# "PixelReasoner_RL_v1"
# "TimeSearch_R"
# "Vero_Qwen3I_8B"
# "VideoChat_R1_7B"
)

EVAL_TAG=""
BATCH_ID=""

# ── Run ───────────────────────────────────────────────────────────────────

LIMIT_FLAG=""
[ "$LIMIT" -gt 0 ] 2>/dev/null && LIMIT_FLAG="--limit $LIMIT"

# Handle extra commands (check/resume)
case "${1:-run}" in
    check)
        python3 openai_bench.py --check-batches --output-dir $OUTPUT_DIR --model $MODEL
        exit 0
        ;;
    resume)
        [ -z "$BATCH_ID" ] && echo "Error: Set BATCH_ID first" && exit 1
        python3 openai_bench.py --resume-batch $BATCH_ID \
            --tasks $TASK_NAME --output-dir $OUTPUT_DIR --model $MODEL \
            --candidate-model "${CANDIDATE_MODELS[0]}" \
            ${EVAL_TAG:+--eval-tag "$EVAL_TAG"}
        exit 0
        ;;
esac

if $GENERATE; then
    echo "=== Generating with $MODEL ($MODE, ${MAX_FRAMES} frames) ==="
    python3 openai_bench.py \
        --tasks $TASK_NAME --output-dir $OUTPUT_DIR --model $MODEL \
        --mode $MODE --max-frames $MAX_FRAMES --detail $DETAIL \
        --num-workers $NUM_WORKERS $LIMIT_FLAG
    echo ""
fi

if $EVALUATE; then
    # Submit and poll eval batches for all candidate models in parallel
    LOG_DIR="/tmp/eval_logs_$$"
    mkdir -p "$LOG_DIR"
    EVAL_PIDS=()
    for CM in "${CANDIDATE_MODELS[@]}"; do
        (
            python3 openai_bench.py --evaluate \
                --tasks $TASK_NAME --output-dir $OUTPUT_DIR --model $MODEL \
                --mode $MODE --candidate-model "$CM" \
                --num-workers $NUM_WORKERS $LIMIT_FLAG \
                ${EVAL_TAG:+--eval-tag "$EVAL_TAG"} \
                2>&1 | sed "s/^/[$CM] /"
        ) &
        EVAL_PIDS+=($!)
        echo "Launched eval for $CM (PID ${EVAL_PIDS[-1]})"
    done

    # Wait for all eval jobs
    FAIL=0
    for i in "${!EVAL_PIDS[@]}"; do
        CM="${CANDIDATE_MODELS[$i]}"
        if wait "${EVAL_PIDS[$i]}"; then
            echo "=== Eval done: $CM ==="
        else
            echo "=== Eval FAILED: $CM (exit $?) ==="
            FAIL=1
        fi
    done
    rm -rf "$LOG_DIR"

    [ "$FAIL" -ne 0 ] && echo "WARNING: Some eval jobs failed" >&2

    # Score sequentially (fast, CPU-only)
    for CM in "${CANDIDATE_MODELS[@]}"; do
        echo "=== Scoring $CM (judge: $MODEL) ==="
        python3 eval.py \
            --tasks $TASK_NAME --output_dir $OUTPUT_DIR --model "$CM" \
            --evaluate false --score true \
            --eval_model "$MODEL" \
            ${EVAL_TAG:+--eval_tag "$EVAL_TAG"} \
            --config-file config.yaml
        echo ""
    done
fi
