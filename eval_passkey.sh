#!/bin/bash

# =============================================================================
# eval_passkey.sh
# -----------------------------------------------------------------------------
# Purpose: Passkey retrieval evaluation script for long-context language models
# -----------------------------------------------------------------------------
# Description:
#   This script evaluates the model's ability to retrieve passkeys embedded
#   in long contexts across different RoPE methods or fine-tuned adapters.
#   Tests long-range dependency and retrieval capabilities.
# -----------------------------------------------------------------------------
# Usage:
#   bash eval_passkey.sh
# -----------------------------------------------------------------------------
# Output:
#   Evaluation results saved to: results/passkey/
# =============================================================================

echo "=========================================="
echo "Passkey Retrieval Evaluation Script"
echo "=========================================="

PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=$PYTHONPATH

export DISABLE_FLASH_ATTN=1
export USE_FLASH_ATTN=0
export HF_ALLOW_CODE_EVAL=1

CUDA_DEVICES="1,2,3"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# ── Model Configuration ──────────────────────────────────────────────────────
MODEL_NAME="huggyllama/llama-7b"
DTYPE="auto"
QUANT="--load-in-4bit"

MAX_LENGTH=65536
MIN_LENGTH=2048
OUTPUT_DIR="results"

# ── Evaluation Mode Flags ────────────────────────────────────────────────────
ROPE=true
ADAPTER=false
ADAPTER_DIR="finetunes/continued_pretrain"

# ── RoPE Methods Configuration ───────────────────────────────────────────────
ROPE_METHODS=(
    "--rope-type none"
    "--rope-type linear --rope-dynamic"
    "--rope-type ntk --rope-dynamic"
    "--rope-type part-ntk --rope-dynamic"
    "--rope-type yarn --rope-dynamic"
    "--rope-type inverse-dual-rope --rope-dynamic"
    "--rope-type inverse-dual-rope-scaled --rope-dynamic"
    # "--rope-type freq-reciprocal --rope-dynamic"
    # "--rope-type freq-reciprocal-scaled --rope-dynamic"
    # "--rope-type freq-reciprocal-scaled-no-layer --rope-dynamic"
    # "--rope-type dual-rope --rope-dynamic"
    # "--rope-type dual-rope-scaled --rope-dynamic"
    # "--rope-type inverse-dual-tangle-rope --rope-dynamic"
    # "--rope-type inverse-dual-tangle-rope-scaled --rope-dynamic"
    # "--rope-type inverse-dual-nopos-rope --rope-dynamic"
    # "--rope-type inverse-dual-nopos-rope-scaled --rope-dynamic"
)

# ── Adapter Paths Configuration ──────────────────────────────────────────────
ADAPTER_PATHS=(
    "--adapter-path ${ADAPTER_DIR}/inverse-dual-rope-scaled_20260406_070155"
    "--adapter-path ${ADAPTER_DIR}/inverse-dual-rope_20260403_103555"
    "--adapter-path ${ADAPTER_DIR}/yarn_20260316_071953"
    "--adapter-path ${ADAPTER_DIR}/part-ntk_20260315_233845"
    "--adapter-path ${ADAPTER_DIR}/ntk_20260315_155711"
    "--adapter-path ${ADAPTER_DIR}/linear_20260315_081529"
    "--adapter-path ${ADAPTER_DIR}/none_20260315_003356"
    # "--adapter-path ${ADAPTER_DIR}/freq-reciprocal-scaled-no-layer_20260324_014910"
    # "--adapter-path ${ADAPTER_DIR}/freq-reciprocal_20260317_001708"
    # "--adapter-path ${ADAPTER_DIR}/dual-rope_20260402_113443"
    # "--adapter-path ${ADAPTER_DIR}/freq-reciprocal-scaled_20260320_003434"
)

# ── Base Adapter Combinations ────────────────────────────────────────────
BASE_COMBOS=()

# ── Build Methods List ───────────────────────────────────────────────────────
METHODS=()
if [ $ROPE = true ]; then
    METHODS+=("${ROPE_METHODS[@]}")
fi
if [ $ADAPTER = true ]; then
    METHODS+=("${ADAPTER_PATHS[@]}")
fi

for combo in "${BASE_COMBOS[@]}"; do
    IFS='|' read -r base_path rope_type rest <<< "$combo"
    METHODS+="--base-adapter-path ${ADAPTER_DIR}/${base_path} ${rope_type} ${rest}"
done

# ── Passkey Arguments ────────────────────────────────────────────────────────
PASSKEY_ARGS="--num-keys 5 \
--iterations 20 \
--data-mode real \
--dataset-name konwoo/RedPajama-Data-1T-Sample-subset1000 \
--split train \
--aggressive-memory True \
--restrict-tokens True \
--save-dir ${OUTPUT_DIR}/passkey"

echo "=========================================="
echo "Configuration"
echo "=========================================="
echo "Model          : ${MODEL_NAME}"
echo "Max length     : ${MAX_LENGTH}"
echo "Min length     : ${MIN_LENGTH}"
echo "Quantization   : ${QUANT}"
echo "Methods        : ${#METHODS[@]}"
echo "=========================================="

# -----------------------------------------------------------------------------
# Function: run_passkey_eval
# -----------------------------------------------------------------------------
run_passkey_eval() {
    local method=$1

    echo ""
    echo "------------------------------------------"
    echo "Eval: passkey | Method: $method"
    echo "------------------------------------------"

    local cmd="python eval/passkey.py \
        --model-name ${MODEL_NAME} \
        ${method} \
        --max-length ${MAX_LENGTH} \
        --min-length ${MIN_LENGTH} \
        --dtype ${DTYPE} \
        ${QUANT} \
        ${PASSKEY_ARGS}"

    echo "Executing: $cmd"
    eval $cmd

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Passkey eval completed: $method"
    else
        echo "[FAILED] Passkey eval failed: $method"
    fi
}

echo "=========================================="
echo "Starting Passkey Evaluation"
echo "=========================================="

for method in "${METHODS[@]}"; do
    run_passkey_eval "$method"
done

echo ""
echo "=========================================="
echo "Passkey Evaluation Completed!"
echo "=========================================="
