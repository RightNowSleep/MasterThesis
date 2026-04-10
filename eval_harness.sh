#!/bin/bash

# =============================================================================
# eval_harness.sh
# -----------------------------------------------------------------------------
# Purpose: LM-Eval-Harness benchmark script for long-context language models
# -----------------------------------------------------------------------------
# Description:
#   This script runs lm-eval-harness benchmarks across different RoPE methods
#   or fine-tuned adapters. Supports various task categories including reasoning,
#   math, code, and long-context tasks.
# -----------------------------------------------------------------------------
# Usage:
#   bash eval_harness.sh
# -----------------------------------------------------------------------------
# Output:
#   Evaluation results saved to: results/eval_harness/
# =============================================================================

echo "=========================================="
echo "LM-Eval-Harness Evaluation Script"
echo "=========================================="

PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=$PYTHONPATH

export DISABLE_FLASH_ATTN=1
export USE_FLASH_ATTN=0
export HF_ALLOW_CODE_EVAL=1

CUDA_DEVICES="2,3"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# ── Model Configuration ──────────────────────────────────────────────────────
MODEL_NAME="huggyllama/llama-7b"
DTYPE="auto"
QUANT="--load-in-4bit"

MAX_LENGTH=65536
MIN_LENGTH=2048
BATCH_SIZE=1
OUTPUT_DIR="results"

# ── Evaluation Mode Flags ────────────────────────────────────────────────────
ROPE=false
ADAPTER=true
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

# ── Eval Harness Arguments ───────────────────────────────────────────────────
# Available tasks:
# niah: niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue
# long_context: longbench,longbench2,longcxt,passkey,ruler,babilong
# reasoning: arc_challenge,truthfulqa,hellaswag,bbh,mmlu
# math: gsm8k,aime,hendrycks_math,mathqa,arithmetic
# code: humaneval,mbpp,codex2text
# Too long tasks: bbh
# false tasks: humaneval_infilling,aime
# new tasks: asdiv,bbq,hendrycks_math500,triviaqa,math_word_problems,hrm8k,agieval_math
TASKS="mbpp"
EVAL_HARNESS_ARGS="--tasks ${TASKS} --batch-size ${BATCH_SIZE} --output-dir ${OUTPUT_DIR}/eval_harness"

# Cap max length at 16384 for eval harness to avoid OOM errors
if [ $MAX_LENGTH -gt 16384 ]; then
    MAX_LENGTH=16384
    echo "Max length is greater than 16384, setting to 16384"
fi

echo "=========================================="
echo "Configuration"
echo "=========================================="
echo "Model          : ${MODEL_NAME}"
echo "Max length     : ${MAX_LENGTH}"
echo "Min length     : ${MIN_LENGTH}"
echo "Quantization   : ${QUANT}"
echo "Methods        : ${#METHODS[@]}"
echo "Tasks          : ${TASKS}"
echo "=========================================="

# -----------------------------------------------------------------------------
# Function: run_eval_harness_eval
# -----------------------------------------------------------------------------
run_eval_harness_eval() {
    local method=$1

    echo ""
    echo "------------------------------------------"
    echo "Eval: eval_harness | Method: $method"
    echo "------------------------------------------"

    local cmd="python eval/eval_harness.py \
        --model-name ${MODEL_NAME} \
        ${method} \
        --max-length ${MAX_LENGTH} \
        --min-length ${MAX_LENGTH} \
        --dtype ${DTYPE} \
        ${QUANT} \
        ${EVAL_HARNESS_ARGS}"

    echo "Executing: $cmd"
    eval $cmd

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Eval harness completed: $method"
    else
        echo "[FAILED] Eval harness failed: $method"
    fi
}

echo "=========================================="
echo "Starting LM-Eval-Harness Evaluation"
echo "=========================================="

for method in "${METHODS[@]}"; do
    run_eval_harness_eval "$method"
done

echo ""
echo "=========================================="
echo "LM-Eval-Harness Evaluation Completed!"
echo "=========================================="
