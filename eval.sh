#!/bin/bash

# =============================================================================
# eval.sh
# -----------------------------------------------------------------------------
# Purpose: Comprehensive evaluation script for long-context language models
# -----------------------------------------------------------------------------
# Description:
#   This script provides a unified evaluation framework that supports multiple
#   evaluation types including perplexity, performance, passkey retrieval, and
#   lm-eval-harness benchmarks. It can evaluate both base models with RoPE
#   methods and fine-tuned adapters.
#
#   Additionally supports base adapter + target RoPE combinations via BASE_COMBOS,
#   allowing evaluation of adapter stacking scenarios (e.g., evaluating
#   inverse-dual-rope-scaled on top of inverse-dual-rope).
# -----------------------------------------------------------------------------
# Usage:
#   bash eval.sh
# -----------------------------------------------------------------------------
# Parameters:
#   None (all configuration is done via variables below)
# -----------------------------------------------------------------------------
# Globals:
#   PYTHONPATH             - Project root added to Python module search path
#   CUDA_VISIBLE_DEVICES   - GPU device IDs for computation (default: "2,3")
#   DISABLE_FLASH_ATTN     - Disable flash attention (set to 1)
#   USE_FLASH_ATTN         - Flash attention flag (set to 0)
#   HF_ALLOW_CODE_EVAL     - Allow HuggingFace code evaluation (set to 1)
# -----------------------------------------------------------------------------
# Output:
#   Evaluation results saved to: results/
# =============================================================================

echo "=========================================="
echo "Evaluation Script"
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
BATCH_SIZE=1
OUTPUT_DIR="results"

# ── Evaluation Mode Flags ────────────────────────────────────────────────────
ROPE=true
ADAPTER=false
ADAPTER_DIR="finetunes/continued_pretrain"

# ── RoPE Methods Configuration ───────────────────────────────────────────────
ROPE_METHODS=(
    # "--rope-type none"
    # "--rope-type linear --rope-dynamic"
    # "--rope-type ntk --rope-dynamic"
    # "--rope-type part-ntk --rope-dynamic"
    # "--rope-type yarn --rope-dynamic"
    # "--rope-type freq-reciprocal --rope-dynamic"
    # "--rope-type freq-reciprocal-scaled --rope-dynamic"
    # "--rope-type freq-reciprocal-scaled-no-layer --rope-dynamic"
    # "--rope-type dual-rope --rope-dynamic"
    # "--rope-type dual-rope-scaled --rope-dynamic"
    "--rope-type inverse-dual-rope --rope-dynamic"
    # "--rope-type inverse-dual-rope-scaled --rope-dynamic"
    # "--rope-type inverse-dual-tangle-rope --rope-dynamic"
    # "--rope-type inverse-dual-tangle-rope-scaled --rope-dynamic"
    # "--rope-type inverse-dual-nopos-rope --rope-dynamic"
    # "--rope-type inverse-dual-nopos-rope-scaled --rope-dynamic"
)

# ── Adapter Paths Configuration ──────────────────────────────────────────────
ADAPTER_PATHS=(
    "--adapter-path ${ADAPTER_DIR}/inverse-dual-rope_20260403_103555"
    "--adapter-path ${ADAPTER_DIR}/yarn_20260316_071953"
    "--adapter-path ${ADAPTER_DIR}/part-ntk_20260315_233845"
    "--adapter-path ${ADAPTER_DIR}/ntk_20260315_155711"
    "--adapter-path ${ADAPTER_DIR}/linear_20260315_081529"
    "--adapter-path ${ADAPTER_DIR}/none_20260315_003356"
    "--adapter-path ${ADAPTER_DIR}/freq-reciprocal-scaled-no-layer_20260324_014910"
    "--adapter-path ${ADAPTER_DIR}/freq-reciprocal_20260317_001708"
    "--adapter-path ${ADAPTER_DIR}/dual-rope_20260402_113443"
    "--adapter-path ${ADAPTER_DIR}/freq-reciprocal-scaled_20260320_003434"
)

# ── Base Adapter Combinations ────────────────────────────────────────────
# Optional: Define base adapter + target RoPE combinations for evaluation.
# Each entry should be: "base_path|target_rope_type|dynamic_flag"
# Format: "--base-combo <base_adapter>|<rope_type>[|--rope-dynamic]"
#
# Example: Evaluate inverse-dual-rope-scaled on top of inverse-dual-rope
# BASE_COMBOS=(
#     "finetunes/inverse-dual-rope_20260403|inverse-dual-rope-scaled|--rope-dynamic"
# )
BASE_COMBOS=()

# ── Build Methods List ───────────────────────────────────────────────────────
METHODS=()
if [ $ROPE = true ]; then
    METHODS+=("${ROPE_METHODS[@]}")
fi
if [ $ADAPTER = true ]; then
    METHODS+=("${ADAPTER_PATHS[@]}")
fi

# Add base adapter combinations (parsed into --base-adapter-path + --rope-type)
for combo in "${BASE_COMBOS[@]}"; do
    IFS='|' read -r base_path rope_type rest <<< "$combo"
    METHODS+="--base-adapter-path ${ADAPTER_DIR}/${base_path} ${rope_type} ${rest}"
done

# ── Evaluation Type Flags ────────────────────────────────────────────────────
PERPLEXITY=true
PERFORMANCE=false
PASSKEY=false
EVAL_HARNESS=false

# ── Evaluation Arguments ─────────────────────────────────────────────────────
PERPLEXITY_ARGS="--dataset-name emozilla/proofpile-test-tokenized \
--split test \
--limit 100 \
--add-start-token True \
--sliding-window 256 \
--truncate True \
--aggressive-memory True \
--save-dir ${OUTPUT_DIR}/perplexity"

PERFORMANCE_ARGS="--save-dir ${OUTPUT_DIR}/performance"

PASSKEY_ARGS="--num-keys 5 \
--iterations 20 \
--data-mode real \
--dataset-name konwoo/RedPajama-Data-1T-Sample-subset1000 \
--split train \
--aggressive-memory True \
--restrict-tokens True \
--save-dir ${OUTPUT_DIR}/passkey"

# Available tasks:
# long_context: longbench,longbench2,longcxt,passkey,ruler,babilong
# reasoning: arc_challenge,truthfulqa,hellaswag,bbh,mmlu
# math: gsm8k,aime,hendrycks_math
# code: humaneval,mbpp,humaneval_infilling
# Too long tasks: bbh
TASKS="humananeval,mbpp,humananeval_infilling"
EVAL_HARNESS_ARGS="--tasks ${TASKS} --batch-size ${BATCH_SIZE} --output-dir ${OUTPUT_DIR}/eval_harness"

echo "=========================================="
echo "Configuration"
echo "=========================================="
echo "Model          : ${MODEL_NAME}"
echo "Max length     : ${MAX_LENGTH}"
echo "Min length     : ${MIN_LENGTH}"
echo "Quantization   : ${QUANT}"
echo "Methods        : ${#METHODS[@]}"
echo "Base Combos    : ${#BASE_COMBOS[@]}"
if [ ${#BASE_COMBOS[@]} -gt 0 ]; then
    echo "  (Will evaluate base adapter + target RoPE combinations)"
fi
echo "Perplexity     : ${PERPLEXITY}"
echo "Performance    : ${PERFORMANCE}"
echo "Passkey        : ${PASSKEY}"
echo "Eval Harness   : ${EVAL_HARNESS}"
echo "=========================================="

# -----------------------------------------------------------------------------
# Function: run_perplexity_eval
# -----------------------------------------------------------------------------
# Purpose: Execute perplexity evaluation for a specific method
# -----------------------------------------------------------------------------
# Args:
#   $1 - method: RoPE method string or adapter path
# -----------------------------------------------------------------------------
# Returns:
#   0 on success, non-zero on failure
#   Stdout: Evaluation progress and result messages
# -----------------------------------------------------------------------------
run_perplexity_eval() {
    local method=$1

    echo ""
    echo "------------------------------------------"
    echo "Eval: perplexity | Method: $method"
    echo "------------------------------------------"

    local cmd="python eval/perplexity.py \
        --model-name ${MODEL_NAME} \
        ${method} \
        --max-length ${MAX_LENGTH} \
        --min-length ${MIN_LENGTH} \
        --dtype ${DTYPE} \
        ${QUANT} \
        ${PERPLEXITY_ARGS}"

    echo "Executing: $cmd"
    eval $cmd

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Perplexity eval completed: $method"
    else
        echo "[FAILED] Perplexity eval failed: $method"
    fi
}

# -----------------------------------------------------------------------------
# Function: run_performance_eval
# -----------------------------------------------------------------------------
# Purpose: Execute performance benchmark for a specific method
# -----------------------------------------------------------------------------
# Args:
#   $1 - method: RoPE method string or adapter path
# -----------------------------------------------------------------------------
# Returns:
#   0 on success, non-zero on failure
#   Stdout: Benchmark progress and result messages
# -----------------------------------------------------------------------------
run_performance_eval() {
    local method=$1

    echo ""
    echo "------------------------------------------"
    echo "Eval: performance | Method: $method"
    echo "------------------------------------------"

    local cmd="python eval/performance.py \
        --model-name ${MODEL_NAME} \
        ${method} \
        --max-length ${MAX_LENGTH} \
        --min-length ${MIN_LENGTH} \
        --dtype ${DTYPE} \
        ${QUANT} \
        ${PERFORMANCE_ARGS}"

    echo "Executing: $cmd"
    eval $cmd

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Performance eval completed: $method"
    else
        echo "[FAILED] Performance eval failed: $method"
    fi
}

# -----------------------------------------------------------------------------
# Function: run_passkey_eval
# -----------------------------------------------------------------------------
# Purpose: Execute passkey retrieval evaluation for a specific method
# -----------------------------------------------------------------------------
# Args:
#   $1 - method: RoPE method string or adapter path
# -----------------------------------------------------------------------------
# Returns:
#   0 on success, non-zero on failure
#   Stdout: Retrieval accuracy and result messages
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

# -----------------------------------------------------------------------------
# Function: run_eval_harness_eval
# -----------------------------------------------------------------------------
# Purpose: Execute lm-eval-harness evaluation for a specific method
# -----------------------------------------------------------------------------
# Args:
#   $1 - method: RoPE method string or adapter path
# -----------------------------------------------------------------------------
# Returns:
#   0 on success, non-zero on failure
#   Stdout: Harness benchmark scores and result messages
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
echo "Starting Evaluation"
echo "=========================================="

if [ "$PERPLEXITY" = true ]; then
    for method in "${METHODS[@]}"; do
        run_perplexity_eval "$method"
    done
fi

if [ "$PERFORMANCE" = true ]; then
    for method in "${METHODS[@]}"; do
        run_performance_eval "$method"
    done
fi

if [ "$PASSKEY" = true ]; then
    for method in "${METHODS[@]}"; do
        run_passkey_eval "$method"
    done
fi

if [ "$EVAL_HARNESS" = true ]; then
    # Cap max length at 16384 for eval harness to avoid OOM errors
    if [ $MAX_LENGTH -gt 16384 ]; then
        MAX_LENGTH=16384
        echo "Max length is greater than 16384, setting to 16384"
    fi
    for method in "${METHODS[@]}"; do
        run_eval_harness_eval "$method"
    done
fi

echo ""
echo "=========================================="
echo "All Evaluations Completed!"
echo "=========================================="
