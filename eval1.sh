#!/bin/bash

# =============================================================================
# eval1.sh
# -----------------------------------------------------------------------------
# Purpose: Run Evaluation Group 1 (Perplexity & Performance) for RoPE methods
# -----------------------------------------------------------------------------
# Description:
#   This script evaluates LLaMA-7b model with various RoPE scaling methods
#   using perplexity and performance benchmarks. It tests the model's ability
#   to handle long contexts with different position embedding strategies.
# -----------------------------------------------------------------------------
# Usage:
#   bash eval1.sh
# -----------------------------------------------------------------------------
# Parameters: None (all configuration is done via variables below)
# -----------------------------------------------------------------------------
# Output:
#   Evaluation results printed to console and saved to respective output files
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH

echo "=========================================="
echo "Eval Group 1: Perplexity & Performance"
echo "=========================================="

export DISABLE_FLASH_ATTN=1
export USE_FLASH_ATTN=0

CUDA_DEVICES="2,3"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# ── Model Configuration ──────────────────────────────────────────────────────
MODEL_NAME="huggyllama/llama-7b"
DTYPE="auto"
QUANT="--load-in-4bit"

MAX_LENGTH=65536
MIN_LENGTH=2048

# ── Evaluation Flags ─────────────────────────────────────────────────────────
ENABLE_PERPLEXITY=true
ENABLE_PERFORMANCE=false

# ── RoPE Methods Configuration ───────────────────────────────────────────────
ROPE_METHODS=(
    # "--rope-type none"
    # "--rope-type linear --rope-dynamic"
    # "--rope-type ntk --rope-dynamic"
    # "--rope-type part-ntk --rope-dynamic"
    # "--rope-type yarn --rope-dynamic"
    "--rope-type freq-reciprocal --rope-dynamic"
    # "--rope-type freq-reciprocal-scaled --rope-dynamic"
    # "--rope-type freq-reciprocal-scaled-no-layer --rope-dynamic"
    # "--rope-type freq-reciprocal-scaled-adaptive --rope-dynamic"
    # "--rope-type freq-reciprocal-scaled-adaptive --rope-factor 4.0"
)

echo "=========================================="
echo "Configuration"
echo "=========================================="
echo "Model           : ${MODEL_NAME}"
echo "Min length      : ${MIN_LENGTH}"
echo "Max length      : ${MAX_LENGTH}"
echo "Quantization    : ${QUANT}"
echo "RoPE methods    : ${#ROPE_METHODS[@]}"
echo "Perplexity      : ${ENABLE_PERPLEXITY}"
echo "Performance     : ${ENABLE_PERFORMANCE}"
echo "=========================================="

# -----------------------------------------------------------------------------
# Function: run_eval
# -----------------------------------------------------------------------------
# Purpose: Execute evaluation for a specific evaluation type and RoPE method
# -----------------------------------------------------------------------------
# Arguments:
#   $1 - eval_type: The type of evaluation (perplexity or performance)
#   $2 - rope_method: The RoPE configuration string
# -----------------------------------------------------------------------------
# Returns:
#   0 on success, non-zero on failure
# -----------------------------------------------------------------------------
run_eval() {
    local eval_type=$1
    local rope_method=$2
    
    echo ""
    echo "------------------------------------------"
    echo "Eval: $eval_type | $rope_method"
    echo "------------------------------------------"
    
    local cmd="python ${SCRIPT_DIR}/eval/${eval_type}.py \
        --model-name ${MODEL_NAME} \
        ${rope_method} \
        --max-length ${MAX_LENGTH} \
        --min-length ${MIN_LENGTH} \
        --dtype ${DTYPE} \
        ${QUANT}"
    
    echo "Executing: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Eval completed: $eval_type with $rope_method"
    else
        echo "[FAILED] Eval failed: $eval_type with $rope_method"
    fi
}

if [ "$ENABLE_PERPLEXITY" = true ]; then
    echo ""
    echo "=========================================="
    echo "Eval Type: perplexity"
    echo "=========================================="
    
    for rope_method in "${ROPE_METHODS[@]}"; do
        run_eval "perplexity" "$rope_method"
    done
fi

if [ "$ENABLE_PERFORMANCE" = true ]; then
    echo ""
    echo "=========================================="
    echo "Eval Type: performance"
    echo "=========================================="
    
    for rope_method in "${ROPE_METHODS[@]}"; do
        run_eval "performance" "$rope_method"
    done
fi

echo ""
echo "=========================================="
echo "Eval Group 1 completed!"
echo "=========================================="
