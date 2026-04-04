#!/bin/bash

# =============================================================================
# eval_harness.sh
# -----------------------------------------------------------------------------
# Purpose: Run lm-eval-harness evaluation for RoPE methods and fine-tuned adapters
# -----------------------------------------------------------------------------
# Description:
#   This script performs a two-phase evaluation using lm-eval-harness:
#   Phase 1: Evaluates base model with various dynamic RoPE methods
#   Phase 2: Evaluates fine-tuned adapters trained with different RoPE methods
# -----------------------------------------------------------------------------
# Usage:
#   bash eval_harness.sh
# -----------------------------------------------------------------------------
# Parameters:
#   None (all configuration is done via variables below)
# -----------------------------------------------------------------------------
# Globals:
#   PYTHONPATH             - Project parent directory added to Python module search path
#   CUDA_VISIBLE_DEVICES   - GPU device IDs for computation (default: "0,1,2,3")
# -----------------------------------------------------------------------------
# Output:
#   Evaluation results saved to: results/harness/
# =============================================================================

PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="huggyllama/llama-7b"
MAX_LENGTH=16384
DTYPE="auto"

# ── Evaluation Tasks Configuration ───────────────────────────────────────────
TASKS="longbench2,arc_challenge,hellaswag,truthfulqa_mc1,mmlu"
BATCH_SIZE=2
OUTPUT_DIR="results/harness"
QUANT="--load-in-4bit"

CUDA_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# ── RoPE Methods Configuration (Phase 1) ──────────────────────────────────────
ROPE_METHODS=(
    "--rope-type none"
    "--rope-type linear --rope-dynamic"
    "--rope-type ntk --rope-dynamic"
    "--rope-type part-ntk --rope-dynamic"
    "--rope-type yarn --rope-dynamic"
    "--rope-type freq-reciprocal --rope-dynamic"
    "--rope-type freq-reciprocal-scaled --rope-dynamic"
)

# ── Adapter Configuration (Phase 2) ───────────────────────────────────────────
ADAPTER_DIR="finetunes/continued_pretrain"
ADAPTERS=(
    "none_20260315_003356"
    "linear_20260315_081529"
    "ntk_20260315_155711"
    "part-ntk_20260315_233845"
    "yarn_20260316_071953"
    "freq-reciprocal_20260317_001708"
    "freq-reciprocal-scaled_20260320_003434"
    "freq-reciprocal-scaled-no-layer_20260324_014910"
)

echo "=========================================="
echo "lm-eval Harness Evaluation"
echo "=========================================="
echo "Base model   : ${MODEL_NAME}"
echo "Max length   : ${MAX_LENGTH}"
echo "Tasks        : ${TASKS}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "RoPE methods : ${#ROPE_METHODS[@]}"
echo "Adapter dir  : ${ADAPTER_DIR}"
echo "Adapters     : ${#ADAPTERS[@]}"
echo "=========================================="

# -----------------------------------------------------------------------------
# Function: run_eval
# -----------------------------------------------------------------------------
# Purpose: Execute lm-eval-harness evaluation with given arguments
# -----------------------------------------------------------------------------
# Args:
#   $1 - args: Additional command-line arguments (RoPE method or adapter path)
# -----------------------------------------------------------------------------
# Returns:
#   0 on success, non-zero on failure
#   Stdout: Evaluation progress and benchmark scores
# -----------------------------------------------------------------------------
run_eval() {
    local args=$1

    echo ""
    echo "------------------------------------------"
    echo "Evaluating: ${args}"
    echo "------------------------------------------"

    local cmd="python eval/eval_harness.py \
        --model-name ${MODEL_NAME} \
        --max-length ${MAX_LENGTH} \
        --dtype ${DTYPE} \
        ${QUANT} \
        ${args} \
        --tasks ${TASKS} \
        --batch-size ${BATCH_SIZE} \
        --output-dir ${OUTPUT_DIR} \
        --log-samples"

    echo "Executing: ${cmd}"
    eval ${cmd}
}

# ── Phase 1: Dynamic RoPE Evaluation ──────────────────────────────────────────
echo ""
echo "=========================================="
echo "Phase 1: Dynamic evaluation"
echo "=========================================="

for rope_method in "${ROPE_METHODS[@]}"; do
    run_eval "${rope_method}"
done

# ── Phase 2: Fine-tuned Adapter Evaluation ────────────────────────────────────
if [ ${#ADAPTERS[@]} -gt 0 ]; then
    echo ""
    echo "=========================================="
    echo "Phase 2: Fine-tuned adapter evaluation"
    echo "=========================================="

    for adapter_entry in "${ADAPTERS[@]}"; do
        run_eval "--adapter_path ${ADAPTER_DIR}/${adapter_entry}"
    done
fi

echo ""
echo "All evaluations complete! Results saved to: ${OUTPUT_DIR}/"
echo "=========================================="
