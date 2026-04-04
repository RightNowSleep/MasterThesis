#!/bin/bash

# =============================================================================
# search_params.sh
# -----------------------------------------------------------------------------
# Purpose: Run attention scale parameter search for RoPE scaling optimization
# -----------------------------------------------------------------------------
# Description:
#   This script searches for optimal parameters in the attention scaling formula:
#       mscale_i(t) = 1 + alpha * max(0, (ln(max(1, floor(t/b_i))) - ln(L_0)) / ln(L_0))
#
#   where alpha (attn_scale_coef) is the parameter to optimize.
#   The optimization objective is to minimize perplexity across multiple context
#   lengths using configurable search strategies (grid, random, Bayesian, BOHB,
#   or adaptive).
# -----------------------------------------------------------------------------
# Usage:
#   bash search_params.sh
# -----------------------------------------------------------------------------
# Parameters:
#   None (all configuration is done via variables below)
# -----------------------------------------------------------------------------
# Globals:
#   PYTHONPATH             - Project root added to Python module search path
#   CUDA_VISIBLE_DEVICES   - GPU device IDs for computation (default: "1,2,3")
#   DISABLE_FLASH_ATTN     - Disable flash attention (set to 1)
#   USE_FLASH_ATTN         - Flash attention flag (set to 0)
# -----------------------------------------------------------------------------
# Output:
#   Search results saved to: results/param_search/
# =============================================================================

echo "=========================================="
echo "Attention Scale Parameter Search"
echo "=========================================="

PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=$PYTHONPATH

export DISABLE_FLASH_ATTN=1
export USE_FLASH_ATTN=0

CUDA_DEVICES="1,2,3"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# ── Model Configuration ──────────────────────────────────────────────────────
MODEL_NAME="huggyllama/llama-7b"
DTYPE="auto"
QUANT="--load-in-4bit"

MAX_LENGTH=65536

# ── RoPE Configuration ───────────────────────────────────────────────────────
ROPE_TYPE="freq-reciprocal-scaled-adaptive"
ROPE_DYNAMIC="--rope-dynamic"

# ── Search Configuration ─────────────────────────────────────────────────────
SEARCH_METHOD="bohb"
OUTPUT_DIR="results/param_search"

# ── Parameter Bounds ──────────────────────────────────────────────────────────
# attn_scale_base is fixed at 1.0

# attn_scale_coef search range (3 decimal places)
COEF_MIN=0.01
COEF_MAX=0.3
COEF_STEPS=100

# ── Adaptive Search Configuration ────────────────────────────────────────────
ADAPTIVE_STAGES=6
ADAPTIVE_REFINEMENT_FACTOR=0.3

# ── Random/Bayesian Search Configuration ─────────────────────────────────────
RANDOM_SAMPLES=50
BAYESIAN_ITERATIONS=100

# ── BOHB Search Configuration ────────────────────────────────────────────────
BOHB_INITIAL_SAMPLES=25
BOHB_ITERATIONS=100
BOHB_EARLY_STOP_FACTOR=3

# ── Evaluation Configuration ─────────────────────────────────────────────────
# Multi-length evaluation: from min to max length
EVAL_MIN_LENGTH=4096
EVAL_MAX_LENGTH=65536
EVAL_DATASET="emozilla/proofpile-test-tokenized"
EVAL_SPLIT="test"
EVAL_LIMIT=50

# ── Resume from previous run (optional) ──────────────────────────────────────
RESUME=""
# RESUME="--resume results/param_search/search_results_20260331_120000.json"

# ── Run Search ───────────────────────────────────────────────────────────────
echo "Model          : ${MODEL_NAME}"
echo "RoPE Type      : ${ROPE_TYPE}"
echo "Search Method  : ${SEARCH_METHOD}"
echo "Eval Length    : ${EVAL_MIN_LENGTH} - ${EVAL_MAX_LENGTH}"
echo "Output Dir     : ${OUTPUT_DIR}"
echo ""
echo "Parameter Space:"
echo "  base         : 1.0 (fixed)"
echo "  coef         : [${COEF_MIN}, ${COEF_MAX}] (steps: ${COEF_STEPS})"
echo "  Known optimal: 0.0707, 0.1"

# Print method-specific configuration details
if [ "$SEARCH_METHOD" = "adaptive" ]; then
    echo "Adaptive Config:"
    echo "  Stages       : ${ADAPTIVE_STAGES}"
    echo "  Refinement   : ${ADAPTIVE_REFINEMENT_FACTOR}"
elif [ "$SEARCH_METHOD" = "bohb" ]; then
    echo "BOHB Config:"
    echo "  Initial Samples: ${BOHB_INITIAL_SAMPLES}"
    echo "  Iterations   : ${BOHB_ITERATIONS}"
    echo "  Early Stop   : ${BOHB_EARLY_STOP_FACTOR}"
fi
echo "=========================================="

python search_attn_scale_params.py \
    --model-name ${MODEL_NAME} \
    --rope-type ${ROPE_TYPE} \
    ${ROPE_DYNAMIC} \
    --max-length ${MAX_LENGTH} \
    --dtype ${DTYPE} \
    ${QUANT} \
    --search-method ${SEARCH_METHOD} \
    --attn-scale-coef-min ${COEF_MIN} \
    --attn-scale-coef-max ${COEF_MAX} \
    --attn-scale-coef-steps ${COEF_STEPS} \
    --adaptive-stages ${ADAPTIVE_STAGES} \
    --adaptive-refinement-factor ${ADAPTIVE_REFINEMENT_FACTOR} \
    --random-samples ${RANDOM_SAMPLES} \
    --bayesian-iterations ${BAYESIAN_ITERATIONS} \
    --bohb-initial-samples ${BOHB_INITIAL_SAMPLES} \
    --bohb-iterations ${BOHB_ITERATIONS} \
    --bohb-early-stop-factor ${BOHB_EARLY_STOP_FACTOR} \
    --eval-min-length ${EVAL_MIN_LENGTH} \
    --eval-max-length ${EVAL_MAX_LENGTH} \
    --eval-dataset ${EVAL_DATASET} \
    --eval-split ${EVAL_SPLIT} \
    --eval-limit ${EVAL_LIMIT} \
    --output-dir ${OUTPUT_DIR} \
    ${RESUME}

echo ""
echo "=========================================="
echo "Search Completed!"
echo "=========================================="
