#!/bin/bash

# =============================================================================
# search_params.sh
# -----------------------------------------------------------------------------
# Purpose: Run Optuna-based hyperparameter search for inverse-dual-rope-scaled
# -----------------------------------------------------------------------------
# Description:
#   This script searches for optimal alpha/beta/gamma parameters in the
#   decomposed scaling function using Optuna TPE sampler:
#       s(t) = (1 + alpha * ln(k+1)) * (1 + beta * e^(-gamma * r))
#
#   where alpha (global growth), beta (boundary jump), gamma (intra-segment decay)
#   are optimized to minimize perplexity across multiple context lengths.
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
echo "Optuna Hyperparameter Search"
echo "inverse-dual-rope-scaled (alpha/beta/gamma)"
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
ROPE_TYPE="inverse-dual-rope-scaled"
ROPE_DYNAMIC="--rope-dynamic"

# Optional: override defaults with explicit values
# ROPE_ALPHA="--rope-alpha 0.1"
# ROPE_BETA="--rope-beta 0.5"
# ROPE_GAMMA="--rope-gamma 2.0"

# ── Optuna Configuration ─────────────────────────────────────────────────────
N_TRIALS=100
STUDY_NAME="inverse-dual-rope-scaled-search"
STORAGE=""  # Empty = in-memory; set e.g. "sqlite:///optuna.db" for persistence
SAMPLER_SEED=42
PRUNER_N_WARMUP_STEPS=10
PRUNER_N_MIN_STEPS=5

# ── Parameter Search Space ────────────────────────────────────────────────────
ALPHA_RANGE="0.05,0.40"
BETA_RANGE="0.20,1.50"
GAMMA_RANGE="0.50,5.00"

# ── Evaluation Configuration ─────────────────────────────────────────────────
EVAL_MIN_LENGTH=4096
EVAL_MAX_LENGTH=65536
EVAL_DATASET="emozilla/proofpile-test-tokenized"
EVAL_SPLIT="test"
EVAL_LIMIT=50

# ── Output Configuration ─────────────────────────────────────────────────────
OUTPUT_DIR="results/param_search"

# ── Print Configuration ──────────────────────────────────────────────────────
echo "Model          : ${MODEL_NAME}"
echo "RoPE Type      : ${ROPE_TYPE}"
echo "Search Method  : Optuna (TPE + MedianPruner)"
echo "Trials         : ${N_TRIALS}"
echo "Eval Length    : ${EVAL_MIN_LENGTH} - ${EVAL_MAX_LENGTH}"
echo "Output Dir     : ${OUTPUT_DIR}"
echo ""
echo "Parameter Space:"
echo "  alpha range : [${ALPHA_RANGE}]"
echo "  beta range  : [${BETA_RANGE}]"
echo "  gamma range : [${GAMMA_RANGE}]"
echo ""

if [ -n "${STORAGE}" ]; then
    echo "Storage        : ${STORAGE}"
else
    echo "Storage        : (in-memory)"
fi
echo "=========================================="

# ── Run Search ───────────────────────────────────────────────────────────────
python search_attn_scale_params.py \
    --model-name ${MODEL_NAME} \
    --rope-type ${ROPE_TYPE} \
    ${ROPE_DYNAMIC} \
    --max-length ${MAX_LENGTH} \
    --dtype ${DTYPE} \
    ${QUANT} \
    --n-trials ${N_TRIALS} \
    --study-name "${STUDY_NAME}" \
    --sampler-seed ${SAMPLER_SEED} \
    --pruner-n-warmup-steps ${PRUNER_N_WARMUP_STEPS} \
    --pruner-n-min-steps ${PRUNER_N_MIN_STEPS} \
    --alpha-range "${ALPHA_RANGE}" \
    --beta-range "${BETA_RANGE}" \
    --gamma-range "${GAMMA_RANGE}" \
    --eval-min-length ${EVAL_MIN_LENGTH} \
    --eval-max-length ${EVAL_MAX_LENGTH} \
    --eval-dataset ${EVAL_DATASET} \
    --eval-split ${EVAL_SPLIT} \
    --eval-limit ${EVAL_LIMIT} \
    --output-dir ${OUTPUT_DIR} \
    ${ROPE_ALPHA:-} \
    ${ROPE_BETA:-} \
    ${ROPE_GAMMA:-}

echo ""
echo "=========================================="
echo "Search Completed!"
echo "=========================================="
