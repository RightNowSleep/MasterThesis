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
#
#   This script now supports searching for optimal alpha/beta/gamma parameters
#   on top of a base RoPE adapter (e.g., inverse-dual-rope), enabling
#   hierarchical composition: base method → scaled variant parameter search.
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

# ── Base Adapter Configuration ─────────────────────────────────────────────
# Optional: Path to a base adapter containing the foundational RoPE method.
# When set, the script loads this adapter first, then applies the target RoPE type.
# This enables hierarchical composition: base method → scaled variant parameter search.
#
# Usage scenarios:
#   1. Hierarchical search: Search scaled params on top of a trained adapter
#   2. Method comparison: Compare different scaling strategies on same base
#   3. Ablation study: Isolate the contribution of parameter search vs base method
#
# Example: BASE_ADAPTER_PATH="finetunes/continued_pretrain/inverse-dual-rope_20260403_103555"
BASE_ADAPTER_PATH="finetunes/continued_pretrain/inverse-dual-rope_20260403_103555"

# Uncomment ONE of the following examples based on your use case:

# Scenario 1: Search inverse-dual-rope-scaled on top of inverse-dual-rope adapter
# (Most common: refine a trained base method with optimal scaling parameters)
# BASE_ADAPTER_PATH="finetunes/continued_pretrain/inverse-dual-rope_20260403_103555"

# Scenario 2: Search dual-rope-scaled on top of dual-rope adapter
# (Alternative base method for comparison)
# BASE_ADAPTER_PATH="finetunes/continued_pretrain/dual-rope_20260402_113443"

# Scenario 3: Traditional mode (no base adapter, search directly on base model)
# (Baseline: pure parameter search without pretrained RoPE adaptation)
# BASE_ADAPTER_PATH=""

# ── LoRA Adapter Configuration (Mode 2) ───────────────────────────────────
# Optional: Path to an existing fine-tuned LoRA adapter to load before search.
# When set, the adapter is merged into the model first, then parameter search
# runs on top. Maps to --adapter-path in model_loader Mode 2.
#
# Example: Search params on top of a previously fine-tuned adapter
# ADAPTER_PATH="finetunes/finetune/some_method_20260401_120000"
ADAPTER_PATH=""

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
STORAGE=""  # Empty = in-memory; auto-generate as {rope_type}_{timestamp}.db
SAMPLER_SEED=42
PRUNER_N_WARMUP_STEPS=10
PRUNER_N_MIN_STEPS=5

# ── Parameter Search Space ────────────────────────────────────────────────────
ALPHA_RANGE="0.05,0.40"
BETA_RANGE="0.20,1.50"
# GAMMA_RANGE="0.50,5.00"
GAMMA_RANGE="1.10,5.00"

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

if [ -n "${BASE_ADAPTER_PATH}" ]; then
    echo "Base Adapter  : ${BASE_ADAPTER_PATH}"
    echo "  Mode        : Hierarchical (base adapter + target scaling)"
    echo "  Strategy    : Load base RoPE weights, then optimize ${ROPE_TYPE} params"
else
    echo "Base Adapter  : (none)"
    echo "  Mode        : Direct search on base model"
    echo "  Strategy    : Optimize ${ROPE_TYPE} params from scratch"
fi
if [ -n "${ADAPTER_PATH}" ]; then
    echo "LoRA Adapter  : ${ADAPTER_PATH}"
    echo "  Mode        : Load adapter first, then search (Mode 2)"
fi
echo ""

if [ -n "${STORAGE}" ]; then
    echo "Storage        : ${STORAGE}"
else
    echo "Storage        : (in-memory)"
fi
echo "=========================================="

# ── Run Search ───────────────────────────────────────────────────────────────
# Build base adapter argument conditionally:
#   - If BASE_ADAPTER_PATH is set: pass --base-adapter-path to enable hierarchical search
#   - If empty: search directly on the base model (traditional mode)
if [ -n "${BASE_ADAPTER_PATH}" ]; then
    BASE_ADAPTER_ARG="--base-adapter-path ${BASE_ADAPTER_PATH}"
else
    BASE_ADAPTER_ARG=""
fi

# Build LoRA adapter argument (Mode 2) conditionally:
#   - If ADAPTER_PATH is set: pass --adapter-path to load adapter before search
if [ -n "${ADAPTER_PATH}" ]; then
    ADAPTER_ARG="--adapter-path ${ADAPTER_PATH}"
else
    ADAPTER_ARG=""
fi

python search_attn_scale_params.py \
    --model-name ${MODEL_NAME} \
    --rope-type ${ROPE_TYPE} \
    ${ROPE_DYNAMIC} \
    ${BASE_ADAPTER_ARG} \
    ${ADAPTER_ARG} \
    --max-length ${MAX_LENGTH} \
    --dtype ${DTYPE} \
    ${QUANT} \
    --n-trials ${N_TRIALS} \
    --study-name "${STUDY_NAME}" \
    --storage "${STORAGE}" \
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
