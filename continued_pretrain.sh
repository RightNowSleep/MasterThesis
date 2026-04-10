#!/bin/bash

# =============================================================================
# continued_pretrain.sh
# -----------------------------------------------------------------------------
# Purpose: Run continued pretraining (QLoRA / LoRA) across all RoPE methods
# -----------------------------------------------------------------------------
# Description:
#   This script performs continued pretraining on a LLaMA-7b model using
#   QLoRA/LoRA techniques. It iterates through multiple RoPE methods to
#   extend the model's context length capabilities through additional training.
#   This script now supports hierarchical RoPE training via --base-adapter-path,
#   enabling training of scaled variants (e.g., inverse-dual-rope-scaled) on top
#   of their foundational methods (e.g., inverse-dual-rope).
# -----------------------------------------------------------------------------
# Usage:
#   bash continued_pretrain.sh
# -----------------------------------------------------------------------------
# Parameters:
#   None (all configuration is done via variables below)
# -----------------------------------------------------------------------------
# Globals:
#   CUDA_VISIBLE_DEVICES  - GPU device IDs for computation (default: "0,1,2,3")
#   WANDB                 - WandB project name for experiment tracking (empty = disabled)
# -----------------------------------------------------------------------------
# Output:
#   Pretrained model checkpoints saved to: finetunes/continued_pretrain/
# =============================================================================

# ── Model Configuration ──────────────────────────────────────────────────────
MODEL_NAME="huggyllama/llama-7b"
MAX_LENGTH=16384
DTYPE="bfloat16"
ROPE_FACTOR=8.0

# ── Training Hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE=1
GRADIENT_ACCUMULATE_EVERY=2
MAX_TRAIN_STEPS=400
WARMUP_STEPS=40
LEARNING_RATE=2e-4
WEIGHT_DECAY=0.01
GRAD_NORM=1.0
LR_SCHEDULE="cosine"
CHECKPOINTING_STEPS=40
MAX_CHECKPOINTS=2
SEED=42

# ── LoRA / Quantization Settings ─────────────────────────────────────────────
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
QUANTIZATION="4bit"

# ── Dataset Configuration ────────────────────────────────────────────────────
DATASET="emozilla/pg_books-tokenized-bos-eos-chunked-65536"

# ── Infrastructure Settings ──────────────────────────────────────────────────
CUDA_DEVICES="1,2,3"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

OUTPUT_DIR="finetunes/continued_pretrain"
WANDB=""                # Set to a WandB project name to enable, e.g. "my-project"

# ── Progressive Length Training ──────────────────────────────────────────────
# Enable progressive length training for gradual context extension.
# When enabled, training goes through [2048, 4096, 8192, ..., MAX_LENGTH]
PROGRESSIVE_LENGTH=false

# ── RoPE Methods Configuration ───────────────────────────────────────────────
# NOTE: --rope-factor and --rope-dynamic are mutually exclusive.
#   --rope-factor F only -> static scaling with fixed ratio F > 1.0  (used here:
#                          pretraining targets a known max length, static is preferred)
#   --rope-dynamic only  -> runtime scaling, no fixed ratio
ROPE_METHODS=(
    # "--rope-type none"
    # "--rope-type linear --rope-factor $ROPE_FACTOR"
    # "--rope-type ntk --rope-factor $ROPE_FACTOR"
    # "--rope-type part-ntk --rope-factor $ROPE_FACTOR"
    # "--rope-type yarn --rope-factor $ROPE_FACTOR"
    # "--rope-type freq-reciprocal --rope-factor $ROPE_FACTOR"
    # "--rope-type freq-reciprocal-scaled --rope-factor $ROPE_FACTOR"
    # "--rope-type freq-reciprocal-scaled-no-layer --rope-factor $ROPE_FACTOR"
    # "--rope-type dual-rope --rope-factor $ROPE_FACTOR"
    # "--rope-type inverse-dual-rope --rope-factor $ROPE_FACTOR"
    # "--rope-type inverse-dual-rope-scaled --rope-factor $ROPE_FACTOR"
    "--rope-type inverse-dual-nopos-rope --rope-factor $ROPE_FACTOR"
    # "--rope-type inverse-dual-nopos-rope-scaled --rope-factor $ROPE_FACTOR"
)

# ── Base Adapter Configuration ─────────────────────────────────────────────
# Optional: Path to a base adapter containing the foundational RoPE method.
# When set, continued pretraining will load this base adapter first, then
# apply the target RoPE type from ROPE_METHODS on top of it.
# This enables hierarchical RoPE composition (e.g., train inverse-dual-rope-scaled
# on top of a trained inverse-dual-rope adapter).
#
# Example: Train inverse-dual-rope-scaled on top of inverse-dual-rope
# BASE_ADAPTER_PATH="finetunes/continued_pretrain/inverse-dual-rope_20260403_103555"
# BASE_ADAPTER_ROPE_TYPE="inverse-dual-rope-scaled"
BASE_ADAPTER_PATH="finetunes/continued_pretrain/inverse-dual-nopos-rope_base"
BASE_ADAPTER_ROPE_TYPE="inverse-dual-nopos-rope"

# Uncomment to enable:
# BASE_ADAPTER_PATH="finetunes/continued_pretrain/inverse-dual-rope_20260403_103555"
# BASE_ADAPTER_ROPE_TYPE="inverse-dual-rope-scaled"

# ── LoRA Adapter Configuration (Mode 2) ───────────────────────────────────
# Optional: Path to an existing fine-tuned LoRA adapter to load before pretraining.
# When set, the adapter is merged into the model first, then new LoRA weights
# are trained on top. Maps to --adapter-path in model_loader Mode 2 (Step 7).
#
# Example: Continue pretraining on top of a previously trained adapter
# ADAPTER_PATH="finetunes/continued_pretrain/some_method_20260401_120000"
ADAPTER_PATH=""

# ── Build Shared Argument String ─────────────────────────────────────────────
BASE_ARGS="--model-name $MODEL_NAME \
  --max-length $MAX_LENGTH \
  --dtype $DTYPE \
  --dataset $DATASET \
  --batch-size $BATCH_SIZE \
  --gradient-accumulate-every $GRADIENT_ACCUMULATE_EVERY \
  --max-train-steps $MAX_TRAIN_STEPS \
  --warmup-steps $WARMUP_STEPS \
  --learning-rate $LEARNING_RATE \
  --weight-decay $WEIGHT_DECAY \
  --grad-norm $GRAD_NORM \
  --lr-schedule $LR_SCHEDULE \
  --checkpointing-steps $CHECKPOINTING_STEPS \
  --max-checkpoints $MAX_CHECKPOINTS \
  --lora-r $LORA_R \
  --lora-alpha $LORA_ALPHA \
  --lora-dropout $LORA_DROPOUT \
  --quantization $QUANTIZATION \
  --seed $SEED \
  --output-dir $OUTPUT_DIR"

# Conditionally append optional arguments to the base command string
if [ -n "$WANDB" ]; then
    BASE_ARGS="$BASE_ARGS --wandb $WANDB"
fi

if [ "$PROGRESSIVE_LENGTH" = true ]; then
    BASE_ARGS="$BASE_ARGS --progressive-length"
fi

echo "=========================================="
echo "Running Continued Pretraining"
echo "=========================================="
echo "Model      : $MODEL_NAME"
echo "Max length : $MAX_LENGTH"
echo "Steps      : $MAX_TRAIN_STEPS  (warmup: $WARMUP_STEPS)"
echo "LR         : $LEARNING_RATE  (schedule: $LR_SCHEDULE)"
echo "Quantization: $QUANTIZATION  |  LoRA r=$LORA_R / alpha=$LORA_ALPHA"
echo "RoPE methods: ${#ROPE_METHODS[@]}"
echo "Progressive : $PROGRESSIVE_LENGTH"
if [ -n "${BASE_ADAPTER_PATH}" ]; then
    echo "Base Adapter   : ${BASE_ADAPTER_PATH}"
    echo "Base→Target    : ${BASE_ADAPTER_ROPE_TYPE} → ${ROPE_METHODS[*]}"
else
    echo "Base Adapter   : (none - training from scratch)"
fi
if [ -n "${ADAPTER_PATH}" ]; then
    echo "LoRA Adapter   : ${ADAPTER_PATH} (Mode 2: load before pretraining)"
fi
echo "Output dir : $OUTPUT_DIR"
echo "=========================================="

# -----------------------------------------------------------------------------
# Function: run_pretrain
# -----------------------------------------------------------------------------
# Purpose: Execute continued pretraining for a specific RoPE method
# -----------------------------------------------------------------------------
# Args:
#   $1 - rope_method: The RoPE configuration string (e.g., "--rope-type linear")
# -----------------------------------------------------------------------------
# Returns:
#   0 on success, non-zero on failure
#   Stdout: Training progress and result messages
# -----------------------------------------------------------------------------
run_pretrain() {
    local rope_method=$1

    echo ""
    echo "------------------------------------------"
    echo "RoPE: $rope_method"
    echo "------------------------------------------"

    # Build command with optional base adapter (Mode 1) and LoRA adapter (Mode 2) support
    local base_adapter_arg=""
    local adapter_arg=""
    if [ -n "${BASE_ADAPTER_PATH}" ]; then
        base_adapter_arg="--base-adapter-path ${BASE_ADAPTER_PATH}"
        # NOTE: rope_method from ROPE_METHODS is preserved as the target RoPE type.
        # model_loader.py Mode 1 uses it to override the base adapter's RoPE config.
        echo "[INFO] Base adapter mode: ${BASE_ADAPTER_PATH}"
        echo "[INFO] Target RoPE (from ROPE_METHODS): ${rope_method}"
    fi
    if [ -n "${ADAPTER_PATH}" ]; then
        adapter_arg="--adapter-path ${ADAPTER_PATH}"
        echo "[INFO] LoRA adapter mode (Mode 2): ${ADAPTER_PATH}"
    fi

    local cmd="python continued_pretrain.py \
      $BASE_ARGS \
      $base_adapter_arg \
      $adapter_arg \
      $rope_method"

    echo "Executing: $cmd"
    eval $cmd

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Pretraining completed: $rope_method"
    else
        echo "[FAILED]  Pretraining failed:    $rope_method"
    fi
}

# Iterate over each configured RoPE method and run pretraining
for rope_method in "${ROPE_METHODS[@]}"; do
    run_pretrain "$rope_method"
done

echo ""
echo "=========================================="
echo "All continued pretraining runs completed!"
echo "=========================================="
