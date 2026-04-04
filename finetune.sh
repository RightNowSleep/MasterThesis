#!/bin/bash

# =============================================================================
# finetune.sh
# -----------------------------------------------------------------------------
# Purpose: Run supervised fine-tuning (QLoRA / LoRA) across all RoPE methods
# -----------------------------------------------------------------------------
# Description:
#   This script performs supervised fine-tuning on a LLaMA-7b model using
#   QLoRA/LoRA techniques. It iterates through multiple RoPE (Rotary Position
#   Embedding) methods to compare their effectiveness for long-context tasks.
# -----------------------------------------------------------------------------
# Usage:
#   bash finetune.sh
# -----------------------------------------------------------------------------
# Parameters:
#   None (all configuration is done via variables below)
# -----------------------------------------------------------------------------
# Globals:
#   CUDA_VISIBLE_DEVICES  - GPU device IDs for computation (default: "0,1,2,3")
#   WANDB                 - WandB project name for experiment tracking (empty = disabled)
# -----------------------------------------------------------------------------
# Output:
#   Fine-tuned model checkpoints saved to: finetunes/finetune/
# =============================================================================

# ── Model Configuration ──────────────────────────────────────────────────────
MODEL_NAME="huggyllama/llama-7b"
MAX_LENGTH=32768
DTYPE="bfloat16"

# ── Training Hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE=1
GRADIENT_ACCUMULATE_EVERY=8
MAX_TRAIN_STEPS=600
WARMUP_STEPS=60
LEARNING_RATE=2e-4
WEIGHT_DECAY=0.01
GRAD_NORM=1.0
LR_SCHEDULE="cosine"
CHECKPOINTING_STEPS=20
MAX_CHECKPOINTS=3
SEED=42

# ── LoRA / Quantization Settings ─────────────────────────────────────────────
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
QUANTIZATION="4bit"

# ── Dataset Configuration ────────────────────────────────────────────────────
DATASET="emozilla/pg_books-tokenized-bos-eos-chunked-65536"

# ── Infrastructure Settings ──────────────────────────────────────────────────
CUDA_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
OUTPUT_DIR="finetunes/finetune"
WANDB=""                # Set to a WandB project name to enable, e.g. "my-project"

# ── RoPE Methods Configuration ───────────────────────────────────────────────
# NOTE: --rope-factor and --rope-dynamic are mutually exclusive.
#   --rope-factor F only -> static scaling with fixed ratio F > 1.0  (used here:
#                          fine-tuning targets a known max length, static is preferred)
#   --rope-dynamic only  -> runtime scaling, no fixed ratio
ROPE_METHODS=(
    "--rope-type none"
    "--rope-type linear --rope-factor 4.0"
    "--rope-type ntk --rope-factor 4.0"
    "--rope-type part-ntk --rope-factor 4.0"
    "--rope-type yarn --rope-factor 4.0"
    "--rope-type my-rope --rope-factor 4.0"
    "--rope-type my-rope-scaled --rope-factor 4.0"
    "--rope-type my-rope2 --rope-factor 4.0"
    "--rope-type my-rope2-scaled --rope-factor 4.0"
    "--rope-type block-layered --rope-factor 4.0"
    "--rope-type block-layered-scaled --rope-factor 4.0"
    "--rope-type freq-smooth --rope-factor 4.0"
    "--rope-type freq-smooth-scaled --rope-factor 4.0"
    "--rope-type freq-reciprocal --rope-factor 4.0"
    "--rope-type freq-reciprocal-scaled --rope-factor 4.0"
)

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

# Conditionally append optional WandB argument if project name is set
if [ -n "$WANDB" ]; then
    BASE_ARGS="$BASE_ARGS --wandb $WANDB"
fi

echo "=========================================="
echo "Running Fine-tuning"
echo "=========================================="
echo "Model      : $MODEL_NAME"
echo "Max length : $MAX_LENGTH"
echo "Steps      : $MAX_TRAIN_STEPS  (warmup: $WARMUP_STEPS)"
echo "LR         : $LEARNING_RATE  (schedule: $LR_SCHEDULE)"
echo "Quantization: $QUANTIZATION  |  LoRA r=$LORA_R / alpha=$LORA_ALPHA"
echo "RoPE methods: ${#ROPE_METHODS[@]}"
echo "Output dir : $OUTPUT_DIR"
echo "=========================================="

# -----------------------------------------------------------------------------
# Function: run_finetune
# -----------------------------------------------------------------------------
# Purpose: Execute fine-tuning for a specific RoPE method
# -----------------------------------------------------------------------------
# Args:
#   $1 - rope_method: The RoPE configuration string (e.g., "--rope-type linear")
# -----------------------------------------------------------------------------
# Returns:
#   0 on success, non-zero on failure
#   Stdout: Training progress and result messages
# -----------------------------------------------------------------------------
run_finetune() {
    local rope_method=$1

    echo ""
    echo "------------------------------------------"
    echo "RoPE: $rope_method"
    echo "------------------------------------------"

    local cmd="python finetune.py \
      $BASE_ARGS \
      $rope_method"

    echo "Executing: $cmd"
    eval $cmd

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Fine-tuning completed: $rope_method"
    else
        echo "[FAILED]  Fine-tuning failed:    $rope_method"
    fi
}

# Iterate over each configured RoPE method and run fine-tuning
for rope_method in "${ROPE_METHODS[@]}"; do
    run_finetune "$rope_method"
done

echo ""
echo "=========================================="
echo "All fine-tuning runs completed!"
echo "=========================================="
