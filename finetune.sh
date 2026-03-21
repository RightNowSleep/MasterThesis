#!/bin/bash

# =============================================================================
# finetune.sh
# Runs supervised fine-tuning (QLoRA / LoRA) across all RoPE methods.
# Usage: bash finetune.sh
# =============================================================================

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME="huggyllama/llama-7b"
MAX_LENGTH=32768
DTYPE="bfloat16"

# ── Training hyperparameters ──────────────────────────────────────────────────
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

# ── LoRA / quantization ───────────────────────────────────────────────────────
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
QUANTIZATION="4bit"

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET="emozilla/pg_books-tokenized-bos-eos-chunked-65536"

# ── Infrastructure ────────────────────────────────────────────────────────────
CUDA_DEVICES="1,2,3"
OUTPUT_DIR="/home/linzhen/workspace/finetunes/finetune"
WANDB=""                # Set to a WandB project name to enable, e.g. "my-project"

# ── RoPE methods ──────────────────────────────────────────────────────────────
# NOTE: --rope-factor and --rope-dynamic are mutually exclusive.
#   --rope-factor F only → static scaling with fixed ratio F > 1.0  (used here:
#                          fine-tuning targets a known max length, static is preferred)
#   --rope-dynamic only  → runtime scaling, no fixed ratio
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

# ── Build shared argument string ──────────────────────────────────────────────
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
  --output-dir $OUTPUT_DIR \
  --cuda-visible-devices $CUDA_DEVICES"

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

run_finetune() {
    local rope_method=$1

    echo ""
    echo "------------------------------------------"
    echo "RoPE: $rope_method"
    echo "------------------------------------------"

    local cmd="CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python finetune.py \
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

for rope_method in "${ROPE_METHODS[@]}"; do
    run_finetune "$rope_method"
done

echo ""
echo "=========================================="
echo "All fine-tuning runs completed!"
echo "=========================================="