#!/bin/bash

# =============================================================================
# continued_pretrain.sh
# Runs continued pretraining (QLoRA / LoRA) across all RoPE methods.
# Usage: bash continued_pretrain.sh
# =============================================================================

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME="huggyllama/llama-7b"
MAX_LENGTH=16384
DTYPE="bfloat16"
ROPE_FACTOR=8.0

# ── Training hyperparameters ─────────────────────────────────────────────────
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

# ── LoRA / quantization ───────────────────────────────────────────────────────
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
QUANTIZATION="4bit"

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET="emozilla/pg_books-tokenized-bos-eos-chunked-65536"

# ── Infrastructure ────────────────────────────────────────────────────────────
CUDA_DEVICES="1,2,3"
OUTPUT_DIR="finetunes/continued_pretrain"
WANDB=""                # Set to a WandB project name to enable, e.g. "my-project"

# ── RoPE methods ─────────────────────────────────────────────────────────────
# NOTE: --rope-factor and --rope-dynamic are mutually exclusive.
#   --rope-factor F only → static scaling with fixed ratio F > 1.0  (used here:
#                          pretraining targets a known max length, static is preferred)
#   --rope-dynamic only  → runtime scaling, no fixed ratio
ROPE_METHODS=(
    "--rope-type none"
    "--rope-type linear --rope-factor $ROPE_FACTOR"
    "--rope-type ntk --rope-factor $ROPE_FACTOR"
    "--rope-type part-ntk --rope-factor $ROPE_FACTOR"
    "--rope-type yarn --rope-factor $ROPE_FACTOR"
)

# ── Build shared argument string ─────────────────────────────────────────────
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
echo "Running Continued Pretraining"
echo "=========================================="
echo "Model      : $MODEL_NAME"
echo "Max length : $MAX_LENGTH"
echo "Steps      : $MAX_TRAIN_STEPS  (warmup: $WARMUP_STEPS)"
echo "LR         : $LEARNING_RATE  (schedule: $LR_SCHEDULE)"
echo "Quantization: $QUANTIZATION  |  LoRA r=$LORA_R / alpha=$LORA_ALPHA"
echo "RoPE methods: ${#ROPE_METHODS[@]}"
echo "Output dir : $OUTPUT_DIR"
echo "=========================================="

run_pretrain() {
    local rope_method=$1

    echo ""
    echo "------------------------------------------"
    echo "RoPE: $rope_method"
    echo "------------------------------------------"

    local cmd="CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python continued_pretrain.py \
      $BASE_ARGS \
      $rope_method"

    echo "Executing: $cmd"
    eval $cmd

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Pretraining completed: $rope_method"
    else
        echo "[FAILED]  Pretraining failed:    $rope_method"
    fi
}

for rope_method in "${ROPE_METHODS[@]}"; do
    run_pretrain "$rope_method"
done

echo ""
echo "=========================================="
echo "All continued pretraining runs completed!"
echo "=========================================="