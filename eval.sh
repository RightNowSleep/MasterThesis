#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export USE_FLASH_ATTN=0

# Model
MODEL="--model-name huggyllama/llama-7b --load-in-4bit --min-length 2048 --max-length 65536 --load-in-4bit"
MODEL_NAME="huggyllama/llama-7b"
MAX_LENGTH=65536
MIN_LENGTH=2048
DTYPE="auto"

# RoPE method
# NOTE: --rope-factor and --rope-dynamic are mutually exclusive.
#   --rope-dynamic only  → runtime scaling (s = seq_len / original_L), no fixed ratio
#   --rope-factor F only → static scaling with a fixed ratio F &gt; 1.0
# Evaluation tests use dynamic mode so the model self-adapts across all length steps.
ROPE_METHODS=(
    "--rope-type none"
    "--rope-type linear --rope-dynamic"
    "--rope-type ntk --rope-dynamic"
    "--rope-type part-ntk --rope-dynamic"
    "--rope-type yarn --rope-dynamic"
    "--rope-type my-rope --rope-dynamic"
    "--rope-type my-rope-scaled --rope-dynamic"
    "--rope-type my-rope2 --rope-dynamic"
    "--rope-type my-rope2-scaled --rope-dynamic"
    "--rope-type block-layered --rope-dynamic"
    "--rope-type block-layered-scaled --rope-dynamic"
    "--rope-type freq-smooth --rope-dynamic"
    "--rope-type freq-smooth-scaled --rope-dynamic"
    "--rope-type freq-reciprocal --rope-dynamic"
    "--rope-type freq-reciprocal-scaled --rope-dynamic"

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

# Evaluation type
TEST_TYPES=(
    "perplexity"
    "passkey"
    "quality"
    "performance"
)

TEST_SET="--length-step 2048"

echo "=========================================="
echo "Running all RoPE method evaluations"
echo "=========================================="
echo "Model: $MODEL"
echo "Test types: ${TEST_TYPES[*]}"
echo "RoPE methods: ${#ROPE_METHODS[@]}"
echo "=========================================="

run_eval() {
    local eval_type=$1
    local rope_type=$2
    
    echo ""
    echo "------------------------------------------"
    echo "Eval: $eval_type | RoPE: $rope_type"
    echo "------------------------------------------"
    
    local cmd="python $SCRIPT_DIR/eval/${eval_type}.py \
        --model-name $MODEL_NAME \
        --rope-type $rope_type \
        --max-length $MAX_LENGTH \
        --min-length $MIN_LENGTH \
        --dtype $DTYPE \
        --load-in-4bit"
    
    echo "Executing: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Eval completed: $eval_type with $rope_type"
    else
        echo "[FAILED] Eval failed: $eval_type with $rope_type"
    fi
}

for rope_method in "${ROPE_METHODS[@]}"; do
    echo ""
    echo "=========================================="
    echo "RoPE Method: $rope_method"
    echo "=========================================="
    
    for test_type in "${TEST_TYPES[@]}"; do
        run_eval "$test_type" "$rope_method"
    done
done

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
