#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export USE_FLASH_ATTN=0

MODEL_NAME="${MODEL_NAME:-huggyllama/llama-7b}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MIN_LENGTH="${MIN_LENGTH:-2048}"
DEVICE="${DEVICE:-}"
DTYPE="${DTYPE:-auto}"

ROPE_TYPES=("none" "linear" "ntk" "part-ntk" "yarn" "my-rope" "dynamic-my-rope")
ROPE_FACTORS=("none" "4.0" "4.0" "4.0" "4.0" "4.0" "4.0")
EVAL_TYPES=("perplexity" "passkey" "quality" "performance")

echo "=========================================="
echo "Running all RoPE method evaluations"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Max Length: $MAX_LENGTH"
echo "Min Length: $MIN_LENGTH"
echo "Device: ${DEVICE:-auto}"
echo "Dtype: $DTYPE"
echo "=========================================="

run_eval() {
    local eval_type=$1
    local rope_type=$2
    local rope_factor=$3
    
    echo ""
    echo "------------------------------------------"
    echo "Eval: $eval_type | RoPE: $rope_type | Factor: $rope_factor"
    echo "------------------------------------------"
    
    local cmd="python $SCRIPT_DIR/eval/${eval_type}.py \
        --model-name $MODEL_NAME \
        --rope-type $rope_type \
        --max-length $MAX_LENGTH \
        --min-length $MIN_LENGTH \
        --dtype $DTYPE"
    
    if [ "$rope_type" != "none" ] && [ "$rope_factor" != "none" ]; then
        cmd="$cmd --rope-factor $rope_factor"
    fi
    
    if [ -n "$DEVICE" ]; then
        cmd="$cmd --device $DEVICE"
    fi
    
    if [ "$rope_type" = "linear" ] || [ "$rope_type" = "ntk" ] || \
       [ "$rope_type" = "part-ntk" ] || [ "$rope_type" = "yarn" ]; then
        cmd="$cmd --rope-dynamic"
    fi
    
    echo "Executing: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Eval completed: $eval_type with $rope_type"
    else
        echo "[FAILED] Eval failed: $eval_type with $rope_type"
    fi
}

for i in "${!ROPE_TYPES[@]}"; do
    rope_type="${ROPE_TYPES[$i]}"
    rope_factor="${ROPE_FACTORS[$i]}"
    
    for eval_type in "${EVAL_TYPES[@]}"; do
        run_eval "$eval_type" "$rope_type" "$rope_factor"
    done
done

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
