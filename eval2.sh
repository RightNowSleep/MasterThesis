#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH

echo "=========================================="
echo "Eval Group 2: Passkey & Harness"
echo "=========================================="

export DISABLE_FLASH_ATTN=1
export USE_FLASH_ATTN=0

CUDA_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

MODEL_NAME="huggyllama/llama-7b"
DTYPE="auto"
QUANT="--load-in-4bit"

MAX_LENGTH=16384
MIN_LENGTH=16384

TASKS="arc_challenge,hellaswag,truthfulqa_mc1,mmlu"
TASK_LIST=(
    "arc_challenge"
    "hellaswag"
    "truthfulqa_mc1"
    "mmlu"
    "longbench2"
)
BATCH_SIZE=1
OUTPUT_DIR="results/harness"

ENABLE_PASSKEY=false
ENABLE_HARNESS=true

ADAPTER_DIR="finetunes/continued_pretrain"
ADAPTERS=(
    "yarn_20260316_071953"
    "freq-reciprocal-scaled_20260320_003434"
    "freq-reciprocal-scaled-no-layer_20260324_014910"
    "freq-reciprocal_20260317_001708"
    "part-ntk_20260315_233845"
    "none_20260315_003356"
    "linear_20260315_081529"
    "ntk_20260315_155711"
)

echo "=========================================="
echo "Configuration"
echo "=========================================="
echo "Model           : ${MODEL_NAME}"
echo "Min length      : ${MIN_LENGTH}"
echo "Max length      : ${MAX_LENGTH}"
echo "Quantization    : ${QUANT}"
echo "Adapter dir     : ${ADAPTER_DIR}"
echo "Adapters        : ${#ADAPTERS[@]}"
echo "Passkey         : ${ENABLE_PASSKEY}"
echo "Harness         : ${ENABLE_HARNESS}"
echo "=========================================="

run_passkey_eval() {
    local adapter_name=$1
    
    echo ""
    echo "------------------------------------------"
    echo "Eval: passkey | Adapter: $adapter_name"
    echo "------------------------------------------"
    
    local cmd="python ${SCRIPT_DIR}/eval/passkey.py \
        --model-name ${MODEL_NAME} \
        --adapter-path ${ADAPTER_DIR}/${adapter_name} \
        --max-length ${MAX_LENGTH} \
        --min-length ${MIN_LENGTH} \
        --dtype ${DTYPE} \
        ${QUANT}"
    
    echo "Executing: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Passkey eval completed: $adapter_name"
    else
        echo "[FAILED] Passkey eval failed: $adapter_name"
    fi
}

run_harness_eval() {
    local adapter_name=$1
    local task=$2
    
    echo ""
    echo "------------------------------------------"
    echo "Eval: harness | Adapter: $adapter_name | Task: $task"
    echo "------------------------------------------"
    
    local cmd="python ${SCRIPT_DIR}/eval/eval_harness.py \
        --model-name ${MODEL_NAME} \
        --max-length ${MAX_LENGTH} \
        --dtype ${DTYPE} \
        ${QUANT} \
        --adapter-path ${ADAPTER_DIR}/${adapter_name} \
        --tasks ${task} \
        --batch-size ${BATCH_SIZE} \
        --output-dir results/${task} \
        --log-samples
        --use-cache"
    
    echo "Executing: ${cmd}"
    eval ${cmd}
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Harness eval completed: $adapter_name | $task"
    else
        echo "[FAILED] Harness eval failed: $adapter_name | $task"
    fi
}

if [ "$ENABLE_PASSKEY" = true ]; then
    echo ""
    echo "=========================================="
    echo "Eval Type: passkey"
    echo "=========================================="
    
    for adapter_name in "${ADAPTERS[@]}"; do
        run_passkey_eval "$adapter_name"
    done
fi

if [ "$ENABLE_HARNESS" = true ]; then
    echo ""
    echo "=========================================="
    echo "Eval Type: harness"
    echo "=========================================="
    
    for task in "${TASK_LIST[@]}"; do
        for adapter_name in "${ADAPTERS[@]}"; do
            run_harness_eval "$adapter_name" "$task"
        done
    done
fi

echo ""
echo "=========================================="
echo "Eval Group 2 completed!"
echo "=========================================="