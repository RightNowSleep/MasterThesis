#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Evaluation Script"
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

TASKS="longbench2,arc_challenge,hellaswag,truthfulqa_mc1,mmlu"
BATCH_SIZE=2
OUTPUT_DIR="results/harness"

ADAPTER_DIR="finetunes/continued_pretrain"

METHODS=(
    "--rope-type none"
    "--rope-type linear --rope-dynamic"
    "--rope-type ntk --rope-dynamic"
    "--rope-type part-ntk --rope-dynamic"
    "--rope-type yarn --rope-dynamic"
    "--rope-type freq-reciprocal --rope-dynamic"
    "--rope-type freq-reciprocal-scaled --rope-dynamic"
    "--adapter-path ${ADAPTER_DIR}/none_20260315_003356"
    "--adapter-path ${ADAPTER_DIR}/linear_20260315_081529"
    "--adapter-path ${ADAPTER_DIR}/ntk_20260315_155711"
    "--adapter-path ${ADAPTER_DIR}/part-ntk_20260315_233845"
    "--adapter-path ${ADAPTER_DIR}/yarn_20260316_071953"
    "--adapter-path ${ADAPTER_DIR}/freq-reciprocal_20260317_001708"
    "--adapter-path ${ADAPTER_DIR}/freq-reciprocal-scaled_20260320_003434"
    "--adapter-path ${ADAPTER_DIR}/freq-reciprocal-scaled-no-layer_20260324_014910"
)

EVAL_TYPES=(
    # "perplexity"
    # "performance"
    "passkey"
    "eval_harness"
)

echo "=========================================="
echo "Configuration"
echo "=========================================="
echo "Model          : ${MODEL_NAME}"
echo "Max length     : ${MAX_LENGTH}"
echo "Quantization   : ${QUANT}"
echo "Methods        : ${#METHODS[@]}"
echo "Adapters       : ${#ADAPTERS[@]}"
echo "Eval types     : ${EVAL_TYPES[*]}"
echo "Run harness    : ${RUN_HARNESS}"
echo "=========================================="

run_eval() {
    local eval_type=$1
    local args=$2
    
    echo ""
    echo "------------------------------------------"
    echo "Eval: $eval_type | $args"
    echo "------------------------------------------"
    
    local cmd="python ${SCRIPT_DIR}/eval/${eval_type}.py \
        --model-name ${MODEL_NAME} \
        ${args} \
        --max-length ${MAX_LENGTH} \
        --min-length ${MIN_LENGTH} \
        --dtype ${DTYPE} \
        ${QUANT}"
    
    echo "Executing: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Eval completed: $eval_type with $args"
    else
        echo "[FAILED] Eval failed: $eval_type with $args"
    fi
}

for eval_type in "${EVAL_TYPES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Eval Type: $eval_type"
    echo "=========================================="
    
    for method in "${METHODS[@]}"; do
        run_eval "$eval_type" "$method"
    done
done

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="