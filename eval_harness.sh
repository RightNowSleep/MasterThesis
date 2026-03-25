PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="huggyllama/llama-7b"
MAX_LENGTH=16384
DTYPE="auto"

TASKS="longbench2,arc_challenge,hellaswag,truthfulqa_mc1,mmlu"
BATCH_SIZE=2
OUTPUT_DIR="results/harness"
QUANT="--load-in-4bit"

CUDA_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

ROPE_METHODS=(
    "--rope-type none"
    "--rope-type linear --rope-dynamic"
    "--rope-type ntk --rope-dynamic"
    "--rope-type part-ntk --rope-dynamic"
    "--rope-type yarn --rope-dynamic"
    "--rope-type freq-reciprocal --rope-dynamic"
    "--rope-type freq-reciprocal-scaled --rope-dynamic"
)

ADAPTER_DIR = "finetunes/continued_pretrain"
ADAPTERS=(
    "none_20260315_003356"
    "linear_20260315_081529"
    "ntk_20260315_155711"
    "part-ntk_20260315_233845"
    "yarn_20260316_071953"
    "freq-reciprocal_20260317_001708"
    "freq-reciprocal-scaled_20260320_003434"
    "freq-reciprocal-scaled-no-layer_20260324_014910"
)

echo "=========================================="
echo "lm-eval Harness Evaluation"
echo "=========================================="
echo "Base model   : ${MODEL_NAME}"
echo "Max length   : ${MAX_LENGTH}"
echo "Tasks        : ${TASKS}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "RoPE methods : ${#ROPE_METHODS[@]}"
echo "Adapter dir  : ${ADAPTER_DIR}"
echo "Adapters     : ${#ADAPTERS[@]}"
echo "=========================================="

run_eval() {
    local args=$1

    echo ""
    echo "------------------------------------------"
    echo "Evaluating: ${args}"
    echo "------------------------------------------"

    local cmd="python eval/eval_harness.py \
        --model-name ${MODEL_NAME} \
        --max-length ${MAX_LENGTH} \
        --dtype ${DTYPE} \
        ${QUANT} \
        ${args} \
        --tasks ${TASKS} \
        --batch-size ${BATCH_SIZE} \
        --output-dir ${OUTPUT_DIR} \
        --log-samples"

    echo "Executing: ${cmd}"
    eval ${cmd}
}

# Phase 1: dynamic evaluation
echo ""
echo "=========================================="
echo "Phase 1: Dynamic evaluation"
echo "=========================================="

for rope_method in "${ROPE_METHODS[@]}"; do
    run_eval "${rope_method}"
done

# Phase 2: fine-tuned adapters
if [ ${#ADAPTERS[@]} -gt 0 ]; then
    echo ""
    echo "=========================================="
    echo "Phase 2: Fine-tuned adapter evaluation"
    echo "=========================================="

    for adapter_entry in "${ADAPTERS[@]}"; do
        run_eval "--adapter_path ${ADAPTER_DIR}/${adapter_entry}"
    done
fi

echo ""
echo "All evaluations complete! Results saved to: ${OUTPUT_DIR}/"
echo "=========================================="