MODEL_NAME="huggyllama/llama-7b"
MAX_LENGTH=16384
DTYPE="auto"
ADAPTER="${ADAPTER:-}"

TASKS="longbench,longbenchv2,arc_challenge,hellaswag,truthfulqa_mc1,mmlu"
BATCH_SIZE=1
OUTPUT_DIR="results/harness"
QUANT="--load-in-4bit"

CUDA_DEVICES="1,2,3"
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

if ! python -c "import lm_eval" 2>/dev/null; then
    echo "[ERROR] lm-eval is not installed.  Run: pip install lm-eval"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "lm-eval Harness Evaluation"
echo "=========================================="
echo "Base model   : ${MODEL_NAME}"
echo "Max length   : ${MAX_LENGTH}"
echo "Tasks        : ${TASKS}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "RoPE methods : ${#ROPE_METHODS[@]}"
[ -n "$ADAPTER" ] && echo "Adapter      : ${ADAPTER}"
echo "=========================================="

run_eval() {
    local label=$1
    local rope_args=$2
    local adapter_args=$3

    echo ""
    echo "------------------------------------------"
    echo "Evaluating: ${label}"
    echo "------------------------------------------"

    local cmd="python eval/run_harness.py \
        --model-name ${MODEL_NAME} \
        --max-length ${MAX_LENGTH} \
        --dtype ${DTYPE} \
        ${QUANT} \
        ${rope_args} \
        ${adapter_args} \
        --tasks ${TASKS} \
        --batch-size ${BATCH_SIZE} \
        --output-dir ${OUTPUT_DIR} \
        --log-samples"

    echo "Executing: ${cmd}"
    eval ${cmd}

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] ${label}"
    else
        echo "[FAILED]  ${label}"
    fi
}

# Phase 1: base model
echo ""
echo "=========================================="
echo "Phase 1: Base model evaluation"
echo "=========================================="

for rope_method in "${ROPE_METHODS[@]}"; do
    run_eval "base | ${rope_method}" "${rope_method}" ""
done

# Phase 2: fine-tuned adapter
if [ -n "$ADAPTER" ]; then
    echo ""
    echo "=========================================="
    echo "Phase 2: Fine-tuned adapter evaluation"
    echo "=========================================="

    ADAPTER_ROPE_TYPE=""
    ADAPTER_ROPE_FACTOR=""
    ADAPTER_ROPE_DYNAMIC=""
    if [ -f "${ADAPTER}/args.json" ]; then
        ADAPTER_ROPE_TYPE=$(python -c "
import json
d = json.load(open('${ADAPTER}/args.json'))
print(d.get('rope_type', 'none'))
" 2>/dev/null)
        ADAPTER_ROPE_FACTOR=$(python -c "
import json
d = json.load(open('${ADAPTER}/args.json'))
v = d.get('rope_factor', None)
print(v if v is not None else '')
" 2>/dev/null)
        ADAPTER_ROPE_DYNAMIC=$(python -c "
import json
d = json.load(open('${ADAPTER}/args.json'))
print('true' if d.get('rope_dynamic', False) else 'false')
" 2>/dev/null)
    fi

    if [ -n "$ADAPTER_ROPE_TYPE" ] && [ "$ADAPTER_ROPE_TYPE" != "none" ]; then
        if [ -n "$ADAPTER_ROPE_FACTOR" ] && [ "$ADAPTER_ROPE_FACTOR" != "None" ] && [ "$ADAPTER_ROPE_FACTOR" != "" ]; then
            ADAPTER_ROPE_ARGS="--rope-type ${ADAPTER_ROPE_TYPE} --rope-factor ${ADAPTER_ROPE_FACTOR}"
        else
            ADAPTER_ROPE_ARGS="--rope-type ${ADAPTER_ROPE_TYPE} --rope-dynamic"
        fi
        echo "Auto-detected adapter RoPE: ${ADAPTER_ROPE_ARGS}"
    else
        ADAPTER_ROPE_ARGS="--rope-type freq-reciprocal --rope-dynamic"
        echo "Could not detect adapter RoPE; using fallback: ${ADAPTER_ROPE_ARGS}"
    fi

    run_eval "adapter | ${ADAPTER_ROPE_ARGS}" \
             "${ADAPTER_ROPE_ARGS}" \
             "--adapter-path ${ADAPTER}"
fi

# Summary
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
OUTPUT_DIR="${OUTPUT_DIR}" python - <<'PYEOF'
import json, os, glob
output_dir = os.environ.get("OUTPUT_DIR", "results/harness")
files = sorted(glob.glob(os.path.join(output_dir, "*.json")))
if not files:
    print("  No result files found.")
else:
    for f in files:
        try:
            data = json.load(open(f))
            results = data.get("results", {})
            print(f"\n  {os.path.basename(f).replace('.json','')}")
            for task, metrics in results.items():
                for k, v in metrics.items():
                    if isinstance(v, float) and "acc" in k:
                        print(f"    {task:30s} {k:20s} {v*100:.2f}%")
                        break
        except Exception as e:
            print(f"  {f}: parse error ({e})")
PYEOF

echo ""
echo "All evaluations complete! Results saved to: ${OUTPUT_DIR}/"
echo "=========================================="