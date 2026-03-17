# Model
MODEL="--model-name huggyllama/llama-7b --load-in-4bit --min-length 2048 --max-length 65536 --load-in-4bit"

# RoPE method
# NOTE: --rope-factor and --rope-dynamic are mutually exclusive.
#   --rope-dynamic only  → runtime scaling (s = seq_len / original_L), no fixed ratio
#   --rope-factor F only → static scaling with a fixed ratio F > 1.0
# Evaluation tests use dynamic mode so the model self-adapts across all length steps.
ROPE_METHODS=(
    # "--rope-type none"
    # "--rope-type linear --rope-dynamic"
    # "--rope-type ntk --rope-dynamic"
    # "--rope-type part-ntk --rope-dynamic"
    # "--rope-type yarn --rope-dynamic"
    # "--rope-type my-rope --rope-dynamic"
    # "--rope-type my-rope-scaled --rope-dynamic"
    # "--rope-type my-rope2 --rope-dynamic"
    # "--rope-type my-rope2-scaled --rope-dynamic"
    # "--rope-type block-layered --rope-dynamic"
    # "--rope-type block-layered-scaled --rope-dynamic"
    # "--rope-type freq-smooth --rope-dynamic"
    # "--rope-type freq-smooth-scaled --rope-dynamic"
    # "--rope-type freq-reciprocal --rope-dynamic"
    "--rope-type freq-reciprocal-scaled --rope-dynamic"
    # "--rope-type linear --rope-factor 4.0"
    # "--rope-type ntk --rope-factor 4.0"
    # "--rope-type part-ntk --rope-factor 4.0"
    # "--rope-type yarn --rope-factor 4.0"
    # "--rope-type my-rope --rope-factor 4.0"
    # "--rope-type my-rope-scaled --rope-factor 4.0"
    # "--rope-type my-rope2 --rope-factor 4.0"
    # "--rope-type my-rope2-scaled --rope-factor 4.0"
    # "--rope-type block-layered --rope-factor 4.0"
    # "--rope-type block-layered-scaled --rope-factor 4.0"
    # "--rope-type freq-smooth --rope-factor 4.0"
    # "--rope-type freq-smooth-scaled --rope-factor 4.0"
    # "--rope-type freq-reciprocal --rope-factor 4.0"
    "--rope-type freq-reciprocal-scaled --rope-factor 4.0"
)

# Evaluation type
TEST_TYPES=(
    "perplexity"
    # "passkey"
    # "quality"
    # "performance"
)

TEST_SET="--length-step 2048"

echo "=========================================="
echo "Running all RoPE method tests"
echo "=========================================="
echo "Model: $MODEL"
echo "Test types: ${TEST_TYPES[*]}"
echo "RoPE methods: ${#ROPE_METHODS[@]}"
echo "=========================================="

run_test() {
    local test_type=$1
    local rope_method=$2
    
    echo ""
    echo "------------------------------------------"
    echo "Test: $test_type | RoPE: $rope_method"
    echo "------------------------------------------"
    
    if [ "$test_type" = "quality" ]; then
        local cmd="python test.py $test_type $MODEL $rope_method"
    else
        local cmd="python test.py $test_type $MODEL $rope_method"
    fi
    echo "Executing: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Test completed: $test_type"
    else
        echo "[FAILED] Test failed: $test_type"
    fi
}

for test_type in "${TEST_TYPES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Test Type: $test_type"
    echo "=========================================="
    
    for rope_method in "${ROPE_METHODS[@]}"; do
        run_test "$test_type" "$rope_method"
    done
done

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="