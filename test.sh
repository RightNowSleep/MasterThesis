#!/bin/bash

# =============================================================================
# test.sh
# -----------------------------------------------------------------------------
# Purpose: Run evaluation tests across all RoPE methods for long-context models
# -----------------------------------------------------------------------------
# Description:
#   This script runs multiple evaluation types (perplexity, passkey, performance)
#   on a LLaMA-7b model with various RoPE scaling methods. It uses dynamic
#   scaling mode to allow the model to self-adapt across all sequence lengths.
# -----------------------------------------------------------------------------
# Usage:
#   bash test.sh
# -----------------------------------------------------------------------------
# Parameters:
#   None (all configuration is done via variables below)
# -----------------------------------------------------------------------------
# Globals:
#   CUDA_VISIBLE_DEVICES   - GPU device IDs for computation (default: "0,1,2,3")
#   DISABLE_FLASH_ATTN     - Disable flash attention (set to 1)
# -----------------------------------------------------------------------------
# Output:
#   Evaluation results printed to console and saved to respective output files
# =============================================================================

export DISABLE_FLASH_ATTN=1
CUDA_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# ── Model Configuration ──────────────────────────────────────────────────────
MODEL="--model-name huggyllama/llama-7b --load-in-4bit --min-length 2048 --max-length 65536"

# ── RoPE Methods Configuration ───────────────────────────────────────────────
# NOTE: --rope-factor and --rope-dynamic are mutually exclusive.
#   --rope-dynamic only  -> runtime scaling (s = seq_len / original_L), no fixed ratio
#   --rope-factor F only -> static scaling with a fixed ratio F > 1.0
# Evaluation tests use dynamic mode so the model self-adapts across all length steps.
ROPE_METHODS=(
    "--rope-type none"
    "--rope-type linear --rope-dynamic"
    "--rope-type ntk --rope-dynamic"
    "--rope-type part-ntk --rope-dynamic"
    "--rope-type yarn --rope-dynamic"
    "--rope-type freq-reciprocal --rope-dynamic"
    "--rope-type freq-reciprocal-scaled --rope-dynamic"
)

# ── Test Types Configuration ─────────────────────────────────────────────────
TEST_TYPES=(
    "perplexity"
    "passkey"
    "performance"
)

TEST_SET="--length-step 2048"

echo "=========================================="
echo "Running all RoPE method tests"
echo "=========================================="
echo "Model: $MODEL"
echo "Test types: ${TEST_TYPES[*]}"
echo "RoPE methods: ${#ROPE_METHODS[@]}"
echo "=========================================="

# -----------------------------------------------------------------------------
# Function: run_test
# -----------------------------------------------------------------------------
# Purpose: Execute a specific test type with a given RoPE method
# -----------------------------------------------------------------------------
# Args:
#   $1 - test_type: The type of test to run (perplexity, passkey, or performance)
#   $2 - rope_method: The RoPE configuration string
# -----------------------------------------------------------------------------
# Returns:
#   0 on success, non-zero on failure
#   Stdout: Test progress and result messages
# -----------------------------------------------------------------------------
run_test() {
    local test_type=$1
    local rope_method=$2

    echo ""
    echo "------------------------------------------"
    echo "Test: $test_type | RoPE: $rope_method"
    echo "------------------------------------------"

    # Build and execute the test command for the given type and RoPE method
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

# Nested loop: iterate over each test type, then over each RoPE method
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
