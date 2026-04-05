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

# ── Base Adapter Testing ──────────────────────────────────────────────────
# Optional: Test models loaded from base adapters with different RoPE configs
# Format: "base_adapter_path|target_rope_type"
BASE_ADAPTER_TEST=""
# Example:
# BASE_ADAPTER_TEST="finetunes/inverse-dual-rope_20260403|inverse-dual-rope-scaled"

# ── Adapter Testing (Mode 2) ─────────────────────────────────────────────
# Optional: Test models loaded from fine-tuned LoRA adapters
# Format: array of "--adapter-path <path>" strings
ADAPTER_PATHS=()
# Example:
# ADAPTER_PATHS=(
#     "--adapter-path finetunes/continued_pretrain/inverse-dual-rope_20260403_103555"
#     "--adapter-path finetunes/continued_pretrain/yarn_20260316_071953"
# )

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
if [ -n "$BASE_ADAPTER_TEST" ]; then
    IFS='|' read -r base_path target_rope <<< "$BASE_ADAPTER_TEST"
    echo "Base Adapter: ${base_path} (RoPE: ${target_rope})"
else
    echo "Base Adapter: (none)"
fi
echo "Adapter Paths: ${#ADAPTER_PATHS[@]}"
if [ ${#ADAPTER_PATHS[@]} -gt 0 ]; then
    echo "  (Will also test with --adapter-path Mode 2)"
fi
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

    # Support three modes: base adapter / adapter-path / direct RoPE
    local cmd=""
    if [ "$test_type" = "quality" ]; then
        cmd="python test.py $test_type $MODEL $rope_method"
    elif [ -n "$BASE_ADAPTER_TEST" ]; then
        IFS='|' read -r base_path target_rope <<< "$BASE_ADAPTER_TEST"
        cmd="python test.py $test_type $MODEL --base-adapter-path ${base_path} --rope-type ${target_rope} --rope-dynamic"
    else
        cmd="python test.py $test_type $MODEL $rope_method"
    fi

    echo "Executing: $cmd"
    eval $cmd

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Test completed: $test_type"
    else
        echo "[FAILED] Test failed: $test_type"
    fi
}

# Mode 2: Run tests with --adapter-path (fine-tuned adapters)
run_adapter_test() {
    local test_type=$1
    local adapter_arg=$2

    echo ""
    echo "------------------------------------------"
    echo "Test: $test_type | Adapter: $adapter_arg"
    echo "------------------------------------------"

    local cmd="python test.py $test_type $MODEL ${adapter_arg}"

    echo "Executing: $cmd"
    eval $cmd

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Adapter test completed: $test_type"
    else
        echo "[FAILED] Adapter test failed: $test_type"
    fi
}

# Nested loop: iterate over each test type, then over each RoPE method
for test_type in "${TEST_TYPES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Test Type: $test_type"
    echo "=========================================="

    # Mode 3 & Mode 1: RoPE methods and base adapter tests
    for rope_method in "${ROPE_METHODS[@]}"; do
        run_test "$test_type" "$rope_method"
    done

    # Mode 2: Adapter path tests
    for adapter_arg in "${ADAPTER_PATHS[@]}"; do
        run_adapter_test "$test_type" "$adapter_arg"
    done
done

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
