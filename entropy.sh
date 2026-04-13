#!/bin/bash

# =============================================================================
# entropy.sh
# -----------------------------------------------------------------------------
# Purpose: Run entropy evaluation pipeline for RoPE method analysis
# -----------------------------------------------------------------------------
# Description:
#   This script performs a two-part evaluation pipeline:
#   Part 1: Computes attention entropy metrics for different RoPE methods
#   Part 2: Generates visualization plots from the computed entropy data
# -----------------------------------------------------------------------------
# Usage:
#   bash entropy.sh
# -----------------------------------------------------------------------------
# Parameters:
#   None (all configuration is done via variables below)
# -----------------------------------------------------------------------------
# Globals:
#   CUDA_VISIBLE_DEVICES  - GPU device IDs for computation (default: "0,1,2,3")
# -----------------------------------------------------------------------------
# Output:
#   Entropy data saved to:     results/entropy/
#   Visualization plots saved to: results/entropy/plots/
# =============================================================================

CUDA_DEVICES="1,2,3"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

set -e

# ── Path Configuration ───────────────────────────────────────────────────────
ENTROPY_SCRIPT="eval/entropy.py"
PLOT_SCRIPT="eval/plot_entropy.py"
SAVE_DIR="results/entropy"
PLOT_DIR="results/entropy/plots"

# ── Evaluation Parameters ─────────────────────────────────────────────────────
MAX_LENGTH=3072
NUM_SAMPLES=100
LOAD_4BIT="--load-in-4bit"
DATASET="emozilla/proofpile-test-tokenized"

# ── Evaluation Mode Flags ─────────────────────────────────────────────────────
ROPE=true
ADAPTER=true
ADAPTER_DIR="finetunes/continued_pretrain"

# ── RoPE Methods Configuration ───────────────────────────────────────────────
ROPE_METHODS=(
    # "--rope-type none"
    # "--rope-type linear --rope-dynamic"
    # "--rope-type ntk --rope-dynamic"
    # "--rope-type part-ntk --rope-dynamic"
    # "--rope-type freq-reciprocal --rope-dynamic"
    # "--rope-type dual-rope --rope-dynamic"
    # "--rope-type inverse-dual-rope --rope-dynamic"
    "--rope-type inverse-dual-rope-scaled --rope-dynamic"
    # "--rope-type inverse-dual-tangle-rope --rope-dynamic"
    # "--rope-type inverse-dual-tangle-rope-scaled --rope-dynamic"
    # "--rope-type inverse-dual-nopos-rope --rope-dynamic"
    # "--rope-type inverse-dual-nopos-rope-scaled --rope-dynamic"
)

# ── Adapter Paths Configuration ──────────────────────────────────────────────
ADAPTER_PATHS=(
    # "--adapter-path ${ADAPTER_DIR}/dual-rope_20260402_113443"
    "--adapter-path ${ADAPTER_DIR}/inverse-dual-rope_20260403_103555"
    "--adapter-path ${ADAPTER_DIR}/inverse-dual-rope-scaled_20260406_070155"
)

# ── Base Adapter Configuration ─────────────────────────────────────────────
# Optional: Base adapter for entropy evaluation with custom RoPE override
# Format: "base_adapter_path|target_rope_type"
BASE_ADAPTER_FOR_ENTROPY=""
# Example:
# BASE_ADAPTER_FOR_ENTROPY="finetunes/inverse-dual-rope_20260403|inverse-dual-rope-scaled"

# ── Build Methods List ───────────────────────────────────────────────────────
# Combine enabled RoPE methods and adapter paths into the final evaluation list
METHODS=()
if [ $ROPE = true ]; then
    METHODS+=("${ROPE_METHODS[@]}")
fi
if [ $ADAPTER = true ]; then
    METHODS+=("${ADAPTER_PATHS[@]}")
fi

# Parse and add base adapter combination if configured
if [ -n "$BASE_ADAPTER_FOR_ENTROPY" ]; then
    IFS='|' read -r base_path target_rope <<< "$BASE_ADAPTER_FOR_ENTROPY"
    METHODS+=("--base-adapter-path ${ADAPTER_DIR}/${base_path} --rope-type ${target_rope} --rope-dynamic")
fi

# ── Pipeline Control Flags ────────────────────────────────────────────────────
PART1=false
PART2=true

# ── Track generated JSON files from Part 1 ──────────────────────────────────────
GENERATED_JSON_FILES=()

mkdir -p "$SAVE_DIR"
mkdir -p "$PLOT_DIR"

echo "=========================================="
echo "Entropy Evaluation Pipeline"
echo "=========================================="
echo "Max Length: $MAX_LENGTH"
echo "Num Samples: $NUM_SAMPLES"
echo "4-bit Quantization: $LOAD_4BIT"
echo "Methods: ${#METHODS[@]}"
echo "ROPE: ${ROPE}"
echo "ADAPTER: ${ADAPTER}"
if [ -n "$BASE_ADAPTER_FOR_ENTROPY" ]; then
    IFS='|' read -r base_path target_rope <<< "$BASE_ADAPTER_FOR_ENTROPY"
    echo "Base Adapter: ${ADAPTER_DIR}/${base_path} (RoPE: ${target_rope})"
else
    echo "Base Adapter: (none)"
fi
echo "Save Directory: $SAVE_DIR"
echo "=========================================="

# ── Part 1: Entropy Evaluation ────────────────────────────────────────────────
if [ $PART1 = true ]; then
    echo ""
    echo "========== Part 1: Running Entropy Evaluation =========="
    echo ""

    for method in "${METHODS[@]}"; do
        echo "--------------------------------------------------"
        echo "Processing method: $method"
        echo "--------------------------------------------------"

        # Run entropy evaluation
        python "$ENTROPY_SCRIPT" \
            --model-name "huggyllama/llama-7b" \
            $method \
            --max-length "$MAX_LENGTH" \
            $LOAD_4BIT \
            --num-samples "$NUM_SAMPLES" \
            --dataset-name "$DATASET" \
            --save-dir "$SAVE_DIR"

        # Extract rope type from method string and generate expected filename
        # Example: "--rope-type inverse-dual-rope --rope-dynamic" → "llama-7b_inverse-dual-rope_dynamic.json"
        ROPE_TYPE=$(echo "$method" | grep -oP '(?<=--rope-type )[^ ]+' || echo "none")
        if echo "$method" | grep -q "--rope-dynamic"; then
            JSON_FILE="llama-7b_${ROPE_TYPE}_dynamic.json"
        elif echo "$method" | grep -q "--rope-factor"; then
            FACTOR=$(echo "$method" | grep -oP '(?<=--rope-factor )[^ ]+' | tr '.' '_')
            JSON_FILE="llama-7b_${ROPE_TYPE}_factor${FACTOR}.json"
        else
            JSON_FILE="llama-7b_${ROPE_TYPE}.json"
        fi

        JSON_PATH="$SAVE_DIR/$JSON_FILE"
        if [ -f "$JSON_PATH" ]; then
            GENERATED_JSON_FILES+=("$JSON_PATH")
            echo "Generated JSON: $JSON_PATH"
        else
            echo "WARNING: Expected JSON not found: $JSON_PATH"
        fi

        echo "Completed: $method"
        echo ""
    done

    echo "========== Part 1 Complete =========="
    echo ""
fi

# ── Part 2: Plot Generation ───────────────────────────────────────────────────
if [ $PART2 = true ]; then
    echo "========== Part 2: Generating Plots =========="
    echo ""

    if [ ${#GENERATED_JSON_FILES[@]} -eq 0 ]; then
        echo "WARNING: No JSON files were generated in Part 1"
        echo "Falling back to scanning directory: $SAVE_DIR"
        JSON_FILES=$(find "$SAVE_DIR" -maxdepth 1 -name "*.json" -type f | sort)
        if [ -z "$JSON_FILES" ]; then
            echo "ERROR: No JSON files found in $SAVE_DIR"
            exit 0
        fi
        # Convert to array
        while IFS= read -r file; do
            GENERATED_JSON_FILES+=("$file")
        done <<< "$JSON_FILES"
    fi

    echo "Processing ${#GENERATED_JSON_FILES[@]} JSON file(s) from Part 1:"
    for f in "${GENERATED_JSON_FILES[@]}"; do
        echo "  - $f"
    done
    echo ""

    for INPUT_FILE in "${GENERATED_JSON_FILES[@]}"; do
        FILENAME=$(basename "$INPUT_FILE" .json)
        METHOD_PLOT_DIR="$PLOT_DIR/$FILENAME"

        echo "--------------------------------------------------"
        echo "Plotting for: $FILENAME"
        echo "Input: $INPUT_FILE"
        echo "Output: $METHOD_PLOT_DIR"
        echo "--------------------------------------------------"

        python "$PLOT_SCRIPT" \
            --input "$INPUT_FILE" \
            --out-dir "$METHOD_PLOT_DIR"

        echo "Plots generated for: $FILENAME"
        echo ""
    done

    echo "========== Part 2 Complete =========="
    echo ""
fi

echo "=========================================="
echo "All evaluations and plots completed!"
echo "=========================================="
echo "Results saved in: $SAVE_DIR"
echo "Plots saved in: $PLOT_DIR"
echo "=========================================="
