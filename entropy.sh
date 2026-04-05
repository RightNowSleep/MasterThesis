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

CUDA_DEVICES="0,1,2,3"
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
    "--rope-type inverse-dual-rope --rope-dynamic"
)

# ── Adapter Paths Configuration ──────────────────────────────────────────────
ADAPTER_PATHS=(
    # "--adapter-path ${ADAPTER_DIR}/dual-rope_20260402_113443"
    "--adapter-path ${ADAPTER_DIR}/inverse-dual-rope_20260403_103555"
)

# ── Build Methods List ───────────────────────────────────────────────────────
# Combine enabled RoPE methods and adapter paths into the final evaluation list
METHODS=()
if [ $ROPE = true ]; then
    METHODS+=("${ROPE_METHODS[@]}")
fi
if [ $ADAPTER = true ]; then
    METHODS+=("${ADAPTER_PATHS[@]}")
fi

# ── Pipeline Control Flags ────────────────────────────────────────────────────
PART1=true
PART2=true

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

        python "$ENTROPY_SCRIPT" \
            --model-name "huggyllama/llama-7b" \
            $method \
            --max-length "$MAX_LENGTH" \
            $LOAD_4BIT \
            --num-samples "$NUM_SAMPLES" \
            --dataset-name "$DATASET" \
            --save-dir "$SAVE_DIR"

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
    echo "Scanning directory: $SAVE_DIR"
    echo ""

    JSON_FILES=$(find "$SAVE_DIR" -maxdepth 1 -name "*.json" -type f | sort)

    if [ -z "$JSON_FILES" ]; then
        echo "WARNING: No JSON files found in $SAVE_DIR"
        exit 0
    fi

    for INPUT_FILE in $JSON_FILES; do
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
