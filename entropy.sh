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
# Parameters: None (all configuration is done via variables below)
# -----------------------------------------------------------------------------
# Output:
#   - Entropy data saved to: results/entropy/
#   - Visualization plots saved to: results/entropy/plots/
# =============================================================================

CUDA_DEVICES="1,2,3"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

set -e

# ── Path Configuration ───────────────────────────────────────────────────────
PROJECT_ROOT="/home/linzhen/workspace/MasterThesis"
ENTROPY_SCRIPT="$PROJECT_ROOT/eval/entropy.py"
PLOT_SCRIPT="$PROJECT_ROOT/eval/plot_entropy.py"
SAVE_DIR="$PROJECT_ROOT/results/entropy"
PLOT_DIR="$PROJECT_ROOT/results/entropy/plots"

# ── Evaluation Parameters ─────────────────────────────────────────────────────
MAX_LENGTH=3072
NUM_SAMPLES=100
LOAD_4BIT="--load-in-4bit"
DATASET="emozilla/proofpile-test-tokenized"
ROPE_METHODS=("linear" "ntk" "part-ntk" "freq-reciprocal")
DYNAMIC="--rope-dynamic"

# ── Pipeline Control Flags ────────────────────────────────────────────────────
PART1=false
PART2=true

mkdir -p "$SAVE_DIR"
mkdir -p "$PLOT_DIR"

echo "=========================================="
echo "Entropy Evaluation Pipeline"
echo "=========================================="
echo "Max Length: $MAX_LENGTH"
echo "Num Samples: $NUM_SAMPLES"
echo "4-bit Quantization: $LOAD_4BIT"
echo "RoPE Methods: ${ROPE_METHODS[*]}"
echo "Save Directory: $SAVE_DIR"
echo "=========================================="

# ── Part 1: Entropy Evaluation ────────────────────────────────────────────────
if [ $PART1 = true ]; then
    echo ""
    echo "========== Part 1: Running Entropy Evaluation =========="
    echo ""

    for method in "${ROPE_METHODS[@]}"; do
        echo "--------------------------------------------------"
        echo "Processing RoPE method: $method"
        echo "--------------------------------------------------"
        
        python "$ENTROPY_SCRIPT" \
            --model-name "huggyllama/llama-7b" \
            --rope-type "$method" \
            $DYNAMIC \
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
