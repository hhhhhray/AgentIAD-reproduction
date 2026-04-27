#!/bin/bash
# Step 4: Evaluation
# Evaluates the trained AgentIAD model on the MMAD eval split.

set -e

cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):$PYTHONPATH"

# --- Configuration ---
MODEL_PATH="${1:-./checkpoints/grpo}"
MODE="${2:-pz_cr}"  # "pz_only" or "pz_cr"
MMAD_ROOT="./data/MMAD"
OUTPUT_DIR="./evaluation/results"

echo "=== Step 4: Evaluation ==="
echo "Model: ${MODEL_PATH}"
echo "Mode: ${MODE}"

python evaluation/evaluate.py \
    --model_path "$MODEL_PATH" \
    --mmad_root "$MMAD_ROOT" \
    --domain_knowledge_path "${MMAD_ROOT}/domain_knowledge.json" \
    --eval_samples_path "./trajectories/eval_samples.json" \
    --mode "$MODE" \
    --max_rounds 3 \
    --output_dir "$OUTPUT_DIR" \
    --use_flash_attn \
    --device cuda

echo "=== Evaluation complete ==="
