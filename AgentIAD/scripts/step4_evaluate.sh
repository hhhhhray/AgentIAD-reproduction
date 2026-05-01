#!/bin/bash
# Step 4: Evaluation
# Evaluates the trained AgentIAD model on the MMAD eval split.

set -e

cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):$PYTHONPATH"

# --- Configuration ---
MODEL_PATH="${1:-./checkpoints/grpo}"
MODE="${2:-pz_cr_sv}"  # "pz_only", "pz_cr", or "pz_cr_sv"
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
    --max_rounds 4 \
    --output_dir "$OUTPUT_DIR" \
    --use_flash_attn \
    --device cuda \
    --grounding_dino_checkpoint "./models/grounded_sam2/grounding_dino_swinb_cogcoor.pth" \
    --sam2_checkpoint "./models/grounded_sam2/sam2_hiera_large.pt" \
    --sam2_model_cfg "configs/sam2.1/sam2.1_hiera_l.yaml"

echo "=== Evaluation complete ==="
