#!/bin/bash
# Step 1: Build SFT trajectories using GPT-4o
# This generates structured reasoning trajectories from the MMAD dataset.
# Requires OpenAI API key.

set -e

cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):$PYTHONPATH"

# --- Configuration ---
OPENAI_API_KEY="${OPENAI_API_KEY:?Please set OPENAI_API_KEY environment variable}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"  # Optional: for proxy/custom endpoint
GPT_MODEL="${GPT_MODEL:-gpt-4o}"
MMAD_ROOT="./data/MMAD"
OUTPUT_DIR="./trajectories"
MAX_CONCURRENT=8

# Which sub-datasets to use. Options: MVTec, VisA, MVTec-LOCO, GoodsAD
# Use comma-separated list, or leave empty for all datasets.
# For MVTec+VisA only (saves disk space): DATASETS="MVTec,VisA"
DATASETS="${DATASETS:-MVTec,VisA}"

# Adjust sample counts proportionally when using subset
# Full MMAD (~8366 samples): sft=1600, grpo=366, pz_cr=112
# MVTec+VisA (~60% of total): sft=960, grpo=220, pz_cr=67
SFT_NUM="${SFT_NUM:-960}"
GRPO_NUM="${GRPO_NUM:-220}"
PZ_CR_NUM="${PZ_CR_NUM:-67}"

echo "=== Step 1: Building SFT Trajectories ==="
echo "GPT Model: ${GPT_MODEL}"
echo "MMAD Root: ${MMAD_ROOT}"
echo "Datasets: ${DATASETS}"
echo "SFT samples: ${SFT_NUM}, GRPO samples: ${GRPO_NUM}, PZ+CR: ${PZ_CR_NUM}"
echo "Output Dir: ${OUTPUT_DIR}"

python trajectories/build_trajectories.py \
    --mmad_root "$MMAD_ROOT" \
    --domain_knowledge_path "${MMAD_ROOT}/domain_knowledge.json" \
    --output_dir "$OUTPUT_DIR" \
    --datasets "$DATASETS" \
    --openai_api_key "$OPENAI_API_KEY" \
    --openai_base_url "$OPENAI_BASE_URL" \
    --gpt_model "$GPT_MODEL" \
    --sft_num "$SFT_NUM" \
    --grpo_num "$GRPO_NUM" \
    --pz_cr_num "$PZ_CR_NUM" \
    --max_concurrent "$MAX_CONCURRENT" \
    --seed 42

echo "=== Trajectory construction complete ==="
