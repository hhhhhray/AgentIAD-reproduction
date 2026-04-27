#!/bin/bash
# Full pipeline: from data download to evaluation
# Usage: bash scripts/run_all.sh

set -e

echo "============================================"
echo "  AgentIAD Full Training & Evaluation Pipeline"
echo "============================================"

# Step 0: Download data (uncomment if needed)
# bash scripts/download_data.sh

# Step 1: Build SFT trajectories
echo ""
echo "[1/4] Building SFT trajectories..."
bash scripts/step1_build_trajectories.sh

# Step 2: Supervised Fine-Tuning
echo ""
echo "[2/4] Running SFT..."
bash scripts/step2_sft_train.sh

# Step 3: GRPO Reinforcement Learning
echo ""
echo "[3/4] Running GRPO..."
bash scripts/step3_grpo_train.sh

# Step 4: Evaluation
echo ""
echo "[4/4] Evaluating..."
bash scripts/step4_evaluate.sh ./checkpoints/grpo pz_cr

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
