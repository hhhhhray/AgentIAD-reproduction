#!/bin/bash
# Step 3: Agentic Reinforcement Learning with GRPO
# Fine-tunes the SFT model using GRPO with two-level reward.
# Paper: 8xA100 80GB with ZeRO-3. Adapted for single A800.

set -e

cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):$PYTHONPATH"

# --- Configuration ---
SFT_MODEL="./checkpoints/sft"
OUTPUT_DIR="./checkpoints/grpo"
MMAD_ROOT="./data/MMAD"
GRPO_SAMPLES="./trajectories/grpo_samples.json"

echo "=== Step 3: GRPO Agentic Reinforcement Learning ==="
echo "SFT model: ${SFT_MODEL}"
echo "Output: ${OUTPUT_DIR}"

python training/grpo_trainer.py \
    --sft_model_path "$SFT_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --mmad_root "$MMAD_ROOT" \
    --domain_knowledge_path "${MMAD_ROOT}/domain_knowledge.json" \
    --grpo_samples_path "$GRPO_SAMPLES" \
    --mode "pz_cr_sv" \
    --num_epochs 3 \
    --rollouts_per_prompt 8 \
    --replay_buffer_size 128 \
    --learning_rate 1e-6 \
    --kl_coeff 0.1 \
    --clip_ratio 0.2 \
    --temperature 1.0 \
    --zero_advantage_filtering \
    --alpha 1.0 \
    --beta 1.0 \
    --lambda_type 0.1 \
    --lambda_1 1.0 \
    --lambda_2 0.5 \
    --lambda_3 0.05 \
    --expected_tool_usage 1.0 \
    --lambda_4 0.3 \
    --max_rounds 4 \
    --use_flash_attn \
    --device cuda \
    --grounding_dino_checkpoint "./models/grounded_sam2/grounding_dino_swinb_cogcoor.pth" \
    --sam2_checkpoint "./models/grounded_sam2/sam2_hiera_large.pt" \
    --sam2_model_cfg "configs/sam2.1/sam2.1_hiera_l.yaml"

echo "=== GRPO training complete ==="
