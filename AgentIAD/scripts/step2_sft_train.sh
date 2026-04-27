#!/bin/bash
# Step 2: Supervised Fine-Tuning (SFT)
# Trains Qwen2.5-VL-3B on structured trajectories.
# Paper: 8xA100 80GB, but adapted here for single A800.

set -e

cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):$PYTHONPATH"

# --- Configuration ---
MODEL_NAME="./models/Qwen2.5-VL-3B-Instruct"
TRAJECTORY_DIR="./trajectories/sft_trajectories"
OUTPUT_DIR="./checkpoints/sft"
NUM_GPUS=1  # Single A800

echo "=== Step 2: Supervised Fine-Tuning ==="
echo "Base model: ${MODEL_NAME}"
echo "Trajectories: ${TRAJECTORY_DIR}"
echo "Output: ${OUTPUT_DIR}"

# For single GPU: no deepspeed, use standard training
# For multi-GPU: use deepspeed
if [ "$NUM_GPUS" -gt 1 ]; then
    deepspeed --num_gpus=$NUM_GPUS training/sft_trainer.py \
        --model_name_or_path "$MODEL_NAME" \
        --trajectory_dir "$TRAJECTORY_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_epochs 20 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --warmup_ratio 0.05 \
        --max_length 4096 \
        --freeze_vision \
        --use_flash_attn \
        --gradient_checkpointing \
        --logging_steps 10 \
        --num_workers 4 \
        --deepspeed ./configs/ds_zero2.json \
        --use_wandb
else
    # Single GPU: adjust batch size and accumulation to match effective batch size
    # Paper: batch_size=4 * grad_accum=4 * 8 GPUs = 128 effective
    # Single GPU: batch_size=2 * grad_accum=64 = 128 effective
    python training/sft_trainer.py \
        --model_name_or_path "$MODEL_NAME" \
        --trajectory_dir "$TRAJECTORY_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_epochs 20 \
        --batch_size 2 \
        --gradient_accumulation_steps 64 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --warmup_ratio 0.05 \
        --max_length 4096 \
        --freeze_vision \
        --use_flash_attn \
        --gradient_checkpointing \
        --logging_steps 10 \
        --num_workers 4 \
        --deepspeed "" \
        --use_wandb
fi

echo "=== SFT training complete ==="
