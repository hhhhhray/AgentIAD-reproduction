"""
Perceptive Supervised Fine-Tuning (SFT) for AgentIAD.

Trains Qwen2.5-VL-3B on structured multi-turn trajectories with:
- Frozen vision encoder
- Loss masking: only final CoT + last tool call contribute to loss (Eq. 1)
- AdamW optimizer with cosine decay
- DeepSpeed ZeRO-2

See paper Section 3.2.3 and Table 4.
"""
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from qwen_vl_utils import process_vision_info


# ============================================================
# SFT Dataset with Loss Masking
# ============================================================

class AgentIADSFTDataset(Dataset):
    """
    SFT dataset that loads pre-built trajectory JSONs and tokenizes them
    with selective loss masking (Eq. 1).

    Loss mask: mt = 1 only for the final assistant reasoning response
    and the last tool invocation output; mt = 0 otherwise.
    """

    ASSISTANT_ROLE = "assistant"

    def __init__(
        self,
        trajectory_dir: str,
        processor: AutoProcessor,
        max_length: int = 4096,
    ):
        self.processor = processor
        self.max_length = max_length

        # Load trajectory files
        self.trajectories = []
        traj_dir = Path(trajectory_dir)
        for traj_file in sorted(traj_dir.glob("traj_*.json")):
            with open(traj_file, "r", encoding="utf-8") as f:
                self.trajectories.append(json.load(f))
        print(f"Loaded {len(self.trajectories)} trajectories from {trajectory_dir}")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = self.trajectories[idx]
        messages = traj["messages"]

        # Load images
        image_paths = traj.get("image_paths", [])
        images = []
        for p in image_paths:
            if os.path.exists(p):
                images.append(Image.open(p).convert("RGB"))

        # Apply chat template to get full text
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Process with processor (handles image tokens)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Build loss mask: only supervise the last two assistant turns
        loss_mask = self._compute_loss_mask(messages, input_ids)
        labels[loss_mask == 0] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Add vision inputs if present
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)

        return result

    def _compute_loss_mask(
        self, messages: List[Dict], input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss mask following the paper's strategy:
        Only the final reasoning step and last tool invocation output get mt=1.
        This means the last two assistant turns in the conversation.
        """
        mask = torch.zeros_like(input_ids)

        # Find positions of assistant turns by tokenizing incrementally
        # We supervise only the last two assistant messages
        assistant_indices = [
            i for i, msg in enumerate(messages) if msg["role"] == self.ASSISTANT_ROLE
        ]

        if len(assistant_indices) == 0:
            return mask

        # We want the last two assistant turns
        supervised_msg_indices = assistant_indices[-2:] if len(assistant_indices) >= 2 else assistant_indices

        # Tokenize prefix up to each supervised message to find token boundaries
        for msg_idx in supervised_msg_indices:
            # Tokenize everything up to this message
            prefix_messages = messages[:msg_idx]
            prefix_text = self.processor.apply_chat_template(
                prefix_messages, tokenize=False, add_generation_prompt=True
            )
            prefix_tokens = self.processor.tokenizer(
                prefix_text, return_tensors="pt", add_special_tokens=False
            )
            start_pos = prefix_tokens["input_ids"].shape[1]

            # Tokenize up to and including this message
            include_messages = messages[: msg_idx + 1]
            include_text = self.processor.apply_chat_template(
                include_messages, tokenize=False, add_generation_prompt=False
            )
            include_tokens = self.processor.tokenizer(
                include_text, return_tensors="pt", add_special_tokens=False
            )
            end_pos = include_tokens["input_ids"].shape[1]

            # Set mask
            start_pos = min(start_pos, len(mask) - 1)
            end_pos = min(end_pos, len(mask))
            mask[start_pos:end_pos] = 1

        return mask


# ============================================================
# Custom Trainer with vision support
# ============================================================

class AgentIADSFTTrainer(Trainer):
    """Custom trainer to handle multi-modal inputs properly."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Standard causal LM loss with our custom labels (already masked)."""
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ============================================================
# Data Collator
# ============================================================

class SFTDataCollator:
    """Collate function that handles variable-length pixel_values."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        # Stack fixed-size tensors
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        batch["labels"] = torch.stack([f["labels"] for f in features])

        # Handle pixel_values (variable sizes)
        if "pixel_values" in features[0] and features[0]["pixel_values"] is not None:
            batch["pixel_values"] = torch.cat(
                [f["pixel_values"] for f in features if f.get("pixel_values") is not None],
                dim=0,
            )
        if "image_grid_thw" in features[0] and features[0]["image_grid_thw"] is not None:
            batch["image_grid_thw"] = torch.cat(
                [f["image_grid_thw"] for f in features if f.get("image_grid_thw") is not None],
                dim=0,
            )

        return batch


# ============================================================
# Main Training Function
# ============================================================

def freeze_vision_encoder(model):
    """Freeze the vision encoder, only train language adapter and tool head."""
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


def run_sft(args):
    """Run Supervised Fine-Tuning."""
    # Load processor and model
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    # Ensure padding token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "sdpa",
        trust_remote_code=True,
    )

    # Freeze vision encoder (Table 4)
    if args.freeze_vision:
        freeze_vision_encoder(model)

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load dataset
    dataset = AgentIADSFTDataset(
        trajectory_dir=args.trajectory_dir,
        processor=processor,
        max_length=args.max_length,
    )

    # Training arguments (Table 4)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=3,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
        report_to="wandb" if args.use_wandb else "none",
    )

    # Initialize trainer
    trainer = AgentIADSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SFTDataCollator(processor),
    )

    # Train
    trainer.train()

    # Save final model and processor
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"SFT training complete. Model saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="AgentIAD SFT Training")
    parser.add_argument("--model_name_or_path", type=str,
                        default="./models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--trajectory_dir", type=str,
                        default="./trajectories/sft_trajectories")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sft")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--freeze_vision", action="store_true", default=True)
    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--deepspeed", type=str, default="./configs/ds_zero2.json")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    args = parser.parse_args()

    run_sft(args)


if __name__ == "__main__":
    main()
