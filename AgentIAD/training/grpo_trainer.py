"""
Agentic Reinforcement Learning with GRPO for AgentIAD.

Implements the Generalized Reinforcement Policy Optimization (GRPO) framework
with multi-round tool-augmented rollouts and two-level reward.

See paper Section 3.3 and Table 5.

This module provides a standalone GRPO trainer that:
1. Generates K rollouts per prompt with tool interaction
2. Computes rewards using the two-level reward function
3. Optimizes using clipped surrogate + KL penalty (Eqs. 2-4)
"""
import argparse
import copy
import gc
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from data.data_utils import (
    is_logical_anomaly_sample,
    load_domain_knowledge,
    load_mask,
    bbox_from_mask,
)
from data.mmad_dataset import MMADGRPODataset, scan_mmad_samples
from tools.visual_tools import ToolExecutor
from training.rewards import (
    RewardComputer,
    compute_group_query_rate,
    compute_group_sv_rate,
)


class GRPORolloutEngine:
    """
    Generates multi-round rollouts with tool interaction for GRPO training.
    Each rollout consists of the agent alternating between reasoning and tool use.
    """

    def __init__(
        self,
        model,
        processor,
        tool_executor: ToolExecutor,
        max_rounds: int = 3,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
    ):
        self.model = model
        self.processor = processor
        self.tool_executor = tool_executor
        self.max_rounds = max_rounds
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def generate_rollout(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        dataset_name: str,
        category: str,
        image_path: str,
    ) -> Tuple[str, List[str], List[Dict]]:
        """
        Generate a single multi-round rollout with tool interaction.

        Returns:
            full_text: Complete concatenated model output.
            step_texts: List of model outputs at each step.
            tool_results: List of tool call results.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        images_in_context = [image]
        step_texts = []
        tool_results = []
        full_text = ""

        for round_idx in range(self.max_rounds):
            # Generate model response
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
            )
            # Decode only the generated part
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            response = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            step_texts.append(response)
            full_text += response

            # Check if response contains a tool call
            tool_call = self.tool_executor.parse_tool_call(response)
            if tool_call is not None:
                # Execute the tool
                result_image, result_text = self.tool_executor.execute(
                    tool_call,
                    images_in_context,
                    dataset_name,
                    category,
                    image_path,
                )
                tool_results.append({
                    "tool_call": tool_call,
                    "result_text": result_text,
                    "has_image": result_image is not None,
                })

                # Add assistant and tool result to conversation
                messages.append({"role": "assistant", "content": response})
                if result_image is not None:
                    images_in_context.append(result_image)
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": result_text},
                            {"type": "image", "image": result_image},
                        ],
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": result_text,
                    })
            else:
                # No tool call - check if there's a final answer
                answer = self.tool_executor.parse_final_answer(response)
                if answer is not None:
                    break
                # If no answer and no tool call, append and continue
                messages.append({"role": "assistant", "content": response})

        return full_text, step_texts, tool_results


class GRPOTrainer:
    """
    GRPO trainer for AgentIAD.

    Implements:
    - K rollouts per prompt (K=8)
    - Group-relative advantage estimation
    - Clipped surrogate loss + KL penalty (Eqs. 2-4)
    - Zero-advantage filtering
    - Two-level reward computation
    """

    def __init__(
        self,
        model,
        ref_model,
        processor,
        tool_executor: ToolExecutor,
        reward_computer: RewardComputer,
        # GRPO hyperparams (Table 5)
        rollouts_per_prompt: int = 8,
        replay_buffer_size: int = 128,
        lr: float = 1e-6,
        kl_coeff: float = 0.1,
        clip_ratio: float = 0.2,
        temperature: float = 1.0,
        zero_advantage_filtering: bool = True,
        max_rounds: int = 3,
        max_new_tokens: int = 512,
        device: str = "cuda",
    ):
        self.model = model
        self.ref_model = ref_model
        self.processor = processor
        self.reward_computer = reward_computer
        self.rollouts_per_prompt = rollouts_per_prompt
        self.replay_buffer_size = replay_buffer_size
        self.kl_coeff = kl_coeff
        self.clip_ratio = clip_ratio
        self.zero_advantage_filtering = zero_advantage_filtering
        self.device = device

        # Rollout engine
        self.rollout_engine = GRPORolloutEngine(
            model=model,
            processor=processor,
            tool_executor=tool_executor,
            max_rounds=max_rounds,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        # Optimizer (Table 5: AdamW, lr=1e-6)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    def collect_rollouts(
        self, prompt_batch: List[Dict]
    ) -> List[Dict]:
        """
        Collect K rollouts for each prompt in the batch.
        Returns a list of rollout data with rewards.
        """
        all_rollouts = []

        for prompt_data in prompt_batch:
            group_rollouts = []
            group_step_texts = []

            for k in range(self.rollouts_per_prompt):
                full_text, step_texts, tool_results = self.rollout_engine.generate_rollout(
                    image=prompt_data["image"],
                    system_prompt=prompt_data["system_prompt"],
                    user_prompt=prompt_data["user_prompt"],
                    dataset_name=prompt_data["ground_truth"]["dataset_name"],
                    category=prompt_data["ground_truth"]["category"],
                    image_path=prompt_data["image_path"],
                )
                group_rollouts.append({
                    "full_text": full_text,
                    "step_texts": step_texts,
                    "tool_results": tool_results,
                    "prompt_data": prompt_data,
                })
                group_step_texts.append(step_texts)

            # Compute group query/SV rates for behavior reward
            group_query_rate = compute_group_query_rate(group_step_texts)
            group_sv_rate = compute_group_sv_rate(group_step_texts)
            is_logical = is_logical_anomaly_sample(prompt_data["ground_truth"])

            # Compute rewards for each rollout
            rewards = []
            for rollout in group_rollouts:
                total_reward, details = self.reward_computer.compute_total_reward(
                    full_text=rollout["full_text"],
                    rollout_texts=rollout["step_texts"],
                    gt=prompt_data["ground_truth"],
                    group_query_rate=group_query_rate,
                    group_sv_rate=group_sv_rate,
                    is_logical=is_logical,
                )
                rollout["reward"] = total_reward
                rollout["reward_details"] = details
                rewards.append(total_reward)

            # Compute group-relative advantages
            mean_r = np.mean(rewards)
            std_r = np.std(rewards) + 1e-8
            for i, rollout in enumerate(group_rollouts):
                advantage = (rewards[i] - mean_r) / std_r
                rollout["advantage"] = advantage

            # Zero-advantage filtering
            if self.zero_advantage_filtering:
                group_rollouts = [
                    r for r in group_rollouts if abs(r["advantage"]) > 1e-6
                ]

            all_rollouts.extend(group_rollouts)

        return all_rollouts

    def compute_log_probs(
        self, model, text: str, image: Image.Image, system_prompt: str, user_prompt: str
    ) -> torch.Tensor:
        """Compute log probabilities of the full response under the given model."""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {"role": "assistant", "content": text},
        ]
        from qwen_vl_utils import process_vision_info
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[full_text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad() if model is self.ref_model else torch.enable_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        # Get per-token log probs for the response portion
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum()

    def train_step(self, rollouts: List[Dict]) -> Dict[str, float]:
        """
        Perform one GRPO optimization step on collected rollouts.
        Implements Eqs. 2-4: L_GRPO = L_clip - beta * L_KL
        """
        self.model.train()
        total_loss = 0.0
        total_clip_loss = 0.0
        total_kl_loss = 0.0
        num_samples = 0

        for rollout in rollouts:
            prompt_data = rollout["prompt_data"]
            advantage = rollout["advantage"]
            full_text = rollout["full_text"]

            # Compute log probs under current policy
            log_prob = self.compute_log_probs(
                self.model,
                full_text,
                prompt_data["image"],
                prompt_data["system_prompt"],
                prompt_data["user_prompt"],
            )

            # Compute log probs under reference policy
            with torch.no_grad():
                ref_log_prob = self.compute_log_probs(
                    self.ref_model,
                    full_text,
                    prompt_data["image"],
                    prompt_data["system_prompt"],
                    prompt_data["user_prompt"],
                )

            # Importance ratio (Eq. 2)
            ratio = torch.exp(log_prob - ref_log_prob)
            advantage_tensor = torch.tensor(
                advantage, device=self.device, dtype=torch.float32
            )

            # Clipped surrogate (Eq. 2)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
            )
            clip_loss = -torch.min(
                ratio * advantage_tensor, clipped_ratio * advantage_tensor
            )

            # KL divergence penalty (Eq. 3)
            kl_div = ref_log_prob - log_prob  # Approximate KL
            kl_loss = self.kl_coeff * kl_div

            # Total GRPO loss (Eq. 4)
            loss = clip_loss + kl_loss

            total_loss += loss.item()
            total_clip_loss += clip_loss.item()
            total_kl_loss += kl_loss.item()
            num_samples += 1

            # Backward
            loss.backward()

        # Optimizer step
        if num_samples > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        metrics = {
            "loss": total_loss / max(num_samples, 1),
            "clip_loss": total_clip_loss / max(num_samples, 1),
            "kl_loss": total_kl_loss / max(num_samples, 1),
            "num_rollouts": num_samples,
            "avg_reward": np.mean([r["reward"] for r in rollouts]) if rollouts else 0,
            "avg_advantage": np.mean([r["advantage"] for r in rollouts]) if rollouts else 0,
        }
        return metrics

    def train(
        self,
        dataset: MMADGRPODataset,
        num_epochs: int = 3,
        save_dir: str = "./checkpoints/grpo",
        log_interval: int = 10,
    ):
        """Full GRPO training loop."""
        os.makedirs(save_dir, exist_ok=True)
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Process one prompt at a time for rollouts
            shuffle=True,
            collate_fn=lambda x: x[0],  # Return single item
        )

        global_step = 0
        for epoch in range(num_epochs):
            epoch_rewards = []
            pbar = tqdm(dataloader, desc=f"GRPO Epoch {epoch + 1}/{num_epochs}")

            replay_buffer = []

            for batch_idx, prompt_data in enumerate(pbar):
                # Collect rollouts for this prompt
                rollouts = self.collect_rollouts([prompt_data])
                replay_buffer.extend(rollouts)

                # Train when buffer is full
                if len(replay_buffer) >= self.replay_buffer_size:
                    metrics = self.train_step(replay_buffer)
                    epoch_rewards.extend([r["reward"] for r in replay_buffer])
                    replay_buffer = []
                    global_step += 1

                    if global_step % log_interval == 0:
                        print(
                            f"Step {global_step} | "
                            f"Loss: {metrics['loss']:.4f} | "
                            f"Reward: {metrics['avg_reward']:.4f} | "
                            f"KL: {metrics['kl_loss']:.4f}"
                        )

                    pbar.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "reward": f"{metrics['avg_reward']:.4f}",
                    })

            # Process remaining buffer
            if replay_buffer:
                metrics = self.train_step(replay_buffer)
                epoch_rewards.extend([r["reward"] for r in replay_buffer])

            avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0
            print(f"Epoch {epoch + 1} complete. Avg reward: {avg_epoch_reward:.4f}")

            # Save checkpoint
            ckpt_dir = os.path.join(save_dir, f"epoch_{epoch + 1}")
            self.model.save_pretrained(ckpt_dir)
            self.processor.save_pretrained(ckpt_dir)

        # Save final model
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)
        print(f"GRPO training complete. Model saved to {save_dir}")


def run_grpo(args):
    """Main GRPO training function."""
    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.sft_model_path, trust_remote_code=True
    )

    # Load SFT model as initial policy
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.sft_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "sdpa",
        trust_remote_code=True,
    ).to(args.device)

    # Load reference model (frozen copy of SFT model)
    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.sft_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "sdpa",
        trust_remote_code=True,
    ).to(args.device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load dataset
    domain_knowledge = load_domain_knowledge(args.domain_knowledge_path)
    # Load pre-split GRPO samples
    with open(args.grpo_samples_path, "r") as f:
        grpo_samples = json.load(f)

    dataset = MMADGRPODataset(
        samples=grpo_samples,
        mmad_root=args.mmad_root,
        domain_knowledge=domain_knowledge,
        mode=args.mode,
    )

    # Initialize components
    sv_config = None
    if "sv" in args.mode:
        sv_config = {
            "grounding_dino_checkpoint": args.grounding_dino_checkpoint,
            "sam2_checkpoint": args.sam2_checkpoint,
            "sam2_model_cfg": args.sam2_model_cfg,
            "device": args.device,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
        }
    tool_executor = ToolExecutor(mmad_root=args.mmad_root, sv_config=sv_config)
    reward_computer = RewardComputer(
        alpha=args.alpha,
        beta=args.beta,
        lambda_type=args.lambda_type,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_3=args.lambda_3,
        expected_tool_usage=args.expected_tool_usage,
        lambda_4=args.lambda_4,
    )

    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        processor=processor,
        tool_executor=tool_executor,
        reward_computer=reward_computer,
        rollouts_per_prompt=args.rollouts_per_prompt,
        replay_buffer_size=args.replay_buffer_size,
        lr=args.learning_rate,
        kl_coeff=args.kl_coeff,
        clip_ratio=args.clip_ratio,
        temperature=args.temperature,
        zero_advantage_filtering=args.zero_advantage_filtering,
        max_rounds=args.max_rounds,
        device=args.device,
    )

    # Train
    trainer.train(
        dataset=dataset,
        num_epochs=args.num_epochs,
        save_dir=args.output_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="AgentIAD GRPO Training")
    parser.add_argument("--sft_model_path", type=str, default="./checkpoints/sft")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/grpo")
    parser.add_argument("--mmad_root", type=str, default="./data/MMAD")
    parser.add_argument("--domain_knowledge_path", type=str,
                        default="./data/MMAD/domain_knowledge.json")
    parser.add_argument("--grpo_samples_path", type=str,
                        default="./trajectories/grpo_samples.json")
    parser.add_argument("--mode", type=str, default="pz_cr_sv",
                        choices=["pz_only", "pz_cr", "pz_cr_sv"])
    # GRPO hyperparams (Table 5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--rollouts_per_prompt", type=int, default=8)
    parser.add_argument("--replay_buffer_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--kl_coeff", type=float, default=0.1)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--zero_advantage_filtering", action="store_true", default=True)
    # Reward coefficients (Table 6)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lambda_type", type=float, default=0.1)
    parser.add_argument("--lambda_1", type=float, default=1.0)
    parser.add_argument("--lambda_2", type=float, default=0.5)
    parser.add_argument("--lambda_3", type=float, default=0.05)
    parser.add_argument("--expected_tool_usage", type=float, default=1.0)
    parser.add_argument("--lambda_4", type=float, default=0.3,
                        help="SV-diversity reward weight")
    # Misc
    parser.add_argument("--max_rounds", type=int, default=4)
    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    # Structural Validator (Grounded SAM 2) arguments
    parser.add_argument("--grounding_dino_checkpoint", type=str,
                        default="./models/grounded_sam2/grounding_dino_swinb_cogcoor.pth")
    parser.add_argument("--sam2_checkpoint", type=str,
                        default="./models/grounded_sam2/sam2_hiera_large.pt")
    parser.add_argument("--sam2_model_cfg", type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.2)
    args = parser.parse_args()

    run_grpo(args)


if __name__ == "__main__":
    main()
