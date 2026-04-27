"""
AgentIAD Configuration - All hyperparameters from the paper.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Dataset and data split configuration."""
    mmad_root: str = "./data/MMAD"
    domain_knowledge_path: str = "./data/MMAD/domain_knowledge.json"
    # 20% train (1600 SFT + 366 GRPO), 80% eval (6400)
    sft_num_samples: int = 1600
    grpo_num_samples: int = 366
    # Among SFT samples, 112 are PZ+CR type, rest are PZ-only
    sft_pz_cr_samples: int = 112
    image_size: int = 1280  # Qwen2.5-VL default
    seed: int = 42


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name_or_path: str = "./models/Qwen2.5-VL-3B-Instruct"
    freeze_vision_encoder: bool = True
    use_flash_attention: bool = True
    dtype: str = "bfloat16"


@dataclass
class SFTConfig:
    """Supervised Fine-Tuning hyperparameters (Table 4)."""
    output_dir: str = "./checkpoints/sft"
    num_train_epochs: int = 20
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    bf16: bool = True
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    dataloader_num_workers: int = 4
    deepspeed: Optional[str] = "./configs/ds_zero2.json"


@dataclass
class GRPOConfig:
    """GRPO Agentic RL hyperparameters (Table 5)."""
    output_dir: str = "./checkpoints/grpo"
    sft_model_path: str = "./checkpoints/sft"
    num_train_epochs: int = 3
    learning_rate: float = 1e-6
    rollouts_per_prompt: int = 8  # K=8
    replay_buffer_size: int = 128
    global_batch_size: int = 128
    temperature: float = 1.0
    kl_coeff: float = 0.1  # beta
    clip_ratio: float = 0.2  # epsilon
    zero_advantage_filtering: bool = True
    bf16: bool = True
    deepspeed: Optional[str] = "./configs/ds_zero3_offload.json"


@dataclass
class RewardConfig:
    """Reward coefficients (Table 6)."""
    # Perception reward weights
    alpha: float = 1.0  # perception reward weight
    # Accuracy reward: 1 if format valid AND prediction correct
    # IoU reward
    iou_threshold: float = 0.5
    iou_reward_above_threshold: float = 1.0
    # Type reward
    lambda_type: float = 0.1  # type reward weight (paper says 0.1 in Table 6)
    type_reward_bonus: float = 0.1
    # Behavior reward weights
    beta: float = 1.0  # behavior reward weight
    lambda_1: float = 1.0  # stepwise correctness
    lambda_2: float = 0.5  # CR-diversity term (0.5 * (query_rate - 1))
    lambda_3: float = 0.05  # tool-call efficiency
    expected_tool_usage: float = 1.0  # n*


@dataclass
class TrajectoryConfig:
    """Trajectory construction configuration."""
    output_dir: str = "./trajectories"
    gpt_model: str = "gpt-4o"
    openai_api_key: str = ""
    openai_base_url: str = ""
    max_concurrent_requests: int = 8
    # Normal sample ROI generation
    normal_roi_prompt_template: str = "normal_roi"
    # CoT generation
    cot1_system_prompt: str = "cot1_system"
    cot2_system_prompt: str = "cot2_system"
    cot3_system_prompt: str = "cot3_system"


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    model_path: str = "./checkpoints/grpo"
    mode: str = "pz_cr"  # "pz_only" or "pz_cr"
    batch_size: int = 1
    max_rounds: int = 3
    output_dir: str = "./evaluation/results"
