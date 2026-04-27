# AgentIAD - Unofficial Reproduction

> **Disclaimer:** This is an **unofficial reproduction** of the paper *"AgentIAD: Tool-Augmented Single-Agent for Industrial Anomaly Detection"*. This project is not affiliated with or endorsed by the original authors.

An unofficial PyTorch implementation of AgentIAD, a tool-driven agentic framework that enables multi-stage visual inspection for industrial anomaly detection. The agent uses a **Perceptive Zoomer (PZ)** for localized fine-grained analysis and a **Comparative Retriever (CR)** for querying normal exemplars.

## Original Paper

- **Title:** AgentIAD: Tool-Augmented Single-Agent for Industrial Anomaly Detection
- **Authors:** Junwen Miao, Penghui Du, Yi Liu, Yu Wang, Yan Wang
- **Affiliations:** AIR, Tsinghua University; Beihang University
- **arXiv:** [2512.13671](https://arxiv.org/abs/2512.13671)

## Project Structure

```
AgentIAD/
├── configs/          # Training configs (DeepSpeed, hyperparameters)
├── data/             # Dataset utilities and MMAD dataloader
├── evaluation/       # Evaluation pipeline
├── models/           # Model definitions
├── scripts/          # Shell scripts for each training stage
├── tools/            # Visual tools (Perceptive Zoomer, Comparative Retriever)
├── trajectories/     # Trajectory construction for SFT
├── training/         # SFT and GRPO trainers
└── requirements.txt
```

## Requirements

- Python >= 3.10
- CUDA >= 12.1
- 1x A100 80GB (minimum for training)

## Installation

```bash
git clone https://github.com/hhhhhray/AgentIAD-reproduction.git
cd AgentIAD-reproduction
pip install -r AgentIAD/requirements.txt
```

## Data Preparation

1. Download the [MMAD dataset](https://github.com/jam-cc/MMAD) and place it under `AgentIAD/data/MMAD/`.
2. Download the base model [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) and place it under `AgentIAD/models/`.

## Usage

The training pipeline has 4 stages:

```bash
# Step 1: Build structured trajectories (requires OpenAI API key)
export OPENAI_API_KEY="your-key-here"
bash AgentIAD/scripts/step1_build_trajectories.sh

# Step 2: Supervised Fine-Tuning (SFT)
bash AgentIAD/scripts/step2_sft_train.sh

# Step 3: GRPO Reinforcement Learning
bash AgentIAD/scripts/step3_grpo_train.sh

# Step 4: Evaluation
bash AgentIAD/scripts/step4_evaluate.sh
```

## Citation

If you find this work useful, please cite the original paper:

```bibtex
@article{miao2025agentiad,
  title={AgentIAD: Tool-Augmented Single-Agent for Industrial Anomaly Detection},
  author={Miao, Junwen and Du, Penghui and Liu, Yi and Wang, Yu and Wang, Yan},
  journal={arXiv preprint arXiv:2512.13671},
  year={2025}
}
```

## License

This reproduction is released under the [MIT License](LICENSE).
