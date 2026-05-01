# AgentIAD - Unofficial Reproduction + Extension

> **Disclaimer:** This is an **unofficial reproduction** of the paper *"AgentIAD: Tool-Augmented Single-Agent for Industrial Anomaly Detection"*. This project is not affiliated with or endorsed by the original authors.

An unofficial PyTorch implementation of AgentIAD, a tool-driven agentic framework that enables multi-stage visual inspection for industrial anomaly detection. The agent uses a **Perceptive Zoomer (PZ)** for localized fine-grained analysis and a **Comparative Retriever (CR)** for querying normal exemplars.

## Our Extension: Structural Validator (SV) for Logical Anomaly Detection

Beyond the original paper, we introduce a new tool — **Structural Validator (SV)** — that uses [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) (Grounding DINO + SAM 2) to address **logical anomaly detection** (missing components, wrong counts, incorrect arrangements), a known weakness in existing IAD methods.

**How it works:**
- The agent generates a text query (e.g., "screws") describing the component to inspect
- Grounded SAM 2 detects and segments all matching instances in the image
- Returns an annotated image with numbered masks + structured count summary
- The agent reasons about whether the structure is normal (the SV tool does NOT make judgments)

**Key design choices:**
- SV tool is only activated for logical anomaly datasets (MVTec-LOCO, GoodsAD) via per-sample mode selection
- New `lambda_4` reward term encourages SV usage on logical anomaly samples during GRPO training
- 80 structural trajectories (PZ -> SV -> answer) are added to SFT training
- `max_rounds` increased from 3 to 4 to accommodate the additional tool call

> **If you only want the original paper reproduction** without the SV extension, use `--mode pz_cr` in all scripts. This disables all SV-related functionality and matches the original AgentIAD behavior.

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
├── models/           # Model definitions + Grounded-SAM-2
├── scripts/          # Shell scripts for each training stage
├── tools/            # Visual tools (PZ, CR, SV)
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

For the SV extension, also install Grounded SAM 2:

```bash
cd AgentIAD/models
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2 && pip install -e .
cd grounding_dino && pip install -e . && cd ../..
```

## Data Preparation

1. Download the [MMAD dataset](https://github.com/jam-cc/MMAD) and place it under `AgentIAD/data/MMAD/`.
2. Download the base model [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) and place it under `AgentIAD/models/`.
3. (For SV extension) Download Grounding DINO and SAM 2 weights into `AgentIAD/models/grounded_sam2/`.

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

**Original reproduction only (no SV):** add `MODE=pz_cr` before Step 3/4, or pass `--mode pz_cr` directly.

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
