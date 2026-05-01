"""
Inference and Evaluation pipeline for AgentIAD.

Performs multi-round tool-augmented reasoning on the MMAD eval split,
computing binary classification accuracy per dataset and overall.
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from data.data_utils import load_domain_knowledge, load_mask, bbox_from_mask
from data.mmad_dataset import MMADGRPODataset, scan_mmad_samples, ALL_DATASET_NAMES
from data.data_utils import split_dataset
from tools.visual_tools import ToolExecutor


class AgentIADInference:
    """
    Multi-round inference engine for AgentIAD.
    The agent iteratively reasons, invokes tools (PZ, CR), and produces a final answer.
    """

    def __init__(
        self,
        model,
        processor,
        tool_executor: ToolExecutor,
        max_rounds: int = 3,
        max_new_tokens: int = 512,
        device: str = "cuda",
    ):
        self.model = model
        self.processor = processor
        self.tool_executor = tool_executor
        self.max_rounds = max_rounds
        self.max_new_tokens = max_new_tokens
        self.device = device

    @torch.no_grad()
    def infer(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        dataset_name: str,
        category: str,
        image_path: str,
    ) -> Dict:
        """
        Run multi-round inference on a single image.

        Returns dict with:
            - prediction: bool (anomaly_present)
            - predicted_type: str
            - full_text: complete reasoning trace
            - num_rounds: number of reasoning rounds
            - tools_used: list of tool names invoked
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
        full_text = ""
        tools_used = []

        for round_idx in range(self.max_rounds):
            # Build input
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
            ).to(self.device)

            # Generate
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,  # Low temperature for deterministic inference
                do_sample=False,
            )
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            response = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            full_text += response

            # Check for tool call
            tool_call = self.tool_executor.parse_tool_call(response)
            if tool_call is not None:
                tools_used.append(tool_call.get("name", "unknown"))
                result_image, result_text = self.tool_executor.execute(
                    tool_call,
                    images_in_context,
                    dataset_name,
                    category,
                    image_path,
                )
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
                    messages.append({"role": "user", "content": result_text})
            else:
                # Check for final answer
                answer = self.tool_executor.parse_final_answer(response)
                if answer is not None:
                    pred = answer.get("anomaly_present", False)
                    if isinstance(pred, str):
                        pred = pred.lower() in ("true", "1", "yes")
                    return {
                        "prediction": pred,
                        "predicted_type": answer.get("top_anomaly", "none"),
                        "visual_descriptions": answer.get("visual_descriptions", []),
                        "full_text": full_text,
                        "num_rounds": round_idx + 1,
                        "tools_used": tools_used,
                    }
                messages.append({"role": "assistant", "content": response})

        # If no answer after max rounds, attempt to parse from last response
        answer = self.tool_executor.parse_final_answer(full_text)
        pred = False
        pred_type = "none"
        if answer is not None:
            pred = answer.get("anomaly_present", False)
            pred_type = answer.get("top_anomaly", "none")

        return {
            "prediction": pred,
            "predicted_type": pred_type,
            "visual_descriptions": [],
            "full_text": full_text,
            "num_rounds": self.max_rounds,
            "tools_used": tools_used,
        }


def evaluate(args):
    """Run evaluation on the MMAD eval split."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    # Load model and processor
    print(f"Loading model from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "sdpa",
        trust_remote_code=True,
    ).to(args.device).eval()

    # Load dataset
    domain_knowledge = load_domain_knowledge(args.domain_knowledge_path)

    # Load pre-split eval samples
    if args.eval_samples_path and os.path.exists(args.eval_samples_path):
        with open(args.eval_samples_path, "r") as f:
            eval_samples = json.load(f)
    else:
        all_samples = scan_mmad_samples(args.mmad_root, domain_knowledge)
        _, _, eval_samples = split_dataset(all_samples)

    print(f"Evaluating on {len(eval_samples)} samples")

    # Build dataset for prompt generation
    eval_dataset = MMADGRPODataset(
        samples=eval_samples,
        mmad_root=args.mmad_root,
        domain_knowledge=domain_knowledge,
        mode=args.mode,
    )

    # Initialize inference engine
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
    engine = AgentIADInference(
        model=model,
        processor=processor,
        tool_executor=tool_executor,
        max_rounds=args.max_rounds,
        device=args.device,
    )

    # Run inference
    results = []
    correct_per_dataset = defaultdict(int)
    total_per_dataset = defaultdict(int)
    correct_total = 0

    os.makedirs(args.output_dir, exist_ok=True)

    for idx in tqdm(range(len(eval_dataset)), desc="Evaluating"):
        sample_data = eval_dataset[idx]
        gt = sample_data["ground_truth"]

        result = engine.infer(
            image=sample_data["image"],
            system_prompt=sample_data["system_prompt"],
            user_prompt=sample_data["user_prompt"],
            dataset_name=gt["dataset_name"],
            category=gt["category"],
            image_path=sample_data["image_path"],
        )

        # Check correctness
        is_correct = result["prediction"] == gt["anomaly_present"]
        if is_correct:
            correct_total += 1
            correct_per_dataset[gt["dataset_name"]] += 1
        total_per_dataset[gt["dataset_name"]] += 1

        results.append({
            "image_path": sample_data["image_path"],
            "dataset": gt["dataset_name"],
            "category": gt["category"],
            "gt_anomaly_present": gt["anomaly_present"],
            "gt_anomaly_type": gt.get("anomaly_type", "none"),
            "prediction": result["prediction"],
            "predicted_type": result["predicted_type"],
            "correct": is_correct,
            "num_rounds": result["num_rounds"],
            "tools_used": result["tools_used"],
        })

    # Compute per-dataset and overall accuracy
    print("\n" + "=" * 60)
    print("Evaluation Results (Binary Classification Accuracy %)")
    print("=" * 60)

    # Per-category accuracy within each dataset
    category_correct = defaultdict(lambda: defaultdict(int))
    category_total = defaultdict(lambda: defaultdict(int))
    for r in results:
        ds = r["dataset"]
        cat = r["category"]
        category_total[ds][cat] += 1
        if r["correct"]:
            category_correct[ds][cat] += 1

    dataset_avg_acc = {}
    for ds in ALL_DATASET_NAMES:
        if ds not in total_per_dataset:
            continue
        # Category-averaged accuracy (as in paper)
        cat_accs = []
        for cat in category_total[ds]:
            total = category_total[ds][cat]
            correct = category_correct[ds][cat]
            acc = 100.0 * correct / total if total > 0 else 0
            cat_accs.append(acc)
        avg_acc = sum(cat_accs) / len(cat_accs) if cat_accs else 0
        dataset_avg_acc[ds] = avg_acc
        print(f"  {ds:15s}: {avg_acc:.2f}% (category-averaged)")

    overall = sum(dataset_avg_acc.values()) / len(dataset_avg_acc) if dataset_avg_acc else 0
    print(f"\n  {'Overall':15s}: {overall:.2f}%")
    print("=" * 60)

    # Also report raw accuracy
    raw_acc = 100.0 * correct_total / len(results) if results else 0
    print(f"  Raw accuracy (not category-averaged): {raw_acc:.2f}%")

    # Tool usage statistics
    pz_count = sum(1 for r in results if "crop_image_normalized" in r["tools_used"])
    cr_count = sum(1 for r in results if "query_image" in r["tools_used"])
    sv_count = sum(1 for r in results if "segment_and_count" in r["tools_used"])
    print(f"\n  Tool usage: PZ={pz_count}, CR={cr_count}, SV={sv_count}")

    # Save results
    output_path = os.path.join(args.output_dir, "eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": args.mode,
            "model_path": args.model_path,
            "num_samples": len(results),
            "category_averaged_accuracy": dataset_avg_acc,
            "overall_accuracy": overall,
            "raw_accuracy": raw_acc,
            "per_sample_results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AgentIAD Evaluation")
    parser.add_argument("--model_path", type=str, default="./checkpoints/grpo")
    parser.add_argument("--mmad_root", type=str, default="./data/MMAD")
    parser.add_argument("--domain_knowledge_path", type=str,
                        default="./data/MMAD/domain_knowledge.json")
    parser.add_argument("--eval_samples_path", type=str, default="./trajectories/eval_samples.json")
    parser.add_argument("--mode", type=str, default="pz_cr_sv",
                        choices=["pz_only", "pz_cr", "pz_cr_sv"])
    parser.add_argument("--max_rounds", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./evaluation/results")
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

    evaluate(args)


if __name__ == "__main__":
    main()
