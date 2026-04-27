"""
MMAD Dataset classes for AgentIAD training and evaluation.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .data_utils import (
    bbox_from_mask,
    crop_image_by_bbox,
    get_anomaly_types_for_category,
    get_normal_reference,
    load_domain_knowledge,
    load_mask,
)


# All sub-datasets in MMAD
ALL_DATASET_NAMES = ["MVTec", "VisA", "MVTec-LOCO", "GoodsAD"]
# Also support common directory naming variants
DATASET_NAME_ALIASES = {
    "MVTec-AD": "MVTec", "mvtec": "MVTec", "mvtec-ad": "MVTec",
    "visa": "VisA", "VisA": "VisA",
    "MVTec-LOCO": "MVTec-LOCO", "mvtec-loco": "MVTec-LOCO",
    "GoodsAD": "GoodsAD", "goodsad": "GoodsAD",
}


def scan_mmad_samples(
    mmad_root: str,
    domain_knowledge: Dict,
    dataset_names: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Scan the MMAD dataset directory and build a list of all samples.

    Args:
        mmad_root: Root directory of MMAD dataset.
        domain_knowledge: Domain knowledge dict.
        dataset_names: List of sub-dataset names to scan. If None, scan all.
            Accepts aliases like "MVTec-AD" for "MVTec".

    Expected MMAD structure:
        {mmad_root}/{dataset_name}/{category}/test/{defect_type_or_good}/{image_files}
        {mmad_root}/{dataset_name}/{category}/ground_truth/{defect_type}/{mask_files}

    Returns list of dicts with keys:
        - image_path, mask_path (None for normal), dataset_name, category,
          anomaly_present, anomaly_type, anomaly_types_list
    """
    if dataset_names is None:
        dataset_names = ALL_DATASET_NAMES

    # Also try to match actual directory names on disk (handle naming variants)
    scan_dirs = []
    for name in dataset_names:
        # Try the name directly first
        candidate = os.path.join(mmad_root, name)
        if os.path.isdir(candidate):
            scan_dirs.append((name, candidate))
            continue
        # Try canonical name
        canonical = DATASET_NAME_ALIASES.get(name, name)
        candidate = os.path.join(mmad_root, canonical)
        if os.path.isdir(candidate):
            scan_dirs.append((canonical, candidate))
            continue
        # Try scanning mmad_root for case-insensitive match
        if os.path.isdir(mmad_root):
            for d in os.listdir(mmad_root):
                if d.lower().replace("-", "").replace("_", "") == name.lower().replace("-", "").replace("_", ""):
                    scan_dirs.append((d, os.path.join(mmad_root, d)))
                    break

    samples = []
    for dataset_name, dataset_dir in scan_dirs:
        for category in sorted(os.listdir(dataset_dir)):
            cat_dir = os.path.join(dataset_dir, category)
            test_dir = os.path.join(cat_dir, "test")
            if not os.path.isdir(test_dir):
                continue
            anomaly_types = get_anomaly_types_for_category(
                domain_knowledge, dataset_name, category
            )
            for defect_type in sorted(os.listdir(test_dir)):
                defect_dir = os.path.join(test_dir, defect_type)
                if not os.path.isdir(defect_dir):
                    continue
                is_normal = defect_type.lower() in ("good", "normal")
                for img_name in sorted(os.listdir(defect_dir)):
                    if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        continue
                    image_path = os.path.join(defect_dir, img_name)
                    mask_path = None
                    if not is_normal:
                        # Try to find corresponding mask
                        mask_name = os.path.splitext(img_name)[0] + "_mask.png"
                        candidate_mask = os.path.join(
                            cat_dir, "ground_truth", defect_type, mask_name
                        )
                        if os.path.exists(candidate_mask):
                            mask_path = candidate_mask
                        else:
                            # Try without _mask suffix
                            candidate_mask2 = os.path.join(
                                cat_dir, "ground_truth", defect_type, img_name
                            )
                            if os.path.exists(candidate_mask2):
                                mask_path = candidate_mask2
                    samples.append({
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "dataset_name": dataset_name,
                        "category": category,
                        "anomaly_present": not is_normal,
                        "anomaly_type": defect_type if not is_normal else "none",
                        "anomaly_types_list": anomaly_types,
                    })
    return samples


class MMADDataset(Dataset):
    """Base MMAD dataset that returns raw sample info."""

    def __init__(self, samples: List[Dict], mmad_root: str):
        self.samples = samples
        self.mmad_root = mmad_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        result = {**sample, "image": image}
        if sample["mask_path"] is not None:
            mask = load_mask(sample["mask_path"])
            result["mask"] = mask
            result["bbox"] = bbox_from_mask(mask)
        else:
            result["mask"] = None
            result["bbox"] = None
        return result


class MMADSFTDataset(Dataset):
    """
    SFT Dataset that loads pre-constructed trajectories.
    Each item is a multi-turn conversation with tool calls formatted for Qwen2.5-VL.
    """

    def __init__(
        self,
        trajectory_dir: str,
        processor: Any,
        max_length: int = 4096,
    ):
        self.processor = processor
        self.max_length = max_length
        # Load all trajectory JSON files
        self.trajectories = []
        traj_dir = Path(trajectory_dir)
        for traj_file in sorted(traj_dir.glob("*.json")):
            with open(traj_file, "r", encoding="utf-8") as f:
                self.trajectories.append(json.load(f))

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict:
        traj = self.trajectories[idx]
        messages = traj["messages"]
        images = []
        for img_path in traj.get("image_paths", []):
            images.append(Image.open(img_path).convert("RGB"))

        # Build the conversation for Qwen2.5-VL
        # The processor handles image tokens automatically
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Build loss mask: only supervise final reasoning + last tool call
        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        loss_mask = self._build_loss_mask(traj, labels)
        labels[loss_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
        }

    def _build_loss_mask(self, traj: Dict, labels) -> np.ndarray:
        """
        Build loss mask per Eq.(1): mt=1 only for final reasoning response
        and last tool invocation output.
        """
        mask = np.zeros(len(labels), dtype=np.int32)
        # Find positions of the last two assistant turns
        # traj["mask_ranges"] contains token ranges for supervised segments
        if "mask_ranges" in traj:
            for start, end in traj["mask_ranges"]:
                start = min(start, len(mask) - 1)
                end = min(end, len(mask))
                mask[start:end] = 1
        else:
            # Fallback: supervise all assistant tokens
            mask[:] = 1
        return mask


class MMADGRPODataset(Dataset):
    """
    GRPO Dataset that provides prompts for rollout generation.
    Each item contains the initial prompt and ground truth for reward computation.
    """

    def __init__(
        self,
        samples: List[Dict],
        mmad_root: str,
        domain_knowledge: Dict,
        mode: str = "pz_cr",
    ):
        self.samples = samples
        self.mmad_root = mmad_root
        self.domain_knowledge = domain_knowledge
        self.mode = mode  # "pz_only" or "pz_cr"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        anomaly_types_str = ", ".join(sample["anomaly_types_list"])

        # Build system prompt based on mode
        system_prompt = self._get_system_prompt()
        user_prompt = self._get_user_prompt(
            sample["category"], anomaly_types_str
        )

        # Ground truth for reward computation
        gt = {
            "anomaly_present": sample["anomaly_present"],
            "anomaly_type": sample["anomaly_type"],
            "bbox": None,
            "dataset_name": sample["dataset_name"],
            "category": sample["category"],
        }
        if sample["mask_path"] is not None:
            mask = load_mask(sample["mask_path"])
            gt["bbox"] = bbox_from_mask(mask)

        return {
            "image": image,
            "image_path": sample["image_path"],
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "ground_truth": gt,
            "mmad_root": self.mmad_root,
        }

    def _get_system_prompt(self) -> str:
        """Get system prompt based on mode (Section 7.2 / 7.3)."""
        base = (
            "You are a vision expert specialized in industrial anomaly detection. "
            "You will evaluate whether the given object image is normal or abnormal. "
            "If abnormal, select the most fitting anomaly label from the candidate types "
            "provided by the user.\n"
            "Output format:\n"
            "<think> Explain your visual reasoning. </think>\n"
            '<answer> {"anomaly_present": true/false, "top_anomaly": "<label or \'none\'>", '
            '"visual_descriptions": ["..."]} </answer>\n'
            "If normal → anomaly_present=false, top_anomaly=\"none\", visual_descriptions=[].\n"
            "If abnormal → include concise visual phrases for visible cues.\n\n"
        )
        pz_tool = (
            '# Tools\nYou may call function to assist with the user query.\n\n'
            'You are provided with function signatures within <tools> </tools> XML tags:\n'
            '<tools>\n'
            '{"type": "function", "function": {"name": "crop_image_normalized", '
            '"description": "Zoom in on the image based on the bounding box coordinates.", '
            '"parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", '
            '"description": "normalized coordinates for bounding box of the region you want '
            'to zoom in. Values should be within [0.0,1.0].", "items": {"type": "number"}}, '
            '"target_image": {"type": "number", "description": "The index of the image to '
            'crop. Index from 1 to the number of images. Choose 1 to operate on original '
            'image."}}, "required": ["bbox_2d", "target_image"]}}}'
        )
        cr_tool = (
            '\n{"type": "function", "function": {"name": "query_image", '
            '"description": "Retrieve a normal reference image of the same class for '
            'comparison. This function does not require any arguments.", '
            '"parameters": {"type": "object", "properties": {}, "required": []}}}'
        )
        tool_call_fmt = (
            '\n</tools>\n'
            'For each function call, return a json object with function name and arguments '
            'within <tool_call></tool_call> XML tags:\n'
            '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n'
            '</tool_call>'
        )
        if self.mode == "pz_only":
            return base + pz_tool + tool_call_fmt
        else:
            return base + pz_tool + cr_tool + tool_call_fmt

    def _get_user_prompt(self, class_name: str, anomaly_types: str) -> str:
        """Get user prompt based on mode (Section 7.3.1 / 7.3.2)."""
        base = (
            f"Evaluate the following image from the class \"{class_name}\". "
            f"Candidate anomaly types: {anomaly_types}. "
            "Determine if the object is normal or abnormal. "
            "Follow the instruction and we can look closer by `crop_image_normalized`."
        )
        if self.mode == "pz_cr":
            base += (
                " If, after inspecting the crop, the evidence is still insufficient, "
                "you may also call `query_image` to retrieve a normal reference image."
            )
        base += (
            "\nReason with the visual information step by step, "
            "and output the final answer in the required XML format."
        )
        return base
