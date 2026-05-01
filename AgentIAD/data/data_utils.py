"""
Data utility functions for MMAD dataset processing.
"""
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def load_domain_knowledge(path: str) -> Dict:
    """Load domain knowledge JSON containing anomaly type descriptions per category."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_anomaly_types_for_category(
    domain_knowledge: Dict, dataset_name: str, category: str
) -> List[str]:
    """Extract anomaly type names for a given dataset and category."""
    if dataset_name not in domain_knowledge:
        return []
    if category not in domain_knowledge[dataset_name]:
        return []
    types = list(domain_knowledge[dataset_name][category].keys())
    # Remove 'good' / 'normal' from anomaly type list
    types = [t for t in types if t.lower() not in ("good", "normal")]
    return types


def bbox_from_mask(mask: np.ndarray, normalize: bool = True) -> List[float]:
    """
    Extract bounding box [x1, y1, x2, y2] from a binary mask.
    If normalize=True, coordinates are in [0, 1] relative to image dimensions.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        # No defect region, return center crop
        h, w = mask.shape
        return [0.25, 0.25, 0.75, 0.75]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = mask.shape
    # Add small padding (5%)
    pad_h, pad_w = int(0.05 * h), int(0.05 * w)
    rmin = max(0, rmin - pad_h)
    rmax = min(h - 1, rmax + pad_h)
    cmin = max(0, cmin - pad_w)
    cmax = min(w - 1, cmax + pad_w)
    if normalize:
        return [
            round(cmin / w, 4),
            round(rmin / h, 4),
            round(cmax / w, 4),
            round(rmax / h, 4),
        ]
    return [cmin, rmin, cmax, rmax]


def crop_image_by_bbox(
    image: Image.Image, bbox: List[float], normalized: bool = True
) -> Image.Image:
    """
    Crop image by bounding box.
    bbox: [x1, y1, x2, y2] in normalized [0,1] coords if normalized=True.
    """
    w, h = image.size
    if normalized:
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)
    else:
        x1, y1, x2, y2 = [int(c) for c in bbox]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return image.crop((x1, y1, x2, y2))


def compute_iou(bbox_pred: List[float], bbox_gt: List[float]) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(bbox_pred[0], bbox_gt[0])
    y1 = max(bbox_pred[1], bbox_gt[1])
    x2 = min(bbox_pred[2], bbox_gt[2])
    y2 = min(bbox_pred[3], bbox_gt[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_pred = max(0, bbox_pred[2] - bbox_pred[0]) * max(0, bbox_pred[3] - bbox_pred[1])
    area_gt = max(0, bbox_gt[2] - bbox_gt[0]) * max(0, bbox_gt[3] - bbox_gt[1])
    union = area_pred + area_gt - inter
    if union <= 0:
        return 0.0
    return inter / union


def get_normal_reference(
    mmad_root: str, dataset_name: str, category: str, exclude_path: Optional[str] = None
) -> Optional[str]:
    """
    Get a random normal reference image path for a given category.
    Used by the Comparative Retriever tool.
    """
    # MMAD organizes normal images under: {dataset_name}/{category}/test/good/
    # or {dataset_name}/{category}/train/good/
    base_dirs = [
        os.path.join(mmad_root, dataset_name, category, "test", "good"),
        os.path.join(mmad_root, dataset_name, category, "train", "good"),
    ]
    normal_images = []
    for base_dir in base_dirs:
        if os.path.isdir(base_dir):
            for fname in os.listdir(base_dir):
                fpath = os.path.join(base_dir, fname)
                if fpath != exclude_path and fname.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp")
                ):
                    normal_images.append(fpath)
    if not normal_images:
        return None
    return random.choice(normal_images)


def split_dataset(
    all_samples: List[Dict],
    sft_num: int = 1600,
    grpo_num: int = 366,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into SFT, GRPO, and eval sets.
    Paper: 20% train (1600 SFT + 366 GRPO), 80% eval (6400).
    """
    rng = random.Random(seed)
    indices = list(range(len(all_samples)))
    rng.shuffle(indices)
    sft_indices = indices[:sft_num]
    grpo_indices = indices[sft_num : sft_num + grpo_num]
    eval_indices = indices[sft_num + grpo_num :]
    sft_set = [all_samples[i] for i in sft_indices]
    grpo_set = [all_samples[i] for i in grpo_indices]
    eval_set = [all_samples[i] for i in eval_indices]
    return sft_set, grpo_set, eval_set


def load_mask(mask_path: str) -> np.ndarray:
    """Load a defect mask as a binary numpy array."""
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask > 127).astype(np.uint8)


# Datasets known to contain logical/structural anomalies
LOGICAL_ANOMALY_DATASETS = {"MVTec-LOCO", "GoodsAD"}


def is_logical_anomaly_sample(sample: Dict) -> bool:
    """
    Check if a sample comes from a logical anomaly context.
    Used for: trajectory selection, reward shaping, mode-aware prompting.
    MVTec-LOCO and GoodsAD are the two MMAD datasets focused on logical anomalies.
    """
    return sample.get("dataset_name", "") in LOGICAL_ANOMALY_DATASETS
