"""
Visual tools for AgentIAD: Perceptive Zoomer (PZ), Comparative Retriever (CR),
and Structural Validator (SV).
"""
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from PIL import Image

from data.data_utils import crop_image_by_bbox, get_normal_reference


class PerceptiveZoomer:
    """
    Perceptive Zoomer (PZ) tool.
    Crops a region from the input image based on normalized bounding box coordinates.
    Corresponds to the `crop_image_normalized` function in the paper.
    """

    def __call__(
        self,
        images: List[Image.Image],
        bbox_2d: List[float],
        target_image: int = 1,
    ) -> Image.Image:
        """
        Args:
            images: List of images in the conversation context.
            bbox_2d: Normalized [x1, y1, x2, y2] in [0, 1].
            target_image: 1-indexed image index to crop.
        Returns:
            Cropped image region.
        """
        idx = target_image - 1  # Convert to 0-indexed
        if idx < 0 or idx >= len(images):
            idx = 0
        image = images[idx]
        # Clamp bbox values to [0, 1]
        bbox_2d = [max(0.0, min(1.0, v)) for v in bbox_2d]
        # Ensure x1 < x2, y1 < y2
        if bbox_2d[0] >= bbox_2d[2]:
            bbox_2d[2] = min(1.0, bbox_2d[0] + 0.1)
        if bbox_2d[1] >= bbox_2d[3]:
            bbox_2d[3] = min(1.0, bbox_2d[1] + 0.1)
        return crop_image_by_bbox(image, bbox_2d, normalized=True)


class ComparativeRetriever:
    """
    Comparative Retriever (CR) tool.
    Retrieves a normal reference image of the same category.
    Corresponds to the `query_image` function in the paper.
    """

    def __init__(self, mmad_root: str):
        self.mmad_root = mmad_root

    def __call__(
        self,
        dataset_name: str,
        category: str,
        exclude_path: Optional[str] = None,
    ) -> Optional[Image.Image]:
        """
        Args:
            dataset_name: Name of the dataset (MVTec, VisA, etc.).
            category: Product category name.
            exclude_path: Path to exclude from selection.
        Returns:
            A normal reference image, or None if unavailable.
        """
        ref_path = get_normal_reference(
            self.mmad_root, dataset_name, category, exclude_path
        )
        if ref_path is None:
            return None
        return Image.open(ref_path).convert("RGB")


class StructuralValidator:
    """
    Structural Validator (SV) tool.
    Uses Grounded SAM 2 to detect and segment all instances of a queried component.
    Returns an annotated image with numbered masks and a structured text summary.
    Corresponds to the `segment_and_count` function.
    """

    # Color palette for mask annotations (up to 10 distinct colors)
    _COLORS_HEX = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    ]

    def __init__(self, sv_config: Dict):
        self._config = sv_config
        self._grounding_model = None
        self._sam2_predictor = None

    def _load_models(self):
        """Lazy-load Grounded SAM 2 models on first call."""
        import groundingdino
        from groundingdino.util.inference import (
            load_model as load_grounding_dino,
        )
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # Resolve config path from the groundingdino package location
        gdino_dir = os.path.dirname(groundingdino.__file__)
        gdino_cfg = os.path.join(
            gdino_dir, "config", "GroundingDINO_SwinB_cfg.py"
        )

        self._grounding_model = load_grounding_dino(
            model_config_path=gdino_cfg,
            model_checkpoint_path=self._config["grounding_dino_checkpoint"],
            device=self._config.get("device", "cuda"),
        )
        sam2_model = build_sam2(
            self._config.get(
                "sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml"
            ),
            self._config["sam2_checkpoint"],
            device=self._config.get("device", "cuda"),
        )
        self._sam2_predictor = SAM2ImagePredictor(sam2_model)

    def __call__(
        self,
        images: List[Image.Image],
        query: str,
        target_image: int = 1,
    ) -> Tuple[Image.Image, str]:
        """
        Args:
            images: List of images in the conversation context.
            query: Text query describing the component to detect (e.g., "screws").
            target_image: 1-indexed image index.
        Returns:
            (annotated_image, structured_summary_text)
        """
        if self._grounding_model is None:
            self._load_models()

        idx = target_image - 1
        if idx < 0 or idx >= len(images):
            idx = 0
        image = images[idx]

        image_np = np.array(image)
        detections = self._run_grounding_dino(image_np, query)

        if len(detections) == 0:
            summary = (
                f'Structural validation for "{query}": 0 instances detected.'
            )
            return image, summary

        masks = self._run_sam2(image_np, detections.xyxy)
        detections.mask = masks

        annotated = self._draw_annotations(image_np, detections)
        summary = self._build_summary(image_np, detections, query)

        return Image.fromarray(annotated), summary

    def _run_grounding_dino(self, image_np: np.ndarray, query: str):
        """Run Grounding DINO for open-vocabulary detection."""
        from groundingdino.util.inference import predict
        import torchvision.transforms.functional as F

        image_tensor = F.to_tensor(image_np).to(
            self._config.get("device", "cuda")
        )
        boxes, confidences, labels = predict(
            model=self._grounding_model,
            image=image_tensor,
            caption=query,
            box_threshold=self._config.get("box_threshold", 0.25),
            text_threshold=self._config.get("text_threshold", 0.2),
        )

        h, w = image_np.shape[:2]
        # Grounding DINO returns normalized cxcywh — convert to xyxy pixel coords
        boxes_np = boxes.cpu().numpy()
        if len(boxes_np) == 0:
            return sv.Detections.empty()
        cx, cy, bw, bh = (
            boxes_np[:, 0], boxes_np[:, 1],
            boxes_np[:, 2], boxes_np[:, 3],
        )
        xyxy = np.stack([
            (cx - bw / 2) * w,
            (cy - bh / 2) * h,
            (cx + bw / 2) * w,
            (cy + bh / 2) * h,
        ], axis=1)

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidences.cpu().numpy(),
        )

    def _run_sam2(
        self, image_np: np.ndarray, boxes_xyxy: np.ndarray
    ) -> np.ndarray:
        """Run SAM 2 segmentation on detected boxes."""
        self._sam2_predictor.set_image(image_np)
        masks_list = []
        for box in boxes_xyxy:
            mask, _, _ = self._sam2_predictor.predict(
                box=box, multimask_output=False
            )
            masks_list.append(mask[0])
        return np.array(masks_list)

    def _draw_annotations(
        self, image_np: np.ndarray, detections
    ) -> np.ndarray:
        """Draw numbered masks and bounding boxes on the image."""
        annotated = image_np.copy()

        # Parse colors (BGR for OpenCV)
        colors_bgr = []
        for hex_c in self._COLORS_HEX:
            r = int(hex_c[1:3], 16)
            g = int(hex_c[3:5], 16)
            b = int(hex_c[5:7], 16)
            colors_bgr.append((b, g, r))

        # Draw semi-transparent masks
        for i, mask in enumerate(detections.mask):
            color = colors_bgr[i % len(colors_bgr)]
            overlay = annotated.copy()
            overlay[mask] = color
            cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

        # Draw bounding boxes
        for i, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = box.astype(int)
            color = colors_bgr[i % len(colors_bgr)]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw numbered labels at mask centroids
        for i, mask in enumerate(detections.mask):
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            label = str(i + 1)
            font_scale = 0.8
            thickness = 2
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            # Black background rectangle for readability
            cv2.rectangle(
                annotated,
                (cx - tw // 2 - 4, cy - th // 2 - 4),
                (cx + tw // 2 + 4, cy + th // 2 + 4),
                (0, 0, 0),
                -1,
            )
            # White text label
            cv2.putText(
                annotated,
                label,
                (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )

        return annotated

    def _build_summary(
        self, image_np: np.ndarray, detections, query: str
    ) -> str:
        """Build structured text summary of detection results."""
        h, w = image_np.shape[:2]
        total_area = h * w
        n = len(detections)
        lines = [
            f'Structural validation for "{query}": {n} instance(s) detected.'
        ]

        for i in range(n):
            box = detections.xyxy[i]
            x1_n, y1_n = box[0] / w, box[1] / h
            x2_n, y2_n = box[2] / w, box[3] / h
            conf = detections.confidence[i]
            mask_area = detections.mask[i].sum() / total_area * 100
            lines.append(
                f"  #{i + 1}: bbox=[{x1_n:.2f},{y1_n:.2f},{x2_n:.2f},{y2_n:.2f}], "
                f"area={mask_area:.1f}%, confidence={conf:.2f}"
            )

        return "\n".join(lines)


class ToolExecutor:
    """
    Executes tool calls parsed from model output during multi-round inference.
    Manages the conversation state including images and tool results.
    """

    def __init__(self, mmad_root: str, sv_config: Optional[Dict] = None):
        self.pz = PerceptiveZoomer()
        self.cr = ComparativeRetriever(mmad_root)
        self.sv = StructuralValidator(sv_config) if sv_config else None

    def parse_tool_call(self, text: str) -> Optional[Dict]:
        """
        Parse a tool call from model output.
        Expected format:
        <tool_call>
        {"name": "crop_image_normalized", "arguments": {"bbox_2d": [...], "target_image": 1}}
        </tool_call>
        """
        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        match = re.search(pattern, text, re.DOTALL)
        if match is None:
            return None
        try:
            call = json.loads(match.group(1))
            return call
        except json.JSONDecodeError:
            return None

    def execute(
        self,
        tool_call: Dict,
        images: List[Image.Image],
        dataset_name: str,
        category: str,
        current_image_path: Optional[str] = None,
    ) -> Tuple[Optional[Image.Image], str]:
        """
        Execute a parsed tool call and return the result image and description text.

        Returns:
            (result_image, result_text) - the image produced and a text description.
        """
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        if name == "crop_image_normalized":
            bbox = args.get("bbox_2d", [0.25, 0.25, 0.75, 0.75])
            target = args.get("target_image", 1)
            cropped = self.pz(images, bbox, target)
            return cropped, "Here is the cropped image:"

        elif name == "query_image":
            ref_image = self.cr(dataset_name, category, current_image_path)
            if ref_image is not None:
                return ref_image, "Here is the normal reference image:"
            else:
                return None, "No normal reference image available for this category."

        elif name == "segment_and_count":
            if self.sv is None:
                return None, "Structural validation tool is not available."
            query = args.get("query", "")
            target = args.get("target_image", 1)
            if not query:
                return None, "No query provided for structural validation."
            result_image, summary = self.sv(images, query, target)
            return (
                result_image,
                f"Here is the structural validation result:\n{summary}",
            )

        else:
            return None, f"Unknown tool: {name}"

    def parse_final_answer(self, text: str) -> Optional[Dict]:
        """
        Parse the final answer from model output.
        Expected format:
        <answer> {"anomaly_present": true/false, "top_anomaly": "...", "visual_descriptions": [...]} </answer>
        """
        pattern = r"<answer>\s*(\{.*?\})\s*</answer>"
        match = re.search(pattern, text, re.DOTALL)
        if match is None:
            return None
        try:
            answer = json.loads(match.group(1))
            return answer
        except json.JSONDecodeError:
            return None

    def extract_bbox_from_tool_call(self, text: str) -> Optional[List[float]]:
        """Extract the bbox from a crop_image_normalized tool call."""
        tool_call = self.parse_tool_call(text)
        if tool_call is None:
            return None
        if tool_call.get("name") != "crop_image_normalized":
            return None
        return tool_call.get("arguments", {}).get("bbox_2d")
