"""
Visual tools for AgentIAD: Perceptive Zoomer (PZ) and Comparative Retriever (CR).
"""
import json
import random
import re
from typing import Dict, List, Optional, Tuple

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


class ToolExecutor:
    """
    Executes tool calls parsed from model output during multi-round inference.
    Manages the conversation state including images and tool results.
    """

    def __init__(self, mmad_root: str):
        self.pz = PerceptiveZoomer()
        self.cr = ComparativeRetriever(mmad_root)

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
