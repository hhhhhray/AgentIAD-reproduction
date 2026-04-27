"""
Trajectory Construction Pipeline for AgentIAD.

Uses GPT-4o to generate multi-step CoT reasoning traces, then assembles them
into structured perceptive (PZ-only) and comparative (PZ+CR) trajectories
for SFT training. See paper Section 3.2 and Supplementary Section 6.
"""
import argparse
import asyncio
import base64
import json
import os
import random
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from data.data_utils import (
    bbox_from_mask,
    crop_image_by_bbox,
    get_normal_reference,
    load_domain_knowledge,
    load_mask,
    split_dataset,
)
from data.mmad_dataset import scan_mmad_samples

# ============================================================
# GPT-4o Prompt Templates (from Supplementary Sections 6.5)
# ============================================================

# --- Normal sample ROI generation (Section 6.1) ---
NORMAL_ROI_PROMPT = """This is a normal {class_name} image without any defects. However, I need you to identify ONE region in this image that you would focus on when verifying it is normal. Choose a region where defects are most likely to occur or that typically requires careful inspection. Please output ONLY the normalized bounding box coordinates in the format:
[x_min, y_min, x_max, y_max]
All values must be between 0 and 1, representing proportions of the image dimensions:
- x_min: left edge (0 = left, 1 = right)
- y_min: top edge (0 = top, 1 = bottom)
- x_max: right edge
- y_max: bottom edge

Example: [0.2, 0.3, 0.6, 0.7]
Output ONLY the bbox coordinates, nothing else."""

# --- CoT-1: Global Reasoning (Section 6.5) ---
COT1_SYSTEM_PROMPT = """You are a vision expert specialized in industrial anomaly detection. You will evaluate whether the given object image is normal or abnormal. You have access to both the original image and a region-of-interest (ROI) image that highlights potential anomaly areas. Explain why you need to examine this ROI region - what caught your attention in the original image that led you to focus on this area, but DO NOT mention the ROI image in your explanation.
ATTENTION: GT ANSWER IS PROVIDED IN THE QUESTION, YOU SHOULD FOLLOW IT."""

COT1_USER_ABNORMAL = """Ground Truth Information:
- Class: {class_name}
- Status: ABNORMAL (defective)
- Specific anomaly type: {anomaly_type}

IMPORTANT: Your analysis MUST align with the Ground Truth provided above. The object is confirmed to be ABNORMAL with the specific anomaly type {anomaly_type}. Please identify and describe the visual evidence that explain why you need to examine this ROI region.

ROI normalized bbox: {bbox_coords}"""

COT1_USER_NORMAL = """Ground Truth Information:
- Class: {class_name}
- Status: NORMAL (no defects)

IMPORTANT: Your analysis MUST align with the Ground Truth provided above. The object is confirmed to be NORMAL with no defects. Please identify and describe the visual evidence that explain why you need to examine this ROI region.

ROI normalized bbox: {bbox_coords}"""

# --- CoT-2: Local Reasoning after PZ crop (Section 6.5) ---
COT2_SYSTEM_PROMPT = """You are a vision expert specialized in industrial anomaly detection.
You will evaluate whether the given object image is normal or abnormal. You have access to both the original image and a region-of-interest (ROI) image that highlights potential anomaly areas. If abnormal, select the most fitting anomaly label from the candidate types provided by the user.

Output format:
<think> Explain your visual reasoning, considering both the original image and the ROI information. </think>
<answer> {"anomaly_present": true/false, "top_anomaly": "<label or 'none'>", "visual_descriptions": ["..."]} </answer>
Guidelines:
- In <think>: Provide detailed analysis of what you observe in both images.
- If normal → anomaly_present=false, top_anomaly="none", visual_descriptions=[].
- If abnormal → include concise visual phrases for visible cues.

ATTENTION: GT ANSWER IS PROVIDED IN THE QUESTION, YOU SHOULD FOLLOW IT."""

COT2_USER_ABNORMAL = """Ground Truth Information:
- Class: {class_name}
- Status: ABNORMAL (defective)
- Specific anomaly type: {anomaly_type}

IMPORTANT: Your analysis MUST align with the Ground Truth provided above. The object is confirmed to be ABNORMAL with the specific anomaly type {anomaly_type}. Please identify and describe the visual evidence that supports this classification.

ROI normalized bbox: {bbox_coords}"""

COT2_USER_NORMAL = """Ground Truth Information:
- Class: {class_name}
- Status: NORMAL (no defects)

IMPORTANT: Your analysis MUST align with the Ground Truth provided above. The object is confirmed to be NORMAL with no defects. Please confirm this by describing why the object appears normal and free from anomalies.

ROI normalized bbox: {bbox_coords}"""

# --- CoT-3: Comparative Reasoning after CR (Section 6.5) ---
COT3_SYSTEM_PROMPT = """You are an industrial anomaly analysis expert.
You will review images of manufactured products and explain the visual evidence that supports the provided ground truth. Focus strictly on verifiable cues visible in the images. Describe contrasts between the target image (with ROI) and the normal reference.
Do not output any final classification or prediction—only deliver the reasoning narrative."""

COT3_USER_ABNORMAL = """Class: {class_name}
You will receive three images in order: (1) the full target image, (2) the cropped ROI highlighting a potential anomaly, (3) a normal reference image from the same class.
Candidate anomaly types: {anomaly_types_str}
Ground truth: the sample is ABNORMAL. Anomaly type: {anomaly_type}
ROI normalized bbox: {bbox_coords}
Explain the concrete visual cues within the ROI that deviate from the normal reference and justify the provided anomaly type.
Describe only the reasoning process, using concise sentences or bullet points referencing observable evidence."""

COT3_USER_NORMAL = """Class: {class_name}
You will receive three images in order: (1) the full target image, (2) the cropped ROI highlighting a potential anomaly, (3) a normal reference image from the same class.
Candidate anomaly types: {anomaly_types_str}
Ground truth: the sample is NORMAL. Anomaly type: none
ROI normalized bbox: {bbox_coords}
Explain the concrete visual cues within the ROI that deviate from the normal reference and justify the provided anomaly type.
Describe only the reasoning process, using concise sentences or bullet points referencing observable evidence."""


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for API calls."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def make_image_content(image: Image.Image) -> Dict:
    """Create OpenAI-format image content block."""
    b64 = image_to_base64(image)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"},
    }


class TrajectoryBuilder:
    """Builds SFT trajectories using GPT-4o for CoT generation."""

    def __init__(
        self,
        mmad_root: str,
        domain_knowledge: Dict,
        openai_api_key: str,
        openai_base_url: str = "",
        gpt_model: str = "gpt-4o",
    ):
        from openai import AsyncOpenAI

        self.mmad_root = mmad_root
        self.domain_knowledge = domain_knowledge
        self.gpt_model = gpt_model
        client_kwargs = {"api_key": openai_api_key}
        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url
        self.client = AsyncOpenAI(**client_kwargs)

    async def _call_gpt(
        self,
        system_prompt: str,
        user_prompt: str,
        images: List[Image.Image],
    ) -> str:
        """Call GPT-4o with text and images."""
        content = [{"type": "text", "text": user_prompt}]
        for img in images:
            content.append(make_image_content(img))
        response = await self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content

    async def get_normal_roi(
        self, image: Image.Image, class_name: str
    ) -> List[float]:
        """For normal samples, use GPT-4o to predict a plausible ROI."""
        prompt = NORMAL_ROI_PROMPT.format(class_name=class_name)
        content = [
            {"type": "text", "text": prompt},
            make_image_content(image),
        ]
        response = await self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=100,
            temperature=0.3,
        )
        text = response.choices[0].message.content.strip()
        # Parse bbox from response
        try:
            bbox = json.loads(text)
            if isinstance(bbox, list) and len(bbox) == 4:
                return [float(v) for v in bbox]
        except (json.JSONDecodeError, ValueError):
            pass
        # Try regex extraction
        import re
        match = re.search(r"\[([0-9.,\s]+)\]", text)
        if match:
            vals = [float(v.strip()) for v in match.group(1).split(",")]
            if len(vals) == 4:
                return vals
        return [0.25, 0.25, 0.75, 0.75]

    async def generate_cot1(
        self,
        image: Image.Image,
        roi_image: Image.Image,
        sample: Dict,
        bbox_coords: List[float],
    ) -> str:
        """Generate CoT-1: Global reasoning about why to examine ROI."""
        bbox_str = str(bbox_coords)
        if sample["anomaly_present"]:
            user_prompt = COT1_USER_ABNORMAL.format(
                class_name=sample["category"],
                anomaly_type=sample["anomaly_type"],
                bbox_coords=bbox_str,
            )
        else:
            user_prompt = COT1_USER_NORMAL.format(
                class_name=sample["category"],
                bbox_coords=bbox_str,
            )
        return await self._call_gpt(
            COT1_SYSTEM_PROMPT, user_prompt, [image, roi_image]
        )

    async def generate_cot2(
        self,
        image: Image.Image,
        roi_image: Image.Image,
        sample: Dict,
        bbox_coords: List[float],
    ) -> str:
        """Generate CoT-2: Local reasoning after PZ crop with <think><answer> format."""
        bbox_str = str(bbox_coords)
        if sample["anomaly_present"]:
            user_prompt = COT2_USER_ABNORMAL.format(
                class_name=sample["category"],
                anomaly_type=sample["anomaly_type"],
                bbox_coords=bbox_str,
            )
        else:
            user_prompt = COT2_USER_NORMAL.format(
                class_name=sample["category"],
                bbox_coords=bbox_str,
            )
        return await self._call_gpt(
            COT2_SYSTEM_PROMPT, user_prompt, [image, roi_image]
        )

    async def generate_cot3(
        self,
        image: Image.Image,
        roi_image: Image.Image,
        ref_image: Image.Image,
        sample: Dict,
        bbox_coords: List[float],
    ) -> str:
        """Generate CoT-3: Comparative reasoning after CR with reference image."""
        anomaly_types_str = ", ".join(sample["anomaly_types_list"])
        bbox_str = str(bbox_coords)
        if sample["anomaly_present"]:
            user_prompt = COT3_USER_ABNORMAL.format(
                class_name=sample["category"],
                anomaly_types_str=anomaly_types_str,
                anomaly_type=sample["anomaly_type"],
                bbox_coords=bbox_str,
            )
        else:
            user_prompt = COT3_USER_NORMAL.format(
                class_name=sample["category"],
                anomaly_types_str=anomaly_types_str,
                bbox_coords=bbox_str,
            )
        return await self._call_gpt(
            COT3_SYSTEM_PROMPT, user_prompt, [image, roi_image, ref_image]
        )

    def _build_perceptive_trajectory(
        self,
        sample: Dict,
        bbox: List[float],
        cot1: str,
        cot2: str,
        image_path: str,
        roi_image_path: str,
    ) -> Dict:
        """
        Assemble a Perceptive Trajectory (PZ-only).
        Structure: system -> user (image + question) -> assistant (cot1 + tool_call)
                   -> user (cropped image result) -> assistant (cot2 with <think><answer>)
        """
        anomaly_types_str = ", ".join(sample["anomaly_types_list"])
        user_question = (
            f'Evaluate the following image from the class "{sample["category"]}". '
            f"Candidate anomaly types: {anomaly_types_str}. "
            "Determine if the object is normal or abnormal. "
            "Follow the instruction and we can look closer by `crop_image_normalized`."
            "\nReason with the visual information step by step, "
            "and output the final answer in the required XML format."
        )
        bbox_str = json.dumps(bbox)
        tool_call_str = (
            f'<tool_call>\n{{"name": "crop_image_normalized", '
            f'"arguments": {{"bbox_2d": {bbox_str}, "target_image": 1}}}}\n</tool_call>'
        )

        messages = [
            {"role": "system", "content": self._get_system_prompt("pz_only")},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_question},
                ],
            },
            {
                "role": "assistant",
                "content": cot1 + "\nNow I will zoom in to look clearer.\n" + tool_call_str,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the cropped image:"},
                    {"type": "image", "image": roi_image_path},
                ],
            },
            {
                "role": "assistant",
                "content": cot2,
            },
        ]

        # mask_ranges: indices of the last two assistant turns for loss masking
        # These will be computed at tokenization time in the dataset class
        return {
            "type": "perceptive",
            "messages": messages,
            "image_paths": [image_path, roi_image_path],
            "sample_info": {
                "image_path": sample["image_path"],
                "dataset_name": sample["dataset_name"],
                "category": sample["category"],
                "anomaly_present": sample["anomaly_present"],
                "anomaly_type": sample["anomaly_type"],
                "bbox": bbox,
            },
        }

    def _build_comparative_trajectory(
        self,
        sample: Dict,
        bbox: List[float],
        cot1: str,
        cot2_intermediate: str,
        cot3: str,
        image_path: str,
        roi_image_path: str,
        ref_image_path: str,
    ) -> Dict:
        """
        Assemble a Comparative Trajectory (PZ+CR).
        After PZ, the agent recognizes uncertainty and calls CR for reference comparison.
        """
        anomaly_types_str = ", ".join(sample["anomaly_types_list"])
        user_question = (
            f'Evaluate the following image from the class "{sample["category"]}". '
            f"Candidate anomaly types: {anomaly_types_str}. "
            "Determine if the object is normal or abnormal. "
            "Follow the instruction and we can look closer by `crop_image_normalized`. "
            "If, after inspecting the crop, the evidence is still insufficient, "
            "you may also call `query_image` to retrieve a normal reference image."
            "\nReason with the visual information step by step, "
            "and output the final answer in the required XML format."
        )
        bbox_str = json.dumps(bbox)
        pz_call = (
            f'<tool_call>\n{{"name": "crop_image_normalized", '
            f'"arguments": {{"bbox_2d": {bbox_str}, "target_image": 1}}}}\n</tool_call>'
        )
        cr_call = '<tool_call>\n{"name": "query_image", "arguments": {}}\n</tool_call>'

        # Build the answer part with <think><answer> from cot3
        if sample["anomaly_present"]:
            answer_json = json.dumps({
                "anomaly_present": True,
                "top_anomaly": sample["anomaly_type"],
                "visual_descriptions": [],  # Will be filled by GPT
            })
        else:
            answer_json = json.dumps({
                "anomaly_present": False,
                "top_anomaly": "none",
                "visual_descriptions": [],
            })

        messages = [
            {"role": "system", "content": self._get_system_prompt("pz_cr")},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_question},
                ],
            },
            {
                "role": "assistant",
                "content": cot1 + "\nNow I will zoom in to look clearer.\n" + pz_call,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the cropped image:"},
                    {"type": "image", "image": roi_image_path},
                ],
            },
            {
                "role": "assistant",
                "content": (
                    cot2_intermediate
                    + "\nTo make a confident decision, I would like to compare it "
                    "with a normal reference image of the same class.\n"
                    + cr_call
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the normal reference image:"},
                    {"type": "image", "image": ref_image_path},
                ],
            },
            {
                "role": "assistant",
                "content": f"<think>\n{cot3}\n</think>\n<answer>\n{answer_json}\n</answer>",
            },
        ]

        return {
            "type": "comparative",
            "messages": messages,
            "image_paths": [image_path, roi_image_path, ref_image_path],
            "sample_info": {
                "image_path": sample["image_path"],
                "dataset_name": sample["dataset_name"],
                "category": sample["category"],
                "anomaly_present": sample["anomaly_present"],
                "anomaly_type": sample["anomaly_type"],
                "bbox": bbox,
            },
        }

    def _get_system_prompt(self, mode: str) -> str:
        """Build system prompt with tool definitions."""
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
        if mode == "pz_only":
            return base + pz_tool + tool_call_fmt
        else:
            return base + pz_tool + cr_tool + tool_call_fmt

    async def build_single_trajectory(
        self,
        sample: Dict,
        traj_type: str,
        output_dir: str,
        idx: int,
    ) -> Optional[str]:
        """
        Build a single trajectory for one sample.
        traj_type: "perceptive" (PZ-only) or "comparative" (PZ+CR)
        """
        try:
            image = Image.open(sample["image_path"]).convert("RGB")

            # Step 1: Get ROI bbox
            if sample["anomaly_present"] and sample.get("mask_path"):
                mask = load_mask(sample["mask_path"])
                bbox = bbox_from_mask(mask)
            else:
                bbox = await self.get_normal_roi(image, sample["category"])

            # Step 2: Crop ROI
            roi_image = crop_image_by_bbox(image, bbox, normalized=True)

            # Save ROI image
            roi_dir = os.path.join(output_dir, "roi_images")
            os.makedirs(roi_dir, exist_ok=True)
            roi_path = os.path.join(roi_dir, f"roi_{idx:06d}.png")
            roi_image.save(roi_path)

            # Step 3: Generate CoT-1 (global reasoning)
            cot1 = await self.generate_cot1(image, roi_image, sample, bbox)

            # Step 4: Generate CoT-2 (local reasoning after PZ)
            cot2 = await self.generate_cot2(image, roi_image, sample, bbox)

            if traj_type == "perceptive":
                traj = self._build_perceptive_trajectory(
                    sample, bbox, cot1, cot2,
                    sample["image_path"], roi_path,
                )
            else:
                # Comparative: also need CoT-3 with reference image
                ref_path = get_normal_reference(
                    self.mmad_root,
                    sample["dataset_name"],
                    sample["category"],
                    sample["image_path"],
                )
                if ref_path is None:
                    # Fallback to perceptive if no reference available
                    traj = self._build_perceptive_trajectory(
                        sample, bbox, cot1, cot2,
                        sample["image_path"], roi_path,
                    )
                else:
                    ref_image = Image.open(ref_path).convert("RGB")
                    # For comparative, CoT-2 should be intermediate (no final answer)
                    cot2_intermediate = cot2.split("<answer>")[0].replace(
                        "<think>", ""
                    ).replace("</think>", "").strip()
                    cot3 = await self.generate_cot3(
                        image, roi_image, ref_image, sample, bbox
                    )
                    traj = self._build_comparative_trajectory(
                        sample, bbox, cot1, cot2_intermediate, cot3,
                        sample["image_path"], roi_path, ref_path,
                    )

            # Save trajectory
            traj_path = os.path.join(output_dir, f"traj_{idx:06d}.json")
            with open(traj_path, "w", encoding="utf-8") as f:
                json.dump(traj, f, ensure_ascii=False, indent=2)
            return traj_path

        except Exception as e:
            print(f"Error building trajectory for sample {idx}: {e}")
            return None


async def build_all_trajectories(args):
    """Main function to build all SFT trajectories."""
    # Load domain knowledge and scan dataset
    domain_knowledge = load_domain_knowledge(args.domain_knowledge_path)
    dataset_names = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    all_samples = scan_mmad_samples(args.mmad_root, domain_knowledge, dataset_names)
    print(f"Datasets: {dataset_names or 'all'}")
    print(f"Total samples found: {len(all_samples)}")

    # Split dataset
    sft_samples, grpo_samples, eval_samples = split_dataset(
        all_samples, args.sft_num, args.grpo_num, args.seed
    )
    print(f"SFT: {len(sft_samples)}, GRPO: {len(grpo_samples)}, Eval: {len(eval_samples)}")

    # Save splits for later use
    os.makedirs(args.output_dir, exist_ok=True)
    for name, samples in [
        ("sft_samples", sft_samples),
        ("grpo_samples", grpo_samples),
        ("eval_samples", eval_samples),
    ]:
        path = os.path.join(args.output_dir, f"{name}.json")
        # Save sample metadata (without images)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

    # Build trajectories for SFT samples
    builder = TrajectoryBuilder(
        mmad_root=args.mmad_root,
        domain_knowledge=domain_knowledge,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        gpt_model=args.gpt_model,
    )

    traj_dir = os.path.join(args.output_dir, "sft_trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    # Determine which samples get comparative trajectories (112 out of 1600)
    rng = random.Random(args.seed)
    comparative_indices = set(rng.sample(range(len(sft_samples)), min(args.pz_cr_num, len(sft_samples))))

    # Process with concurrency control
    semaphore = asyncio.Semaphore(args.max_concurrent)
    results = []

    async def process_one(i, sample):
        async with semaphore:
            traj_type = "comparative" if i in comparative_indices else "perceptive"
            return await builder.build_single_trajectory(
                sample, traj_type, traj_dir, i
            )

    tasks = [process_one(i, s) for i, s in enumerate(sft_samples)]
    for coro in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Building trajectories"
    ):
        result = await coro
        results.append(result)

    successful = sum(1 for r in results if r is not None)
    print(f"Successfully built {successful}/{len(sft_samples)} trajectories")


def main():
    parser = argparse.ArgumentParser(description="Build AgentIAD SFT trajectories")
    parser.add_argument("--mmad_root", type=str, default="./data/MMAD")
    parser.add_argument("--domain_knowledge_path", type=str,
                        default="./data/MMAD/domain_knowledge.json")
    parser.add_argument("--output_dir", type=str, default="./trajectories")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated sub-dataset names, e.g. 'MVTec,VisA'. Default: all")
    parser.add_argument("--openai_api_key", type=str, required=True)
    parser.add_argument("--openai_base_url", type=str, default="")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o")
    parser.add_argument("--sft_num", type=int, default=1600)
    parser.add_argument("--grpo_num", type=int, default=366)
    parser.add_argument("--pz_cr_num", type=int, default=112,
                        help="Number of comparative (PZ+CR) trajectories among SFT samples")
    parser.add_argument("--max_concurrent", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    asyncio.run(build_all_trajectories(args))


if __name__ == "__main__":
    main()
