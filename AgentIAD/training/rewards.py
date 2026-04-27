"""
Reward functions for AgentIAD Agentic Reinforcement Learning.

Two-level reward design (Section 3.3, Table 6):
  R = alpha * R_perc + beta * R_beh

Perception Reward (R_perc = R_acc + R_iou + R_type):
  - R_acc: format validity + classification accuracy (Eq. 7)
  - R_iou: spatial alignment of predicted crop vs GT defect (Eq. 8)
  - R_type: defect category correctness (Eq. 9)

Behavior Reward (R_beh, Eq. 10):
  - Stepwise correctness, CR-diversity, tool-call efficiency
"""
import json
import re
from typing import Dict, List, Optional, Tuple

from data.data_utils import compute_iou


class RewardComputer:
    """Computes the full reward for a single rollout trajectory."""

    def __init__(
        self,
        # Perception reward params
        alpha: float = 1.0,
        iou_threshold: float = 0.5,
        iou_reward_above: float = 1.0,
        lambda_type: float = 0.1,
        type_reward_bonus: float = 0.1,
        # Behavior reward params
        beta: float = 1.0,
        lambda_1: float = 1.0,
        lambda_2: float = 0.5,
        lambda_3: float = 0.05,
        expected_tool_usage: float = 1.0,
    ):
        self.alpha = alpha
        self.iou_threshold = iou_threshold
        self.iou_reward_above = iou_reward_above
        self.lambda_type = lambda_type
        self.type_reward_bonus = type_reward_bonus
        self.beta = beta
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.expected_tool_usage = expected_tool_usage

    def parse_answer(self, text: str) -> Optional[Dict]:
        """Parse <answer> JSON from model output."""
        pattern = r"<answer>\s*(\{.*?\})\s*</answer>"
        match = re.search(pattern, text, re.DOTALL)
        if match is None:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    def parse_tool_calls(self, text: str) -> List[Dict]:
        """Parse all <tool_call> blocks from model output."""
        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        calls = []
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                calls.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                continue
        return calls

    def compute_accuracy_reward(
        self, answer: Optional[Dict], gt_anomaly_present: bool
    ) -> float:
        """
        R_acc (Eq. 7): I[format valid] * I[prediction correct]
        Score of 1 only when format is valid AND classification matches GT.
        """
        if answer is None:
            return 0.0
        # Check format validity
        if "anomaly_present" not in answer:
            return 0.0
        pred = answer["anomaly_present"]
        if not isinstance(pred, bool):
            # Try to interpret string
            if isinstance(pred, str):
                pred = pred.lower() in ("true", "1", "yes")
            else:
                return 0.0
        # Check correctness
        if pred == gt_anomaly_present:
            return 1.0
        return 0.0

    def compute_iou_reward(
        self,
        tool_calls: List[Dict],
        gt_bbox: Optional[List[float]],
    ) -> float:
        """
        R_iou (Eq. 8): IoU-based spatial alignment reward.
        1.0 if IoU > threshold, else graded IoU value.
        Only applies when GT bbox is available (anomalous samples with masks).
        """
        if gt_bbox is None:
            return 0.0
        # Find the crop_image_normalized call
        pred_bbox = None
        for call in tool_calls:
            if call.get("name") == "crop_image_normalized":
                pred_bbox = call.get("arguments", {}).get("bbox_2d")
                break
        if pred_bbox is None:
            return 0.0
        iou = compute_iou(pred_bbox, gt_bbox)
        if iou > self.iou_threshold:
            return self.iou_reward_above
        return iou

    def compute_type_reward(
        self,
        answer: Optional[Dict],
        gt_anomaly_present: bool,
        gt_anomaly_type: str,
    ) -> float:
        """
        R_type (Eq. 9): Defect category correctness.
        lambda_type * I[gt=1] * I[pred_type == gt_type]
        Only provides reward when anomaly is actually present.
        """
        if not gt_anomaly_present:
            return 0.0
        if answer is None:
            return 0.0
        pred_type = answer.get("top_anomaly", "").lower().strip()
        gt_type = gt_anomaly_type.lower().strip()
        if pred_type == gt_type:
            return self.type_reward_bonus
        return 0.0

    def compute_perception_reward(
        self,
        text: str,
        gt: Dict,
    ) -> Tuple[float, Dict]:
        """
        R_perc = R_acc + R_iou + R_type (Eq. 6)
        """
        answer = self.parse_answer(text)
        tool_calls = self.parse_tool_calls(text)

        r_acc = self.compute_accuracy_reward(answer, gt["anomaly_present"])
        r_iou = self.compute_iou_reward(tool_calls, gt.get("bbox"))
        r_type = self.compute_type_reward(
            answer, gt["anomaly_present"], gt.get("anomaly_type", "none")
        )
        r_perc = r_acc + r_iou + r_type
        details = {"r_acc": r_acc, "r_iou": r_iou, "r_type": r_type, "r_perc": r_perc}
        return r_perc, details

    def compute_behavior_reward(
        self,
        rollout_texts: List[str],
        gt: Dict,
        group_query_rate: float,
    ) -> Tuple[float, Dict]:
        """
        R_beh (Eq. 10):
        (1/K) * sum_t [ lambda_1 * I(y_t == y_gt) + lambda_2 * q_t - lambda_3 * max(0, n_t - n*) ]

        Args:
            rollout_texts: List of model output texts at each reasoning step.
            gt: Ground truth dict.
            group_query_rate: Normalized frequency of CR tool usage in the rollout group.
        """
        K = len(rollout_texts)
        if K == 0:
            return 0.0, {"r_beh": 0.0}

        total = 0.0
        for text in rollout_texts:
            # Stepwise correctness
            answer = self.parse_answer(text)
            step_correct = 0.0
            if answer is not None:
                pred = answer.get("anomaly_present", None)
                if isinstance(pred, bool) and pred == gt["anomaly_present"]:
                    step_correct = 1.0

            # Tool call count at this step
            tool_calls = self.parse_tool_calls(text)
            n_t = len(tool_calls)

            # CR diversity: lambda_2 * (query_rate - 1)
            # Encourages using CR when under-used
            cr_diversity = self.lambda_2 * (group_query_rate - 1)

            # Efficiency: penalize excess tool calls
            efficiency_penalty = self.lambda_3 * max(0, n_t - self.expected_tool_usage)

            step_reward = (
                self.lambda_1 * step_correct + cr_diversity - efficiency_penalty
            )
            total += step_reward

        r_beh = total / K
        return r_beh, {"r_beh": r_beh}

    def compute_total_reward(
        self,
        full_text: str,
        rollout_texts: List[str],
        gt: Dict,
        group_query_rate: float,
    ) -> Tuple[float, Dict]:
        """
        R = alpha * R_perc + beta * R_beh (Eq. 5)
        """
        r_perc, perc_details = self.compute_perception_reward(full_text, gt)
        r_beh, beh_details = self.compute_behavior_reward(
            rollout_texts, gt, group_query_rate
        )
        total = self.alpha * r_perc + self.beta * r_beh
        details = {**perc_details, **beh_details, "total_reward": total}
        return total, details


def compute_group_query_rate(rollout_texts_group: List[List[str]]) -> float:
    """
    Compute the normalized frequency of CR (query_image) tool usage
    across a group of rollouts for one prompt.
    """
    total_calls = 0
    query_calls = 0
    for rollout in rollout_texts_group:
        for text in rollout:
            pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
            for match in re.finditer(pattern, text, re.DOTALL):
                total_calls += 1
                try:
                    call = json.loads(match.group(0).replace("<tool_call>", "").replace("</tool_call>", "").strip())
                    if call.get("name") == "query_image":
                        query_calls += 1
                except json.JSONDecodeError:
                    continue
    if total_calls == 0:
        return 0.0
    return query_calls / total_calls
