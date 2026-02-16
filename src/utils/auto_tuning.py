"""Report-driven auto-tuning helpers for depth-only accuracy settings."""

from __future__ import annotations

from statistics import mean
from typing import Dict, List, Tuple


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _extract_metrics(report: Dict) -> Dict[str, float]:
    scenes: List[Dict] = report.get("scenes", [])
    quality_scores: List[float] = []
    overlaps: List[float] = []
    confidences: List[float] = []

    for scene in scenes:
        capture_quality = scene.get("capture_quality", {})
        if "quality_score" in capture_quality:
            quality_scores.append(float(capture_quality["quality_score"]))
        if "overlap_median" in capture_quality:
            overlaps.append(float(capture_quality["overlap_median"]))
        if "confidence" in scene:
            confidences.append(float(scene["confidence"]))

    return {
        "num_scenes": float(len(scenes)),
        "mean_mape_percent": float(report.get("mean_mape_percent", 0.0)),
        "median_mape_percent": float(report.get("median_mape_percent", 0.0)),
        "mean_confidence": float(report.get("mean_confidence", 0.0)),
        "scene_confidence_mean": float(mean(confidences)) if confidences else 0.0,
        "capture_quality_mean": float(mean(quality_scores)) if quality_scores else 0.0,
        "capture_overlap_mean": float(mean(overlaps)) if overlaps else 0.0,
    }


def recommend_tuning(
    report: Dict,
    current: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float], List[str], Dict[str, float]]:
    """Return (recommended, deltas, rationale, metrics) from one benchmark report.

    Expected `current` keys:
      - capture_quality_threshold
      - quality_drop_fraction
      - adaptive_quality_drop_min
      - adaptive_quality_drop_max
      - depth_confidence_min
      - depth_confidence_weight_power
    """
    metrics = _extract_metrics(report)

    recommended = {
        "capture_quality_threshold": float(current["capture_quality_threshold"]),
        "quality_drop_fraction": float(current["quality_drop_fraction"]),
        "adaptive_quality_drop_min": float(current["adaptive_quality_drop_min"]),
        "adaptive_quality_drop_max": float(current["adaptive_quality_drop_max"]),
        "depth_confidence_min": float(current["depth_confidence_min"]),
        "depth_confidence_weight_power": float(current["depth_confidence_weight_power"]),
    }

    rationale: List[str] = []

    mean_mape = metrics["mean_mape_percent"]
    mean_conf = metrics["mean_confidence"]
    quality_mean = metrics["capture_quality_mean"]
    overlap_mean = metrics["capture_overlap_mean"]

    if mean_mape > 12.0:
        recommended["depth_confidence_min"] += 0.05
        recommended["depth_confidence_weight_power"] += 0.15
        recommended["quality_drop_fraction"] += 0.06
        recommended["capture_quality_threshold"] += 0.03
        recommended["adaptive_quality_drop_max"] += 0.05
        rationale.append("High MAPE detected (>12%): tighten confidence filtering and remove more low-quality views.")
    elif mean_mape > 9.0:
        recommended["depth_confidence_min"] += 0.03
        recommended["depth_confidence_weight_power"] += 0.10
        recommended["quality_drop_fraction"] += 0.04
        rationale.append("MAPE above target (>9%): apply moderate tightening of depth and quality filtering.")
    elif mean_mape < 7.0 and mean_conf > 0.70:
        recommended["depth_confidence_min"] -= 0.03
        recommended["quality_drop_fraction"] -= 0.03
        recommended["capture_quality_threshold"] -= 0.02
        rationale.append("Strong baseline (<7% MAPE, high confidence): relax filters slightly to preserve coverage.")

    if mean_conf < 0.45:
        recommended["depth_confidence_min"] -= 0.04
        recommended["depth_confidence_weight_power"] -= 0.10
        rationale.append("Low confidence detected (<0.45): relax depth confidence gate to recover usable support.")

    if quality_mean < 0.45:
        recommended["quality_drop_fraction"] += 0.05
        recommended["capture_quality_threshold"] = max(
            recommended["capture_quality_threshold"],
            quality_mean + 0.03,
        )
        rationale.append("Low capture quality mean: increase pruning pressure and slightly raise quality threshold.")

    if overlap_mean < 0.35:
        recommended["adaptive_quality_drop_max"] += 0.05
        recommended["adaptive_quality_drop_min"] += 0.02
        rationale.append("Weak overlap detected: strengthen adaptive drop range to reduce unstable frames.")

    if not rationale:
        rationale.append("Metrics are stable; retain current settings.")

    recommended["capture_quality_threshold"] = _clamp(recommended["capture_quality_threshold"], 0.25, 0.70)
    recommended["quality_drop_fraction"] = _clamp(recommended["quality_drop_fraction"], 0.05, 0.50)
    recommended["adaptive_quality_drop_min"] = _clamp(recommended["adaptive_quality_drop_min"], 0.05, 0.40)
    recommended["adaptive_quality_drop_max"] = _clamp(recommended["adaptive_quality_drop_max"], 0.15, 0.60)
    recommended["depth_confidence_min"] = _clamp(recommended["depth_confidence_min"], 0.15, 0.70)
    recommended["depth_confidence_weight_power"] = _clamp(recommended["depth_confidence_weight_power"], 0.8, 2.2)

    if recommended["adaptive_quality_drop_min"] > recommended["adaptive_quality_drop_max"]:
        midpoint = (recommended["adaptive_quality_drop_min"] + recommended["adaptive_quality_drop_max"]) * 0.5
        recommended["adaptive_quality_drop_min"] = _clamp(midpoint - 0.03, 0.05, 0.40)
        recommended["adaptive_quality_drop_max"] = _clamp(midpoint + 0.03, 0.15, 0.60)

    deltas = {
        key: float(recommended[key] - float(current[key]))
        for key in recommended
    }

    return recommended, deltas, rationale, metrics
