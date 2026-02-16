"""Capture quality analysis utilities for image-only measurement workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


@dataclass
class ImageQualityMetrics:
    """Per-image quality metrics."""

    path: str
    blur_score: float
    exposure_mean: float
    underexposed_ratio: float
    overexposed_ratio: float


@dataclass
class CaptureQualitySummary:
    """Aggregate quality summary for an image set."""

    num_images: int
    blur_mean: float
    blur_median: float
    exposure_mean: float
    underexposed_ratio_mean: float
    overexposed_ratio_mean: float
    overlap_mean: float
    overlap_median: float
    overlap_std: float
    quality_score: float
    quality_level: str


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def _blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _exposure_stats(gray: np.ndarray) -> Dict[str, float]:
    normalized = gray.astype(np.float32) / 255.0
    underexposed = float((normalized < 0.05).mean())
    overexposed = float((normalized > 0.95).mean())
    exposure_mean = float(normalized.mean())
    return {
        "mean": exposure_mean,
        "underexposed_ratio": underexposed,
        "overexposed_ratio": overexposed,
    }


def _pair_overlap_score(img_a: np.ndarray, img_b: np.ndarray, orb: cv2.ORB) -> float:
    """Estimate overlap score in [0, 1] using ORB feature match consistency."""
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    keypoints_a, descriptors_a = orb.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(gray_b, None)

    if descriptors_a is None or descriptors_b is None:
        return 0.0
    if len(keypoints_a) < 20 or len(keypoints_b) < 20:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.78 * n.distance:
            good_matches.append(m)

    denom = max(min(len(keypoints_a), len(keypoints_b)), 1)
    return float(np.clip(len(good_matches) / denom, 0.0, 1.0))


def _quality_level(score: float) -> str:
    if score >= 0.80:
        return "excellent"
    if score >= 0.65:
        return "good"
    if score >= 0.45:
        return "fair"
    return "poor"


def analyze_capture_quality(image_paths: List[Path]) -> Dict:
    """Analyze quality of an ordered image set and return report dict."""
    if len(image_paths) < 3:
        raise ValueError("At least 3 images required for capture quality analysis")

    metrics: List[ImageQualityMetrics] = []
    loaded_images: List[np.ndarray] = []

    for image_path in image_paths:
        image = _load_image(image_path)
        loaded_images.append(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        exposure = _exposure_stats(gray)

        metrics.append(
            ImageQualityMetrics(
                path=str(image_path),
                blur_score=_blur_score(gray),
                exposure_mean=exposure["mean"],
                underexposed_ratio=exposure["underexposed_ratio"],
                overexposed_ratio=exposure["overexposed_ratio"],
            )
        )

    overlap_scores = []
    orb = cv2.ORB_create(nfeatures=1200)
    for index in range(len(loaded_images) - 1):
        overlap_scores.append(_pair_overlap_score(loaded_images[index], loaded_images[index + 1], orb))

    blur_values = np.array([item.blur_score for item in metrics], dtype=np.float64)
    exposure_values = np.array([item.exposure_mean for item in metrics], dtype=np.float64)
    under_values = np.array([item.underexposed_ratio for item in metrics], dtype=np.float64)
    over_values = np.array([item.overexposed_ratio for item in metrics], dtype=np.float64)
    overlap_values = np.array(overlap_scores, dtype=np.float64) if overlap_scores else np.array([0.0], dtype=np.float64)

    blur_norm = np.clip(np.log1p(np.median(blur_values)) / np.log1p(350.0), 0.0, 1.0)
    exposure_balance = np.clip(1.0 - abs(float(np.mean(exposure_values)) - 0.5) / 0.5, 0.0, 1.0)
    clipping_penalty = np.clip(1.0 - (float(np.mean(under_values)) + float(np.mean(over_values))), 0.0, 1.0)
    overlap_quality = np.clip(float(np.median(overlap_values)) / 0.20, 0.0, 1.0)

    quality_score = float(
        0.35 * blur_norm
        + 0.20 * exposure_balance
        + 0.20 * clipping_penalty
        + 0.25 * overlap_quality
    )

    summary = CaptureQualitySummary(
        num_images=len(metrics),
        blur_mean=float(np.mean(blur_values)),
        blur_median=float(np.median(blur_values)),
        exposure_mean=float(np.mean(exposure_values)),
        underexposed_ratio_mean=float(np.mean(under_values)),
        overexposed_ratio_mean=float(np.mean(over_values)),
        overlap_mean=float(np.mean(overlap_values)),
        overlap_median=float(np.median(overlap_values)),
        overlap_std=float(np.std(overlap_values)),
        quality_score=quality_score,
        quality_level=_quality_level(quality_score),
    )

    return {
        "summary": summary.__dict__,
        "images": [metric.__dict__ for metric in metrics],
        "overlap_scores": [float(value) for value in overlap_scores],
    }
