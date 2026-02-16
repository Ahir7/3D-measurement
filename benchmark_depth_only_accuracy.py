#!/usr/bin/env python3
"""Benchmark depth-only measurement accuracy from a manifest of scenes.

Manifest JSON format:
[
  {
    "name": "box_scene_1",
    "images_glob": "examples/box1/*.jpg",
    "ground_truth_cm": {"width": 21.0, "height": 15.0, "depth": 7.8},
    "config": "configs/rtx2060_config.py"
  }
]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Tuple

from src.utils.capture_quality import analyze_capture_quality


def _quality_gate_decision(
    quality_score: float,
    threshold: Optional[float],
    policy: str,
) -> str:
    """Return one of: proceed, skip, fail."""
    if threshold is None:
        return "proceed"
    if quality_score >= threshold:
        return "proceed"

    if policy == "skip":
        return "skip"
    if policy == "fail":
        return "fail"
    return "proceed"


def _resolve_images(entry: Dict) -> List[str]:
    """Resolve image inputs from either images_glob or explicit images list."""
    if "images" in entry:
        images = [str(Path(path)) for path in entry["images"]]
    else:
        pattern = entry["images_glob"]
        images = sorted(str(path) for path in Path(".").glob(pattern))

    if not images:
        raise RuntimeError(
            f"No images found for scene '{entry.get('name', '<unknown>')}'. "
            "Provide 'images_glob' or explicit 'images'."
        )
    return images


def _validate_entry(entry: Dict) -> None:
    required_fields = ["name", "ground_truth_cm"]
    for field_name in required_fields:
        if field_name not in entry:
            raise RuntimeError(f"Manifest entry missing required field: {field_name}")

    if "images_glob" not in entry and "images" not in entry:
        raise RuntimeError("Manifest entry must contain either 'images_glob' or 'images'")

    ground_truth = entry["ground_truth_cm"]
    for dimension in ("width", "height", "depth"):
        if dimension not in ground_truth:
            raise RuntimeError(f"Manifest entry '{entry['name']}' missing ground_truth_cm.{dimension}")
        if float(ground_truth[dimension]) <= 0:
            raise RuntimeError(f"Manifest entry '{entry['name']}' has non-positive ground_truth_cm.{dimension}")


def _run_scene(
    entry: Dict,
    num_runs: int,
    output_root: Path,
    quality_threshold: Optional[float],
    quality_policy: str,
) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    _validate_entry(entry)

    name = entry["name"]
    images = _resolve_images(entry)
    image_paths = [Path(path) for path in images]
    ground_truth = entry["ground_truth_cm"]
    config = entry.get("config")

    scene_output = output_root / name
    scene_output.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "main.py",
        "measure",
        *images,
        "--num-runs",
        str(num_runs),
        "--output",
        str(scene_output),
    ]
    if config:
        command.extend(["--config", config])

    print(f"\n[SCENE] {name}")
    print(f"[INFO] Images: {len(images)}")

    capture_quality = analyze_capture_quality(image_paths)
    quality_summary = capture_quality["summary"]
    print(
        f"[INFO] Capture quality: score={quality_summary['quality_score']:.3f} "
        f"({quality_summary['quality_level']}), overlap={quality_summary['overlap_median']:.3f}"
    )

    decision = _quality_gate_decision(
        quality_score=float(quality_summary["quality_score"]),
        threshold=quality_threshold,
        policy=quality_policy,
    )
    if decision == "skip":
        print(
            f"[WARN] Skipping scene due to quality gate: score={quality_summary['quality_score']:.3f} "
            f"< threshold={quality_threshold:.3f} (policy=skip)"
        )
        skipped = {
            "name": name,
            "reason": "quality_gate",
            "quality_score": float(quality_summary["quality_score"]),
            "quality_threshold": float(quality_threshold),
            "quality_policy": quality_policy,
            "capture_quality": quality_summary,
        }
        return None, None, skipped

    if decision == "fail":
        raise RuntimeError(
            f"Quality gate failed for scene '{name}': "
            f"score={quality_summary['quality_score']:.3f} < threshold={quality_threshold:.3f}"
        )

    if quality_summary["quality_level"] == "poor":
        print("[WARN] Capture quality is poor; measurement error may be high")

    print(f"[INFO] Running: {' '.join(command)}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Measurement command failed for scene '{name}' (exit={completed.returncode})"
        )

    results_path = scene_output / "results.json"
    if not results_path.exists():
        raise RuntimeError(f"Missing results for scene '{name}': {results_path}")

    with open(results_path, "r") as handle:
        result = json.load(handle)

    measured = result["measurements"]
    errors = {}
    mape_values = []

    for dimension in ("width", "height", "depth"):
        gt = float(ground_truth[dimension])
        pred = float(measured[dimension])
        abs_error = abs(pred - gt)
        ape = (abs_error / gt * 100.0) if gt > 0 else 0.0
        errors[f"{dimension}_abs_error_cm"] = abs_error
        errors[f"{dimension}_ape_percent"] = ape
        mape_values.append(ape)

    summary = {
        "name": name,
        "ground_truth_cm": ground_truth,
        "num_images": len(images),
        "capture_quality": quality_summary,
        "measured_cm": {
            "width": float(measured["width"]),
            "height": float(measured["height"]),
            "depth": float(measured["depth"]),
        },
        "scene_mape_percent": float(mean(mape_values)),
        "confidence": float(result.get("confidence", 0.0)),
        "errors": errors,
        "results_path": str(results_path),
    }

    if result.get("error_bounds"):
        summary["error_bounds"] = result["error_bounds"]

    print(
        f"[RESULT] MAPE={summary['scene_mape_percent']:.2f}% | "
        f"W={summary['measured_cm']['width']:.2f}, "
        f"H={summary['measured_cm']['height']:.2f}, "
        f"D={summary['measured_cm']['depth']:.2f}"
    )
    return summary, result, None


def main():
    parser = argparse.ArgumentParser(description="Benchmark depth-only accuracy across multiple scenes")
    parser.add_argument("--manifest", required=True, help="Path to benchmark manifest JSON")
    parser.add_argument("--num-runs", type=int, default=3, help="Measurement repetitions per scene")
    parser.add_argument("--output", default="output/baseline", help="Output directory for benchmark reports")
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.45,
        help="Capture quality score threshold in [0,1]. Use a negative value to disable gating.",
    )
    parser.add_argument(
        "--quality-policy",
        choices=["warn", "skip", "fail"],
        default="fail",
        help="Behavior when quality score is below threshold.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining scenes if one scene fails",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)

    if not isinstance(manifest, list) or not manifest:
        raise RuntimeError("Manifest must be a non-empty list")

    quality_threshold = args.quality_threshold
    if quality_threshold < 0:
        quality_threshold = None
    elif quality_threshold > 1.0:
        raise RuntimeError("--quality-threshold must be in [0, 1], or negative to disable")

    summaries = []
    raw_results = []
    failures = []
    skipped = []
    for entry in manifest:
        try:
            summary, raw_result, skip_record = _run_scene(
                entry,
                args.num_runs,
                output_root,
                quality_threshold=quality_threshold,
                quality_policy=args.quality_policy,
            )
            if skip_record:
                skipped.append(skip_record)
                continue

            summaries.append(summary)
            raw_results.append({"name": summary["name"], "result": raw_result})
        except Exception as error:
            failure = {"name": entry.get("name", "<unknown>"), "error": str(error)}
            failures.append(failure)
            print(f"[ERROR] {failure['name']}: {failure['error']}")
            if not args.continue_on_error:
                raise

    if not summaries:
        raise RuntimeError("No successful scenes were processed")

    mapes = [item["scene_mape_percent"] for item in summaries]
    confidences = [item["confidence"] for item in summaries]

    aggregate = {
        "num_scenes": len(summaries),
        "num_runs_per_scene": args.num_runs,
        "quality_threshold": quality_threshold,
        "quality_policy": args.quality_policy,
        "mean_mape_percent": float(mean(mapes)),
        "median_mape_percent": float(median(mapes)),
        "mean_confidence": float(mean(confidences)),
        "scenes": summaries,
        "skipped": skipped,
        "failures": failures,
    }

    report_path = output_root / "accuracy_report.json"
    with open(report_path, "w") as handle:
        json.dump(aggregate, handle, indent=2)

    raw_report_path = output_root / "raw_results.json"
    with open(raw_report_path, "w") as handle:
        json.dump(raw_results, handle, indent=2)

    print("\n" + "=" * 70)
    print("DEPTH-ONLY BASELINE SUMMARY")
    print("=" * 70)
    print(f"Scenes: {aggregate['num_scenes']}")
    print(f"Runs per scene: {aggregate['num_runs_per_scene']}")
    print(f"Mean MAPE: {aggregate['mean_mape_percent']:.2f}%")
    print(f"Median MAPE: {aggregate['median_mape_percent']:.2f}%")
    print(f"Mean Confidence: {aggregate['mean_confidence']:.3f}")
    if skipped:
        print(f"Skipped: {len(skipped)}")
    if failures:
        print(f"Failures: {len(failures)}")
    print(f"Report: {report_path}")
    print(f"Raw results: {raw_report_path}")


if __name__ == "__main__":
    main()
