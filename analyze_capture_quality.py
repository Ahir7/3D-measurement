#!/usr/bin/env python3
"""Analyze capture quality for an image set and export a JSON report."""

import argparse
import json
from pathlib import Path

from src.utils.capture_quality import analyze_capture_quality


def main():
    parser = argparse.ArgumentParser(description="Analyze capture quality for depth-only measurement")
    parser.add_argument("images", nargs="+", help="Input image paths")
    parser.add_argument("--output", default="output/capture_quality.json", help="Output JSON report path")
    args = parser.parse_args()

    image_paths = [Path(image) for image in args.images]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing images: {missing}")

    report = analyze_capture_quality(image_paths)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2)

    summary = report["summary"]
    print("=" * 70)
    print("CAPTURE QUALITY SUMMARY")
    print("=" * 70)
    print(f"Images: {summary['num_images']}")
    print(f"Quality score: {summary['quality_score']:.3f} ({summary['quality_level']})")
    print(f"Blur median: {summary['blur_median']:.2f}")
    print(f"Exposure mean: {summary['exposure_mean']:.3f}")
    print(f"Overlap median: {summary['overlap_median']:.3f}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
