#!/usr/bin/env python3
"""Compare two depth-only accuracy reports and print improvement/regression summary."""

import argparse
import json
from pathlib import Path


def _load_report(path: Path):
    with open(path, "r") as handle:
        return json.load(handle)


def _format_delta(new_value: float, old_value: float, unit: str = "%") -> str:
    delta = new_value - old_value
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}{unit}"


def main():
    parser = argparse.ArgumentParser(description="Compare depth-only accuracy reports")
    parser.add_argument("--old", required=True, help="Path to old/baseline report JSON")
    parser.add_argument("--new", required=True, help="Path to new/candidate report JSON")
    parser.add_argument("--max-mape-regression", type=float, default=0.5, help="Allowed MAPE regression percentage points")
    args = parser.parse_args()

    old_report = _load_report(Path(args.old))
    new_report = _load_report(Path(args.new))

    old_mean = float(old_report.get("mean_mape_percent", 0.0))
    new_mean = float(new_report.get("mean_mape_percent", 0.0))

    old_median = float(old_report.get("median_mape_percent", 0.0))
    new_median = float(new_report.get("median_mape_percent", 0.0))

    old_conf = float(old_report.get("mean_confidence", 0.0))
    new_conf = float(new_report.get("mean_confidence", 0.0))

    print("=" * 70)
    print("DEPTH-ONLY ACCURACY REPORT COMPARISON")
    print("=" * 70)
    print(f"Old report: {args.old}")
    print(f"New report: {args.new}")
    print()

    print(f"Mean MAPE:   {old_mean:.2f}% -> {new_mean:.2f}% ({_format_delta(new_mean, old_mean)})")
    print(f"Median MAPE: {old_median:.2f}% -> {new_median:.2f}% ({_format_delta(new_median, old_median)})")
    print(f"Confidence:  {old_conf:.3f} -> {new_conf:.3f} ({_format_delta(new_conf, old_conf, unit='')})")

    accepted = (new_mean - old_mean) <= args.max_mape_regression

    print()
    if accepted:
        print("[PASS] Candidate report accepted (no significant MAPE regression)")
    else:
        print("[FAIL] Candidate report rejected (MAPE regression exceeds threshold)")


if __name__ == "__main__":
    main()
