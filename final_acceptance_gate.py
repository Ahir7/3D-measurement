#!/usr/bin/env python3
"""Final production acceptance gate for depth-only pipeline."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _load_json(path: Path) -> Dict:
    with open(path, "r") as handle:
        return json.load(handle)


def _evaluate_accuracy(
    baseline: Dict,
    candidate: Dict,
    max_mape_regression: float,
) -> Tuple[bool, List[str], Dict[str, float]]:
    checks: List[str] = []

    baseline_mean = float(baseline.get("mean_mape_percent", 0.0))
    candidate_mean = float(candidate.get("mean_mape_percent", 0.0))
    candidate_confidence = float(candidate.get("mean_confidence", 0.0))

    regression = candidate_mean - baseline_mean
    pass_regression = regression <= max_mape_regression
    checks.append(
        f"MAPE regression {regression:+.2f}pp (threshold <= {max_mape_regression:.2f}pp): "
        f"{'PASS' if pass_regression else 'FAIL'}"
    )

    details = {
        "baseline_mean_mape_percent": baseline_mean,
        "candidate_mean_mape_percent": candidate_mean,
        "candidate_mean_confidence": candidate_confidence,
        "mape_regression_pp": regression,
    }
    return pass_regression, checks, details


def _evaluate_soak(
    soak_report: Dict,
    max_failure_rate: float,
    max_p95_latency_seconds: float,
) -> Tuple[bool, List[str], Dict[str, float]]:
    checks: List[str] = []

    failure_rate = float(soak_report.get("failure_rate", 1.0))
    failures = int(soak_report.get("failures", 0))
    p95 = float(soak_report.get("latency_seconds", {}).get("p95", 1e9))

    pass_failure_rate = failure_rate <= max_failure_rate
    pass_no_exceptions = True
    for failure in soak_report.get("failure_details", []):
        if str(failure.get("status", "")).lower() == "exception":
            pass_no_exceptions = False
            break

    pass_p95 = p95 <= max_p95_latency_seconds

    checks.append(
        f"Failure rate {failure_rate * 100:.2f}% (threshold <= {max_failure_rate * 100:.2f}%): "
        f"{'PASS' if pass_failure_rate else 'FAIL'}"
    )
    checks.append(
        f"Unhandled exceptions: {'PASS' if pass_no_exceptions else 'FAIL'}"
    )
    checks.append(
        f"P95 latency {p95:.2f}s (threshold <= {max_p95_latency_seconds:.2f}s): "
        f"{'PASS' if pass_p95 else 'FAIL'}"
    )

    details = {
        "failure_rate": failure_rate,
        "failures": float(failures),
        "p95_latency_seconds": p95,
    }
    return pass_failure_rate and pass_no_exceptions and pass_p95, checks, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Final acceptance gate for depth-only production readiness")
    parser.add_argument("--baseline-report", required=True, help="Baseline accuracy_report.json")
    parser.add_argument("--candidate-report", required=True, help="Candidate accuracy_report.json")
    parser.add_argument("--soak-report", required=True, help="Soak test report JSON")
    parser.add_argument("--max-mape-regression", type=float, default=0.5, help="Allowed MAPE regression in percentage points")
    parser.add_argument("--max-failure-rate", type=float, default=0.05, help="Allowed soak failure rate in [0,1]")
    parser.add_argument("--max-p95-latency", type=float, default=45.0, help="Allowed p95 latency in seconds")
    parser.add_argument("--output", default="output/final_acceptance_report.json", help="Output gate report path")
    args = parser.parse_args()

    baseline_path = Path(args.baseline_report)
    candidate_path = Path(args.candidate_report)
    soak_path = Path(args.soak_report)

    baseline = _load_json(baseline_path)
    candidate = _load_json(candidate_path)
    soak = _load_json(soak_path)

    accuracy_ok, accuracy_checks, accuracy_details = _evaluate_accuracy(
        baseline,
        candidate,
        max_mape_regression=args.max_mape_regression,
    )
    soak_ok, soak_checks, soak_details = _evaluate_soak(
        soak,
        max_failure_rate=args.max_failure_rate,
        max_p95_latency_seconds=args.max_p95_latency,
    )

    accepted = accuracy_ok and soak_ok

    report = {
        "accepted": accepted,
        "inputs": {
            "baseline_report": str(baseline_path),
            "candidate_report": str(candidate_path),
            "soak_report": str(soak_path),
        },
        "thresholds": {
            "max_mape_regression": args.max_mape_regression,
            "max_failure_rate": args.max_failure_rate,
            "max_p95_latency": args.max_p95_latency,
        },
        "accuracy": {
            "ok": accuracy_ok,
            "checks": accuracy_checks,
            "details": accuracy_details,
        },
        "soak": {
            "ok": soak_ok,
            "checks": soak_checks,
            "details": soak_details,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print("=" * 70)
    print("FINAL ACCEPTANCE GATE")
    print("=" * 70)
    print("Accuracy checks:")
    for check in accuracy_checks:
        print(f"- {check}")
    print("Soak checks:")
    for check in soak_checks:
        print(f"- {check}")
    print()
    print(f"Verdict: {'PASS' if accepted else 'FAIL'}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
