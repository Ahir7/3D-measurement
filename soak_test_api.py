#!/usr/bin/env python3
"""Soak test for /measure endpoint stability under repeated requests.

Example:
python soak_test_api.py \
  --url http://localhost:8000 \
  --iterations 50 \
  --images examples/original/resized/1.jpg examples/original/resized/2.jpg examples/original/resized/3.jpg
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import List

import requests


def _build_files(image_paths: List[Path]):
    handles = []
    files = []
    for path in image_paths:
        handle = open(path, "rb")
        handles.append(handle)
        files.append(("files", (path.name, handle, "image/jpeg")))
    return files, handles


def _close_handles(handles):
    for handle in handles:
        try:
            handle.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Soak test /measure endpoint")
    parser.add_argument("--url", default="http://localhost:8000", help="Base API URL")
    parser.add_argument("--iterations", type=int, default=30, help="Number of /measure calls")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout seconds")
    parser.add_argument("--max-failures", type=int, default=3, help="Abort after this many failures")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between requests (seconds)")
    parser.add_argument("--images", nargs="+", required=True, help="Input image paths")
    parser.add_argument("--output", default="output/soak_test_report.json", help="Output report path")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    endpoint = f"{base_url}/measure"

    image_paths = [Path(path) for path in args.images]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing images: {missing}")
    if len(image_paths) < 3:
        raise RuntimeError("At least 3 images are required")

    latencies = []
    failures = []
    status_counts = {}

    for iteration in range(1, args.iterations + 1):
        files, handles = _build_files(image_paths)
        started = time.perf_counter()

        try:
            response = requests.post(endpoint, files=files, timeout=args.timeout)
            latency = time.perf_counter() - started
            latencies.append(latency)

            status = response.status_code
            status_counts[status] = status_counts.get(status, 0) + 1

            if status != 200:
                detail = None
                try:
                    detail = response.json().get("detail")
                except Exception:
                    detail = response.text[:400]
                failures.append({
                    "iteration": iteration,
                    "status": status,
                    "detail": detail,
                })
                print(f"[FAIL] iter={iteration} status={status} latency={latency:.2f}s detail={detail}")
            else:
                print(f"[OK]   iter={iteration} status=200 latency={latency:.2f}s")

        except Exception as error:
            latency = time.perf_counter() - started
            latencies.append(latency)
            failures.append({
                "iteration": iteration,
                "status": "exception",
                "detail": str(error),
            })
            print(f"[EXC]  iter={iteration} latency={latency:.2f}s error={error}")

        finally:
            _close_handles(handles)

        if len(failures) >= args.max_failures:
            print(f"[ABORT] Reached max failures ({args.max_failures})")
            break

        if args.sleep > 0:
            time.sleep(args.sleep)

    total_runs = len(latencies)
    successful = total_runs - len(failures)

    summary = {
        "endpoint": endpoint,
        "requested_iterations": args.iterations,
        "completed_iterations": total_runs,
        "successful": successful,
        "failures": len(failures),
        "failure_rate": (len(failures) / total_runs) if total_runs else 1.0,
        "status_counts": status_counts,
        "latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": sorted(latencies)[max(int(0.95 * len(latencies)) - 1, 0)] if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "failure_details": failures,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 70)
    print("SOAK TEST SUMMARY")
    print("=" * 70)
    print(f"Completed: {summary['completed_iterations']}/{summary['requested_iterations']}")
    print(f"Failures: {summary['failures']} ({summary['failure_rate'] * 100:.1f}%)")
    print(f"Mean latency: {summary['latency_seconds']['mean']:.2f}s" if summary['latency_seconds']['mean'] is not None else "Mean latency: N/A")
    print(f"P95 latency: {summary['latency_seconds']['p95']:.2f}s" if summary['latency_seconds']['p95'] is not None else "P95 latency: N/A")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
