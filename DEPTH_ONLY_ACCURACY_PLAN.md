# Depth-Only Accuracy Plan (Image-Only)

This project now runs in image-only mode (no markers, IMU, AR/VR inputs).

## 1) Baseline Protocol

- Use at least **3 objects/scenes** with known dimensions.
- For each scene, capture **15-25 images** with overlap and varied angles.
- Run each scene with `--num-runs 3` to measure repeatability.

### Ground-truth benchmark

Create a manifest JSON and run:

```bash
python benchmark_depth_only_accuracy.py --manifest benchmark_manifest.json --num-runs 3 --quality-threshold 0.45 --quality-policy fail
```

Use `benchmark_manifest.template.json` as the starting point.

Optional pre-check (single scene):

```bash
python analyze_capture_quality.py examples/benchmark/scene_box_daylight/*.jpg --output output/capture_quality_scene_box_daylight.json
```

This reports blur/exposure/overlap quality before measurement.

Output report:

- `output/baseline/accuracy_report.json`
- `output/baseline/raw_results.json`

Each scene in `accuracy_report.json` also includes `capture_quality` summary.
If a quality gate is active, report also includes `quality_threshold`, `quality_policy`, and any `skipped` scenes.

If you want to continue processing even when one scene fails:

```bash
python benchmark_depth_only_accuracy.py --manifest benchmark_manifest.json --num-runs 3 --quality-threshold 0.45 --quality-policy fail --continue-on-error
```

Quality policy modes:

- `fail`: stop with error when a scene is below threshold.
- `skip`: skip low-quality scenes and continue.
- `warn`: run all scenes; only emit warning.

Disable gate entirely:

```bash
python benchmark_depth_only_accuracy.py --manifest benchmark_manifest.json --quality-threshold -1
```

## 2) Quality Targets

- Median MAPE (W/H/D): **<= 10%** initially
- Stretch target: **<= 7%**
- Repeatability (run-to-run std): **<= 5%**

## 3) Capture Rules (Most Important)

- Keep object fully in frame in all images.
- Avoid motion blur and heavy exposure changes.
- Ensure texture/feature richness around object.
- Prefer diffuse lighting and stable background.

## 4) Calibration Rule

- Keep `metric3d.depth_scale_factor = 1.0`.
- Use post-scale correction only via `scale_recovery.depth_only_calibration`.

## 5) Auto Quality Prefilter (Accuracy Stage)

Pipeline now auto-analyzes capture quality before reconstruction and may drop the lowest-quality images when the set quality score is low.

Key config knobs (`SystemConfig`):

- `enable_capture_quality_filter`
- `capture_quality_threshold` (default `0.45`)
- `quality_drop_fraction` (default `0.20`)
- `enable_adaptive_quality_drop`
- `adaptive_quality_drop_min` / `adaptive_quality_drop_max`
- `min_images_after_quality_filter`

Behavior:

- If capture quality is above threshold, all images are used.
- If below threshold, lowest-quality images are pruned (bounded by `min_images_after_quality_filter`).
- Adaptive mode increases/decreases pruning strength based on overlap median + overlap dispersion.
- Result diagnostics include quality before/after and number of removed images.

## 6) Regression Tracking

After every algorithm/config change:

1. Run the same baseline manifest.
2. Compare MAPE and confidence vs previous report.
3. Accept change only if metrics improve or remain stable.

Comparison command:

```bash
python compare_accuracy_reports.py --old output/baseline_prev/accuracy_report.json --new output/baseline/accuracy_report.json
```

Default acceptance rule in comparator:

- Reject if mean MAPE regresses by more than `0.5` percentage points.

## 7) Soak Stability Check

Before production rollout, run repeated `/measure` calls to detect crashes/timeouts:

```bash
python soak_test_api.py --url http://localhost:8000 --iterations 30 --max-failures 3 --images examples/original/resized/1.jpg examples/original/resized/2.jpg examples/original/resized/3.jpg
```

Suggested acceptance:

- Failure rate <= 5%
- No unhandled exceptions
- P95 latency stable across runs

## 8) Confidence-Aware Depth Alignment (Next Stage)

Depth-aligned scale fusion now uses Metric3D confidence maps to down-weight unreliable pixels during COLMAP/depth ratio fitting.

Key knobs (`ScaleRecoveryConfig`):

- `depth_confidence_min` (default `0.35`): minimum confidence for a pixel to contribute.
- `depth_confidence_weight_power` (default `1.25`): emphasizes high-confidence pixels when fusing per-view ratios.

Practical tuning guidance:

- Raise `depth_confidence_min` to `0.45-0.55` when scenes have reflective/textureless regions.
- Lower to `0.25-0.35` if too many views are rejected.
- Increase `depth_confidence_weight_power` to `1.5-1.8` for noisy scenes with enough remaining coverage.

## 9) Auto-Tuning From Benchmark Report

Use report-driven bounded tuning after each benchmark cycle:

```bash
python auto_tune_accuracy_config.py --report output/baseline/accuracy_report.json --base-config configs/rtx2060_config.py --output output/tuning/tuning_recommendation.json --write-override-config configs/rtx2060_tuned_auto.py
```

Then re-run benchmark with tuned config in manifest entries (set `"config": "configs/rtx2060_tuned_auto.py"`) and compare:

```bash
python compare_accuracy_reports.py --old output/baseline/accuracy_report.json --new output/baseline_tuned/accuracy_report.json
```

Safety properties of auto-tuning:

- Only adjusts bounded knobs: quality threshold/drop and depth confidence controls.
- Keeps adaptive drop range consistent (`min <= max`).
- Never mutates base config file in place; generates a separate override config.

## 10) API Production Guardrails

The API now includes reliability controls for sustained production traffic:

- Concurrency backpressure for `/measure` (single GPU slot, short queue timeout).
- OOM circuit breaker: after repeated CUDA OOM failures, new measurement/benchmark requests are temporarily rejected.
- `/health` now exposes breaker + queue state under `reliability`.

Operational guidance:

- If `/measure` returns `429`, reduce client concurrency and retry with jitter.
- If `/measure` or `/benchmark` returns `503` with GPU protection active, wait for `Retry-After` and reduce input size/count.
- Keep soak testing enabled before release (`soak_test_api.py`) and monitor failure rate + p95 latency.

## 11) Final Go/No-Go Acceptance Gate

After baseline + tuned benchmark and soak runs are available, run one final gate:

```bash
python final_acceptance_gate.py --baseline-report output/baseline/accuracy_report.json --candidate-report output/baseline_tuned/accuracy_report.json --soak-report output/soak_test_report.json --max-mape-regression 0.5 --max-failure-rate 0.05 --max-p95-latency 45.0 --output output/final_acceptance_report.json
```

Default gate rules:

- MAPE regression <= `0.5` percentage points.
- Soak failure rate <= `5%`.
- No unhandled exceptions in soak failures.
- Soak p95 latency <= `45s`.
