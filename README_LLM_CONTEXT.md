# README_LLM_CONTEXT.md

## Purpose

This document is a **full context handoff** for another LLM/agent so it can continue implementation without re-discovering architecture and recent changes.

Current state date: **16 Feb 2026**

---

## 1) Project Snapshot

- Project: GPU-accelerated 3D dimension measurement from multi-image input.
- Primary mode in active implementation: **depth-only, image-only**.
- Core output: width/height/depth/volume with confidence + error bounds.
- Main stack:
  - Reconstruction: COLMAP/pycolmap
  - Depth: DPT-based Metric3D path
  - Scale: depth-aligned multi-view fusion (confidence-aware)
  - API: FastAPI
  - Runtime: CUDA PyTorch

Repository root highlights:

- Entry points: `main.py`, `src/api/rest_api.py`
- Core orchestration: `src/core/measurement_system_gpu.py`
- Config: `src/core/config.py`
- Depth model: `src/depth/metric3d_gpu.py`
- Scale fusion: `src/scale/scale_optimizer.py`
- Capture quality: `src/utils/capture_quality.py`
- Accuracy tools:
  - `benchmark_depth_only_accuracy.py`
  - `compare_accuracy_reports.py`
  - `analyze_capture_quality.py`
  - `auto_tune_accuracy_config.py`
  - `soak_test_api.py`
  - `final_acceptance_gate.py`

---

## 2) Architecture (Current)

### End-to-end path

1. Input images loaded.
2. Optional quality prefilter removes weakest images when quality is low.
3. COLMAP sparse reconstruction produces points + camera poses + intrinsics.
4. Depth model predicts per-image depth maps (+ confidence maps).
5. Scale optimizer runs depth-aligned multi-view ratio fitting.
6. Confidence-aware fusion computes metric scale.
7. Scaled point cloud is filtered and measured.
8. API/CLI return dimensions + confidence + diagnostics.

### Core design decisions

- `metric3d.depth_scale_factor` is kept at `1.0`; absolute correction is done via `scale_recovery.depth_only_calibration`.
- Depth-only default weights:
  - marker=0, imu=0, depth=1, object=0
- Robustness emphasis:
  - MAD filtering on per-view scales
  - low-confidence handling in error bounds
  - adaptive quality pruning based on overlap statistics

---

## 3) Key Recent Enhancements (Implemented)

## Stage 8: Confidence-aware depth alignment

- Depth confidence maps are propagated from depth inference to scale optimization.
- New knobs in `ScaleRecoveryConfig`:
  - `depth_confidence_min`
  - `depth_confidence_weight_power`
- In `ScaleOptimizer` depth-aligned fitting:
  - low-confidence pixels are dropped
  - weighted median is used in per-view ratio fusion
  - view weights include confidence signal

Files:
- `src/core/measurement_system_gpu.py`
- `src/scale/scale_optimizer.py`
- `src/core/config.py`

## Stage 9: Report-driven auto-tuning

- Added bounded tuning policy from `accuracy_report.json` metrics.
- Tool emits:
  - recommendation JSON
  - generated override config module
- Never mutates base config in-place.

Files:
- `src/utils/auto_tuning.py`
- `auto_tune_accuracy_config.py`
- `tests/test_auto_tuning.py`

## Stage 10: API guardrails

- Added request backpressure for `/measure` (single GPU slot queue).
- Added OOM circuit breaker with cooldown.
- `/health` includes reliability state payload.

Files:
- `src/api/reliability.py`
- `src/api/rest_api.py`
- `tests/test_api_reliability.py`

## Stage 11: Final acceptance gate

- Added pass/fail gate that combines:
  - benchmark MAPE regression threshold
  - soak failure rate threshold
  - no unhandled soak exceptions
  - soak p95 latency threshold

Files:
- `final_acceptance_gate.py`
- `tests/test_final_acceptance_gate.py`

---

## 4) Important Config Knobs (Current)

In `SystemConfig` / `ScaleRecoveryConfig`:

Quality prefilter:
- `enable_capture_quality_filter`
- `capture_quality_threshold`
- `quality_drop_fraction`
- `enable_adaptive_quality_drop`
- `adaptive_quality_drop_min`
- `adaptive_quality_drop_max`
- `min_images_after_quality_filter`

Depth confidence fusion:
- `depth_confidence_min`
- `depth_confidence_weight_power`

Reliability/confidence behavior:
- `depth_aligned_min_views`
- `low_confidence_threshold`
- `depth_only_calibration`

---

## 5) Operational Workflow (Production-Oriented)

1. Build baseline report:
   - run `benchmark_depth_only_accuracy.py` with manifest.
2. Auto-tune from baseline:
   - run `auto_tune_accuracy_config.py`.
3. Benchmark tuned config:
   - update manifest entries to tuned config and rerun baseline script.
4. Regression compare:
   - run `compare_accuracy_reports.py`.
5. API soak stability:
   - run `soak_test_api.py`.
6. Final verdict:
   - run `final_acceptance_gate.py`.

Reference thresholds currently used:
- MAPE regression <= 0.5pp
- Soak failure rate <= 5%
- No unhandled soak exceptions
- Soak p95 latency <= 45s

---

## 6) Known Environment Notes

- In this workspace shell, `pytest` may be unavailable; syntax checks used fallback `python3 -m py_compile`.
- In non-GPU/non-torch environments, some scripts can still run in degraded mode (for report processing), but end-to-end measurement requires CUDA stack.

---

## 7) Tests Added for New Logic

- `tests/test_scale_fusion.py`
- `tests/test_config_validation.py`
- `tests/test_quality_prefilter.py`
- `tests/test_capture_quality.py`
- `tests/test_benchmark_quality_gate.py`
- `tests/test_auto_tuning.py`
- `tests/test_api_reliability.py`
- `tests/test_final_acceptance_gate.py`

---

## 8) Suggested Next Engineering Steps

1. Execute real benchmark + soak on target hardware/datasets and produce artifacts under `output/`.
2. Tune thresholds empirically from observed failure modes.
3. Add queue metrics and breaker counters to external monitoring.
4. Optional: add controlled multi-GPU or worker sharding strategy if throughput scaling is needed.

---

## 9) Handoff Summary for Next LLM

If continuing from here, prioritize this order:

1. Generate real baseline/tuned/soak artifacts.
2. Run `final_acceptance_gate.py` for true production verdict.
3. Only then change thresholds; avoid speculative tuning without report evidence.
4. Preserve depth-only defaults unless product requirements explicitly re-enable marker/IMU fusion.

This project is already in a production-hardening phase; changes should remain surgical and metrics-driven.
