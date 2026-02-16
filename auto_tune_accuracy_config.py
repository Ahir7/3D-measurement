#!/usr/bin/env python3
"""Auto-tune depth-only accuracy knobs from benchmark report(s)."""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, Optional

from src.utils.auto_tuning import recommend_tuning


DEFAULT_KNOBS = {
    "capture_quality_threshold": 0.45,
    "quality_drop_fraction": 0.20,
    "adaptive_quality_drop_min": 0.10,
    "adaptive_quality_drop_max": 0.35,
    "depth_confidence_min": 0.35,
    "depth_confidence_weight_power": 1.25,
}


def _load_report(path: Path) -> Dict:
    with open(path, "r") as handle:
        return json.load(handle)


def _load_config(config_path: Optional[Path]):
    if config_path is None:
        return None

    spec = importlib.util.spec_from_file_location("tune_base_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config module: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "get_config"):
        raise RuntimeError(f"Config file must define get_config(): {config_path}")

    return module.get_config()


def _current_knobs(config) -> Dict[str, float]:
    if config is None:
        return dict(DEFAULT_KNOBS)

    return {
        "capture_quality_threshold": float(getattr(config, "capture_quality_threshold", DEFAULT_KNOBS["capture_quality_threshold"])),
        "quality_drop_fraction": float(getattr(config, "quality_drop_fraction", DEFAULT_KNOBS["quality_drop_fraction"])),
        "adaptive_quality_drop_min": float(getattr(config, "adaptive_quality_drop_min", DEFAULT_KNOBS["adaptive_quality_drop_min"])),
        "adaptive_quality_drop_max": float(getattr(config, "adaptive_quality_drop_max", DEFAULT_KNOBS["adaptive_quality_drop_max"])),
        "depth_confidence_min": float(
            getattr(getattr(config, "scale_recovery", object()), "depth_confidence_min", DEFAULT_KNOBS["depth_confidence_min"])
        ),
        "depth_confidence_weight_power": float(
            getattr(
                getattr(config, "scale_recovery", object()),
                "depth_confidence_weight_power",
                DEFAULT_KNOBS["depth_confidence_weight_power"],
            )
        ),
    }


def _write_override_config(
    path: Path,
    base_config_path: str,
    recommended: Dict[str, float],
) -> None:
    content = f'''"""Auto-generated tuned config override."""

import importlib.util
from pathlib import Path


def get_config():
    base_path = Path("{base_config_path}")
    spec = importlib.util.spec_from_file_location("base_config", base_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.get_config()

    config.capture_quality_threshold = {recommended["capture_quality_threshold"]:.4f}
    config.quality_drop_fraction = {recommended["quality_drop_fraction"]:.4f}
    config.adaptive_quality_drop_min = {recommended["adaptive_quality_drop_min"]:.4f}
    config.adaptive_quality_drop_max = {recommended["adaptive_quality_drop_max"]:.4f}
    config.scale_recovery.depth_confidence_min = {recommended["depth_confidence_min"]:.4f}
    config.scale_recovery.depth_confidence_weight_power = {recommended["depth_confidence_weight_power"]:.4f}
    return config
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-tune depth-only accuracy settings from a benchmark report")
    parser.add_argument("--report", required=True, help="Path to accuracy_report.json")
    parser.add_argument("--base-config", default="configs/rtx2060_config.py", help="Base config python file")
    parser.add_argument("--output", default="output/tuning/tuning_recommendation.json", help="Path to output recommendation JSON")
    parser.add_argument(
        "--write-override-config",
        default="configs/rtx2060_tuned_auto.py",
        help="Path to generated tuned config module",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    base_config_path = Path(args.base_config)

    report = _load_report(report_path)
    config = None
    config_source = "defaults"
    try:
        config = _load_config(base_config_path)
        config_source = "base_config"
    except Exception as error:
        print(f"[WARN] Could not load base config ({base_config_path}): {error}")
        print("[WARN] Falling back to internal default knobs for recommendation.")

    current = _current_knobs(config)
    recommended, deltas, rationale, metrics = recommend_tuning(report, current)

    recommendation = {
        "report": str(report_path),
        "base_config": str(base_config_path),
        "knob_source": config_source,
        "metrics": metrics,
        "current": current,
        "recommended": recommended,
        "deltas": deltas,
        "rationale": rationale,
        "next_step": "Run benchmark_depth_only_accuracy.py with --config pointing to the generated override config.",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(recommendation, handle, indent=2)

    override_path = Path(args.write_override_config)
    _write_override_config(
        override_path,
        base_config_path=str(base_config_path),
        recommended=recommended,
    )

    print("=" * 70)
    print("AUTO-TUNING RECOMMENDATION")
    print("=" * 70)
    print(f"Report: {report_path}")
    print(f"Base config: {base_config_path}")
    print(f"Mean MAPE: {metrics['mean_mape_percent']:.2f}%")
    print(f"Mean confidence: {metrics['mean_confidence']:.3f}")
    print()
    for key in sorted(recommended.keys()):
        print(
            f"{key}: {current[key]:.4f} -> {recommended[key]:.4f} "
            f"(delta {deltas[key]:+.4f})"
        )
    print()
    print("Rationale:")
    for item in rationale:
        print(f"- {item}")
    print()
    print(f"Recommendation JSON: {output_path}")
    print(f"Generated override config: {override_path}")


if __name__ == "__main__":
    main()
