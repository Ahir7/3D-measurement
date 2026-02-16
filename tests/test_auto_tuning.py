from src.utils.auto_tuning import recommend_tuning


def _current():
    return {
        "capture_quality_threshold": 0.45,
        "quality_drop_fraction": 0.20,
        "adaptive_quality_drop_min": 0.10,
        "adaptive_quality_drop_max": 0.35,
        "depth_confidence_min": 0.35,
        "depth_confidence_weight_power": 1.25,
    }


def test_recommend_tightens_when_high_mape_and_low_quality():
    report = {
        "mean_mape_percent": 13.8,
        "median_mape_percent": 13.2,
        "mean_confidence": 0.52,
        "scenes": [
            {
                "confidence": 0.50,
                "capture_quality": {"quality_score": 0.40, "overlap_median": 0.30},
            },
            {
                "confidence": 0.54,
                "capture_quality": {"quality_score": 0.42, "overlap_median": 0.33},
            },
        ],
    }

    recommended, deltas, rationale, _ = recommend_tuning(report, _current())

    assert recommended["depth_confidence_min"] > 0.35
    assert recommended["quality_drop_fraction"] > 0.20
    assert recommended["capture_quality_threshold"] >= 0.45
    assert recommended["adaptive_quality_drop_max"] >= 0.35
    assert any("High MAPE" in item for item in rationale)
    assert deltas["depth_confidence_min"] > 0


def test_recommend_relaxes_when_strong_metrics():
    report = {
        "mean_mape_percent": 6.1,
        "median_mape_percent": 6.0,
        "mean_confidence": 0.78,
        "scenes": [
            {
                "confidence": 0.79,
                "capture_quality": {"quality_score": 0.62, "overlap_median": 0.55},
            },
            {
                "confidence": 0.77,
                "capture_quality": {"quality_score": 0.60, "overlap_median": 0.52},
            },
        ],
    }

    recommended, _, rationale, _ = recommend_tuning(report, _current())

    assert recommended["depth_confidence_min"] < 0.35
    assert recommended["quality_drop_fraction"] < 0.20
    assert any("Strong baseline" in item for item in rationale)


def test_recommend_respects_bounds_on_extreme_inputs():
    report = {
        "mean_mape_percent": 40.0,
        "median_mape_percent": 38.0,
        "mean_confidence": 0.10,
        "scenes": [
            {
                "confidence": 0.12,
                "capture_quality": {"quality_score": 0.10, "overlap_median": 0.10},
            }
        ],
    }

    current = _current()
    current["capture_quality_threshold"] = 0.68
    current["quality_drop_fraction"] = 0.49
    current["adaptive_quality_drop_min"] = 0.39
    current["adaptive_quality_drop_max"] = 0.59
    current["depth_confidence_min"] = 0.69
    current["depth_confidence_weight_power"] = 2.15

    recommended, _, _, _ = recommend_tuning(report, current)

    assert 0.25 <= recommended["capture_quality_threshold"] <= 0.70
    assert 0.05 <= recommended["quality_drop_fraction"] <= 0.50
    assert 0.05 <= recommended["adaptive_quality_drop_min"] <= 0.40
    assert 0.15 <= recommended["adaptive_quality_drop_max"] <= 0.60
    assert recommended["adaptive_quality_drop_min"] <= recommended["adaptive_quality_drop_max"]
    assert 0.15 <= recommended["depth_confidence_min"] <= 0.70
    assert 0.8 <= recommended["depth_confidence_weight_power"] <= 2.2
