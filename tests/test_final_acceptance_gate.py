from final_acceptance_gate import _evaluate_accuracy, _evaluate_soak


def test_evaluate_accuracy_passes_within_regression_limit():
    baseline = {"mean_mape_percent": 8.0}
    candidate = {"mean_mape_percent": 8.4, "mean_confidence": 0.66}

    ok, checks, details = _evaluate_accuracy(baseline, candidate, max_mape_regression=0.5)

    assert ok is True
    assert details["mape_regression_pp"] == 0.4
    assert any("PASS" in item for item in checks)


def test_evaluate_accuracy_fails_on_excessive_regression():
    baseline = {"mean_mape_percent": 8.0}
    candidate = {"mean_mape_percent": 9.0, "mean_confidence": 0.66}

    ok, checks, details = _evaluate_accuracy(baseline, candidate, max_mape_regression=0.5)

    assert ok is False
    assert details["mape_regression_pp"] == 1.0
    assert any("FAIL" in item for item in checks)


def test_evaluate_soak_passes_on_stable_report():
    soak = {
        "failure_rate": 0.02,
        "failures": 1,
        "latency_seconds": {"p95": 18.0},
        "failure_details": [{"status": 500}],
    }

    ok, checks, details = _evaluate_soak(soak, max_failure_rate=0.05, max_p95_latency_seconds=30.0)

    assert ok is True
    assert details["failure_rate"] == 0.02
    assert any("PASS" in item for item in checks)


def test_evaluate_soak_fails_when_exception_present():
    soak = {
        "failure_rate": 0.01,
        "failures": 1,
        "latency_seconds": {"p95": 12.0},
        "failure_details": [{"status": "exception", "detail": "timeout"}],
    }

    ok, checks, _ = _evaluate_soak(soak, max_failure_rate=0.05, max_p95_latency_seconds=30.0)

    assert ok is False
    assert any("Unhandled exceptions: FAIL" in item for item in checks)
