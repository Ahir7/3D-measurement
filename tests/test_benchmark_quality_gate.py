from benchmark_depth_only_accuracy import _quality_gate_decision


def test_quality_gate_disabled_proceeds():
    assert _quality_gate_decision(quality_score=0.1, threshold=None, policy="fail") == "proceed"


def test_quality_gate_above_threshold_proceeds():
    assert _quality_gate_decision(quality_score=0.7, threshold=0.45, policy="fail") == "proceed"


def test_quality_gate_below_threshold_skip():
    assert _quality_gate_decision(quality_score=0.2, threshold=0.45, policy="skip") == "skip"


def test_quality_gate_below_threshold_fail():
    assert _quality_gate_decision(quality_score=0.2, threshold=0.45, policy="fail") == "fail"


def test_quality_gate_below_threshold_warn_proceeds():
    assert _quality_gate_decision(quality_score=0.2, threshold=0.45, policy="warn") == "proceed"
