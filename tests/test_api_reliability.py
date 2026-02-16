from src.api.reliability import OOMCircuitBreaker, OOMCircuitBreakerConfig


def test_breaker_opens_after_threshold_and_resets_after_cooldown(monkeypatch):
    config = OOMCircuitBreakerConfig(max_consecutive_oom=2, cooldown_seconds=10.0)
    breaker = OOMCircuitBreaker(config)

    clock = {"t": 100.0}
    monkeypatch.setattr(breaker, "_now", lambda: clock["t"])

    assert breaker.is_open() is False
    assert breaker.record_oom() is False
    assert breaker.is_open() is False

    assert breaker.record_oom() is True
    assert breaker.is_open() is True
    assert breaker.retry_after_seconds() > 0.0

    clock["t"] = 111.0
    assert breaker.is_open() is False
    assert breaker.retry_after_seconds() == 0.0


def test_breaker_record_success_clears_state(monkeypatch):
    config = OOMCircuitBreakerConfig(max_consecutive_oom=2, cooldown_seconds=30.0)
    breaker = OOMCircuitBreaker(config)

    monkeypatch.setattr(breaker, "_now", lambda: 50.0)
    breaker.record_oom()
    breaker.record_oom()
    assert breaker.is_open() is True

    breaker.record_success()
    assert breaker.is_open() is False
    state = breaker.state()
    assert state["open"] is False
    assert state["retry_after_seconds"] == 0.0
    assert state["consecutive_oom"] == 0.0
