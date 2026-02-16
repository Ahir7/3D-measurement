"""Reliability primitives for API request protection."""

import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict


@dataclass
class OOMCircuitBreakerConfig:
    """Configuration for OOM circuit breaker."""

    max_consecutive_oom: int = 3
    cooldown_seconds: float = 45.0


class OOMCircuitBreaker:
    """Simple thread-safe circuit breaker for repeated GPU OOM failures."""

    def __init__(self, config: OOMCircuitBreakerConfig):
        self.config = config
        self._lock = Lock()
        self._consecutive_oom = 0
        self._opened_at = 0.0

    def _now(self) -> float:
        return time.time()

    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at <= 0:
                return False
            elapsed = self._now() - self._opened_at
            if elapsed >= self.config.cooldown_seconds:
                self._opened_at = 0.0
                self._consecutive_oom = 0
                return False
            return True

    def retry_after_seconds(self) -> float:
        with self._lock:
            if self._opened_at <= 0:
                return 0.0
            remaining = self.config.cooldown_seconds - (self._now() - self._opened_at)
            return max(float(remaining), 0.0)

    def record_oom(self) -> bool:
        """Record OOM. Returns True if breaker is now open."""
        with self._lock:
            if self._opened_at > 0:
                return True

            self._consecutive_oom += 1
            if self._consecutive_oom >= self.config.max_consecutive_oom:
                self._opened_at = self._now()
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_oom = 0
            self._opened_at = 0.0

    def state(self) -> Dict[str, float]:
        """Expose breaker state for diagnostics."""
        open_state = self.is_open()
        return {
            "open": open_state,
            "retry_after_seconds": self.retry_after_seconds() if open_state else 0.0,
            "consecutive_oom": float(self._consecutive_oom),
            "max_consecutive_oom": float(self.config.max_consecutive_oom),
            "cooldown_seconds": float(self.config.cooldown_seconds),
        }
