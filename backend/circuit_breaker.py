"""Circuit Breaker pattern for downstream AI provider resilience.

Protects the inference gateway from cascading failures, hanging connections,
and quota exhaustion by tracking provider health state:

- CLOSED: Normal operation. Requests pass through to the provider. Consecutive
  failures above `failure_threshold` trip the state to OPEN.
- OPEN: Provider is failing. Requests immediately fast-fail with `CircuitOpenError`
  without executing network calls, saving latency and downstream pressure.
- HALF-OPEN: After `recovery_timeout` expires, probe requests are admitted to test
  provider recovery. Success transitions back to CLOSED; failure resets to OPEN.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is attempted against a provider whose circuit is OPEN."""


class CircuitBreaker(Generic[T]):
    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 3,
        recovery_timeout: float = 10.0,
        half_open_success_threshold: int = 2,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_success_threshold = half_open_success_threshold

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_state_change = time.monotonic()
        self._last_failure_time: float | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        # Check if OPEN has transitioned into HALF_OPEN based on recovery_timeout
        if self._state == CircuitState.OPEN and (
            time.monotonic() - self._last_state_change
        ) >= self.recovery_timeout:
            return CircuitState.HALF_OPEN
        return self._state


    @property
    def is_available(self) -> bool:
        return self.state != CircuitState.OPEN

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Execute `fn` guarded by the circuit breaker state machine."""
        async with self._lock:
            current = self.state
            if current == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. Provider is temporarily unavailable."
                )
            if current == CircuitState.HALF_OPEN and self._state == CircuitState.OPEN:
                self._state = CircuitState.HALF_OPEN
                self._consecutive_successes = 0

        try:
            result = await fn()
            await self._on_success()
            return result
        except Exception as exc:
            await self._on_failure(exc)
            raise

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self.half_open_success_threshold:
                    self._state = CircuitState.CLOSED
                    self._consecutive_failures = 0
                    self._consecutive_successes = 0
                    self._last_state_change = time.monotonic()
            elif self._state == CircuitState.CLOSED:
                self._consecutive_failures = 0

    async def _on_failure(self, exc: Exception) -> None:
        async with self._lock:
            self._last_failure_time = time.monotonic()
            self._consecutive_failures += 1
            if self._state == CircuitState.HALF_OPEN:
                # Failed probe in HALF_OPEN trips immediately back to OPEN
                self._state = CircuitState.OPEN
                self._consecutive_successes = 0
                self._last_state_change = time.monotonic()
            elif self._state == CircuitState.CLOSED:
                if self._consecutive_failures >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._last_state_change = time.monotonic()

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self._consecutive_failures,
            "consecutive_successes": self._consecutive_successes,
            "is_available": self.is_available,
        }
