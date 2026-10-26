"""SingleFlight: in-flight request deduplicator.

Coalesces multiple concurrent executions of the same key into a single
in-flight task. When N concurrent callers request the same key, only the first
caller triggers the underlying async worker function; the other N-1 callers await
the same shared result without duplicating work or hitting downstream providers.

This is the standard architectural solution to the cache-stampede / thundering-herd
problem documented in system-design/architecture.md and ADR 0003.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class _Flight(Generic[T]):
    task: asyncio.Task[T] | None = None


class SingleFlight(Generic[T]):
    """Manages in-flight deduplication keyed by string identifier."""

    def __init__(self) -> None:
        self._flights: dict[str, _Flight[T]] = {}
        self._lock = asyncio.Lock()
        self.total_calls = 0
        self.shared_calls = 0

    async def execute(self, key: str, fn: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        """Execute `fn` if no flight with `key` is currently in flight.

        If a flight is already running for `key`, await its completion.
        Returns `(result, is_shared)` where `is_shared` is True if this invocation
        shared an already-running flight.
        """
        flight: _Flight[T]
        is_initiator = False

        async with self._lock:
            self.total_calls += 1
            if key in self._flights:
                self.shared_calls += 1
                flight = self._flights[key]
            else:
                flight = _Flight()
                self._flights[key] = flight
                flight.task = asyncio.create_task(self._run_flight(key, flight, fn))
                flight.task.add_done_callback(self._consume_task_result)
                is_initiator = True

        # The worker task belongs to the flight, not to whichever caller created it.
        # Shield every caller's wait so cancellation only affects that caller.
        assert flight.task is not None
        result = await asyncio.shield(flight.task)
        return result, not is_initiator

    async def _run_flight(
        self,
        key: str,
        flight: _Flight[T],
        fn: Callable[[], Awaitable[T]],
    ) -> T:
        try:
            return await fn()
        finally:
            async with self._lock:
                if self._flights.get(key) is flight:
                    del self._flights[key]

    @staticmethod
    def _consume_task_result(task: asyncio.Future[T]) -> None:
        """Observe failures if every caller cancels before the flight completes."""
        if not task.cancelled():
            task.exception()

    @property
    def in_flight_count(self) -> int:
        return len(self._flights)

    @property
    def deduplication_rate(self) -> float:
        return self.shared_calls / self.total_calls if self.total_calls else 0.0

    def reset_stats(self) -> None:
        self.total_calls = 0
        self.shared_calls = 0
