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
    future: asyncio.Future[T]
    callers: int = 1


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
        flight: _Flight[T] | None = None
        is_initiator = False

        async with self._lock:
            self.total_calls += 1
            if key in self._flights:
                self.shared_calls += 1
                flight = self._flights[key]
                flight.callers += 1
            else:
                loop = asyncio.get_running_loop()
                flight = _Flight(future=loop.create_future(), callers=1)
                self._flights[key] = flight
                is_initiator = True

        if not is_initiator:
            # Join existing flight
            result = await flight.future
            return result, True

        # Initiator runs the worker function
        try:
            res = await fn()
            flight.future.set_result(res)
            return res, False
        except BaseException as exc:
            if flight.callers > 1:
                flight.future.set_exception(exc)
            else:
                flight.future.cancel()
            raise
        finally:
            async with self._lock:
                self._flights.pop(key, None)

    @property
    def in_flight_count(self) -> int:
        return len(self._flights)

    @property
    def deduplication_rate(self) -> float:
        return self.shared_calls / self.total_calls if self.total_calls else 0.0

    def reset_stats(self) -> None:
        self.total_calls = 0
        self.shared_calls = 0
