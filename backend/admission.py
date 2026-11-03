"""Single-event-loop admission budgets for the public demonstration."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator


class AdmissionRejected(RuntimeError):
    """Work could not be admitted; contains no request or provider details."""


class RequestPermit:
    def __init__(self, budget: AdmissionBudget) -> None:
        self.budget = budget
        self.released = False

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.budget.requests -= 1


class WorkPermit:
    def __init__(self, budget: AdmissionBudget, active: bool) -> None:
        self.budget = budget
        self.active = active
        self.released = False
        self.ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        if active:
            self.ready.set_result(None)

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        if self.active:
            self.budget.active -= 1
        else:
            self.budget._pending.remove(self)
            self.ready.cancel()
        while self.budget._pending and self.budget.active < self.budget.max_active:
            permit = self.budget._pending.popleft()
            if permit.ready.cancelled():
                # Event-loop teardown may cancel a worker before its finally
                # runs. Never promote a cancelled readiness future.
                permit.released = True
                continue
            permit.active = True
            self.budget.active += 1
            permit.ready.set_result(None)


class AdmissionBudget:
    """Reserve synchronously and atomically; only bounded accepted work waits.

    An active completion includes its micro-batcher residence and execution,
    conservatively bounding actual provider concurrency. Streams never queue.
    """

    def __init__(self, max_active: int = 4, max_pending: int = 16, max_requests: int = 64) -> None:
        if max_active < 1 or max_pending < 0 or max_requests < 1:
            raise ValueError("Admission limits require active >= 1, pending >= 0, requests >= 1")
        self.max_active = max_active
        self.max_pending = max_pending
        self.max_requests = max_requests
        self.active = 0
        self.requests = 0
        self.closing = False
        self._pending: deque[WorkPermit] = deque()
        self._streams: set[AdmittedStream] = set()

    @property
    def pending(self) -> int:
        return len(self._pending)

    def reserve_request(self) -> RequestPermit:
        if self.closing or self.requests >= self.max_requests:
            raise AdmissionRejected("Public demo capacity unavailable")
        self.requests += 1
        return RequestPermit(self)

    def reserve_work(self, count: int, *, queue: bool = True) -> list[WorkPermit]:
        free_active = self.max_active - self.active
        available = free_active + (self.max_pending - self.pending if queue else 0)
        if self.closing or count > available:
            raise AdmissionRejected("Public demo capacity unavailable")
        permits = []
        for _ in range(count):
            permit = WorkPermit(self, active=self.active < self.max_active)
            if permit.active:
                self.active += 1
            else:
                self._pending.append(permit)
            permits.append(permit)
        return permits

    async def close_streams(self) -> None:
        """Stop admission and terminate streams, including unused reservations."""
        self.closing = True
        streams = tuple(self._streams)
        owners = {
            stream.owner
            for stream in streams
            if stream.owner is not None and stream.owner is not asyncio.current_task()
        }
        for task in owners:
            if not task.done():
                task.cancel()
        if owners:
            await asyncio.gather(*owners, return_exceptions=True)
        for stream in streams:
            await stream.aclose()


class AdmittedStream(AsyncIterator[str]):
    """Own a stream reservation through iteration and response termination."""

    def __init__(self, iterator: AsyncGenerator[str, None], budget: AdmissionBudget | None) -> None:
        self.iterator = iterator
        self.budget = budget
        self.owner: asyncio.Task | None = None
        self.request: RequestPermit | None = None
        self.work: WorkPermit | None = None
        self.closed = False
        if budget is not None:
            self.request = budget.reserve_request()
            try:
                self.work = budget.reserve_work(1, queue=False)[0]
            except BaseException:
                self.request.release()
                raise
            budget._streams.add(self)

    def start(self) -> None:
        if self.closed:
            raise AdmissionRejected("Public demo capacity unavailable")
        if self.owner is None:
            self.owner = asyncio.current_task()

    async def __anext__(self) -> str:
        self.start()
        return await anext(self.iterator)

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            await self.iterator.aclose()
        finally:
            if self.work is not None:
                self.work.release()
            if self.request is not None:
                self.request.release()
            if self.budget is not None:
                self.budget._streams.discard(self)
