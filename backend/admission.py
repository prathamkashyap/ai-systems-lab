"""Single-event-loop admission budgets for the public demonstration."""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator

from backend.deadlines import InferenceDeadlineExceeded, ShutdownIncomplete, observe, settled


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
        self.transferred = False
        self.deadline = asyncio.get_running_loop().time() + budget.work_timeout_s
        self.ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.ready.add_done_callback(observe)
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

    def __init__(
        self, max_active: int = 4, max_pending: int = 16, max_requests: int = 64, *,
        work_timeout_s: float = 10, stream_timeout_s: float = 15,
        shutdown_grace_s: float = 5, cancel_grace_s: float = 1,
    ) -> None:
        if max_active < 1 or max_pending < 0 or max_requests < 1:
            raise ValueError("Admission limits require active >= 1, pending >= 0, requests >= 1")
        self.max_active = max_active
        self.max_pending = max_pending
        self.max_requests = max_requests
        for value in (work_timeout_s, stream_timeout_s, shutdown_grace_s, cancel_grace_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Deadline durations must be finite and positive")
        self.work_timeout_s = work_timeout_s
        self.stream_timeout_s = stream_timeout_s
        self.shutdown_grace_s = shutdown_grace_s
        self.cancel_grace_s = cancel_grace_s
        self.active = 0
        self.requests = 0
        self.closing = False
        self._pending: deque[WorkPermit] = deque()
        self._streams: set[AdmittedStream] = set()
        self._response_tasks: set[asyncio.Task] = set()

    def fail_closed(self) -> None:
        self.closing = True
        logging.getLogger(__name__).critical(
            "Inference cancellation cleanup exceeded its bound; supervisor termination required"
        )

    def expire_pending(self) -> None:
        for permit in tuple(self._pending):
            if not permit.ready.done():
                permit.ready.set_exception(InferenceDeadlineExceeded())
            permit.release()

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
        self.expired = False
        self.response_owned = False
        self.deadline: float | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._step_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        if budget is not None:
            self.request = budget.reserve_request()
            try:
                self.work = budget.reserve_work(1, queue=False)[0]
            except BaseException:
                self.request.release()
                raise
            budget._streams.add(self)
            self.deadline = asyncio.get_running_loop().time() + budget.stream_timeout_s
            self._timer = asyncio.get_running_loop().call_at(self.deadline, self._expire)

    def _expire(self) -> None:
        self.expired = True
        self._ensure_cleanup()

    def _ensure_cleanup(self) -> asyncio.Task:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup())
            self._cleanup_task.add_done_callback(observe)
        return self._cleanup_task

    async def _cleanup(self) -> None:
        # This tracked owner retains capacity if cancellation is resisted.
        # Observers (including shutdown) always use bounded waits on this task.
        if self._step_task is not None:
            if not self._step_task.done():
                self._step_task.cancel()
            await asyncio.gather(self._step_task, return_exceptions=True)
        await self.iterator.aclose()
        self.closed = True
        if self._timer is not None:
            self._timer.cancel()
        if self.work is not None:
            self.work.release()
        if not self.response_owned or self.request.released:
            self.finish_response()

    def finish_response(self) -> None:
        if self.request is not None:
            self.request.release()
        if self.budget is not None and self.closed:
            self.budget._streams.discard(self)

    def start(self) -> None:
        if self.expired or (self.deadline is not None and self.deadline <= asyncio.get_running_loop().time()):
            raise InferenceDeadlineExceeded()
        if self.closed:
            raise AdmissionRejected("Public demo capacity unavailable")
        if self.owner is None:
            self.owner = asyncio.current_task()

    async def __anext__(self) -> str:
        self.start()
        if self.budget is None:
            return await anext(self.iterator)
        self._step_task = asyncio.create_task(anext(self.iterator))
        self._step_task.add_done_callback(observe)
        remaining = max(0, self.deadline - asyncio.get_running_loop().time())
        if not await settled({self._step_task}, remaining):
            self._expire()
            raise InferenceDeadlineExceeded()
        if self.expired:
            raise InferenceDeadlineExceeded()
        return self._step_task.result()

    async def aclose(self) -> None:
        if self.budget is None:
            await self.iterator.aclose()
            self.closed = True
            return
        task = self._ensure_cleanup()
        if not await settled({task}, self.budget.cancel_grace_s):
            self.budget.fail_closed()
            raise ShutdownIncomplete()
        task.result()
