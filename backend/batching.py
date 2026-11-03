"""Micro-batcher: coalesces concurrent single-item requests arriving within a
short window into one batch call to the underlying provider, trading a small
amount of added latency for much higher throughput under load.

This mirrors the batching strategy used by real inference servers (e.g.
dynamic/continuous batching in vLLM, Triton) at a scale appropriate for a
demo/portfolio service.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from backend.deadlines import InferenceDeadlineExceeded, observe

Req = TypeVar("Req")
Res = TypeVar("Res")


@dataclass
class _PendingItem(Generic[Req, Res]):
    request: Req
    future: asyncio.Future[Res]
    deadline: float | None = None
    on_finish: Callable[[], None] | None = None
    on_result: Callable[[Res], None] | None = None
    timer: asyncio.TimerHandle | None = None
    execution: _Execution | None = None
    finished: bool = False


@dataclass
class _Execution:
    items: list
    task: asyncio.Task | None = None
    error: Exception | None = None
    check: asyncio.TimerHandle | None = None


@dataclass
class MicroBatcher(Generic[Req, Res]):
    """Collects requests for up to `window_ms` (or until `max_batch_size` is
    reached, whichever comes first), then executes them all via
    `batch_fn` in a single call.
    """

    batch_fn: Callable[[list[Req]], Awaitable[list[Res]]]
    max_batch_size: int = 16
    window_ms: float = 10.0
    cancel_grace_s: float = 1.0
    on_stuck: Callable[[], None] | None = None

    _queue: list[_PendingItem[Req, Res]] = field(default_factory=list, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _close_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _flush_task: asyncio.Task[None] | None = field(default=None, init=False)
    _flush_tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False)
    _batch_tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False)
    _executions: dict[asyncio.Task, _Execution] = field(default_factory=dict, init=False)
    _closing: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False)
    _batches_executed: int = field(default=0, init=False)
    _items_processed: int = field(default=0, init=False)

    async def submit(
        self, request: Req, *, deadline: float | None = None,
        on_finish: Callable[[], None] | None = None,
        on_result: Callable[[Res], None] | None = None,
    ) -> Res:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Res] = loop.create_future()
        item = _PendingItem(request, future, deadline, on_finish, on_result)
        if on_finish is not None:
            future.add_done_callback(observe)
        try:
            async with self._lock:
                if self._closing:
                    raise RuntimeError("MicroBatcher is closing and no longer accepts submissions")
                if deadline is not None and deadline <= loop.time():
                    raise InferenceDeadlineExceeded()
                self._queue.append(item)
                if deadline is not None:
                    item.timer = loop.call_at(deadline, self._expire, item)
                if len(self._queue) >= self.max_batch_size:
                    self._cancel_flush_locked()
                    self._launch_batch_locked(self._drain())
                elif self._flush_task is None:
                    task = asyncio.create_task(self._delayed_flush())
                    self._flush_task = task
                    self._flush_tasks.add(task)
                    task.add_done_callback(self._flush_task_done)
        except BaseException:
            future.cancel()
            self._finish(item)
            raise
        # Managed items transfer release ownership to the actual batch task.
        # Cancelling a waiter must never make still-running provider work free.
        return await asyncio.shield(future) if on_finish is not None else await future

    def _finish(self, item: _PendingItem) -> None:
        if item.finished:
            return
        item.finished = True
        if item.timer is not None:
            item.timer.cancel()
        if item.on_finish is not None:
            item.on_finish()

    def _expire(self, item: _PendingItem) -> None:
        if item.finished:
            return
        if item.execution is not None:
            self._terminate(item.execution, InferenceDeadlineExceeded())
        else:
            self._queue.remove(item)
            self._fail_unresolved([item], InferenceDeadlineExceeded())
            self._finish(item)
            if not self._queue:
                self._cancel_flush_locked()

    def _terminate(self, execution: _Execution, error: Exception) -> None:
        if execution.error is not None:
            return
        execution.error = error
        self._fail_unresolved(execution.items, error)
        for item in execution.items:
            if item.timer is not None:
                item.timer.cancel()
        execution.task.cancel()
        if self.on_stuck is not None:
            def check() -> None:
                if not execution.task.done():
                    self.on_stuck()
            execution.check = asyncio.get_running_loop().call_later(self.cancel_grace_s, check)

    def abort(self) -> None:
        """Signal all work to stop; release callbacks still await actual termination."""
        self._closing = True
        self._cancel_flush_locked()
        for task in tuple(self._flush_tasks):
            task.cancel()
        for item in self._drain():
            self._fail_unresolved([item], InferenceDeadlineExceeded())
            self._finish(item)
        for task in tuple(self._batch_tasks):
            execution = self._executions[task]
            self._terminate(execution, InferenceDeadlineExceeded())

    async def _delayed_flush(self) -> None:
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(self.window_ms / 1000)
            async with self._lock:
                if self._closing:
                    return
                batch = self._drain()
                if self._flush_task is current_task:
                    self._flush_task = None
                if batch:
                    self._launch_batch_locked(batch)
        finally:
            async with self._lock:
                if self._flush_task is current_task:
                    self._flush_task = None

    def _drain(self) -> list[_PendingItem[Req, Res]]:
        batch, self._queue = self._queue, []
        return batch

    def _cancel_flush_locked(self) -> asyncio.Task[None] | None:
        task, self._flush_task = self._flush_task, None
        if task is not None and not task.done():
            task.cancel()
        return task

    def _launch_batch_locked(self, batch: list[_PendingItem[Req, Res]]) -> None:
        if not batch:
            return
        execution = _Execution(batch)
        for item in batch:
            item.execution = execution
        task = asyncio.create_task(self._execute(batch))
        execution.task = task
        self._executions[task] = execution
        self._batch_tasks.add(task)
        task.add_done_callback(self._batch_task_done)

    def _batch_task_done(self, task: asyncio.Task[None]) -> None:
        self._batch_tasks.discard(task)
        execution = self._executions.pop(task)
        if execution.check is not None:
            execution.check.cancel()
        self._fail_unresolved(execution.items, execution.error or InferenceDeadlineExceeded())
        for item in execution.items:
            self._finish(item)
        if not task.cancelled():
            # Observe unexpected internal errors as well as batch_fn failures,
            # which _execute already transfers to the request futures.
            task.exception()

    def _flush_task_done(self, task: asyncio.Task[None]) -> None:
        self._flush_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def _execute(self, batch: list[_PendingItem[Req, Res]]) -> None:
        self._batches_executed += 1
        self._items_processed += len(batch)
        try:
            if self._past_deadline(batch):
                raise InferenceDeadlineExceeded()
            results = await self.batch_fn([item.request for item in batch])
            if self._past_deadline(batch) or any(
                item.future.done() and not item.future.cancelled() for item in batch
            ):
                raise InferenceDeadlineExceeded()
            if len(results) != len(batch):
                raise ValueError(
                    "batch_fn returned "
                    f"{len(results)} results for {len(batch)} submitted items"
                )
        except asyncio.CancelledError as exc:
            self._fail_unresolved(batch, exc)
            raise
        except Exception as exc:  # noqa: BLE001 - propagate to all waiters
            error = InferenceDeadlineExceeded() if self._past_deadline(batch) else exc
            self._fail_unresolved(batch, error)
            return

        for item, result in zip(batch, results, strict=True):
            if not item.future.done():
                if item.on_result is not None:
                    item.on_result(result)
                item.future.set_result(result)

    @staticmethod
    def _past_deadline(batch: list[_PendingItem]) -> bool:
        now = asyncio.get_running_loop().time()
        return any(item.deadline is not None and item.deadline <= now for item in batch)

    @staticmethod
    def _fail_unresolved(batch: list[_PendingItem[Req, Res]], exc: BaseException) -> None:
        for item in batch:
            if not item.future.done():
                item.future.set_exception(exc)

    async def aclose(self) -> None:
        """Stop accepting work, flush accepted items, and await all batch tasks."""
        async with self._close_lock:
            if self._closed:
                return

            async with self._lock:
                self._closing = True
                self._cancel_flush_locked()
                flush_tasks = tuple(self._flush_tasks)
                for task in flush_tasks:
                    if not task.done():
                        task.cancel()
                self._launch_batch_locked(self._drain())

            if flush_tasks:
                await asyncio.gather(*flush_tasks, return_exceptions=True)

            # The delayed flush is now stopped and submissions are rejected,
            # so this snapshot contains every batch task shutdown must await.
            batch_tasks = tuple(self._batch_tasks)
            if batch_tasks:
                await asyncio.shield(
                    asyncio.gather(*batch_tasks, return_exceptions=True)
                )

            async with self._lock:
                self._closed = True

    @property
    def stats(self) -> dict:
        avg = self._items_processed / self._batches_executed if self._batches_executed else 0.0
        return {
            "batches_executed": self._batches_executed,
            "items_processed": self._items_processed,
            "avg_batch_size": avg,
        }
