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

Req = TypeVar("Req")
Res = TypeVar("Res")


@dataclass
class _PendingItem(Generic[Req, Res]):
    request: Req
    future: asyncio.Future[Res]


@dataclass
class MicroBatcher(Generic[Req, Res]):
    """Collects requests for up to `window_ms` (or until `max_batch_size` is
    reached, whichever comes first), then executes them all via
    `batch_fn` in a single call.
    """

    batch_fn: Callable[[list[Req]], Awaitable[list[Res]]]
    max_batch_size: int = 16
    window_ms: float = 10.0

    _queue: list[_PendingItem[Req, Res]] = field(default_factory=list, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _close_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _flush_task: asyncio.Task[None] | None = field(default=None, init=False)
    _flush_tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False)
    _batch_tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False)
    _closing: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False)
    _batches_executed: int = field(default=0, init=False)
    _items_processed: int = field(default=0, init=False)

    async def submit(self, request: Req) -> Res:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Res] = loop.create_future()
        async with self._lock:
            if self._closing:
                raise RuntimeError("MicroBatcher is closing and no longer accepts submissions")
            self._queue.append(_PendingItem(request=request, future=future))
            if len(self._queue) >= self.max_batch_size:
                self._cancel_flush_locked()
                self._launch_batch_locked(self._drain())
            elif self._flush_task is None:
                task = asyncio.create_task(self._delayed_flush())
                self._flush_task = task
                self._flush_tasks.add(task)
                task.add_done_callback(self._flush_task_done)
        return await future

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
        task = asyncio.create_task(self._execute(batch))
        self._batch_tasks.add(task)
        task.add_done_callback(self._batch_task_done)

    def _batch_task_done(self, task: asyncio.Task[None]) -> None:
        self._batch_tasks.discard(task)
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
            results = await self.batch_fn([item.request for item in batch])
            if len(results) != len(batch):
                raise ValueError(
                    "batch_fn returned "
                    f"{len(results)} results for {len(batch)} submitted items"
                )
        except asyncio.CancelledError as exc:
            self._fail_unresolved(batch, exc)
            raise
        except Exception as exc:  # noqa: BLE001 - propagate to all waiters
            self._fail_unresolved(batch, exc)
            return

        for item, result in zip(batch, results, strict=True):
            if not item.future.done():
                item.future.set_result(result)

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
