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
    _flush_scheduled: bool = field(default=False, init=False)
    _batches_executed: int = field(default=0, init=False)
    _items_processed: int = field(default=0, init=False)

    async def submit(self, request: Req) -> Res:
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Res] = loop.create_future()
        async with self._lock:
            self._queue.append(_PendingItem(request=request, future=future))
            should_flush_now = len(self._queue) >= self.max_batch_size
            if should_flush_now:
                batch = self._drain()
            elif not self._flush_scheduled:
                self._flush_scheduled = True
                asyncio.create_task(self._delayed_flush())
                batch = None
            else:
                batch = None
        if batch is not None:
            await self._execute(batch)
        return await future

    def _drain(self) -> list[_PendingItem[Req, Res]]:
        batch, self._queue = self._queue, []
        self._flush_scheduled = False
        return batch

    async def _delayed_flush(self) -> None:
        await asyncio.sleep(self.window_ms / 1000)
        async with self._lock:
            batch = self._drain()
        if batch:
            await self._execute(batch)

    async def _execute(self, batch: list[_PendingItem[Req, Res]]) -> None:
        self._batches_executed += 1
        self._items_processed += len(batch)
        try:
            results = await self.batch_fn([item.request for item in batch])
            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)
        except Exception as exc:  # noqa: BLE001 - propagate to all waiters
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)

    @property
    def stats(self) -> dict:
        avg = self._items_processed / self._batches_executed if self._batches_executed else 0.0
        return {
            "batches_executed": self._batches_executed,
            "items_processed": self._items_processed,
            "avg_batch_size": avg,
        }
