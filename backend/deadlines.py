"""Sanitized deadline errors and bounded observation of owned tasks."""

from __future__ import annotations

import asyncio


class InferenceDeadlineExceeded(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Inference deadline exceeded")


class ShutdownIncomplete(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Inference cleanup incomplete; supervisor must terminate this process")


class StreamFinalizationError(RuntimeError):
    """The owned generator terminated, but reported a finalization failure."""

    def __init__(self) -> None:
        super().__init__("Stream finalization failed")


def observe(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()


async def settled(tasks: set[asyncio.Task], seconds: float) -> bool:
    """Bound the observer, without pretending to kill cancellation-resistant code."""
    if not tasks:
        return True
    _, pending = await asyncio.wait(tasks, timeout=seconds)
    return not pending
