import asyncio

import pytest

from backend.batching import MicroBatcher


async def _wait_for_queue_size(batcher: MicroBatcher, size: int) -> None:
    async with asyncio.timeout(1):
        while True:
            async with batcher._lock:
                if len(batcher._queue) >= size:
                    return
            await asyncio.sleep(0)


async def _wait_for_closing(batcher: MicroBatcher) -> None:
    async with asyncio.timeout(1):
        while not batcher._closing:
            await asyncio.sleep(0)


async def test_submit_returns_results_in_submission_order() -> None:
    seen_batches: list[list[int]] = []

    async def batch_fn(requests: list[int]) -> list[str]:
        seen_batches.append(requests)
        return [f"result-{request}" for request in requests]

    batcher = MicroBatcher(batch_fn, max_batch_size=3, window_ms=60_000)
    submissions = [asyncio.create_task(batcher.submit(value)) for value in (3, 1, 2)]

    results = await asyncio.wait_for(asyncio.gather(*submissions), timeout=1)

    assert results == ["result-3", "result-1", "result-2"]
    assert seen_batches == [[3, 1, 2]]
    await batcher.aclose()
    assert batcher._flush_tasks == set()


@pytest.mark.parametrize("result_count", [1, 4], ids=["too-few", "too-many"])
async def test_result_count_mismatch_fails_every_submission(result_count: int) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def batch_fn(requests: list[int]) -> list[int]:
        started.set()
        await release.wait()
        return list(range(result_count))

    batcher = MicroBatcher(batch_fn, max_batch_size=3, window_ms=60_000)
    submissions = [asyncio.create_task(batcher.submit(value)) for value in range(3)]
    await asyncio.wait_for(started.wait(), timeout=1)
    release.set()

    outcomes = await asyncio.wait_for(
        asyncio.gather(*submissions, return_exceptions=True), timeout=1
    )

    assert all(isinstance(outcome, ValueError) for outcome in outcomes)
    assert all(
        f"returned {result_count} results for 3 submitted items" in str(outcome)
        for outcome in outcomes
    )
    assert batcher.stats["items_processed"] == 3
    await batcher.aclose()


async def test_batch_function_exception_fails_every_submission() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def batch_fn(requests: list[int]) -> list[int]:
        started.set()
        await release.wait()
        raise RuntimeError("batch failed")

    batcher = MicroBatcher(batch_fn, max_batch_size=3, window_ms=60_000)
    submissions = [asyncio.create_task(batcher.submit(value)) for value in range(3)]
    await asyncio.wait_for(started.wait(), timeout=1)
    release.set()

    outcomes = await asyncio.wait_for(
        asyncio.gather(*submissions, return_exceptions=True), timeout=1
    )

    assert all(isinstance(outcome, RuntimeError) for outcome in outcomes)
    assert all(str(outcome) == "batch failed" for outcome in outcomes)
    await batcher.aclose()


async def test_failed_batch_does_not_poison_later_batch() -> None:
    attempts = 0

    async def batch_fn(requests: list[str]) -> list[str]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("first batch failed")
        return [request.upper() for request in requests]

    batcher = MicroBatcher(batch_fn, max_batch_size=2, window_ms=60_000)
    failed_submissions = [
        asyncio.create_task(batcher.submit(value)) for value in ("bad-a", "bad-b")
    ]
    failed = await asyncio.wait_for(
        asyncio.gather(*failed_submissions, return_exceptions=True), timeout=1
    )
    assert all(isinstance(outcome, RuntimeError) for outcome in failed)

    later_submission = asyncio.create_task(batcher.submit("good"))
    await _wait_for_queue_size(batcher, 1)
    await asyncio.wait_for(batcher.aclose(), timeout=1)

    assert await asyncio.wait_for(later_submission, timeout=1) == "GOOD"
    assert attempts == 2


async def test_cancelled_submitter_does_not_interrupt_batch_completion() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    completed: list[list[str]] = []

    async def batch_fn(requests: list[str]) -> list[str]:
        started.set()
        await release.wait()
        completed.append(requests)
        return ["done"]

    batcher = MicroBatcher(batch_fn, max_batch_size=1)
    submission = asyncio.create_task(batcher.submit("request"))
    await asyncio.wait_for(started.wait(), timeout=1)

    submission.cancel()
    with pytest.raises(asyncio.CancelledError):
        await submission

    release.set()
    await asyncio.wait_for(batcher.aclose(), timeout=1)

    assert completed == [["request"]]
    assert batcher._flush_tasks == set()
    assert batcher._batch_tasks == set()


async def test_shutdown_flushes_accepted_queued_submissions() -> None:
    async def batch_fn(requests: list[str]) -> list[str]:
        return [request.upper() for request in requests]

    batcher = MicroBatcher(batch_fn, max_batch_size=10, window_ms=60_000)
    submissions = [asyncio.create_task(batcher.submit(value)) for value in ("a", "b", "c")]
    await _wait_for_queue_size(batcher, 3)

    await asyncio.wait_for(batcher.aclose(), timeout=1)
    results = await asyncio.wait_for(asyncio.gather(*submissions), timeout=1)

    assert results == ["A", "B", "C"]
    assert batcher._queue == []
    assert batcher._flush_task is None
    assert batcher._flush_tasks == set()
    assert batcher._batch_tasks == set()


async def test_shutdown_waits_for_executing_batch_and_rejects_new_work() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def batch_fn(requests: list[str]) -> list[str]:
        started.set()
        await release.wait()
        return ["finished"]

    batcher = MicroBatcher(batch_fn, max_batch_size=1)
    submission = asyncio.create_task(batcher.submit("accepted"))
    await asyncio.wait_for(started.wait(), timeout=1)
    closing = asyncio.create_task(batcher.aclose())
    await _wait_for_closing(batcher)

    with pytest.raises(RuntimeError, match="no longer accepts submissions"):
        await batcher.submit("rejected")

    release.set()
    await asyncio.wait_for(closing, timeout=1)
    assert await asyncio.wait_for(submission, timeout=1) == "finished"
    assert batcher._batch_tasks == set()
    assert batcher._flush_tasks == set()
    assert batcher._queue == []


async def test_concurrent_and_repeated_shutdown_execute_pending_batch_once() -> None:
    calls: list[list[str]] = []

    async def batch_fn(requests: list[str]) -> list[str]:
        calls.append(requests)
        return [request.upper() for request in requests]

    batcher = MicroBatcher(batch_fn, max_batch_size=10, window_ms=60_000)
    submission = asyncio.create_task(batcher.submit("once"))
    await _wait_for_queue_size(batcher, 1)

    await asyncio.wait_for(asyncio.gather(batcher.aclose(), batcher.aclose()), timeout=1)
    assert await asyncio.wait_for(submission, timeout=1) == "ONCE"
    await asyncio.wait_for(batcher.aclose(), timeout=1)

    assert calls == [["once"]]
    assert batcher._flush_task is None
    assert batcher._flush_tasks == set()
    assert batcher._batch_tasks == set()


async def test_application_shutdown_closes_gateway_batcher(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend import main

    closed = asyncio.Event()

    class GatewayStub:
        async def aclose(self) -> None:
            closed.set()

    monkeypatch.setattr(main, "gateway", GatewayStub())
    async with asyncio.timeout(1):
        async with main.app.router.lifespan_context(main.app):
            pass

    assert closed.is_set()
