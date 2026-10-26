import asyncio

import pytest

from backend.singleflight import SingleFlight


async def _wait_for_shared_calls(sf: SingleFlight, expected: int) -> None:
    async def wait_for_count() -> None:
        while sf.shared_calls < expected:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_count(), timeout=1)


async def test_singleflight_coalesces_concurrent_calls():
    sf: SingleFlight[str] = SingleFlight()
    execution_count = 0

    async def slow_work():
        nonlocal execution_count
        execution_count += 1
        await asyncio.sleep(0.05)
        return "computed-result"

    results = await asyncio.gather(
        sf.execute("key-1", slow_work),
        sf.execute("key-1", slow_work),
        sf.execute("key-1", slow_work),
        sf.execute("key-1", slow_work),
    )

    # Exactly one execution occurred
    assert execution_count == 1
    assert all(r[0] == "computed-result" for r in results)

    # 1 initiator (is_shared=False), 3 shared (is_shared=True)
    shared_flags = [r[1] for r in results]
    assert shared_flags.count(False) == 1
    assert shared_flags.count(True) == 3
    assert sf.deduplication_rate == 0.75


async def test_singleflight_different_keys_execute_independently():
    sf: SingleFlight[str] = SingleFlight()
    execution_count = 0

    async def work(val: str):
        nonlocal execution_count
        execution_count += 1
        await asyncio.sleep(0.01)
        return f"result-{val}"

    results = await asyncio.gather(
        sf.execute("key-A", lambda: work("A")),
        sf.execute("key-B", lambda: work("B")),
    )

    assert execution_count == 2
    assert results[0][0] == "result-A"
    assert results[1][0] == "result-B"
    assert results[0][1] is False
    assert results[1][1] is False


async def test_singleflight_exception_propagates_to_active_callers_and_retries():
    sf: SingleFlight[str] = SingleFlight()
    started = asyncio.Event()
    release = asyncio.Event()
    execution_count = 0

    async def failing_work():
        nonlocal execution_count
        execution_count += 1
        started.set()
        await release.wait()
        raise ValueError("simulated worker failure")

    initiator = asyncio.create_task(sf.execute("fail-key", failing_work))
    await asyncio.wait_for(started.wait(), timeout=1)
    joiner = asyncio.create_task(sf.execute("fail-key", failing_work))
    cancelled_joiner = asyncio.create_task(sf.execute("fail-key", failing_work))
    await _wait_for_shared_calls(sf, 2)

    cancelled_joiner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cancelled_joiner, timeout=1)

    release.set()
    results = await asyncio.wait_for(
        asyncio.gather(initiator, joiner, return_exceptions=True), timeout=1
    )

    assert all(isinstance(result, ValueError) for result in results)
    assert all(str(result) == "simulated worker failure" for result in results)
    assert execution_count == 1
    assert sf.in_flight_count == 0

    async def retry_work():
        nonlocal execution_count
        execution_count += 1
        return "retry-result"

    assert await asyncio.wait_for(sf.execute("fail-key", retry_work), timeout=1) == (
        "retry-result",
        False,
    )
    assert execution_count == 2
    assert sf.in_flight_count == 0


async def test_singleflight_cancelling_joiner_preserves_other_callers():
    sf: SingleFlight[str] = SingleFlight()
    started = asyncio.Event()
    release = asyncio.Event()
    execution_count = 0

    async def slow_work():
        nonlocal execution_count
        execution_count += 1
        started.set()
        await release.wait()
        return "computed-result"

    initiator = asyncio.create_task(sf.execute("cancel-joiner-key", slow_work))
    await asyncio.wait_for(started.wait(), timeout=1)
    cancelled_joiner = asyncio.create_task(sf.execute("cancel-joiner-key", slow_work))
    remaining_joiner = asyncio.create_task(sf.execute("cancel-joiner-key", slow_work))
    await _wait_for_shared_calls(sf, 2)

    cancelled_joiner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cancelled_joiner, timeout=1)

    release.set()
    results = await asyncio.wait_for(
        asyncio.gather(initiator, remaining_joiner), timeout=1
    )

    assert results == [("computed-result", False), ("computed-result", True)]
    assert execution_count == 1
    assert sf.in_flight_count == 0


async def test_singleflight_cancelling_initiator_preserves_joiner():
    sf: SingleFlight[str] = SingleFlight()
    started = asyncio.Event()
    release = asyncio.Event()
    execution_count = 0

    async def slow_work():
        nonlocal execution_count
        execution_count += 1
        started.set()
        await release.wait()
        return "computed-result"

    initiator = asyncio.create_task(sf.execute("cancel-initiator-key", slow_work))
    await asyncio.wait_for(started.wait(), timeout=1)
    joiner = asyncio.create_task(sf.execute("cancel-initiator-key", slow_work))
    await _wait_for_shared_calls(sf, 1)

    initiator.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(initiator, timeout=1)

    release.set()
    assert await asyncio.wait_for(joiner, timeout=1) == ("computed-result", True)
    assert execution_count == 1
    assert sf.in_flight_count == 0


async def test_singleflight_work_finishes_after_all_callers_cancel_without_warning():
    sf: SingleFlight[str] = SingleFlight()
    started = asyncio.Event()
    release = asyncio.Event()
    about_to_fail = asyncio.Event()
    execution_count = 0
    loop = asyncio.get_running_loop()
    loop_errors = []
    previous_exception_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

    async def failing_work():
        nonlocal execution_count
        execution_count += 1
        started.set()
        await release.wait()
        about_to_fail.set()
        raise ValueError("unobserved worker failure")

    try:
        initiator = asyncio.create_task(sf.execute("all-cancel-key", failing_work))
        await asyncio.wait_for(started.wait(), timeout=1)
        joiner = asyncio.create_task(sf.execute("all-cancel-key", failing_work))
        await _wait_for_shared_calls(sf, 1)

        initiator.cancel()
        joiner.cancel()
        for caller in (initiator, joiner):
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(caller, timeout=1)

        release.set()
        await asyncio.wait_for(about_to_fail.wait(), timeout=1)

        async def wait_for_cleanup() -> None:
            while sf.in_flight_count:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_cleanup(), timeout=1)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    finally:
        release.set()
        loop.set_exception_handler(previous_exception_handler)

    assert execution_count == 1
    assert sf.in_flight_count == 0
    assert loop_errors == []
