import asyncio

import pytest

from backend.singleflight import SingleFlight


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


async def test_singleflight_exception_propagates_to_all_callers():
    sf: SingleFlight[str] = SingleFlight()

    async def failing_work():
        await asyncio.sleep(0.02)
        raise ValueError("simulated worker failure")

    with pytest.raises(ValueError, match="simulated worker failure"):
        await asyncio.gather(
            sf.execute("fail-key", failing_work),
            sf.execute("fail-key", failing_work),
        )
