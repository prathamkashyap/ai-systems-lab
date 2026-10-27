import asyncio
import time

import pytest

from backend.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


async def _open_and_expire(cb: CircuitBreaker) -> None:
    async def fail():
        raise RuntimeError("initial provider failure")

    with pytest.raises(RuntimeError, match="initial provider failure"):
        await cb.call(fail)
    assert cb.state == CircuitState.OPEN

    # Drive the existing lazy timeout transition deterministically without
    # adding a clock-injection API to this focused fix.
    async with cb._lock:
        cb._last_state_change = time.monotonic() - cb.recovery_timeout - 1


async def test_circuit_breaker_closed_under_normal_operation():
    cb = CircuitBreaker[str](name="test", failure_threshold=2, recovery_timeout=0.1)
    assert cb.state == CircuitState.CLOSED
    assert cb.is_available is True

    result = await cb.call(lambda: asyncio.sleep(0.001, result="ok"))
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED


async def test_circuit_breaker_trips_to_open_after_failures():
    cb = CircuitBreaker[str](name="test", failure_threshold=2, recovery_timeout=0.05)

    async def fail():
        raise RuntimeError("provider error")

    with pytest.raises(RuntimeError):
        await cb.call(fail)
    assert cb.state == CircuitState.CLOSED

    # Second failure reaches threshold -> trips to OPEN
    with pytest.raises(RuntimeError):
        await cb.call(fail)
    assert cb.state == CircuitState.OPEN
    assert cb.is_available is False

    # Subsequent call immediately fast-fails with CircuitOpenError
    with pytest.raises(CircuitOpenError):
        await cb.call(lambda: asyncio.sleep(0.001, result="ok"))


async def test_circuit_breaker_recovers_via_half_open():
    cb = CircuitBreaker[str](
        name="test", failure_threshold=1, recovery_timeout=0.05, half_open_success_threshold=1
    )

    async def fail():
        raise RuntimeError("err")

    with pytest.raises(RuntimeError):
        await cb.call(fail)
    assert cb.state == CircuitState.OPEN

    # Wait for recovery cooldown
    await asyncio.sleep(0.06)
    assert cb.state == CircuitState.HALF_OPEN

    # Probe call succeeds -> closes circuit
    res = await cb.call(lambda: asyncio.sleep(0.001, result="probe-success"))
    assert res == "probe-success"
    assert cb.state == CircuitState.CLOSED


async def test_half_open_allows_only_one_concurrent_probe_and_keeps_success_threshold():
    cb = CircuitBreaker[str](
        name="test", failure_threshold=1, recovery_timeout=60, half_open_success_threshold=2
    )
    await _open_and_expire(cb)

    started = asyncio.Event()
    release = asyncio.Event()
    protected_calls = 0

    async def protected():
        nonlocal protected_calls
        protected_calls += 1
        if protected_calls == 1:
            started.set()
            await release.wait()
        return f"probe-{protected_calls}"

    probe = asyncio.create_task(cb.call(protected))
    await asyncio.wait_for(started.wait(), timeout=1)

    rejected = await asyncio.wait_for(
        asyncio.gather(
            *(cb.call(protected) for _ in range(10)), return_exceptions=True
        ),
        timeout=1,
    )
    assert all(isinstance(result, CircuitOpenError) for result in rejected)
    assert protected_calls == 1
    assert cb._consecutive_failures == 1  # rejected calls are not provider failures

    release.set()
    assert await asyncio.wait_for(probe, timeout=1) == "probe-1"
    assert cb.state == CircuitState.HALF_OPEN

    # A second, sequential successful probe satisfies the configured threshold.
    assert await asyncio.wait_for(cb.call(protected), timeout=1) == "probe-2"
    assert protected_calls == 2
    assert cb.state == CircuitState.CLOSED
    assert cb._half_open_probe_in_flight is False


async def test_failed_half_open_probe_reopens_and_allows_later_recovery():
    cb = CircuitBreaker[str](
        name="test", failure_threshold=1, recovery_timeout=60, half_open_success_threshold=1
    )
    await _open_and_expire(cb)

    started = asyncio.Event()
    release = asyncio.Event()
    protected_calls = 0

    async def fail_probe():
        nonlocal protected_calls
        protected_calls += 1
        started.set()
        await release.wait()
        raise RuntimeError("probe failed")

    probe = asyncio.create_task(cb.call(fail_probe))
    await asyncio.wait_for(started.wait(), timeout=1)
    with pytest.raises(CircuitOpenError):
        await cb.call(lambda: asyncio.sleep(0, result="must not run"))
    release.set()

    with pytest.raises(RuntimeError, match="probe failed"):
        await asyncio.wait_for(probe, timeout=1)
    assert cb.state == CircuitState.OPEN
    assert cb._half_open_probe_in_flight is False
    assert cb._consecutive_failures == 2
    with pytest.raises(CircuitOpenError):
        await cb.call(lambda: asyncio.sleep(0, result="must not run"))
    assert protected_calls == 1

    # After another recovery window, a new probe can be admitted and close it.
    async with cb._lock:
        cb._last_state_change = time.monotonic() - cb.recovery_timeout - 1
    assert (
        await asyncio.wait_for(cb.call(lambda: asyncio.sleep(0, result="recovered")), timeout=1)
        == "recovered"
    )
    assert cb.state == CircuitState.CLOSED


async def test_cancelled_half_open_probe_releases_permit_without_counting_failure():
    cb = CircuitBreaker[str](
        name="test", failure_threshold=1, recovery_timeout=60, half_open_success_threshold=1
    )
    await _open_and_expire(cb)

    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_probe():
        started.set()
        await release.wait()
        return "probe result"

    probe = asyncio.create_task(cb.call(blocked_probe))
    await asyncio.wait_for(started.wait(), timeout=1)
    probe.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(probe, timeout=1)

    assert cb._half_open_probe_in_flight is False
    assert cb._consecutive_failures == 1
    assert (
        await asyncio.wait_for(cb.call(lambda: asyncio.sleep(0, result="recovered")), timeout=1)
        == "recovered"
    )
    assert cb.state == CircuitState.CLOSED
