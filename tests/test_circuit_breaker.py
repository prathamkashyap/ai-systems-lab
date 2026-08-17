import asyncio

import pytest

from backend.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


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
