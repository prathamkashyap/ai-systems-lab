import asyncio
import time

import pytest

from backend.admission import AdmissionBudget, AdmissionRejected
from backend.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from backend.deadlines import InferenceDeadlineExceeded
from backend.gateway import Gateway
from backend.models import CompletionRequest
from backend.providers.mock_provider import MockProvider


class ObservedLock(asyncio.Lock):
    def __init__(self):
        super().__init__()
        self.contended = asyncio.Event()

    async def acquire(self):
        if self.locked():
            self.contended.set()
        return await super().acquire()


class ProbeOperation:
    def __init__(self):
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.stopped = asyncio.Event()
        self.cleanup = asyncio.Event()
        self.cleanup.set()
        self.cancellations = asyncio.Queue(maxsize=8)

    async def run(self):
        self.calls += 1
        call = self.calls
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            await self.release.wait()
            return f"probe-{call}"
        except asyncio.CancelledError:
            self.cancellations.put_nowait(None)
            while not self.cleanup.is_set():
                try:
                    await self.cleanup.wait()
                except asyncio.CancelledError:
                    self.cancellations.put_nowait(None)
            raise
        finally:
            self.active -= 1
            self.stopped.set()


class ProbeProvider(MockProvider):
    def __init__(self, name):
        super().__init__(0, 0)
        self.name = name
        self.operation = ProbeOperation()

    async def complete(self, *args, **kwargs):
        return await self.operation.run(), 1, 1


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


async def test_cancelled_closed_call_does_not_count_as_provider_failure():
    cb = CircuitBreaker[str](name="test", failure_threshold=1)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_operation():
        started.set()
        await release.wait()
        return "cancelled result"

    call = asyncio.create_task(cb.call(blocked_operation))
    await asyncio.wait_for(started.wait(), timeout=1)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(call, timeout=1)

    assert cb.state == CircuitState.CLOSED
    assert cb._consecutive_failures == 0
    assert await asyncio.wait_for(cb.call(lambda: asyncio.sleep(0, result="ok")), timeout=1) == "ok"
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

    provider = ProbeOperation()
    probe = asyncio.create_task(cb.call(provider.run))
    await asyncio.wait_for(provider.started.wait(), timeout=1)

    rejected = await asyncio.wait_for(
        asyncio.gather(
            *(cb.call(provider.run) for _ in range(10)), return_exceptions=True
        ),
        timeout=1,
    )
    assert all(isinstance(result, CircuitOpenError) for result in rejected)
    assert provider.calls == provider.active == provider.max_active == 1
    assert cb._consecutive_failures == 1  # rejected calls are not provider failures

    provider.release.set()
    assert await asyncio.wait_for(probe, timeout=1) == "probe-1"
    assert cb.state == CircuitState.HALF_OPEN

    # A second, sequential successful probe satisfies the configured threshold.
    assert await asyncio.wait_for(cb.call(provider.run), timeout=1) == "probe-2"
    assert provider.calls == 2 and provider.active == 0 and provider.max_active == 1
    assert cb.state == CircuitState.CLOSED
    assert cb._half_open_probe_in_flight is False


async def test_failed_half_open_probe_reopens_and_allows_later_recovery():
    cb = CircuitBreaker[str](
        name="test", failure_threshold=1, recovery_timeout=60, half_open_success_threshold=1
    )
    await _open_and_expire(cb)

    provider = ProbeOperation()

    async def fail_probe():
        await provider.run()
        raise RuntimeError("probe failed")

    probe = asyncio.create_task(cb.call(fail_probe))
    await asyncio.wait_for(provider.started.wait(), timeout=1)
    with pytest.raises(CircuitOpenError):
        await cb.call(lambda: asyncio.sleep(0, result="must not run"))
    provider.release.set()

    with pytest.raises(RuntimeError, match="probe failed"):
        await asyncio.wait_for(probe, timeout=1)
    assert cb.state == CircuitState.OPEN
    assert cb._half_open_probe_in_flight is False
    assert cb._consecutive_failures == 2
    with pytest.raises(CircuitOpenError):
        await cb.call(lambda: asyncio.sleep(0, result="must not run"))
    assert provider.calls == 1 and provider.active == 0 and provider.max_active == 1

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


@pytest.mark.parametrize("repeat_cancel", [False, True])
async def test_probe_cleanup_survives_contended_lock_and_repeated_cancellation(repeat_cancel):
    async with asyncio.timeout(2):
        cb = CircuitBreaker[str](failure_threshold=1, recovery_timeout=60, half_open_success_threshold=1)
        await _open_and_expire(cb)
        cb._lock = lock = ObservedLock()
        provider = ProbeOperation()
        baseline = asyncio.all_tasks()
        task = asyncio.create_task(cb.call(provider.run))
        try:
            await provider.started.wait()
            await lock.acquire()  # force the original release helper to suspend
            task.cancel()
            await provider.stopped.wait()
            assert provider.active == 0  # not merely cancellation requested
            assert task.done() or lock.contended.is_set()
            if repeat_cancel:
                # The old implementation is now waiting in cleanup; synchronous
                # cleanup may already have finished, which is also safe.
                task.cancel()
            else:
                lock.release()
            with pytest.raises(asyncio.CancelledError):
                await task
            if lock.locked():
                lock.release()
            permit_stuck = cb._half_open_probe_in_flight
            failures = cb._consecutive_failures
            state = cb.state
            provider.release.set()
            try:
                recovered = await cb.call(provider.run)
            except CircuitOpenError:
                recovered = "rejected"
            assert (permit_stuck, failures, state, recovered, provider.calls, provider.max_active) == (
                False, 1, CircuitState.HALF_OPEN, "probe-2", 2, 1
            ), f"cleanup_lock_waited={lock.contended.is_set()}, provider_active={provider.active}"
            assert cb.state == CircuitState.CLOSED
            snapshot = cb.snapshot()
            cb._release_half_open_probe()
            cb._release_half_open_probe()  # already released: no state/counter changes
            assert cb.snapshot() == snapshot
            assert asyncio.all_tasks() == baseline
        finally:
            if lock.locked():
                lock.release()
            provider.release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("cancellations", [2, 5])
async def test_repeated_cancellation_retains_probe_until_provider_cleanup_finishes(cancellations):
    async with asyncio.timeout(2):
        cb = CircuitBreaker[str](
            failure_threshold=1, recovery_timeout=60, half_open_success_threshold=1
        )
        await _open_and_expire(cb)
        failed_at = cb._last_failure_time
        provider = ProbeOperation()
        provider.cleanup.clear()
        baseline = asyncio.all_tasks()
        task = asyncio.create_task(cb.call(provider.run))
        try:
            await provider.started.wait()
            for _ in range(cancellations):
                task.cancel()
                await provider.cancellations.get()  # cancellation reached live cleanup
                assert not task.done() and provider.active == 1
                assert cb._half_open_probe_in_flight
                rejected = await asyncio.gather(
                    *(cb.call(provider.run) for _ in range(10)), return_exceptions=True
                )
                assert all(isinstance(result, CircuitOpenError) for result in rejected)
                assert provider.calls == provider.max_active == 1
                assert cb._consecutive_failures == 1 and cb._last_failure_time == failed_at
            provider.cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert provider.active == 0 and not cb._half_open_probe_in_flight
            assert cb.state == CircuitState.HALF_OPEN and cb._consecutive_successes == 0
            provider.release.set()
            assert await cb.call(provider.run) == "probe-2"
            assert cb.state == CircuitState.CLOSED and provider.max_active == 1
            assert asyncio.all_tasks() == baseline
        finally:
            provider.cleanup.set()
            provider.release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_cancellation_during_success_bookkeeping_releases_stopped_probe():
    async with asyncio.timeout(2):
        cb = CircuitBreaker[str](
            failure_threshold=1, recovery_timeout=60, half_open_success_threshold=1
        )
        await _open_and_expire(cb)
        cb._lock = lock = ObservedLock()
        provider = ProbeOperation()
        task = asyncio.create_task(cb.call(provider.run))
        try:
            await provider.started.wait()
            await lock.acquire()
            provider.release.set()
            await provider.stopped.wait()
            await lock.contended.wait()  # provider finished, success update still waits
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task  # release must not deadlock on the still-held lock
            assert provider.active == 0 and not cb._half_open_probe_in_flight
            assert cb._consecutive_failures == 1 and cb._consecutive_successes == 0
            lock.release()
            assert await cb.call(provider.run) == "probe-2"
            assert provider.max_active == 1 and cb.state == CircuitState.CLOSED
        finally:
            if lock.locked():
                lock.release()
            provider.release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("mode", ["developer", "public-demo"])
async def test_gateway_falls_back_while_one_half_open_probe_is_active(mode):
    async with asyncio.timeout(3):
        primary, fallback = ProbeProvider("primary"), ProbeProvider("fallback")
        fallback.operation.release.set()
        admission = AdmissionBudget(max_active=3, max_pending=0) if mode == "public-demo" else None
        gateway = Gateway([primary, fallback], admission=admission, max_batch_size=1)
        cb = gateway._circuit_breakers[primary.name]
        cb.failure_threshold = 1
        await _open_and_expire(cb)
        first = asyncio.create_task(gateway.complete(CompletionRequest(prompt="first", cache=False)))
        try:
            await primary.operation.started.wait()
            other = await gateway.complete(CompletionRequest(prompt="other", cache=False))
            assert other.provider == fallback.name
            assert primary.operation.calls == primary.operation.active == primary.operation.max_active == 1
            assert fallback.operation.calls == 1 and cb._consecutive_failures == 1
            primary.operation.release.set()
            assert (await first).provider == primary.name
            assert cb.state == CircuitState.HALF_OPEN  # default threshold still requires two successes
            recovered = await gateway.complete(CompletionRequest(prompt="recovered", cache=False))
            assert recovered.provider == primary.name and cb.state == CircuitState.CLOSED
            assert primary.operation.calls == 2 and primary.operation.max_active == 1
        finally:
            primary.operation.release.set()
            await asyncio.gather(first, return_exceptions=True)
            await gateway.aclose()
        # Developer close may return with completed task callbacks still queued.
        # Observe those callbacks explicitly, rather than sleeping or changing
        # the batcher's lifecycle as part of this breaker-only correction.
        completed = tuple(gateway._batcher._batch_tasks)
        assert all(task.done() for task in completed)
        for task in completed:
            after_callbacks = asyncio.get_running_loop().create_future()
            task.add_done_callback(
                lambda _, done=after_callbacks: None if done.done() else done.set_result(None)
            )
            await after_callbacks
        assert gateway._batcher._batch_tasks == set() and gateway._singleflight.in_flight_count == 0
        if admission is not None:
            assert (admission.active, admission.pending, admission.requests) == (0, 0, 0)


async def test_public_deadline_retains_probe_and_work_until_repeated_cancel_cleanup_stops():
    async with asyncio.timeout(3):
        provider = ProbeProvider("primary")
        provider.operation.cleanup.clear()
        budget = AdmissionBudget(max_active=1, max_pending=0, work_timeout_s=0.1, cancel_grace_s=1)
        gateway = Gateway([provider], admission=budget, max_batch_size=1)
        cb = gateway._circuit_breakers[provider.name]
        cb.failure_threshold = cb.half_open_success_threshold = 1
        await _open_and_expire(cb)
        first = asyncio.create_task(gateway.complete(CompletionRequest(prompt="timeout", cache=False)))
        try:
            await provider.operation.started.wait()
            execution = next(iter(gateway._batcher._batch_tasks))
            with pytest.raises(InferenceDeadlineExceeded):
                await first
            await provider.operation.cancellations.get()
            execution.cancel()  # another cancel during real provider cleanup
            await provider.operation.cancellations.get()
            assert provider.operation.active == budget.active == 1
            assert cb._half_open_probe_in_flight and cb._consecutive_failures == 1
            with pytest.raises(CircuitOpenError):
                await cb.call(provider.operation.run)
            with pytest.raises(AdmissionRejected):
                await gateway.complete(CompletionRequest(prompt="overlap", cache=False))
            assert provider.operation.calls == 1 and not budget.closing
            provider.operation.cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await execution
            assert provider.operation.active == budget.active == 0
            assert not cb._half_open_probe_in_flight and cb._consecutive_failures == 1
            provider.operation.release.set()
            recovered = await gateway.complete(CompletionRequest(prompt="after timeout", cache=False))
            assert recovered.provider == provider.name and cb.state == CircuitState.CLOSED
            assert provider.operation.calls == 2 and provider.operation.max_active == 1
            assert gateway.metrics.snapshot()["total_requests"] == 1  # expired work was not success
        finally:
            provider.operation.cleanup.set()
            provider.operation.release.set()
            await asyncio.gather(first, return_exceptions=True)
            await gateway.aclose()
        assert (budget.active, budget.pending, budget.requests) == (0, 0, 0)
        assert gateway._batcher._batch_tasks == set() and gateway._singleflight.in_flight_count == 0
