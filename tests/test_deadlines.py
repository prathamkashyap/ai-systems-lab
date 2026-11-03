import asyncio

import httpx
import pytest

from backend.admission import AdmissionBudget, AdmissionRejected
from backend.config import build_admission
from backend.gateway import Gateway
from backend.models import CompletionRequest
from backend.providers import MockProvider
from tests.test_app_config import _load_application


class StalledProvider(MockProvider):
    def __init__(self):
        super().__init__(0, 0)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.cleanup = asyncio.Event()
        self.cleanup.set()
        self.active = 0
        self.calls = 0

    async def complete(self, *args, **kwargs):
        self.calls += 1
        self.active += 1
        self.started.set()
        try:
            await self.release.wait()
            return "mock result", 1, 1
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.cleanup.wait()
            raise
        finally:
            self.active -= 1

    async def stream_complete(self, *args, **kwargs):
        yield "first "
        await self.complete(*args, **kwargs)
        yield "last"


def request(text="test", cache=True):
    return CompletionRequest(prompt=text, cache=cache)


def gateway_for(provider, **limits):
    return Gateway(
        [provider],
        max_batch_size=1,
        admission=AdmissionBudget(max_active=1, max_pending=1, **limits),
    )


async def wait_until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


async def test_provider_stops_before_expired_capacity_is_reused():
    async with asyncio.timeout(3):
        from backend.deadlines import InferenceDeadlineExceeded

        provider = StalledProvider()
        provider.cleanup.clear()
        gateway = gateway_for(provider, work_timeout_s=0.05, cancel_grace_s=1)
        gateway.admission.max_pending = 0
        task = asyncio.create_task(gateway.complete(request()))
        try:
            await asyncio.wait_for(provider.started.wait(), 1)
            with pytest.raises(InferenceDeadlineExceeded):
                await asyncio.wait_for(task, 1)
            await provider.cancelled.wait()
            assert provider.active == 1  # cancellation cleanup has not finished
            assert gateway.admission.active == 1
            with pytest.raises(AdmissionRejected):
                await gateway.complete(request("second"))
            assert provider.calls == 1
            provider.cleanup.set()
            await wait_until(lambda: gateway.admission.active == 0)
            provider.release.set()
            assert (await gateway.complete(request("second"))).provider == "mock"
            assert provider.active == 0
        finally:
            provider.cleanup.set()
            provider.release.set()
            await gateway.aclose()


async def test_pending_and_preflush_items_expire_without_provider_calls():
    async with asyncio.timeout(3):
        from backend.deadlines import InferenceDeadlineExceeded

        provider = StalledProvider()
        gateway = gateway_for(provider, work_timeout_s=0.05)
        gateway._batcher.max_batch_size = 16
        gateway._batcher.window_ms = 60_000
        tasks = [asyncio.create_task(gateway.complete(request(str(i)))) for i in range(2)]
        outcomes = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 1)
        assert all(isinstance(outcome, InferenceDeadlineExceeded) for outcome in outcomes)
        assert provider.calls == 0
        await gateway.aclose()
        assert (gateway.admission.active, gateway.admission.pending, gateway.admission.requests) == (
            0,
            0,
            0,
        )
        assert gateway._batcher._queue == []


async def test_shared_flight_deadline_is_not_reset_by_joiners_or_caller_cancellation():
    async with asyncio.timeout(3):
        from backend.deadlines import InferenceDeadlineExceeded

        provider = StalledProvider()
        gateway = gateway_for(provider, work_timeout_s=0.1)
        first = asyncio.create_task(gateway.complete(request()))
        await asyncio.wait_for(provider.started.wait(), 1)
        joined = asyncio.create_task(gateway.complete(request()))
        await wait_until(lambda: gateway._singleflight.shared_calls == 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert gateway.admission.active == 1 and provider.active == 1
        with pytest.raises(InferenceDeadlineExceeded):
            await asyncio.wait_for(joined, 1)
        await gateway.aclose()
        assert provider.calls == 1 and provider.active == 0
        assert gateway.metrics.snapshot()["total_requests"] == 0


async def test_expiration_terminates_entire_running_flush_without_false_success():
    async with asyncio.timeout(3):
        from backend.deadlines import InferenceDeadlineExceeded

        provider = StalledProvider()
        gateway = gateway_for(provider, work_timeout_s=0.05)
        gateway.admission.max_active = 3
        gateway._batcher.max_batch_size = 3
        task = asyncio.create_task(
            gateway.complete_many([request(str(i), cache=False) for i in range(3)])
        )
        await asyncio.wait_for(provider.started.wait(), 1)
        with pytest.raises(InferenceDeadlineExceeded):
            await asyncio.wait_for(task, 1)
        await gateway.aclose()
        assert provider.calls == 1  # no later sequential provider calls in the expired flush
        assert provider.active == 0 and gateway.admission.active == 0
        assert gateway._batcher._batch_tasks == set()
        assert gateway._singleflight.in_flight_count == 0
        assert gateway.metrics.snapshot()["total_requests"] == 0


async def test_partial_flush_results_are_not_cached_or_counted_after_expiration():
    async with asyncio.timeout(3):
        from backend.deadlines import InferenceDeadlineExceeded

        class PartialProvider(StalledProvider):
            async def complete(self, *args, **kwargs):
                if self.calls == 0:
                    self.calls += 1
                    return "completed first member", 1, 1
                return await super().complete(*args, **kwargs)

        provider = PartialProvider()
        gateway = gateway_for(provider, work_timeout_s=0.05)
        gateway.admission.max_active = 3
        gateway._batcher.max_batch_size = 3
        with pytest.raises(InferenceDeadlineExceeded):
            await gateway.complete_many([request(str(i)) for i in range(3)])
        await gateway.aclose()
        assert provider.calls == 2
        assert gateway.cache.size == 0
        assert gateway.metrics.snapshot()["total_requests"] == 0


async def test_unfinished_cancellation_retains_capacity_and_fails_shutdown_closed():
    async with asyncio.timeout(3):
        from backend.deadlines import InferenceDeadlineExceeded, ShutdownIncomplete

        provider = StalledProvider()
        provider.cleanup.clear()
        gateway = gateway_for(provider, work_timeout_s=0.03, shutdown_grace_s=0.03, cancel_grace_s=0.02)
        task = asyncio.create_task(gateway.complete(request()))
        try:
            await asyncio.wait_for(provider.started.wait(), 1)
            with pytest.raises(InferenceDeadlineExceeded):
                await asyncio.wait_for(task, 1)
            await wait_until(lambda: gateway.admission.closing)
            assert provider.active == 1 and gateway.admission.active == 1
            with pytest.raises(AdmissionRejected):
                await gateway.complete(request("must not execute"))
            with pytest.raises(ShutdownIncomplete):
                await asyncio.wait_for(gateway.aclose(), 1)
            assert provider.active == 1 and gateway.admission.active == 1
        finally:
            # Release the fake's resistant cleanup so the test process can exit.
            provider.cleanup.set()
            await wait_until(lambda: provider.active == 0 and gateway.admission.active == 0)
        assert gateway.metrics.snapshot()["total_requests"] == 0
        assert gateway.admission.closing  # operator restart required; do not silently reopen


async def test_late_return_from_cancelled_provider_cannot_become_success():
    async with asyncio.timeout(3):
        from backend.deadlines import InferenceDeadlineExceeded

        class LateProvider(StalledProvider):
            async def complete(self, *args, **kwargs):
                try:
                    return await super().complete(*args, **kwargs)
                except asyncio.CancelledError:
                    return "late success must be discarded", 1, 1

        provider = LateProvider()
        gateway = gateway_for(provider, work_timeout_s=0.03)
        gateway.admission.max_active = 3
        gateway._batcher.max_batch_size = 3
        try:
            with pytest.raises(InferenceDeadlineExceeded):
                await asyncio.wait_for(
                    gateway.complete_many([request(str(i)) for i in range(3)]), 1
                )
            assert provider.calls == 1  # swallowing cancellation must not start later members
        finally:
            provider.release.set()
            await asyncio.wait_for(gateway.aclose(), 1)
        assert gateway.cache.size == 0
        assert gateway.metrics.snapshot()["total_requests"] == 0
        assert gateway.admission.active == 0


async def test_shutdown_grace_expires_and_cancels_cooperative_provider():
    async with asyncio.timeout(3):
        provider = StalledProvider()
        gateway = gateway_for(provider, work_timeout_s=30, shutdown_grace_s=0.05, cancel_grace_s=0.5)
        task = asyncio.create_task(gateway.complete(request()))
        await asyncio.wait_for(provider.started.wait(), 1)
        pending = asyncio.create_task(gateway.complete(request("pending")))
        await wait_until(lambda: gateway.admission.pending == 1)
        await asyncio.wait_for(asyncio.gather(gateway.aclose(), gateway.aclose()), 1)
        outcome = (await asyncio.gather(task, return_exceptions=True))[0]
        assert isinstance(outcome, Exception)
        assert isinstance((await asyncio.gather(pending, return_exceptions=True))[0], Exception)
        assert provider.calls == 1
        assert provider.cancelled.is_set() and provider.active == 0
        assert (gateway.admission.active, gateway.admission.pending, gateway.admission.requests) == (
            0,
            0,
            0,
        )
        await gateway.aclose()
        with pytest.raises(AdmissionRejected):
            await gateway.complete(request("late"))


async def test_normal_completion_and_graceful_drain_preserve_success():
    async with asyncio.timeout(3):
        provider = StalledProvider()
        gateway = gateway_for(provider, work_timeout_s=2, shutdown_grace_s=1)
        task = asyncio.create_task(gateway.complete(request()))
        await asyncio.wait_for(provider.started.wait(), 1)
        closing = asyncio.create_task(gateway.aclose())
        await wait_until(lambda: gateway.admission.closing)
        provider.release.set()
        assert (await task).text == "mock result"
        await asyncio.wait_for(closing, 1)
        assert provider.active == 0 and gateway.admission.active == 0


async def test_cancelled_shutdown_waiter_does_not_abandon_owned_cleanup():
    async with asyncio.timeout(3):
        provider = StalledProvider()
        gateway = gateway_for(provider, work_timeout_s=2, shutdown_grace_s=1)
        task = asyncio.create_task(gateway.complete(request()))
        await asyncio.wait_for(provider.started.wait(), 1)
        closing = asyncio.create_task(gateway.aclose())
        await wait_until(lambda: gateway.admission.closing)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        provider.release.set()
        await asyncio.wait_for(gateway.aclose(), 1)
        assert (await task).text == "mock result"
        assert (gateway.admission.active, gateway.admission.requests) == (0, 0)


async def test_stream_success_before_deadline_releases_capacity():
    async with asyncio.timeout(3):
        provider = StalledProvider()
        provider.release.set()
        gateway = gateway_for(provider, stream_timeout_s=1)
        assert [part async for part in gateway.stream_complete(request())] == ["first ", "last"]
        await gateway.aclose()
        assert (gateway.admission.active, gateway.admission.requests) == (0, 0)


async def test_stream_deadline_retains_capacity_until_provider_cleanup_finishes():
    async with asyncio.timeout(3):
        from backend.deadlines import InferenceDeadlineExceeded

        provider = StalledProvider()
        provider.cleanup.clear()
        gateway = gateway_for(provider, stream_timeout_s=0.05, cancel_grace_s=1)
        gateway.admission.max_pending = 0
        stream = gateway.open_stream(request())
        try:
            assert await anext(stream) == "first "
            step = asyncio.create_task(anext(stream))
            await asyncio.wait_for(provider.started.wait(), 1)
            with pytest.raises(InferenceDeadlineExceeded):
                await asyncio.wait_for(step, 1)
            await asyncio.wait_for(provider.cancelled.wait(), 1)
            assert provider.active == 1 and gateway.admission.active == 1
            with pytest.raises(AdmissionRejected):
                await gateway.complete(request("must not execute"))
            assert provider.calls == 1
        finally:
            provider.cleanup.set()
            await asyncio.wait_for(stream.aclose(), 1)
            await gateway.aclose()
        assert (provider.active, gateway.admission.active, gateway.admission.requests) == (0, 0, 0)


@pytest.mark.parametrize("mode", ["developer", "public-demo"])
async def test_http_and_sse_deadline_contracts(monkeypatch, mode):
    async with asyncio.timeout(3):
        monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", mode)
        monkeypatch.setenv("AI_SYSTEMS_LAB_WORK_TIMEOUT_S", ".05")
        monkeypatch.setenv("AI_SYSTEMS_LAB_STREAM_TIMEOUT_S", ".05")
        module = _load_application(monkeypatch)
        provider = StalledProvider()
        module.gateway = Gateway([provider], max_batch_size=1, admission=build_admission())
        async with asyncio.timeout(3):
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=module.app), base_url="http://test"
                ) as client:
                    if mode == "developer":
                        provider.release.set()
                    response = await client.post("/v1/completions", json={"prompt": "http"})
                    assert response.status_code == (504 if mode == "public-demo" else 200)
                    if mode == "public-demo":
                        assert response.json() == {
                            "error": {
                                "code": "inference_deadline_exceeded",
                                "message": "Inference deadline exceeded",
                            }
                        }
                    stream = await client.post("/v1/completions/stream", json={"prompt": "stream"})
                    assert stream.status_code == 200
                    if mode == "public-demo":
                        assert "event: error" in stream.text
                        assert '"code": "inference_deadline_exceeded"' in stream.text
                        assert "[DONE]" not in stream.text and '"text": "last"' not in stream.text
                    else:
                        assert "[DONE]" in stream.text
            finally:
                provider.release.set()
                await module.gateway.aclose()


@pytest.mark.parametrize("before_headers", [False, True])
async def test_response_deadline_covers_headers_and_stalled_client_send(
    monkeypatch, before_headers
):
    async with asyncio.timeout(3):
        monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "public-demo")
        monkeypatch.setenv("AI_SYSTEMS_LAB_STREAM_TIMEOUT_S", ".05")
        module = _load_application(monkeypatch)
        response = await module.create_streaming_completion(request())
        messages = []
        if before_headers:
            response.stream.deadline = asyncio.get_running_loop().time() - 1

        async def send(message):
            if (
                not before_headers
                and message["type"] == "http.response.body"
                and not message["body"].startswith(b"event: error")
            ):
                await asyncio.Event().wait()  # cooperative slow-client backpressure
            messages.append(message)

        async def receive():
            await asyncio.Event().wait()

        async with asyncio.timeout(2):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        assert messages[0]["status"] == (504 if before_headers else 200)
        body = b"".join(message.get("body", b"") for message in messages)
        assert b"inference_deadline_exceeded" in body
        assert b"[DONE]" not in body
        assert (module.gateway.admission.active, module.gateway.admission.requests) == (0, 0)
        await module.gateway.aclose()


@pytest.mark.parametrize(
    "name",
    [
        "AI_SYSTEMS_LAB_WORK_TIMEOUT_S",
        "AI_SYSTEMS_LAB_STREAM_TIMEOUT_S",
        "AI_SYSTEMS_LAB_SHUTDOWN_GRACE_S",
        "AI_SYSTEMS_LAB_CANCEL_GRACE_S",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "", "secret-invalid"])
def test_invalid_deadline_configuration(monkeypatch, name, value):
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "public-demo")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name) as error:
        build_admission()
    assert "secret-invalid" not in str(error.value)


def test_default_policy_and_invalid_deadlines_ignored_in_developer_mode(monkeypatch):
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "public-demo")
    for name in (
        "AI_SYSTEMS_LAB_WORK_TIMEOUT_S",
        "AI_SYSTEMS_LAB_STREAM_TIMEOUT_S",
        "AI_SYSTEMS_LAB_SHUTDOWN_GRACE_S",
        "AI_SYSTEMS_LAB_CANCEL_GRACE_S",
    ):
        monkeypatch.delenv(name, raising=False)
    budget = build_admission()
    assert (
        budget.work_timeout_s,
        budget.stream_timeout_s,
        budget.shutdown_grace_s,
        budget.cancel_grace_s,
    ) == (10, 15, 5, 1)
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "developer")
    monkeypatch.setenv("AI_SYSTEMS_LAB_WORK_TIMEOUT_S", "invalid")
    assert build_admission() is None
