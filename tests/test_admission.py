import asyncio
import importlib.util
import sys
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from backend.admission import AdmissionBudget, AdmissionRejected
from backend.config import build_admission
from backend.gateway import Gateway, NoProviderAvailable
from backend.models import CompletionRequest
from backend.providers import MockProvider


class GatedProvider(MockProvider):
    def __init__(self):
        super().__init__(0, 0)
        self.release = asyncio.Event()
        self.stream_release = asyncio.Event()
        self.calls = []
        self.active = 0
        self.peak = 0
        self.fail = False

    async def complete(self, messages, **kwargs):
        self.calls.append(messages[-1].content)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await self.release.wait()
            if self.fail:
                raise RuntimeError("simulated failure")
            return "mock result", 1, 1
        finally:
            self.active -= 1

    async def stream_complete(self, messages, **kwargs):
        self.calls.append("stream:" + messages[-1].content)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            yield "first "
            await self.stream_release.wait()
            yield "last"
        finally:
            self.active -= 1


async def wait_until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


def req(prompt, *, cache=False):
    return CompletionRequest(prompt=prompt, cache=cache)


async def test_cancelled_readiness_future_is_not_promoted_or_left_queued():
    budget = AdmissionBudget(max_active=1, max_pending=1)
    active, pending = budget.reserve_work(2)
    pending.ready.cancel()
    active.release()
    pending.release()
    assert (budget.active, budget.pending) == (0, 0)


@pytest_asyncio.fixture
async def bounded():
    provider = GatedProvider()
    budget = AdmissionBudget(max_active=2, max_pending=2, max_requests=8)
    gateway = Gateway([provider], max_batch_size=1, admission=budget)
    async with asyncio.timeout(5):
        try:
            yield gateway, provider, budget
        finally:
            provider.release.set()
            provider.stream_release.set()
            await gateway.aclose()


async def test_saturation_bounds_actual_execution_and_pending_work(bounded):
    gateway, provider, budget = bounded
    tasks = [asyncio.create_task(gateway.complete(req(str(i)))) for i in range(4)]
    await wait_until(lambda: len(provider.calls) == 2 and budget.pending == 2)
    assert budget.active == 2
    assert provider.active == 2
    before = gateway.metrics.snapshot()
    for i in range(20):
        with pytest.raises(AdmissionRejected):
            await gateway.complete(req(f"rejected-{i}", cache=True))
    assert len(provider.calls) == 2
    assert len(gateway._singleflight._flights) == 4
    assert budget.pending == 2
    assert gateway._batcher._queue == []
    assert gateway.metrics.snapshot() == before
    assert gateway.cache.misses == 0
    assert gateway._circuit_breakers["mock"]._consecutive_failures == 0
    provider.release.set()
    assert len(await asyncio.gather(*tasks)) == 4
    assert len(provider.calls) == 4
    assert provider.peak <= 2
    assert (budget.active, budget.pending, budget.requests) == (0, 0, 0)
    assert (await gateway.complete(req("reused"))).provider == "mock"
    assert len(provider.calls) == 5


async def test_cache_hit_bypasses_full_downstream_budget(bounded):
    gateway, provider, budget = bounded
    provider.release.set()
    await gateway.complete(req("cached", cache=True))
    provider.release.clear()
    tasks = [asyncio.create_task(gateway.complete(req(str(i)))) for i in range(4)]
    await wait_until(lambda: budget.active == 2 and budget.pending == 2)
    before = len(provider.calls)
    assert (await gateway.complete(req("cached", cache=True))).cached
    assert len(provider.calls) == before
    provider.release.set()
    await asyncio.gather(*tasks)


async def test_joiners_share_work_but_request_waiters_are_bounded(bounded):
    gateway, provider, budget = bounded
    budget.max_requests = 3
    first = asyncio.create_task(gateway.complete(req("same", cache=True)))
    await wait_until(lambda: len(provider.calls) == 1)
    joiners = [asyncio.create_task(gateway.complete(req("same", cache=True))) for _ in range(2)]
    await wait_until(lambda: gateway._singleflight.shared_calls == 2)
    assert (budget.active, budget.pending, budget.requests) == (1, 0, 3)
    with pytest.raises(AdmissionRejected):
        await gateway.complete(req("same", cache=True))
    provider.release.set()
    responses = await asyncio.gather(first, *joiners)
    assert sum(r.deduplicated for r in responses) == 2
    assert len(provider.calls) == 1
    assert budget.requests == 0


@pytest.mark.parametrize("cache", [False, True])
async def test_completion_cancellation_does_not_release_executing_work_early(bounded, cache):
    gateway, provider, budget = bounded
    budget.max_active = 1
    budget.max_pending = 1
    first = asyncio.create_task(gateway.complete(req("active", cache=cache)))
    await wait_until(lambda: len(provider.calls) == 1)
    pending = asyncio.create_task(gateway.complete(req("pending", cache=cache)))
    await wait_until(lambda: budget.pending == 1)
    for task in (first, pending):
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert (budget.active, budget.pending, budget.requests) == (1, 1, 0)
    with pytest.raises(AdmissionRejected):
        await gateway.complete(req("must not overtake"))
    provider.release.set()
    await gateway.aclose()
    assert provider.calls == ["active", "pending"]
    assert provider.peak == 1
    assert (budget.active, budget.pending, budget.requests) == (0, 0, 0)
    assert gateway._singleflight.in_flight_count == 0


async def test_failure_releases_capacity_and_allows_retry(bounded):
    gateway, provider, budget = bounded
    provider.fail = True
    provider.release.set()
    with pytest.raises(NoProviderAvailable):
        await gateway.complete(req("failure", cache=True))
    assert (budget.active, budget.pending, budget.requests) == (0, 0, 0)
    provider.fail = False
    assert (await gateway.complete(req("failure", cache=True))).provider == "mock"
    assert provider.calls == ["failure", "failure"]


async def test_batch_reserves_all_new_items_or_none(bounded):
    gateway, provider, budget = bounded
    budget.max_pending = 0
    active = asyncio.create_task(gateway.complete(req("occupied")))
    await wait_until(lambda: len(provider.calls) == 1)
    with pytest.raises(AdmissionRejected):
        await gateway.complete_many([req("new-a"), req("new-b")])
    assert provider.calls == ["occupied"]
    assert budget.active == 1
    assert gateway._singleflight.in_flight_count == 1
    # Duplicate members need one new execution reservation, not two.
    batch = asyncio.create_task(
        gateway.complete_many([req("same", cache=True), req("same", cache=True)])
    )
    await wait_until(lambda: len(provider.calls) == 2)
    provider.release.set()
    await active
    responses = await batch
    assert len(responses) == 2 and responses[1].deduplicated
    assert provider.calls == ["occupied", "same"]
    assert budget.active == 0


async def test_atomic_batch_combines_cache_existing_flight_and_new_keys(bounded):
    gateway, provider, budget = bounded
    provider.release.set()
    await gateway.complete(req("cached", cache=True))
    provider.release.clear()
    active = asyncio.create_task(gateway.complete(req("inflight", cache=True)))
    await wait_until(lambda: provider.calls == ["cached", "inflight"])
    batch = asyncio.create_task(
        gateway.complete_many(
            [req("cached", cache=True), req("inflight", cache=True), req("new", cache=True)]
        )
    )
    await wait_until(lambda: provider.calls == ["cached", "inflight", "new"])
    assert budget.active == 2 and budget.pending == 0
    provider.release.set()
    await active
    responses = await batch
    assert [r.cached for r in responses] == [True, True, False]
    assert [r.deduplicated for r in responses] == [False, True, False]
    assert len(provider.calls) == 3


async def test_shutdown_drains_queued_completions_and_rejects_new_work(bounded):
    gateway, provider, budget = bounded
    budget.max_active = 1
    tasks = [asyncio.create_task(gateway.complete(req(str(i)))) for i in range(3)]
    await wait_until(lambda: len(provider.calls) == 1 and budget.pending == 2)
    closing = asyncio.create_task(gateway.aclose())
    await wait_until(lambda: budget.closing)
    with pytest.raises(AdmissionRejected):
        await gateway.complete(req("new"))
    provider.release.set()
    await closing
    assert len(await asyncio.gather(*tasks)) == 3
    assert provider.calls == ["0", "1", "2"]
    assert (budget.active, budget.pending, budget.requests) == (0, 0, 0)
    assert gateway._batcher._batch_tasks == set()
    assert gateway._batcher._flush_tasks == set()
    assert gateway._singleflight.in_flight_count == 0


async def test_microbatch_residence_keeps_active_budget_reserved(bounded):
    gateway, provider, budget = bounded
    gateway._batcher.max_batch_size = 2
    gateway._batcher.window_ms = 60_000
    one = asyncio.create_task(gateway.complete(req("one")))
    await wait_until(lambda: len(gateway._batcher._queue) == 1)
    assert budget.active == 1 and provider.calls == []
    two = asyncio.create_task(gateway.complete(req("two")))
    await wait_until(lambda: len(provider.calls) == 1)
    assert budget.active == 2  # both members held until batch-wide delivery
    provider.release.set()
    await asyncio.gather(one, two)
    assert (budget.active, budget.pending) == (0, 0)
    assert provider.calls == ["one", "two"]


async def test_stream_holds_shared_capacity_until_closed(bounded):
    gateway, provider, budget = bounded
    budget.max_active = 1
    stream = gateway.stream_complete(req("stream"))
    assert await anext(stream) == "first "
    assert budget.active == 1
    queued = asyncio.create_task(gateway.complete(req("queued")))
    await wait_until(lambda: budget.pending == 1)
    with pytest.raises(AdmissionRejected):
        gateway.open_stream(req("rejected-stream"))
    assert provider.calls == ["stream:stream"]
    await stream.aclose()
    await wait_until(lambda: provider.calls == ["stream:stream", "queued"])
    provider.release.set()
    await queued
    assert provider.peak == 1
    assert (budget.active, budget.requests) == (0, 0)


async def test_stream_success_and_failure_both_release_capacity(bounded, monkeypatch):
    gateway, provider, budget = bounded
    provider.stream_release.set()
    assert [part async for part in gateway.stream_complete(req("success"))] == ["first ", "last"]
    assert (budget.active, budget.requests) == (0, 0)

    async def fail_stream(*args, **kwargs):
        raise RuntimeError("stream failed")
        yield  # make this an async generator, matching the provider interface

    monkeypatch.setattr(provider, "stream_complete", fail_stream)
    with pytest.raises(NoProviderAvailable):
        async for _ in gateway.stream_complete(req("failure")):
            pass
    assert (budget.active, budget.requests) == (0, 0)


async def test_shutdown_closes_unused_and_running_streams(bounded):
    gateway, provider, budget = bounded
    unused = gateway.open_stream(req("unused"))
    started = asyncio.Event()

    async def consume():
        async for chunk in gateway.stream_complete(req("running")):
            started.set()

    consumer = asyncio.create_task(consume())
    await started.wait()
    await asyncio.gather(gateway.aclose(), gateway.aclose())
    assert consumer.cancelled()
    assert unused.closed
    assert provider.active == 0
    assert (budget.active, budget.pending, budget.requests) == (0, 0, 0)
    assert budget._streams == set()
    with pytest.raises(AdmissionRejected):
        await gateway.complete(req("closed"))
    await gateway.aclose()


@pytest.mark.parametrize(
    "name,invalid",
    [
        ("AI_SYSTEMS_LAB_MAX_ACTIVE", "0"),
        ("AI_SYSTEMS_LAB_MAX_ACTIVE", "-1"),
        ("AI_SYSTEMS_LAB_MAX_PENDING", "-1"),
        ("AI_SYSTEMS_LAB_MAX_PENDING", ""),
        ("AI_SYSTEMS_LAB_MAX_REQUESTS", "0"),
        ("AI_SYSTEMS_LAB_MAX_REQUESTS", "secret-not-an-integer"),
        ("AI_SYSTEMS_LAB_MAX_ACTIVE", "1.5"),
    ],
)
def test_invalid_public_capacity_config_fails_without_echoing_value(monkeypatch, name, invalid):
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "public-demo")
    monkeypatch.setenv(name, invalid)
    with pytest.raises(ValueError, match=name) as exc:
        build_admission()
    assert "secret-not-an-integer" not in str(exc.value)


def test_configuration_defaults_overrides_and_developer_compatibility(monkeypatch):
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "public-demo")
    for name in (
        "AI_SYSTEMS_LAB_MAX_ACTIVE",
        "AI_SYSTEMS_LAB_MAX_PENDING",
        "AI_SYSTEMS_LAB_MAX_REQUESTS",
    ):
        monkeypatch.delenv(name, raising=False)
    budget = build_admission()
    assert (budget.max_active, budget.max_pending, budget.max_requests) == (4, 16, 64)
    monkeypatch.setenv("AI_SYSTEMS_LAB_MAX_PENDING", "0")
    assert build_admission().max_pending == 0
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "developer")
    monkeypatch.setenv("AI_SYSTEMS_LAB_MAX_ACTIVE", "invalid")
    assert build_admission() is None
    assert Gateway([MockProvider()]).admission is None


async def test_developer_gateway_does_not_apply_public_budgets():
    provider = GatedProvider()
    gateway = Gateway([provider], max_batch_size=1)
    tasks = [asyncio.create_task(gateway.complete(req(str(i)))) for i in range(8)]
    try:
        await wait_until(lambda: len(provider.calls) == 8)
        assert provider.active == 8
    finally:
        provider.release.set()
        await asyncio.gather(*tasks)
        await gateway.aclose()


@pytest_asyncio.fixture
async def public_api(monkeypatch):
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "public-demo")
    monkeypatch.setenv("AI_SYSTEMS_LAB_MAX_ACTIVE", "1")
    monkeypatch.setenv("AI_SYSTEMS_LAB_MAX_PENDING", "0")
    monkeypatch.setenv("AI_SYSTEMS_LAB_MAX_REQUESTS", "4")
    spec = importlib.util.spec_from_file_location(
        "backend._admission_test_main", Path(__file__).resolve().parents[1] / "backend/main.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    provider = GatedProvider()
    module.gateway = Gateway([provider], max_batch_size=1, admission=build_admission())
    async with asyncio.timeout(5):
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=module.app), base_url="http://test"
            ) as client:
                yield module, client, provider
        finally:
            provider.release.set()
            provider.stream_release.set()
            await module.gateway.aclose()


async def test_http_overload_contract_covers_completion_batch_and_stream(public_api):
    module, client, provider = public_api
    first = asyncio.create_task(client.post("/v1/completions", json={"prompt": "held"}))
    await wait_until(lambda: len(provider.calls) == 1)
    for path, payload in [
        ("/v1/completions", {"prompt": "excess"}),
        ("/v1/completions/batch", {"requests": [{"prompt": "new"}, {"prompt": "held"}]}),
        ("/v1/completions/stream", {"prompt": "stream"}),
    ]:
        response = await client.post(path, json=payload)
        assert response.status_code == 503
        assert response.json() == {
            "error": {"code": "capacity_unavailable", "message": "Public demo is busy"}
        }
        assert not response.headers["content-type"].startswith("text/event-stream")
    assert provider.calls == ["held"]
    assert module.gateway.metrics.snapshot()["total_requests"] == 0
    assert module.gateway.metrics.snapshot()["errors"] == 0
    assert module.gateway.cache.misses == 1
    assert module.gateway._singleflight.total_calls == 1
    assert module.gateway._singleflight.shared_calls == 0
    assert module.gateway.admission.requests == 1
    provider.release.set()
    assert (await first).status_code == 200


async def test_sse_response_lifetime_includes_header_send_and_disconnect(public_api):
    module, _, provider = public_api
    response = await module.create_streaming_completion(req("stream"))
    sent_first = asyncio.Event()
    disconnected = asyncio.Event()

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            sent_first.set()

    async def receive():
        await disconnected.wait()
        return {"type": "http.disconnect"}

    task = asyncio.create_task(
        response({"type": "http", "asgi": {"spec_version": "2.0"}}, receive, send)
    )
    await sent_first.wait()
    assert module.gateway.admission.active == 1
    with pytest.raises(AdmissionRejected):
        await module.create_streaming_completion(req("must reject before headers"))
    disconnected.set()
    await task
    assert module.gateway.admission.active == 0
    assert module.gateway.admission.requests == 0
    assert provider.active == 0
    assert provider.calls == ["stream:stream"]


async def test_response_cancelled_before_iteration_releases_reservation(public_api):
    module, _, provider = public_api
    response = await module.create_streaming_completion(req("not started"))

    # Simulate cancellation while sending the initial headers.
    async def send(message):
        raise asyncio.CancelledError

    async def receive():
        await asyncio.Event().wait()

    with pytest.raises(asyncio.CancelledError):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert module.gateway.admission.active == 0
    assert module.gateway.admission.requests == 0
    assert provider.calls == []
