import asyncio
import hashlib
import json
import math
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest

from backend import cache as cache_module
from backend.admission import AdmissionBudget
from backend.cache import TTLLRUCache
from backend.gateway import Gateway
from backend.models import CompletionRequest, Message
from backend.providers.mock_provider import MockProvider
from tests.test_app_config import _load_application


def collision_pair(kind):
    if kind == "temperature":
        return (
            CompletionRequest(prompt="same input", temperature=0.7001),
            CompletionRequest(prompt="same input", temperature=0.7002),
        )
    return (
        CompletionRequest(messages=[Message(role="user", content="hello\x1fassistant:world")]),
        CompletionRequest(
            messages=[
                Message(role="user", content="hello"),
                Message(role="assistant", content="world"),
            ]
        ),
    )


def effective_inputs(req):
    return (
        req.model,
        req.temperature,
        req.max_tokens,
        tuple((message.role, message.content) for message in req.as_messages()),
    )


class CountingProvider(MockProvider):
    def __init__(self):
        super().__init__(0, 0)
        self.calls = []
        self.started = asyncio.Event()
        self.changed = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def complete(self, messages, *, model, max_tokens, temperature):
        self.calls.append(
            (model, temperature, max_tokens, tuple((m.role, m.content) for m in messages))
        )
        text = f"execution-{len(self.calls)}"
        self.started.set()
        self.changed.set()
        await self.release.wait()
        return text, 1, 1

    async def wait_for_calls(self, count):
        while len(self.calls) < count:
            self.changed.clear()
            await self.changed.wait()


@asynccontextmanager
async def gateway_case(mode, *, cache=None):
    provider = CountingProvider()
    admission = AdmissionBudget(max_active=4, max_pending=4) if mode == "public-demo" else None
    gateway = Gateway([provider], cache=cache, admission=admission, max_batch_size=1)
    async with asyncio.timeout(3):
        try:
            yield gateway, provider
        finally:
            provider.release.set()
            await gateway.aclose()
        assert gateway._singleflight.in_flight_count == 0
        if admission is not None:
            assert (admission.active, admission.pending, admission.requests) == (0, 0, 0)


@pytest.mark.parametrize("mode", ["developer", "public-demo"])
@pytest.mark.parametrize("kind", ["temperature", "messages"])
async def test_distinct_inputs_never_reuse_another_cached_result(mode, kind):
    async with gateway_case(mode) as (gateway, provider):
        first, second = collision_pair(kind)
        a = await gateway.complete(first)
        b = await gateway.complete(second)
        assert (len(provider.calls), b.cached, [a.text, b.text]) == (
            2,
            False,
            ["execution-1", "execution-2"],
        )
        assert provider.calls == [effective_inputs(first), effective_inputs(second)]
        assert (await gateway.complete(first)).text == a.text
        assert (await gateway.complete(second)).text == b.text
        assert len(provider.calls) == 2
        assert (gateway.cache.hits, gateway.cache.misses, gateway.cache.size) == (2, 2, 2)
        snapshot = gateway.metrics.snapshot()
        assert (snapshot["total_requests"], snapshot["cache_hits"], snapshot["cache_misses"]) == (
            4,
            2,
            2,
        )


@pytest.mark.parametrize("mode", ["developer", "public-demo"])
@pytest.mark.parametrize("kind", ["temperature", "messages"])
async def test_distinct_inputs_never_join_another_singleflight(mode, kind, monkeypatch):
    async with gateway_case(mode) as (gateway, provider):
        first, second = collision_pair(kind)
        prepared = asyncio.Event()
        original = gateway._singleflight.prepare_many

        async def track(*args, **kwargs):
            result = await original(*args, **kwargs)
            if gateway._singleflight.total_calls == 2:
                prepared.set()
            return result

        monkeypatch.setattr(gateway._singleflight, "prepare_many", track)
        provider.release.clear()
        a = asyncio.create_task(gateway.complete(first))
        tasks = [a]
        try:
            await provider.started.wait()
            tasks.append(asyncio.create_task(gateway.complete(second)))
            await prepared.wait()
            provider.release.set()
            responses = await asyncio.gather(*tasks)
            assert (
                len(provider.calls),
                gateway._singleflight.shared_calls,
                [r.text for r in responses],
                [r.deduplicated for r in responses],
            ) == (2, 0, ["execution-1", "execution-2"], [False, False])
            assert provider.calls == [effective_inputs(first), effective_inputs(second)]
        finally:
            provider.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize(
    "updates",
    [
        {"model": "mock-large"},
        {"model": "mock-small\x1f0.700:user:injected"},
        {"max_tokens": 257},
        {"temperature": math.nextafter(0.7, 1)},
        {"messages": [{"role": "assistant", "content": "same"}]},
        {"messages": [{"role": "system", "content": "same"}]},
        {"messages": [{"role": "user", "content": "same\n"}]},
        {"messages": [{"role": "user", "content": "same\x1f"}]},
    ],
)
def test_all_effective_provider_fields_distinguish_keys(updates):
    gateway = Gateway([MockProvider(0, 0)])
    original = CompletionRequest(messages=[Message(role="user", content="same")])
    changed = CompletionRequest.model_validate(original.model_dump() | updates)
    assert gateway._cache_key(original) != gateway._cache_key(changed)


@pytest.mark.parametrize(
    "left,right",
    [
        ("line:one\nline:two", "line:one\\nline:two"),
        ("hello\x1fassistant:world", "hello\\u001fassistant:world"),
        ("😀 café", "😀 cafe\u0301"),
        ("a:b\x1fc\n雪", "a:b\x1fc\n雪 "),
    ],
)
def test_exact_content_is_not_escaped_normalized_or_trimmed_into_equivalence(left, right):
    gateway = Gateway([MockProvider(0, 0)])
    assert gateway._cache_key(CompletionRequest(prompt=left)) != gateway._cache_key(
        CompletionRequest(prompt=right)
    )


def test_role_order_and_message_boundaries_remain_distinct():
    gateway = Gateway([MockProvider(0, 0)])
    user = Message(role="user", content="x")
    assistant = Message(role="assistant", content="x")
    structures = [[user, assistant], [assistant, user], [user, user], [user]]
    assert len({gateway._cache_key(CompletionRequest(messages=m)) for m in structures}) == 4


@pytest.mark.parametrize("left,right", [(0.0, math.nextafter(0.0, 1)), (0.0, -0.0)])
def test_full_float_representation_is_preserved(left, right):
    gateway = Gateway([MockProvider(0, 0)])
    assert gateway._cache_key(
        CompletionRequest(prompt="same", temperature=left)
    ) != gateway._cache_key(CompletionRequest(prompt="same", temperature=right))


def test_equivalent_effective_inputs_share_key_despite_input_form_and_flags():
    gateway = Gateway([MockProvider(0, 0)])
    content = "  :\x1f\n😀 café 雪  "
    requests = [
        CompletionRequest(prompt=content),
        CompletionRequest(messages=[Message(role="user", content=content)]),
        CompletionRequest.model_validate(
            {
                "messages": [{"content": content, "role": "user"}],
                "temperature": "0.7",
                "max_tokens": "256",
                "cache": False,
                "stream": True,
            }
        ),
    ]
    assert len({gateway._cache_key(req) for req in requests}) == 1


def test_versioned_format_is_stable_and_passed_as_one_hash_payload(monkeypatch):
    gateway = Gateway([MockProvider(0, 0)])
    calls = []
    original = gateway.cache.make_key

    def capture(*parts):
        calls.append(parts)
        return original(*parts)

    monkeypatch.setattr(gateway.cache, "make_key", capture)
    request = CompletionRequest(prompt="hello\x1f😀\n", temperature=0.7001)
    key = gateway._cache_key(request)
    wire = (
        r'{"format":"ai-systems-lab.inference.v1","max_tokens":256,"messages":'
        r'[{"content":"hello\u001f\ud83d\ude00\n","role":"user"}],'
        r'"model":"mock-small","temperature":0.7001}'
    )
    assert calls == [(wire,)]
    assert json.loads(wire)["messages"][0]["content"] == request.prompt
    assert key == hashlib.sha256(wire.encode("ascii")).hexdigest()
    assert len(key) == 64


@pytest.mark.parametrize("mode", ["developer", "public-demo"])
@pytest.mark.parametrize("cancel_initiator", [False, True])
async def test_identical_effective_requests_share_one_actual_execution(
    mode, cancel_initiator, monkeypatch
):
    async with gateway_case(mode) as (gateway, provider):
        prepared = asyncio.Event()
        original = gateway._singleflight.prepare_many

        async def track(*args, **kwargs):
            result = await original(*args, **kwargs)
            if gateway._singleflight.total_calls == 8:
                prepared.set()
            return result

        monkeypatch.setattr(gateway._singleflight, "prepare_many", track)
        request = CompletionRequest(prompt="same :\x1f\n😀")
        equivalent = CompletionRequest(messages=request.as_messages())
        provider.release.clear()
        tasks = [asyncio.create_task(gateway.complete(request))]
        try:
            await provider.started.wait()
            tasks += [asyncio.create_task(gateway.complete(equivalent)) for _ in range(7)]
            await prepared.wait()
            assert len(provider.calls) == 1 and gateway._singleflight.shared_calls == 7
            if cancel_initiator:
                tasks[0].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await tasks[0]
            if gateway.admission is not None:
                assert (gateway.admission.active, gateway.admission.pending) == (1, 0)
            provider.release.set()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            assert all(r.text == "execution-1" and r.deduplicated for r in responses[1:])
            if not cancel_initiator:
                assert not responses[0].cached and not responses[0].deduplicated
            cached = await gateway.complete(equivalent)
            assert cached.cached and not cached.deduplicated and cached.text == "execution-1"
            assert len(provider.calls) == 1
        finally:
            provider.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("mode", ["developer", "public-demo"])
async def test_cache_disabled_calls_bypass_reads_writes_and_deduplication(mode):
    async with gateway_case(mode) as (gateway, provider):
        request = CompletionRequest(prompt="bypass")
        initial = await gateway.complete(request)
        lookups = gateway.cache.hits, gateway.cache.misses
        provider.release.clear()
        uncached = request.model_copy(update={"cache": False})
        tasks = [asyncio.create_task(gateway.complete(uncached)) for _ in range(2)]
        try:
            await provider.wait_for_calls(3)
            assert (gateway.cache.hits, gateway.cache.misses) == lookups
            assert gateway._singleflight.shared_calls == 0
            provider.release.set()
            fresh = await asyncio.gather(*tasks)
            assert all(not r.cached and not r.deduplicated for r in fresh)
            assert {r.text for r in fresh} == {"execution-2", "execution-3"}
            cached = await gateway.complete(request)
            assert cached.cached and cached.text == initial.text == "execution-1"
            assert len(provider.calls) == 3 and gateway.cache.size == 1
        finally:
            provider.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("mode", ["developer", "public-demo"])
async def test_uncached_request_does_not_join_an_active_cached_flight(mode):
    async with gateway_case(mode) as (gateway, provider):
        request = CompletionRequest(prompt="active flight")
        provider.release.clear()
        tasks = [asyncio.create_task(gateway.complete(request))]
        try:
            await provider.started.wait()
            tasks.append(
                asyncio.create_task(gateway.complete(request.model_copy(update={"cache": False})))
            )
            await provider.wait_for_calls(2)
            assert gateway._singleflight.shared_calls == 0
            provider.release.set()
            a, b = await asyncio.gather(*tasks)
            assert [a.text, b.text] == ["execution-1", "execution-2"]
            assert not a.cached and not b.cached
            assert (await gateway.complete(request)).text == a.text
            assert len(provider.calls) == 2 and gateway.cache.size == 1
        finally:
            provider.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("mode", ["developer", "public-demo"])
async def test_gateway_keys_preserve_ttl_and_lru_behavior(mode, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(cache_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    cache = TTLLRUCache(max_size=2, ttl_seconds=1)
    async with gateway_case(mode, cache=cache) as (gateway, provider):
        a, b, c = [CompletionRequest(prompt=value) for value in ("a", "b", "c")]
        await gateway.complete(a)
        assert (await gateway.complete(a)).cached
        clock.now = 2
        assert not (await gateway.complete(a)).cached  # expired entry starts new work
        await gateway.complete(b)
        assert (await gateway.complete(a)).cached  # refresh A, so C evicts B
        await gateway.complete(c)
        assert (await gateway.complete(a)).cached
        assert not (await gateway.complete(b)).cached
        assert len(provider.calls) == 5 and cache.size == 2


@pytest.mark.parametrize("mode", ["developer", "public-demo"])
@pytest.mark.parametrize("kind", ["temperature", "messages"])
async def test_explicit_http_batch_separates_distinct_inputs_and_shares_duplicates(
    mode, kind, monkeypatch
):
    async with asyncio.timeout(4):
        monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", mode)
        for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
            monkeypatch.delenv(name, raising=False)
        module = _load_application(monkeypatch)
        provider = next(p for p in module.gateway.providers if isinstance(p, MockProvider))
        probe = CountingProvider()
        probe.release.clear()
        monkeypatch.setattr(provider, "complete", probe.complete)
        module.gateway._batcher.max_batch_size = 1
        prepared = asyncio.Event()
        original = module.gateway._singleflight.prepare_many

        async def track(*args, **kwargs):
            result = await original(*args, **kwargs)
            if module.gateway._singleflight.total_calls == 3:
                prepared.set()
            return result

        monkeypatch.setattr(module.gateway._singleflight, "prepare_many", track)
        first, second = collision_pair(kind)
        payload = {"requests": [r.model_dump() for r in (first, second, first)]}
        async with (
            module.app.router.lifespan_context(module.app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=module.app), base_url="http://test"
            ) as client,
        ):
            task = asyncio.create_task(client.post("/v1/completions/batch", json=payload))
            try:
                await probe.started.wait()
                await prepared.wait()
                probe.release.set()
                response = await task
                assert response.status_code == 200
                results = response.json()["responses"]
                assert [r["text"] for r in results] == ["execution-1", "execution-2", "execution-1"]
                assert [r["deduplicated"] for r in results] == [False, False, True]
                assert probe.calls == [effective_inputs(first), effective_inputs(second)]
                again = await client.post("/v1/completions/batch", json=payload)
                assert again.status_code == 200
                assert all(r["cached"] and not r["deduplicated"] for r in again.json()["responses"])
                assert len(probe.calls) == 2
                assert (module.gateway.cache.hits, module.gateway.cache.misses) == (3, 3)
                if module.gateway.admission is not None:
                    budget = module.gateway.admission
                    assert (budget.active, budget.pending, budget.requests) == (0, 0, 0)
            finally:
                probe.release.set()
                await asyncio.gather(task, return_exceptions=True)
