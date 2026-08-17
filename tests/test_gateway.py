import asyncio

import pytest

from backend.gateway import Gateway, NoProviderAvailable
from backend.models import CompletionRequest
from backend.providers.base import Provider
from backend.providers.mock_provider import MockProvider


class FailingProvider(Provider):
    name = "failing"

    async def complete(self, messages, *, model, max_tokens, temperature):
        raise RuntimeError("simulated provider outage")

    async def health(self) -> bool:
        return False


async def test_basic_completion():
    gw = Gateway(providers=[MockProvider(base_latency_ms=1, jitter_ms=1)])
    resp = await gw.complete(CompletionRequest(prompt="hello world", model="mock-small"))
    assert resp.provider == "mock"
    assert not resp.cached
    assert resp.output_tokens > 0


async def test_cache_hit_on_repeated_request():
    gw = Gateway(providers=[MockProvider(base_latency_ms=1, jitter_ms=1)])
    req = CompletionRequest(prompt="repeat me", model="mock-small", cache=True)
    first = await gw.complete(req)
    second = await gw.complete(req)
    assert not first.cached
    assert second.cached
    assert second.text == first.text


async def test_cache_disabled_bypasses_cache():
    gw = Gateway(providers=[MockProvider(base_latency_ms=1, jitter_ms=1)])
    req = CompletionRequest(prompt="no cache please", model="mock-small", cache=False)
    first = await gw.complete(req)
    second = await gw.complete(req)
    assert not first.cached
    assert not second.cached


async def test_fallback_to_next_provider_on_failure():
    gw = Gateway(providers=[FailingProvider(), MockProvider(base_latency_ms=1, jitter_ms=1)])
    resp = await gw.complete(CompletionRequest(prompt="fallback test", model="mock-small"))
    assert resp.provider == "mock"


async def test_no_provider_available_raises():
    gw = Gateway(providers=[FailingProvider()])
    with pytest.raises(NoProviderAvailable):
        await gw.complete(CompletionRequest(prompt="doomed", model="mock-small"))


async def test_concurrent_requests_are_batched():
    gw = Gateway(providers=[MockProvider(base_latency_ms=5, jitter_ms=0)], batch_window_ms=20, max_batch_size=32)
    reqs = [CompletionRequest(prompt=f"p{i}", model="mock-small", cache=False) for i in range(10)]
    responses = await asyncio.gather(*(gw.complete(r) for r in reqs))
    assert len(responses) == 10
    assert gw._batcher.stats["batches_executed"] <= 3  # should coalesce, not fire 10 separate batches


async def test_concurrent_duplicate_requests_singleflight():
    gw = Gateway(providers=[MockProvider(base_latency_ms=10, jitter_ms=0)], batch_window_ms=10, max_batch_size=32)
    reqs = [CompletionRequest(prompt="same-prompt", model="mock-small", cache=True) for _ in range(10)]
    responses = await asyncio.gather(*(gw.complete(r) for r in reqs))
    assert len(responses) == 10
    # Exactly one execution executed by the underlying provider
    assert gw._batcher.stats["batches_executed"] == 1
    # 9 requests were deduplicated in-flight
    dedup_count = sum(1 for r in responses if r.deduplicated)
    assert dedup_count == 9


async def test_metrics_recorded():
    gw = Gateway(providers=[MockProvider(base_latency_ms=1, jitter_ms=1)])
    await gw.complete(CompletionRequest(prompt="metric me", model="mock-small"))
    snap = gw.metrics.snapshot()
    assert snap["total_requests"] == 1
    assert snap["requests_by_provider"]["mock"] == 1
