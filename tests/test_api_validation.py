import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from pydantic import ValidationError

from backend.models import CompletionRequest
from backend.providers import MockProvider

_INVALID_COMPLETIONS = [
    pytest.param({}, id="missing"),
    pytest.param({"prompt": None, "messages": None}, id="null-inputs"),
    pytest.param({"prompt": ""}, id="empty-prompt"),
    pytest.param({"prompt": " \t\n"}, id="blank-prompt"),
    pytest.param({"messages": []}, id="empty-messages"),
    pytest.param({"messages": [{"role": "user", "content": ""}]}, id="empty-message"),
    pytest.param({"messages": [{"role": "user", "content": " \n"}]}, id="blank-message"),
    pytest.param(
        {"prompt": "prompt", "messages": [{"role": "user", "content": "message"}]},
        id="conflicting-inputs",
    ),
    pytest.param({"prompt": "valid", "messages": []}, id="prompt-and-empty-messages"),
    pytest.param({"prompt": "x" * 8001}, id="long-prompt"),
    pytest.param({"messages": [{"role": "user", "content": "x" * 8001}]}, id="long-message"),
    pytest.param({"messages": [{"role": "user", "content": "valid"}] * 33}, id="too-many-messages"),
    pytest.param({"messages": [{"role": "invalid", "content": "text"}]}, id="invalid-role"),
    pytest.param({"prompt": "text", "temperature": -1}, id="invalid-temperature"),
    pytest.param({"prompt": "text", "max_tokens": 0}, id="invalid-max-tokens"),
]


@pytest_asyncio.fixture(params=["developer", "public-demo"])
async def api(request, monkeypatch):
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", request.param)
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    name = "backend._validation_test_main"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "backend" / "main.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    calls = {"provider": 0, "retrieval": 0}

    async def complete(*args, **kwargs):
        calls["provider"] += 1
        return "mock boundary result", 1, 1

    provider = next(p for p in module.gateway.providers if isinstance(p, MockProvider))
    monkeypatch.setattr(provider, "complete", complete)
    original_query = module.rag_pipeline.query

    def query(*args, **kwargs):
        calls["retrieval"] += 1
        return original_query(*args, **kwargs)

    monkeypatch.setattr(module.rag_pipeline, "query", query)
    async with (
        asyncio.timeout(5),
        module.app.router.lifespan_context(module.app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=module.app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client,
    ):
        yield module, client, calls


def _assert_no_execution(module, calls):
    assert calls == {"provider": 0, "retrieval": 0}
    assert module.gateway._batcher.stats["items_processed"] == 0
    assert module.gateway._batcher._queue == []
    assert module.gateway._singleflight.total_calls == 0
    snapshot = module.gateway.metrics.snapshot()
    assert snapshot["total_requests"] == 0
    assert snapshot["errors"] == 0
    assert snapshot["cache_hits"] == 0
    assert snapshot["cache_misses"] == 0


@pytest.mark.parametrize("payload", _INVALID_COMPLETIONS)
def test_completion_model_rejects_invalid_input(payload):
    with pytest.raises(ValidationError):
        CompletionRequest.model_validate(payload)


@pytest.mark.parametrize("payload", _INVALID_COMPLETIONS)
@pytest.mark.parametrize("path", ["/v1/completions", "/v1/completions/stream"])
async def test_invalid_completion_rejected_before_execution(api, payload, path):
    module, client, calls = api
    response = await client.post(path, json=payload)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert "Traceback" not in response.text
    _assert_no_execution(module, calls)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"requests": []}, id="empty-batch"),
        pytest.param({"requests": [{}]}, id="invalid-member"),
        pytest.param({"requests": [{"prompt": "a"}] * 17}, id="oversized-batch"),
    ],
)
async def test_invalid_explicit_batch_rejected_before_execution(api, payload):
    module, client, calls = api
    response = await client.post("/v1/completions/batch", json=payload)
    assert response.status_code == 422
    _assert_no_execution(module, calls)


async def test_mixed_batch_rejection_has_no_partial_effects_or_poisoned_futures(api):
    module, client, calls = api
    # If admitted by the old code, these two items fill one flush: the first
    # executes before the second raises, failing both requests batch-wide.
    module.gateway._batcher.max_batch_size = 2
    malformed = {
        "requests": [{"prompt": "valid but must not execute", "cache": False}, {"cache": False}]
    }
    response = await client.post("/v1/completions/batch", json=malformed)
    assert response.status_code == 422
    _assert_no_execution(module, calls)

    # Malformed and valid independent HTTP requests must not share poisoned work.
    bad, good = await asyncio.gather(
        client.post("/v1/completions/batch", json=malformed),
        client.post("/v1/completions", json={"prompt": "independent", "cache": False}),
    )
    assert bad.status_code == 422
    assert good.status_code == 200
    assert calls["provider"] == 1
    assert module.gateway.metrics.snapshot()["total_requests"] == 1
    assert module.gateway.metrics.snapshot()["errors"] == 0
    await module.gateway.aclose()
    assert module.gateway._singleflight.in_flight_count == 0
    assert module.gateway._batcher._queue == []
    assert module.gateway._batcher._batch_tasks == set()
    assert module.gateway._batcher._flush_tasks == set()


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({}, id="missing-query"),
        pytest.param({"query": ""}, id="empty-query"),
        pytest.param({"query": " \t\n"}, id="blank-query"),
        pytest.param({"query": "x" * 2001}, id="long-query"),
        pytest.param({"query": "cache", "top_k": 0}, id="zero-top-k"),
        pytest.param({"query": "cache", "top_k": -1}, id="negative-top-k"),
        pytest.param({"query": "cache", "top_k": 21}, id="large-top-k"),
        pytest.param({"query": "cache", "top_k": "1.5"}, id="fractional-top-k"),
        pytest.param({"query": "cache", "mode": "dense"}, id="unsupported-mode"),
    ],
)
async def test_invalid_retrieval_rejected_before_search(api, params):
    module, client, calls = api
    response = await client.get("/v1/rag/search", params=params)
    assert response.status_code == 422
    _assert_no_execution(module, calls)


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "hello"},
        {"prompt": "x" * 8000},
        {"messages": [{"role": "user", "content": "x" * 8000}]},
        {"messages": [{"role": "user", "content": "hello"}] * 32},
        {
            "prompt": None,
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            "temperature": 0,
            "max_tokens": 8192,
        },
        {"prompt": "hello", "messages": None},
    ],
)
async def test_valid_completion_boundaries_preserve_http_and_sse_contract(api, payload):
    _, client, calls = api
    response = await client.post("/v1/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "mock boundary result"
    assert data["provider"] == "mock"
    assert data["model"] == "mock-small"
    assert data["cached"] is False
    assert data["deduplicated"] is False
    assert {"input_tokens", "output_tokens", "latency_ms"} <= data.keys()
    stream = await client.post("/v1/completions/stream", json=payload)
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")
    events = [line[6:] for line in stream.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    assert "".join(json.loads(event)["text"] for event in events[:-1]) == "mock boundary result"
    assert calls["provider"] == 2


async def test_maximum_explicit_batch_is_accepted(api):
    module, client, calls = api
    response = await client.post(
        "/v1/completions/batch",
        json={"requests": [{"prompt": f"batch-{i}", "cache": False} for i in range(16)]},
    )
    assert response.status_code == 200
    assert len(response.json()["responses"]) == 16
    assert calls["provider"] == 16
    assert module.gateway.metrics.snapshot()["total_requests"] == 16


@pytest.mark.parametrize("mode", ["hybrid", "vector", "bm25"])
@pytest.mark.parametrize("top_k", [1, 20])
async def test_valid_retrieval_limits_and_modes(api, mode, top_k):
    _, client, calls = api
    response = await client.get(
        "/v1/rag/search", params={"query": "cache" + " " * 1995, "mode": mode, "top_k": top_k}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) <= top_k
    assert calls == {"provider": 0, "retrieval": 1}


def test_valid_text_is_preserved_without_trimming():
    prompt = CompletionRequest(prompt="  keep this text \n")
    assert prompt.prompt == "  keep this text \n"
    assert prompt.as_messages()[0].content == "  keep this text \n"
    messages = CompletionRequest(messages=[{"role": "user", "content": "  keep this too \n"}])
    assert messages.as_messages()[0].content == "  keep this too \n"
