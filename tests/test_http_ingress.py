import asyncio
import json

import httpx
import pytest
import pytest_asyncio
from starlette.requests import Request

from backend.config import build_body_limit
from backend.ingress import BodyLimitMiddleware
from backend.models import BatchCompletionRequest
from backend.providers import MockProvider
from tests.test_app_config import _load_application


@pytest_asyncio.fixture
async def api(request, monkeypatch):
    settings = getattr(request, "param", {})
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", settings.get("mode", "public-demo"))
    limit = settings.get("limit", 512)
    if limit is None:
        monkeypatch.delenv("AI_SYSTEMS_LAB_MAX_BODY_BYTES", raising=False)
    else:
        monkeypatch.setenv("AI_SYSTEMS_LAB_MAX_BODY_BYTES", str(limit))
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    module = _load_application(monkeypatch)
    calls = dict.fromkeys(("provider", "retrieval", "parsing", "submit", "request", "work"), 0)

    async def complete(*args, **kwargs):
        calls["provider"] += 1
        return "mock result", 1, 1

    provider = next(p for p in module.gateway.providers if isinstance(p, MockProvider))
    monkeypatch.setattr(provider, "complete", complete)
    original_query = module.rag_pipeline.query
    original_json = Request.json
    original_submit = module.gateway._batcher.submit

    def query(*args, **kwargs):
        calls["retrieval"] += 1
        return original_query(*args, **kwargs)

    async def parse(self):
        calls["parsing"] += 1
        return await original_json(self)

    async def submit(*args, **kwargs):
        calls["submit"] += 1
        return await original_submit(*args, **kwargs)

    monkeypatch.setattr(module.rag_pipeline, "query", query)
    monkeypatch.setattr(Request, "json", parse)
    monkeypatch.setattr(module.gateway._batcher, "submit", submit)
    if module.gateway.admission is not None:
        original_request = module.gateway.admission.reserve_request
        original_work = module.gateway.admission.reserve_work

        def reserve_request():
            calls["request"] += 1
            return original_request()

        def reserve_work(*args, **kwargs):
            calls["work"] += 1
            return original_work(*args, **kwargs)

        monkeypatch.setattr(module.gateway.admission, "reserve_request", reserve_request)
        monkeypatch.setattr(module.gateway.admission, "reserve_work", reserve_work)

    async with (
        asyncio.timeout(5),
        module.app.router.lifespan_context(module.app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=module.app), base_url="http://test"
        ) as client,
    ):
        yield module, client, calls


def assert_no_execution(module, calls, *, parsing=0):
    assert calls == {name: parsing if name == "parsing" else 0 for name in calls}, calls
    assert module.gateway._singleflight.total_calls == 0
    assert module.gateway._singleflight.in_flight_count == 0
    assert module.gateway._batcher._queue == []
    assert module.gateway._batcher._batch_tasks == set()
    assert module.gateway._batcher.stats["items_processed"] == 0
    metrics = module.gateway.metrics.snapshot()
    assert metrics["total_requests"] == metrics["errors"] == 0
    assert metrics["cache_hits"] == metrics["cache_misses"] == 0
    budget = module.gateway.admission
    assert (budget.active, budget.pending, budget.requests) == (0, 0, 0)


@pytest.mark.parametrize("api", [{"limit": None}], indirect=True)
async def test_public_default_rejects_oversized_json_before_parsing(api):
    module, client, calls = api
    body = json.dumps({"prompt": "test", "padding": "x" * 262144}).encode()
    response = await client.post(
        "/v1/completions", content=body, headers={"content-type": "application/json"}
    )
    assert (response.status_code, calls["provider"]) == (413, 0)
    assert response.json() == {
        "error": {"code": "request_too_large", "message": "Request body too large"}
    }
    assert_no_execution(module, calls)


async def test_public_preflight_does_not_advertise_wildcard_access(api):
    _, client, _ = api
    response = await client.options(
        "/v1/completions",
        headers={"origin": "https://foreign.invalid", "access-control-request-method": "POST"},
    )
    assert (response.status_code, response.headers.get("access-control-allow-origin")) in (
        (400, None),
        (405, None),
    )


def padded_body(size, prompt="valid"):
    body = json.dumps({"prompt": prompt, "cache": False}, ensure_ascii=False).encode()
    assert len(body) <= size
    return body + b" " * (size - len(body))


def scope_for(path="/v1/completions", method="POST", headers=()):
    path, _, query = path.partition("?")
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "server": ("test", 80),
        "client": ("127.0.0.1", 1234),
        "headers": [(b"content-type", b"application/json"), *headers],
    }


async def drive(app, chunks, *, headers=(), path="/v1/completions", method="POST"):
    """Supply real ASGI chunks; observe exactly how far ingress reads."""
    frames = [
        {"type": "http.request", "body": chunk, "more_body": i < len(chunks) - 1}
        for i, chunk in enumerate(chunks)
    ]
    messages = []
    reads = 0

    async def receive():
        nonlocal reads
        reads += 1
        if frames:
            return frames.pop(0)
        raise AssertionError("Read beyond the supplied HTTP body")

    async def send(message):
        messages.append(message)

    async with asyncio.timeout(2):
        await app(scope_for(path, method, headers), receive, send)
    assert sum(message["type"] == "http.response.start" for message in messages) == 1
    assert messages[-1]["type"] == "http.response.body"
    assert not messages[-1].get("more_body", False)
    response = httpx.Response(
        messages[0]["status"],
        headers=messages[0].get("headers", []),
        content=b"".join(message.get("body", b"") for message in messages),
    )
    return response, reads


@pytest.mark.parametrize("size,status", [(511, 200), (512, 200), (513, 413)])
@pytest.mark.parametrize("length", ["absent", "accurate", "underreported"])
async def test_actual_byte_boundaries_and_misleading_lengths(api, size, status, length):
    module, _, calls = api
    headers = (
        []
        if length == "absent"
        else [(b"content-length", str(size if length == "accurate" else 1).encode())]
    )
    response, _ = await drive(module.app, [padded_body(size)], headers=headers)
    expected = 400 if size <= 512 and length == "underreported" else status
    assert response.status_code == expected
    if expected != 200:
        assert_no_execution(module, calls)
    else:
        assert calls["provider"] == 1


@pytest.mark.parametrize("length", [None, b"1"])
async def test_chunked_oversize_stops_reading_before_tail_and_execution(api, length):
    module, _, calls = api
    headers = [] if length is None else [(b"content-length", length)]
    response, reads = await drive(
        module.app,
        [b"x" * 256, b"x" * 257, b"unread tail"],
        headers=headers,
    )
    assert response.status_code == 413 and reads == 2
    assert response.json() == {
        "error": {"code": "request_too_large", "message": "Request body too large"}
    }
    assert_no_execution(module, calls)


@pytest.mark.parametrize("length", [b"513", b"9" * 10000, b"0" * 10000 + b"513"])
async def test_declared_oversize_is_rejected_without_receiving_body(api, length):
    module, _, calls = api
    response, reads = await drive(module.app, [], headers=[(b"content-length", length)])
    assert response.status_code == 413 and reads == 0
    assert_no_execution(module, calls)


@pytest.mark.parametrize(
    "headers",
    [
        [(b"content-length", value)]
        for value in (
            b"",
            b" ",
            b"-1",
            b"+1",
            b"1.5",
            b"1e3",
            b"1,1",
            b"secret-invalid",
            b"\xff",
            b"1\n",
        )
    ]
    + [
        [(b"content-length", b"1"), (b"content-length", b"2")],
        [(b"content-length", b"1"), (b"Content-Length", b"1")],
        [(b"content-length", b"1"), (b"transfer-encoding", b"chunked")],
    ],
)
async def test_bad_or_conflicting_lengths_are_sanitized_before_body_read(api, headers):
    module, _, calls = api
    response, reads = await drive(module.app, [], headers=headers)
    assert response.status_code == 400 and reads == 0
    assert response.json() == {
        "error": {"code": "invalid_content_length", "message": "Invalid Content-Length"}
    }
    assert_no_execution(module, calls)


@pytest.mark.parametrize("actual,declared", [(511, b"512"), (512, b"511"), (512, b"0")])
async def test_completed_body_length_must_match_declaration(api, actual, declared):
    module, _, calls = api
    response, _ = await drive(
        module.app, [padded_body(actual)], headers=[(b"content-length", declared)]
    )
    assert response.status_code == 400
    assert_no_execution(module, calls)


@pytest.mark.parametrize(
    "headers", [[], [(b"content-length", b"\t000512 ")], [(b"transfer-encoding", b"chunked")]]
)
async def test_utf8_split_inside_codepoint_is_accepted_at_exact_limit(api, headers):
    module, _, calls = api
    body = padded_body(512, "😀" * 10)
    split = body.index("😀".encode()) + 1
    response, reads = await drive(module.app, [body[:split], body[split:]], headers=headers)
    assert response.status_code == 200 and reads == 2
    assert calls["provider"] == 1


async def test_utf8_is_limited_by_bytes_rather_than_characters(api):
    module, _, calls = api
    text = json.dumps({"prompt": "é" * 300}, ensure_ascii=False)
    assert len(text) < 512 < len(text.encode())
    response, _ = await drive(module.app, [text.encode()])
    assert response.status_code == 413
    assert_no_execution(module, calls)


@pytest.mark.parametrize(
    "path,method",
    [
        ("/v1/completions", "POST"),
        ("/v1/completions/stream", "POST"),
        ("/v1/completions/batch", "POST"),
        ("/v1/rag/search?query=cache", "GET"),
        ("/health", "GET"),
        ("/unknown", "PUT"),
    ],
)
async def test_every_http_route_is_protected_before_any_execution(api, path, method):
    module, _, calls = api
    response, _ = await drive(module.app, [b"private body" * 50], path=path, method=method)
    assert response.status_code == 413 and b"private" not in response.content
    assert_no_execution(module, calls)


@pytest.mark.parametrize("body,status,parsing", [(b"", 422, 0), (b"{}", 422, 1), (b"{", 422, 1)])
async def test_small_malformed_requests_keep_existing_validation(api, body, status, parsing):
    module, _, calls = api
    response, _ = await drive(
        module.app, [body], headers=[(b"content-length", str(len(body)).encode())]
    )
    assert response.status_code == status
    assert_no_execution(module, calls, parsing=parsing)


async def test_empty_health_body_and_small_explicit_batch_remain_valid(api):
    module, client, calls = api
    response, _ = await drive(
        module.app, [b""], path="/health", method="GET", headers=[(b"content-length", b"0")]
    )
    assert response.status_code == 200
    batch = await client.post(
        "/v1/completions/batch", json={"requests": [{"prompt": "one"}, {"prompt": "two"}]}
    )
    assert batch.status_code == 200 and len(batch.json()["responses"]) == 2
    assert calls["provider"] == 2


@pytest.mark.parametrize("api", [{"limit": None}], indirect=True)
async def test_model_valid_batch_can_exceed_total_body_budget(api):
    module, client, calls = api
    payload = {"requests": [{"prompt": "😀" * 8000}] * 16}
    BatchCompletionRequest.model_validate(payload)
    response = await client.post("/v1/completions/batch", json=payload)
    assert response.status_code == 413
    assert_no_execution(module, calls)


async def test_public_same_origin_completion_and_sse_work_without_cors_headers(api):
    _, client, calls = api
    response = await client.get("/")
    assert response.status_code == 200
    for path in ("/v1/completions", "/v1/completions/stream"):
        response = await client.post(path, json={"prompt": path}, headers={"origin": "http://test"})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers
        assert "access-control-allow-credentials" not in response.headers
        if path.endswith("stream"):
            assert response.headers["content-type"].startswith("text/event-stream")
            assert "[DONE]" in response.text
    assert calls["provider"] == 2


async def test_actual_app_sse_disconnect_survives_body_replay(api):
    module, _, calls = api
    sent_first = asyncio.Event()
    body_delivered = False
    baseline = asyncio.all_tasks()

    async def receive():
        nonlocal body_delivered
        if not body_delivered:
            body_delivered = True
            return {"type": "http.request", "body": b'{"prompt":"stream"}', "more_body": False}
        await sent_first.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            sent_first.set()

    scope = scope_for("/v1/completions/stream")
    scope["asgi"]["spec_version"] = "2.0"  # exercise the ASGI disconnect listener
    await module.app(scope, receive, send)
    assert sent_first.is_set() and calls["provider"] == 1
    budget = module.gateway.admission
    assert (budget.active, budget.pending, budget.requests) == (0, 0, 0)
    assert budget._streams == set() and budget._response_tasks == set()
    assert asyncio.all_tasks() == baseline


async def test_cors_does_not_authenticate_direct_http_clients(api):
    _, client, calls = api
    response = await client.post(
        "/v1/completions", json={"prompt": "valid"}, headers={"origin": "https://foreign.invalid"}
    )
    assert response.status_code == 200 and calls["provider"] == 1
    assert not any(name.startswith("access-control-allow-") for name in response.headers)


@pytest.mark.parametrize("api", [{"mode": "developer", "limit": "secret-invalid"}], indirect=True)
async def test_developer_body_and_cors_compatibility(api):
    _, client, calls = api
    headers = {"origin": "https://foreign.invalid"}
    response = await client.post(
        "/v1/completions",
        content=padded_body(262145),
        headers={**headers, "content-type": "application/json"},
    )
    assert response.status_code == 200 and calls["provider"] == 1
    assert response.headers["access-control-allow-origin"] == "*"
    preflight = await client.options(
        "/v1/completions", headers={**headers, "access-control-request-method": "POST"}
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in preflight.headers


async def test_body_limit_is_startup_state(api, monkeypatch):
    module, client, calls = api
    monkeypatch.setenv("AI_SYSTEMS_LAB_MAX_BODY_BYTES", "1024")
    response = await client.post(
        "/v1/completions", content=padded_body(513), headers={"content-type": "application/json"}
    )
    assert response.status_code == 413
    assert_no_execution(module, calls)


@pytest.mark.parametrize(
    "value",
    ["", "0", "-1", "+1", "1.5", " 512", "nan", "inf", "1048577", "secret-invalid", "9" * 5000],
)
def test_invalid_public_body_configuration_fails_startup(monkeypatch, value):
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "public-demo")
    monkeypatch.setenv("AI_SYSTEMS_LAB_MAX_BODY_BYTES", value)
    with pytest.raises(ValueError) as error:
        _load_application(monkeypatch)
    assert (
        str(error.value) == "AI_SYSTEMS_LAB_MAX_BODY_BYTES must be an integer between 1 and 1048576"
    )


@pytest.mark.parametrize("value,expected", [(None, 262144), ("1", 1), ("1048576", 1048576)])
def test_public_body_configuration_boundaries(monkeypatch, value, expected):
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "public-demo")
    if value is None:
        monkeypatch.delenv("AI_SYSTEMS_LAB_MAX_BODY_BYTES", raising=False)
    else:
        monkeypatch.setenv("AI_SYSTEMS_LAB_MAX_BODY_BYTES", value)
    assert build_body_limit() == expected


async def test_disconnect_before_body_end_never_executes_app():
    async with asyncio.timeout(2):
        calls = []
        frames = iter(
            [
                {"type": "http.request", "body": b"partial", "more_body": True},
                {"type": "http.disconnect"},
            ]
        )

        async def receive():
            return next(frames)

        async def downstream(*args):
            calls.append("app")

        async def send(message):
            calls.append(message)

        await BodyLimitMiddleware(downstream, 64)(scope_for(), receive, send)
        assert calls == []


async def test_cancelled_body_receive_leaves_no_reader_tasks():
    async with asyncio.timeout(2):
        receiving = asyncio.Event()
        baseline = asyncio.all_tasks()

        async def receive():
            receiving.set()
            await asyncio.Event().wait()

        async def forbidden(*args):
            raise AssertionError("No app execution or response after cancellation")

        task = asyncio.create_task(
            BodyLimitMiddleware(forbidden, 64)(scope_for(), receive, forbidden)
        )
        await receiving.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert asyncio.all_tasks() == baseline


async def test_replay_preserves_disconnect_channel():
    async with asyncio.timeout(2):
        frames = iter(
            [
                {"type": "http.request", "body": b"ab", "more_body": True},
                {"type": "http.request", "body": b"cd", "more_body": False},
                {"type": "http.disconnect"},
            ]
        )

        async def receive():
            return next(frames)

        async def downstream(scope, replay, send):
            assert await replay() == {"type": "http.request", "body": b"abcd", "more_body": False}
            assert await replay() == {"type": "http.disconnect"}

        await BodyLimitMiddleware(downstream, 4)(scope_for(), receive, None)


@pytest.mark.parametrize("kind", ["lifespan", "websocket"])
async def test_non_http_scopes_are_passed_through(kind):
    async with asyncio.timeout(2):

        async def receive():
            return {"type": kind + ".test"}

        async def downstream(scope, original_receive, send):
            assert scope == {"type": kind} and original_receive is receive

        await BodyLimitMiddleware(downstream, 4)({"type": kind}, receive, None)


async def test_downstream_error_cannot_generate_a_second_ingress_response():
    async with asyncio.timeout(2):
        messages = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        async def downstream(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})
            raise ValueError("internal failure after headers")

        with pytest.raises(ValueError, match="internal failure"):
            await BodyLimitMiddleware(downstream, 4)(scope_for(), receive, send)
        assert [
            message["status"] for message in messages if message["type"] == "http.response.start"
        ] == [200]


async def test_ingress_preserves_public_overload_and_deadline_contracts(api, monkeypatch):
    module, client, calls = api
    started = asyncio.Event()
    release = asyncio.Event()
    stopped = asyncio.Event()

    async def held(*args, **kwargs):
        calls["provider"] += 1
        started.set()
        try:
            await release.wait()
            return "mock result", 1, 1
        finally:
            stopped.set()

    monkeypatch.setattr(module.gateway.providers[0], "complete", held)
    budget = module.gateway.admission
    budget.max_active, budget.max_pending = 1, 0
    budget.work_timeout_s = 0.1
    first = asyncio.create_task(client.post("/v1/completions", json={"prompt": "held"}))
    try:
        await started.wait()
        overloaded = await client.post("/v1/completions", json={"prompt": "excess"})
        assert overloaded.status_code == 503
        assert overloaded.json()["error"]["code"] == "capacity_unavailable"
        timed_out = await first
        assert timed_out.status_code == 504
        assert timed_out.json()["error"]["code"] == "inference_deadline_exceeded"
        await stopped.wait()
        assert calls["provider"] == 1
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)
        await module.gateway.aclose()
    assert (budget.active, budget.pending, budget.requests) == (0, 0, 0)
