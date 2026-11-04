import asyncio

import httpx
import pytest
import pytest_asyncio
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from backend.config import build_rate_limiter
from backend.ingress import BodyLimitMiddleware
from backend.rate_limit import RateLimitMiddleware, TokenBucketLimiter
from tests.test_http_ingress import (
    api as ingress_fixture,
)
from tests.test_http_ingress import (
    assert_no_execution,
    padded_body,
    scope_for,
)
from tests.test_http_ingress import (
    test_ingress_preserves_public_overload_and_deadline_contracts as check_admission_deadlines,
)

api = ingress_fixture  # reuse the isolated composition root and execution spies


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture(autouse=True)
def rate_environment(monkeypatch):
    monkeypatch.setenv("AI_SYSTEMS_LAB_RATE_BURST", "2")
    monkeypatch.setenv("AI_SYSTEMS_LAB_RATE_REFILL_PER_S", "1")
    monkeypatch.setenv("AI_SYSTEMS_LAB_RATE_MAX_CLIENTS", "8")


@pytest_asyncio.fixture
async def limited_api(api, monkeypatch):
    module, client, calls = api
    clock = Clock()
    limiter = getattr(module, "_RATE_LIMITER", None)
    if limiter is not None:
        monkeypatch.setattr(limiter, "_clock", clock)
    return module, client, calls, clock


@pytest.mark.parametrize("retrieval", [False, True])
async def test_public_http_burst_rejects_excess_without_execution(limited_api, retrieval):
    async with asyncio.timeout(3):
        _, client, calls, _ = limited_api
        responses = []
        for i in range(3):
            if retrieval:
                responses.append(await client.get("/v1/rag/search", params={"query": "cache"}))
            else:
                responses.append(
                    await client.post("/v1/completions", json={"prompt": str(i), "cache": False})
                )
        key = "retrieval" if retrieval else "provider"
        assert ([r.status_code for r in responses], calls[key]) == ([200, 200, 429], 2)
        assert responses[-1].json() == {
            "error": {"code": "rate_limited", "message": "Too many requests"}
        }


async def raw_request(
    app,
    *,
    client=("127.0.0.1", 1234),
    headers=(),
    path="/metrics",
    method="GET",
    body=b"",
    forbid_read=False,
):
    scope = scope_for(path, method, headers)
    scope["client"] = client
    messages = []
    reads = 0

    async def receive():
        nonlocal reads
        reads += 1
        assert not forbid_read, "Rate rejection must precede any body read"
        assert reads == 1
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    async with asyncio.timeout(3):
        await app(scope, receive, send)
    assert sum(message["type"] == "http.response.start" for message in messages) == 1
    assert not messages[-1].get("more_body", False)
    return httpx.Response(
        messages[0]["status"],
        headers=messages[0].get("headers", []),
        content=b"".join(message.get("body", b"") for message in messages),
    ), reads


def test_token_boundary_fractional_refill_and_retry_after():
    clock = Clock()
    limiter = TokenBucketLimiter(2, 0.5, clock=clock)
    assert limiter.check(b"a").allowed
    assert limiter.check(b"a").allowed
    assert limiter.check(b"a").retry_after == 2
    clock.advance(0.5)
    assert limiter.check(b"a").retry_after == 2  # ceil(0.75 / 0.5)
    clock.advance(0.5)
    assert limiter.check(b"a").retry_after == 1
    clock.advance(1)
    assert limiter.check(b"a").allowed  # exactly one replenished token
    clock.advance(100)
    assert [limiter.check(b"a").allowed for _ in range(3)] == [True, True, False]


def test_clock_rewind_cannot_mint_tokens():
    clock = Clock()
    limiter = TokenBucketLimiter(1, 1, clock=clock)
    assert limiter.check(b"a").allowed
    clock.advance(0.5)
    assert not limiter.check(b"a").allowed
    clock.advance(-0.5)
    assert not limiter.check(b"a").allowed
    clock.advance(1)
    assert limiter.check(b"a").allowed
    assert not limiter.check(b"a").allowed


async def test_concurrent_checks_admit_exactly_the_burst():
    async with asyncio.timeout(2):
        limiter = TokenBucketLimiter(3, 1, clock=Clock())

        async def check():
            return limiter.check(b"same").allowed

        outcomes = await asyncio.gather(*(check() for _ in range(100)))
        assert sum(outcomes) == 3
        assert len(limiter._buckets) == 1
        assert limiter._buckets[b"same"].tokens == 0


def test_bounded_registry_preserves_active_quotas_under_identity_churn():
    clock = Clock()
    limiter = TokenBucketLimiter(2, 1, 2, clock=clock)
    for key in (b"a", b"b"):
        assert [limiter.check(key).allowed for _ in range(2)] == [True, True]
    for i in range(1000):
        decision = limiter.check(str(i).encode())
        assert not decision.allowed and decision.retry_after is None
    assert set(limiter._buckets) == {b"a", b"b"}
    assert not limiter.check(b"a").allowed
    clock.advance(60)
    assert [limiter.check(b"a").allowed for _ in range(2)] == [True, True]
    assert limiter.check(b"c").allowed  # only idle, fully replenished b is reclaimed
    assert set(limiter._buckets) == {b"a", b"c"}
    assert not limiter.check(b"a").allowed  # active quota was never reset
    assert not limiter.check(b"d").allowed
    assert len(limiter._buckets) == 2


def test_reclaim_waits_until_full_replenishment_as_well_as_idle_time():
    clock = Clock()
    limiter = TokenBucketLimiter(120, 1, 1, clock=clock)
    assert all(limiter.check(b"a").allowed for _ in range(120))
    clock.advance(60)
    assert not limiter.check(b"b").allowed
    assert list(limiter._buckets) == [b"a"]
    clock.advance(60)
    assert limiter.check(b"b").allowed
    assert list(limiter._buckets) == [b"b"]


async def test_rate_middleware_precedes_body_read_and_has_no_execution_effects(limited_api):
    async with asyncio.timeout(3):
        module, client, calls, _ = limited_api
        assert [middleware.cls for middleware in module.app.user_middleware] == [
            RateLimitMiddleware,
            BodyLimitMiddleware,
        ]
        assert (await client.get("/metrics")).status_code == 200
        assert (await client.get("/metrics")).status_code == 200
        response, reads = await raw_request(
            module.app,
            path="/v1/completions",
            method="POST",
            forbid_read=True,
            headers=[(b"content-length", b"999999")],
        )
        assert response.status_code == 429 and reads == 0  # before even declared-size rejection
        assert response.headers["retry-after"] == "1"
        assert response.json() == {
            "error": {"code": "rate_limited", "message": "Too many requests"}
        }
        assert "access-control-allow-origin" not in response.headers
        assert_no_execution(module, calls)


async def test_concurrent_application_requests_read_only_admitted_bodies(limited_api):
    async with asyncio.timeout(3):
        module, _, calls, _ = limited_api
        results = await asyncio.gather(*(raw_request(module.app) for _ in range(32)))
        assert sum(response.status_code == 200 for response, _ in results) == 2
        assert sum(response.status_code == 429 for response, _ in results) == 30
        assert sum(reads for _, reads in results) == 2
        assert_no_execution(module, calls)


@pytest.mark.parametrize(
    "path,method",
    [
        ("/v1/completions", "POST"),
        ("/v1/completions/batch", "POST"),
        ("/v1/completions/stream", "POST"),
        ("/v1/rag/search?query=cache", "GET"),
        ("/metrics/prometheus", "GET"),
        ("/", "GET"),
        ("/unknown", "GET"),
        ("/health/", "GET"),
        ("/health", "POST"),
        ("/health", "HEAD"),
    ],
)
async def test_protected_routes_reject_before_body_or_streaming(limited_api, path, method):
    async with asyncio.timeout(3):
        module, client, calls, _ = limited_api
        await client.get("/metrics")
        await client.get("/metrics")
        response, reads = await raw_request(module.app, path=path, method=method, forbid_read=True)
        assert response.status_code == 429 and reads == 0
        assert response.headers["content-type"].startswith("application/json")
        assert "[DONE]" not in response.text and "event:" not in response.text
        assert_no_execution(module, calls)


async def test_retry_after_and_success_after_controlled_replenishment(limited_api):
    async with asyncio.timeout(3):
        module, client, calls, clock = limited_api
        await client.get("/metrics")
        await client.get("/metrics")
        denied = await client.post("/v1/completions", json={"prompt": "denied"})
        assert denied.status_code == 429 and denied.headers["retry-after"] == "1"
        assert_no_execution(module, calls)
        clock.advance(1)
        assert (
            await client.post("/v1/completions", json={"prompt": "accepted"})
        ).status_code == 200
        assert calls["provider"] == 1


async def test_explicit_batch_counts_once_and_accepted_sse_remains_normal(limited_api):
    async with asyncio.timeout(3):
        _, client, calls, _ = limited_api
        batch = await client.post(
            "/v1/completions/batch", json={"requests": [{"prompt": "a"}, {"prompt": "b"}]}
        )
        assert batch.status_code == 200 and calls["provider"] == 2
        stream = await client.post("/v1/completions/stream", json={"prompt": "stream"})
        assert stream.status_code == 200 and "[DONE]" in stream.text
        assert calls["provider"] == 3
        denied = await client.get("/v1/rag/search", params={"query": "cache"})
        assert denied.status_code == 429 and calls["retrieval"] == 0


async def test_health_is_exact_cheap_exemption_even_when_registry_is_full(limited_api):
    async with asyncio.timeout(3):
        module, client, calls, _ = limited_api
        module._RATE_LIMITER.max_clients = 1
        await client.get("/metrics")
        await client.get("/metrics")
        for _ in range(5):
            response, _ = await raw_request(
                module.app, client=("203.0.113.1", 3000), path="/health"
            )
            assert response.status_code == 200
        assert len(module._RATE_LIMITER._buckets) == 1
        denied, _ = await raw_request(
            module.app,
            client=("203.0.113.1", 3000),
            path="/v1/rag/search?query=cache",
            forbid_read=True,
        )
        assert denied.status_code == 429 and "retry-after" not in denied.headers
        assert_no_execution(module, calls)


@pytest.mark.parametrize(
    "first,alias",
    [
        (("198.51.100.1", 1000), ("198.51.100.1", 5000)),
        (("2001:db8::1", 1000), ("2001:0DB8:0:0:0:0:0:1", 5000)),
        (("198.51.100.1", 1000), ("::ffff:198.51.100.1", 5000)),
    ],
)
async def test_canonical_ip_identity_ignores_port_and_aliases(limited_api, first, alias):
    async with asyncio.timeout(3):
        module, _, calls, _ = limited_api
        for _ in range(2):
            assert (await raw_request(module.app, client=first))[0].status_code == 200
        assert (await raw_request(module.app, client=alias, forbid_read=True))[0].status_code == 429
        assert (await raw_request(module.app, client=("203.0.113.2", 1000)))[0].status_code == 200
        assert len(module._RATE_LIMITER._buckets) == 2
        assert all(len(key) in (4, 16) for key in module._RATE_LIMITER._buckets)
        assert_no_execution(module, calls)


@pytest.mark.parametrize("header", [b"x-forwarded-for", b"x-real-ip", b"forwarded"])
async def test_rotating_forwarding_headers_cannot_reset_socket_quota(limited_api, header):
    async with asyncio.timeout(3):
        module, client, calls, _ = limited_api
        await client.get("/metrics")
        await client.get("/metrics")
        for value in (b"198.51.100.1", b"203.0.113.2", b"for=2001:db8::1", b""):
            response, reads = await raw_request(
                module.app, headers=[(header, value)], forbid_read=True
            )
            assert response.status_code == 429 and reads == 0
        assert len(module._RATE_LIMITER._buckets) == 1
        assert_no_execution(module, calls)


async def test_uvicorn_already_rewritten_identity_fails_closed_without_bucket_creation(limited_api):
    async with asyncio.timeout(3):
        module, _, calls, _ = limited_api
        app = ProxyHeadersMiddleware(module.app, trusted_hosts="*")
        for value in (b"198.51.100.1", b"203.0.113.2", b"2001:db8::1", b"unknown"):
            response, reads = await raw_request(
                app, headers=[(b"x-forwarded-for", value)], forbid_read=True
            )
            assert response.status_code == 429 and reads == 0
            assert "retry-after" not in response.headers
        assert module._RATE_LIMITER._buckets == {}
        assert_no_execution(module, calls)


@pytest.mark.parametrize(
    "client_info",
    [
        None,
        (),
        "bad",
        ("bad-host", 1000),
        ("127.0.0.1", "bad-port"),
        ("::1", 0),
        ("::1", -1),
        ("::1", 70000),
        ("fe80::1%zone", 1000),
        ("x" * 1000, 1000),
    ],
)
async def test_unidentified_and_malformed_clients_share_one_bounded_bucket(
    limited_api, client_info
):
    async with asyncio.timeout(3):
        module, _, calls, _ = limited_api
        for _ in range(2):
            assert (await raw_request(module.app, client=None))[0].status_code == 200
        denied, _ = await raw_request(module.app, client=client_info, forbid_read=True)
        assert denied.status_code == 429
        assert list(module._RATE_LIMITER._buckets) == [b""]
        assert_no_execution(module, calls)


async def test_registry_capacity_rejection_has_no_retry_hint_or_downstream_effects(limited_api):
    async with asyncio.timeout(3):
        module, _, calls, clock = limited_api
        module._RATE_LIMITER.max_clients = 1
        assert (await raw_request(module.app, client=("198.51.100.1", 1000)))[0].status_code == 200
        denied, _ = await raw_request(module.app, client=("203.0.113.1", 1000), forbid_read=True)
        assert denied.status_code == 429 and "retry-after" not in denied.headers
        assert len(module._RATE_LIMITER._buckets) == 1
        assert_no_execution(module, calls)
        clock.advance(60)
        assert (await raw_request(module.app, client=("203.0.113.1", 1000)))[0].status_code == 200
        assert len(module._RATE_LIMITER._buckets) == 1


async def test_rejections_do_not_cancel_or_release_already_accepted_work(limited_api, monkeypatch):
    async with asyncio.timeout(3):
        module, client, calls, _ = limited_api
        started, release = asyncio.Event(), asyncio.Event()

        async def held(*args, **kwargs):
            calls["provider"] += 1
            started.set()
            await release.wait()
            return "held result", 1, 1

        monkeypatch.setattr(module.gateway.providers[0], "complete", held)
        first = asyncio.create_task(client.post("/v1/completions", json={"prompt": "held"}))
        try:
            await started.wait()
            await client.get("/metrics")
            before = dict(calls)
            denied = await client.post("/v1/completions/stream", json={"prompt": "reject"})
            assert denied.status_code == 429 and calls == before
            assert module.gateway.admission.active == 1 and calls["provider"] == 1
            assert not first.done()
        finally:
            release.set()
            assert (await first).status_code == 200
        assert module.gateway.admission.active == 0


async def test_shared_waiters_and_cache_hits_still_cost_http_tokens(limited_api, monkeypatch):
    async with asyncio.timeout(3):
        module, client, calls, clock = limited_api
        started, release = asyncio.Event(), asyncio.Event()

        async def held(*args, **kwargs):
            calls["provider"] += 1
            started.set()
            await release.wait()
            return "shared result", 1, 1

        monkeypatch.setattr(module.gateway.providers[0], "complete", held)
        first = asyncio.create_task(client.post("/v1/completions", json={"prompt": "same"}))
        joined = None
        try:
            await started.wait()
            joined = asyncio.create_task(client.post("/v1/completions", json={"prompt": "same"}))
            while module.gateway._singleflight.shared_calls != 1:
                await asyncio.sleep(0)  # ordering only; the enclosing test is bounded
            rejected = await client.post("/v1/completions", json={"prompt": "same"})
            assert rejected.status_code == 429
            assert calls["provider"] == 1 and module.gateway.admission.active == 1
        finally:
            release.set()
            responses = await asyncio.gather(
                *([first, joined] if joined is not None else [first]), return_exceptions=True
            )
        assert all(response.status_code == 200 for response in responses)
        assert responses[1].json()["deduplicated"] is True
        clock.advance(1)
        cached = await client.post("/v1/completions", json={"prompt": "same"})
        assert cached.status_code == 200 and cached.json()["cached"] is True
        assert (await client.post("/v1/completions", json={"prompt": "same"})).status_code == 429
        assert calls["provider"] == 1 and module.gateway.admission.active == 0


async def test_admitted_oversized_body_preserves_413_and_still_costs_one_token(limited_api):
    async with asyncio.timeout(3):
        module, client, calls, _ = limited_api
        too_large = await client.post(
            "/v1/completions",
            content=padded_body(513),
            headers={"content-type": "application/json"},
        )
        assert too_large.status_code == 413
        assert_no_execution(module, calls)
        assert (await client.post("/v1/completions", json={"prompt": "valid"})).status_code == 200
        assert (await client.get("/metrics")).status_code == 429
        assert calls["provider"] == 1


async def test_accepted_requests_preserve_503_and_504(limited_api, monkeypatch):
    async with asyncio.timeout(3):
        module, client, calls, _ = limited_api
        await check_admission_deadlines((module, client, calls), monkeypatch)


@pytest.mark.parametrize("api", [{"mode": "developer"}], indirect=True)
async def test_developer_mode_has_no_rate_middleware_and_ignores_invalid_settings(
    limited_api, monkeypatch
):
    async with asyncio.timeout(3):
        module, client, calls, _ = limited_api
        assert module._RATE_LIMITER is None
        for _ in range(4):
            assert (
                await client.post("/v1/completions", json={"prompt": "dev", "cache": False})
            ).status_code == 200
        assert calls["provider"] == 4
        monkeypatch.setenv("AI_SYSTEMS_LAB_RATE_BURST", "private-invalid")
        monkeypatch.setenv("AI_SYSTEMS_LAB_RATE_REFILL_PER_S", "nan")
        monkeypatch.setenv("AI_SYSTEMS_LAB_RATE_MAX_CLIENTS", "0")
        assert build_rate_limiter() is None


@pytest.mark.parametrize(
    "name,value",
    [
        (name, value)
        for name in ("AI_SYSTEMS_LAB_RATE_BURST", "AI_SYSTEMS_LAB_RATE_MAX_CLIENTS")
        for value in ("", "0", "-1", "1.5", "nan", "inf", "private-invalid", "9" * 5000)
    ]
    + [
        ("AI_SYSTEMS_LAB_RATE_BURST", "121"),
        ("AI_SYSTEMS_LAB_RATE_MAX_CLIENTS", "4097"),
    ]
    + [
        ("AI_SYSTEMS_LAB_RATE_REFILL_PER_S", value)
        for value in (
            "",
            "0",
            "-1",
            "nan",
            "inf",
            "0.001",
            "11",
            "private-invalid",
            "1e999",
            "1e-999",
        )
    ],
)
def test_invalid_rate_settings_fail_startup_without_echoing_values(monkeypatch, name, value):
    from tests.test_app_config import _load_application

    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "public-demo")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name) as error:
        _load_application(monkeypatch)
    assert "private-invalid" not in str(error.value)


def test_default_configuration_and_startup_snapshot(monkeypatch):
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "public-demo")
    for name in (
        "AI_SYSTEMS_LAB_RATE_BURST",
        "AI_SYSTEMS_LAB_RATE_REFILL_PER_S",
        "AI_SYSTEMS_LAB_RATE_MAX_CLIENTS",
    ):
        monkeypatch.delenv(name, raising=False)
    limiter = build_rate_limiter()
    assert (limiter.burst, limiter.refill_per_s, limiter.max_clients) == (60, 2, 1024)
    assert limiter.reclaim_after_s == 60
    monkeypatch.setenv("AI_SYSTEMS_LAB_RATE_BURST", "1")
    assert limiter.burst == 60


@pytest.mark.parametrize(
    "name,value,attribute,expected",
    [
        ("AI_SYSTEMS_LAB_RATE_BURST", "1", "burst", 1),
        ("AI_SYSTEMS_LAB_RATE_BURST", "120", "burst", 120),
        ("AI_SYSTEMS_LAB_RATE_REFILL_PER_S", "0.01", "refill_per_s", 0.01),
        ("AI_SYSTEMS_LAB_RATE_REFILL_PER_S", "10", "refill_per_s", 10),
        ("AI_SYSTEMS_LAB_RATE_MAX_CLIENTS", "1", "max_clients", 1),
        ("AI_SYSTEMS_LAB_RATE_MAX_CLIENTS", "4096", "max_clients", 4096),
    ],
)
def test_valid_configuration_boundaries(monkeypatch, name, value, attribute, expected):
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "public-demo")
    monkeypatch.setenv(name, value)
    assert getattr(build_rate_limiter(), attribute) == expected


@pytest.mark.parametrize("kind", ["lifespan", "websocket"])
async def test_rate_limiter_does_not_touch_non_http_scopes(kind):
    async with asyncio.timeout(2):
        limiter = TokenBucketLimiter(clock=Clock())
        called = []

        async def downstream(scope, receive, send):
            called.append(scope["type"])

        await RateLimitMiddleware(downstream, limiter)({"type": kind}, None, None)
        assert called == [kind] and limiter._buckets == {}
