import asyncio
from types import SimpleNamespace

import httpx
import pytest
from starlette.responses import JSONResponse

from backend import ingress
from backend.config import build_body_read_limits
from backend.ingress import BodyLimitMiddleware
from tests.test_app_config import _load_application
from tests.test_http_ingress import api as ingress_fixture
from tests.test_http_ingress import assert_no_execution, scope_for

api = ingress_fixture


@pytest.fixture(autouse=True)
def ingress_environment(monkeypatch):
    monkeypatch.setenv("AI_SYSTEMS_LAB_MAX_BODY_READERS", "1")
    monkeypatch.setenv("AI_SYSTEMS_LAB_BODY_TIMEOUT_S", "0.1")


class Upload:
    def __init__(self, app, *, path="/v1/completions", method="POST", headers=()):
        self.app = app
        self.scope = scope_for(path, method, headers)
        self.frames = asyncio.Queue(maxsize=16)
        self.reading = asyncio.Event()
        self.reads = 0
        self.messages = []

    async def receive(self):
        self.reads += 1
        self.reading.set()
        return await self.frames.get()

    async def send(self, message):
        self.messages.append(message)

    def feed(self, body=b"", *, more=False):
        self.frames.put_nowait({"type": "http.request", "body": body, "more_body": more})

    def start(self):
        return asyncio.create_task(self.app(self.scope, self.receive, self.send))

    @property
    def response(self):
        assert sum(m["type"] == "http.response.start" for m in self.messages) == 1
        assert not self.messages[-1].get("more_body", False)
        return httpx.Response(
            self.messages[0]["status"],
            headers=self.messages[0].get("headers", []),
            content=b"".join(m.get("body", b"") for m in self.messages),
        )


async def test_public_rejects_excess_body_readers_before_receiving(api):
    async with asyncio.timeout(2):
        module, _, calls = api
        first = Upload(module.app)
        task = first.start()
        try:
            await first.reading.wait()
            second = Upload(module.app)
            second.feed(b'{"prompt":"must not execute"}')
            await second.start()
            assert (second.response.status_code, second.reads, calls["provider"]) == (503, 0, 0)
            assert second.response.json() == {
                "error": {
                    "code": "ingress_capacity_unavailable",
                    "message": "Request ingress is busy",
                }
            }
            assert_no_execution(module, calls)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_public_stalled_upload_has_finite_total_deadline(api):
    async with asyncio.timeout(2):
        module, _, calls = api
        upload = Upload(module.app)
        upload.feed(b'{"prompt":', more=True)
        task = upload.start()
        try:
            done, _ = await asyncio.wait({task}, timeout=0.25)
            assert task in done, "Upload still reading after the configured 0.1s total deadline"
            await task
            assert upload.response.status_code == 408
            assert upload.response.json() == {
                "error": {"code": "request_timeout", "message": "Request body timed out"}
            }
            assert_no_execution(module, calls)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def accepted(scope, receive, send):
    await receive()  # consume replay, not the original body channel
    await JSONResponse({"accepted": True})(scope, receive, send)


def middleware(**settings):
    return BodyLimitMiddleware(accepted, 64, **settings)


def app_ingress(module):
    layer = module.app.middleware_stack
    while not isinstance(layer, BodyLimitMiddleware):
        layer = layer.app
    return layer


class ControlledTime:
    """Advance phase time or explicitly fire the real timeout without sleeps."""

    def __init__(self, monkeypatch):
        self.now = asyncio.get_running_loop().time()
        self.contexts = []
        self.deadlines = []

        def timeout_at(deadline):
            self.deadlines.append(deadline)
            context = asyncio.timeout_at(deadline)
            self.contexts.append(context)
            return context

        monkeypatch.setattr(
            ingress,
            "asyncio",
            SimpleNamespace(
                get_running_loop=lambda: self,
                current_task=asyncio.current_task,
                CancelledError=asyncio.CancelledError,
                sleep=asyncio.sleep,
                timeout_at=timeout_at,
            ),
        )

    def time(self):
        return self.now

    def expire(self):
        self.now = self.deadlines[-1]
        self.contexts[-1].reschedule(asyncio.get_running_loop().time())


async def test_measured_concurrent_readers_are_bounded_without_waiters():
    async with asyncio.timeout(2):
        app = middleware(max_body_readers=2)
        baseline = asyncio.all_tasks()
        uploads = [Upload(app) for _ in range(2)]
        tasks = [u.start() for u in uploads]
        try:
            await asyncio.gather(*(u.reading.wait() for u in uploads))
            assert app.active_readers == 2
            contenders = [Upload(app) for _ in range(100)]
            await asyncio.gather(*(u.start() for u in contenders))
            assert all(u.response.status_code == 503 and u.reads == 0 for u in contenders)
            assert app.active_readers == 2
            assert asyncio.all_tasks() == baseline | set(tasks)
            for u in uploads:
                u.feed(b"ok")
            await asyncio.gather(*tasks)
            assert all(u.response.status_code == 200 for u in uploads)
            assert app.active_readers == 0
            again = Upload(app)
            again.feed(b"reusable")
            await again.start()
            assert again.response.status_code == 200 and app.active_readers == 0
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize(
    "outcome", ["mismatch", "frame", "oversize", "disconnect", "exception", "cancel"]
)
async def test_reader_capacity_recovers_from_every_upload_exit(outcome):
    async with asyncio.timeout(2):
        executed = []

        async def downstream(*args):
            executed.append(True)
            await accepted(*args)

        app = BodyLimitMiddleware(downstream, 64, max_body_readers=1)
        headers = [(b"content-length", b"4")] if outcome == "mismatch" else []
        upload = Upload(app, headers=headers)
        if outcome == "mismatch":
            upload.feed(b"bad")
        elif outcome == "frame":
            upload.frames.put_nowait({"type": "unexpected.event"})
        elif outcome == "oversize":
            upload.feed(b"x" * 65)
        elif outcome == "disconnect":
            upload.feed(b"partial", more=True)
            upload.frames.put_nowait({"type": "http.disconnect"})
        elif outcome == "exception":

            async def broken_receive():
                assert app.active_readers == 1
                raise RuntimeError("private failure")

            upload.receive = broken_receive
        task = upload.start()
        if outcome == "cancel":
            await upload.reading.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome == "exception":
            with pytest.raises(RuntimeError, match="private failure"):
                await task
        else:
            await task
        assert app.active_readers == 0 and not executed
        if outcome in ("mismatch", "frame", "oversize"):
            assert upload.response.status_code == (413 if outcome == "oversize" else 400)
        else:
            assert upload.messages == []
        again = Upload(app)
        again.feed(b"ok")
        await again.start()
        assert again.response.status_code == 200 and app.active_readers == 0
        assert executed == [True]


@pytest.mark.parametrize("headers", [[], [(b"transfer-encoding", b"chunked")]])
@pytest.mark.parametrize("chunk", [b"", b"x"])
async def test_periodic_partial_frames_do_not_reset_absolute_deadline(monkeypatch, headers, chunk):
    async with asyncio.timeout(2):
        clock = ControlledTime(monkeypatch)
        app = middleware(body_timeout_s=1)
        upload = Upload(app, headers=headers)

        async def receive():
            upload.reads += 1
            clock.now += 0.26
            return {"type": "http.request", "body": chunk, "more_body": True}

        upload.receive = receive
        await upload.start()
        assert upload.response.status_code == 408
        assert upload.reads == 4 and len(clock.deadlines) == 1
        assert app.active_readers == 0
        again = Upload(app)
        again.feed(b"ok")
        await again.start()
        assert again.response.status_code == 200 and app.active_readers == 0


async def test_complete_body_just_inside_deadline_succeeds(monkeypatch):
    async with asyncio.timeout(2):
        clock = ControlledTime(monkeypatch)
        app = middleware(body_timeout_s=1)
        upload = Upload(app)

        async def receive():
            clock.now = clock.deadlines[-1] - 0.000001
            return {"type": "http.request", "body": b"ok", "more_body": False}

        upload.receive = receive
        await upload.start()
        assert upload.response.status_code == 200 and app.active_readers == 0
        assert len(clock.deadlines) == 1


@pytest.mark.parametrize("chunk", [b"", b"x"])
async def test_immediately_ready_frames_yield_and_obey_timeout(monkeypatch, chunk):
    async with asyncio.timeout(2):
        clock = ControlledTime(monkeypatch)
        app = middleware(body_timeout_s=1)
        upload = Upload(app)
        first = asyncio.Event()

        async def receive():
            upload.reads += 1
            assert upload.reads <= 16, "Ready-frame loop did not yield to the timeout controller"
            if upload.reads == 1:
                first.set()
            return {"type": "http.request", "body": chunk, "more_body": True}

        upload.receive = receive
        task = upload.start()
        try:
            await first.wait()  # must run before the receive-side frame guard
            clock.expire()
            await task
            assert upload.response.status_code == 408 and app.active_readers == 0
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("timeout", [False, True])
async def test_slot_retained_until_cancellation_resistant_receive_unwinds(monkeypatch, timeout):
    async with asyncio.timeout(2):
        clock = ControlledTime(monkeypatch)
        app = middleware(max_body_readers=1, body_timeout_s=1)
        upload = Upload(app)
        cleaning = asyncio.Event()
        release = asyncio.Event()

        async def receive():
            upload.reading.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleaning.set()
                await release.wait()
                return {"type": "http.request", "body": b"late", "more_body": False}

        upload.receive = receive
        task = upload.start()
        try:
            await upload.reading.wait()
            if timeout:
                clock.expire()
            else:
                task.cancel()
            await cleaning.wait()
            assert app.active_readers == 1
            rejected = Upload(app)
            await rejected.start()
            assert rejected.response.status_code == 503 and rejected.reads == 0
            assert app.active_readers == 1
            release.set()
            if timeout:
                await task
                assert upload.response.status_code == 408
            else:
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert upload.messages == []
            assert app.active_readers == 0
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize(
    "path,method",
    [
        ("/health", "GET"),
        ("/v1/completions", "POST"),
        ("/v1/completions/batch", "POST"),
        ("/v1/completions/stream", "POST"),
        ("/v1/rag/search?query=cache", "GET"),
    ],
)
async def test_saturation_protects_all_routes_without_execution(api, path, method):
    async with asyncio.timeout(2):
        module, _, calls = api
        held = Upload(module.app, path="/health", method="GET")
        task = held.start()
        try:
            await held.reading.wait()
            rejected = Upload(module.app, path=path, method=method)
            await rejected.start()
            assert rejected.response.status_code == 503 and rejected.reads == 0
            assert rejected.response.json()["error"]["code"] == "ingress_capacity_unavailable"
            assert "retry-after" not in rejected.response.headers
            assert app_ingress(module).active_readers == 1
            assert_no_execution(module, calls)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert app_ingress(module).active_readers == 0


@pytest.mark.parametrize(
    "headers,status",
    [
        ([(b"content-length", b"513")], 413),
        ([(b"content-length", b"invalid-private")], 400),
        ([], 503),
    ],
)
async def test_header_errors_precede_reader_saturation(api, headers, status):
    async with asyncio.timeout(2):
        module, _, calls = api
        held = Upload(module.app)
        task = held.start()
        try:
            await held.reading.wait()
            request = Upload(module.app, headers=headers)
            await request.start()
            assert request.response.status_code == status and request.reads == 0
            assert app_ingress(module).active_readers == 1
            assert_no_execution(module, calls)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_rate_rejection_precedes_declared_size_and_reader_saturation(api):
    async with asyncio.timeout(2):
        module, _, calls = api
        held = Upload(module.app)
        task = held.start()
        try:
            await held.reading.wait()
            limiter = module._RATE_LIMITER
            key = b"\x7f\x00\x00\x01"
            for _ in range(limiter.burst):
                limiter.check(key)
            request = Upload(module.app, headers=[(b"content-length", b"999999")])
            await request.start()
            assert request.response.status_code == 429 and request.reads == 0
            assert_no_execution(module, calls)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_upload_timeout_rejects_before_parsing_or_execution(api, monkeypatch):
    async with asyncio.timeout(2):
        clock = ControlledTime(monkeypatch)
        module, _, calls = api
        upload = Upload(module.app)
        task = upload.start()
        try:
            await upload.reading.wait()
            clock.expire()
            await task
            assert upload.response.status_code == 408
            assert app_ingress(module).active_readers == 0
            assert_no_execution(module, calls)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_body_slot_and_timeout_end_before_inference_or_sse(api, monkeypatch):
    async with asyncio.timeout(2):
        clock = ControlledTime(monkeypatch)
        module, client, _ = api
        started = asyncio.Event()
        release = asyncio.Event()
        provider = module.gateway.providers[0]

        async def stream(*args, **kwargs):
            started.set()
            yield "first"
            await release.wait()
            yield "last"

        monkeypatch.setattr(provider, "stream_complete", stream)
        task = asyncio.create_task(client.post("/v1/completions/stream", json={"prompt": "stream"}))
        try:
            await started.wait()
            assert app_ingress(module).active_readers == 0
            assert clock.contexts[0].when() is not None
            assert not clock.contexts[0].expired()
            # Advancing beyond upload deadline must not cancel the SSE owner.
            clock.now = clock.deadlines[0] + 1
            assert (await client.get("/health")).status_code == 200
            assert not task.done()
            assert module.gateway.admission.active == 1
            release.set()
            response = await task
            assert response.status_code == 200 and "[DONE]" in response.text
            assert module.gateway.admission.active == 0
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("api", [{"mode": "developer"}], indirect=True)
async def test_developer_ignores_invalid_reader_settings(api, monkeypatch):
    async with asyncio.timeout(2):
        module, client, _ = api
        assert module._BODY_READ_LIMITS is None
        monkeypatch.setenv("AI_SYSTEMS_LAB_MAX_BODY_READERS", "private-invalid")
        monkeypatch.setenv("AI_SYSTEMS_LAB_BODY_TIMEOUT_S", "nan")
        assert build_body_read_limits() is None
        response = await client.post("/v1/completions", json={"prompt": "compatible"})
        assert response.status_code == 200


@pytest.mark.parametrize(
    "name,values,message",
    [
        (
            "AI_SYSTEMS_LAB_MAX_BODY_READERS",
            ["", "0", "-1", "+1", "1.5", "65", "nan", "inf", " 8", "private-invalid", "9" * 5000],
            "AI_SYSTEMS_LAB_MAX_BODY_READERS must be an integer between 1 and 64",
        ),
        (
            "AI_SYSTEMS_LAB_BODY_TIMEOUT_S",
            ["", "0", "-1", "nan", "inf", "-inf", "0.099", "30.001", "1e999", "private-invalid"],
            "AI_SYSTEMS_LAB_BODY_TIMEOUT_S must be finite and between 0.1 and 30 seconds",
        ),
    ],
)
def test_invalid_public_reader_settings_fail_startup(monkeypatch, name, values, message):
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "public-demo")
    for value in values:
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError) as error:
            _load_application(monkeypatch)
        assert str(error.value) == message


@pytest.mark.parametrize(
    "readers,seconds,expected",
    [
        (None, None, (8, 5)),
        ("1", "0.1", (1, 0.1)),
        ("64", "30", (64, 30)),
    ],
)
def test_reader_configuration_boundaries(monkeypatch, readers, seconds, expected):
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "public-demo")
    for name, value in (
        ("AI_SYSTEMS_LAB_MAX_BODY_READERS", readers),
        ("AI_SYSTEMS_LAB_BODY_TIMEOUT_S", seconds),
    ):
        if value is None:
            monkeypatch.delenv(name)
        else:
            monkeypatch.setenv(name, value)
    assert build_body_read_limits() == expected


async def test_reader_settings_are_fixed_at_startup(api, monkeypatch):
    async with asyncio.timeout(2):
        module, client, _ = api
        monkeypatch.setenv("AI_SYSTEMS_LAB_MAX_BODY_READERS", "64")
        monkeypatch.setenv("AI_SYSTEMS_LAB_BODY_TIMEOUT_S", "30")
        assert (await client.get("/health")).status_code == 200
        instance = app_ingress(module)
        assert (instance.max_body_readers, instance.body_timeout_s) == (1, 0.1)


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "http.request", "body": "not bytes"},
        {"type": "http.request", "body": b"ok", "more_body": "false"},
    ],
)
async def test_bad_asgi_body_frames_release_capacity(frame):
    async with asyncio.timeout(2):
        app = middleware(max_body_readers=1)
        upload = Upload(app)
        upload.frames.put_nowait(frame)
        await upload.start()
        assert upload.response.status_code == 400 and app.active_readers == 0


async def test_rejection_send_failure_cannot_send_a_second_response():
    async with asyncio.timeout(2):
        app = middleware(max_body_readers=1)
        held = Upload(app)
        task = held.start()
        starts = []
        try:
            await held.reading.wait()
            rejected = Upload(app)

            async def broken_send(message):
                starts.append(message)
                raise TimeoutError("failed write")

            rejected.send = broken_send
            with pytest.raises(TimeoutError, match="failed write"):
                await rejected.start()
            assert len(starts) == 1 and starts[0]["status"] == 503
            assert app.active_readers == 1 and rejected.reads == 0
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert app.active_readers == 0


@pytest.mark.parametrize("delta", [0, 0.000001])
async def test_body_observed_at_or_after_deadline_does_not_execute(monkeypatch, delta):
    async with asyncio.timeout(2):
        clock = ControlledTime(monkeypatch)
        app = middleware(body_timeout_s=1)
        upload = Upload(app)

        async def receive():
            clock.now = clock.deadlines[-1] + delta
            return {"type": "http.request", "body": b"ok", "more_body": False}

        upload.receive = receive
        await upload.start()
        assert upload.response.status_code == 408 and app.active_readers == 0


async def test_upload_expiry_does_not_cancel_other_accepted_inference(api, monkeypatch):
    async with asyncio.timeout(2):
        clock = ControlledTime(monkeypatch)
        module, client, calls = api
        started = asyncio.Event()
        release = asyncio.Event()

        async def complete(*args, **kwargs):
            calls["provider"] += 1
            started.set()
            await release.wait()
            return "accepted inference", 1, 1

        monkeypatch.setattr(module.gateway.providers[0], "complete", complete)
        execution = asyncio.create_task(client.post("/v1/completions", json={"prompt": "first"}))
        upload = Upload(module.app)
        task = None
        try:
            await started.wait()
            assert app_ingress(module).active_readers == 0
            task = upload.start()
            await upload.reading.wait()
            clock.expire()
            await task
            assert upload.response.status_code == 408
            assert not execution.done() and module.gateway.admission.active == 1
            assert calls["provider"] == 1 and app_ingress(module).active_readers == 0
            release.set()
            assert (await execution).status_code == 200
        finally:
            release.set()
            execution.cancel()
            if task is not None:
                task.cancel()
            await asyncio.gather(execution, *([task] if task else []), return_exceptions=True)


async def test_non_http_scope_does_not_reserve_or_read_body():
    async with asyncio.timeout(2):
        called = []

        async def downstream(scope, receive, send):
            called.append(scope["type"])

        async def forbidden():
            raise AssertionError("Unexpected HTTP receive")

        app = BodyLimitMiddleware(downstream, 64)
        await app({"type": "lifespan"}, forbidden, forbidden)
        assert called == ["lifespan"] and app.active_readers == 0


async def test_developer_startup_ignores_both_invalid_settings(monkeypatch):
    async with asyncio.timeout(2):
        monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "developer")
        monkeypatch.setenv("AI_SYSTEMS_LAB_MAX_BODY_READERS", "private-invalid")
        monkeypatch.setenv("AI_SYSTEMS_LAB_BODY_TIMEOUT_S", "inf")
        for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
            monkeypatch.delenv(name, raising=False)
        module = _load_application(monkeypatch)
        assert module._BODY_READ_LIMITS is None
        async with (
            module.app.router.lifespan_context(module.app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=module.app), base_url="http://test"
            ) as client,
        ):
            response = await client.post("/v1/completions", json={"prompt": "valid"})
            assert response.status_code == 200
