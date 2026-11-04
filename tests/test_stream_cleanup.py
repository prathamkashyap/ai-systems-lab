import asyncio

import pytest

from backend.admission import AdmissionBudget, AdmissionRejected, AdmittedStream
from backend.deadlines import ShutdownIncomplete, StreamFinalizationError
from backend.gateway import Gateway
from backend.models import CompletionRequest
from backend.providers import MockProvider
from tests.test_app_config import _load_application


def snapshot(stream, budget):
    return {
        "closed": stream.closed,
        "active": budget.active,
        "requests": budget.requests,
        "tracked": stream in budget._streams,
        "timer_cancelled": stream._timer.cancelled(),
        "closing": budget.closing,
    }


@pytest.mark.parametrize("failing", [False, True])
async def test_terminated_generator_finalizes_bookkeeping_even_on_close_error(failing, caplog):
    async with asyncio.timeout(3):
        stopped = asyncio.Event()

        async def chunks():
            try:
                yield "first"
            finally:
                stopped.set()
                if failing:
                    raise RuntimeError("private finalizer detail")

        budget = AdmissionBudget(max_active=2)
        iterator = chunks()
        stream = AdmittedStream(iterator, budget)
        assert await anext(stream) == "first"
        errors = await asyncio.gather(stream.aclose(), stream.aclose(), return_exceptions=True)
        state = snapshot(stream, budget)
        try:
            assert stopped.is_set() and iterator.ag_frame is None
            assert state == {
                "closed": True,
                "active": 0,
                "requests": 0,
                "tracked": False,
                "timer_cancelled": True,
                "closing": False,
            }, state
            if failing:
                assert all(isinstance(error, StreamFinalizationError) for error in errors)
                assert all("private finalizer detail" not in str(error) for error in errors)
                assert "private finalizer detail" not in caplog.text
            else:
                assert errors == [None, None]
            repeated = (await asyncio.gather(stream.aclose(), return_exceptions=True))[0]
            assert type(repeated) is type(errors[0])
            assert snapshot(stream, budget) == state
            permit = budget.reserve_work(1, queue=False)[0]
            permit.release()  # conclusively stopped work can safely be replaced
        finally:
            # Also clean up leaks when this test runs against the pre-fix code.
            stream._timer.cancel()
            stream.work.release()
            stream.finish_response()


@pytest.mark.parametrize("terminated", [True, False])
async def test_http_owner_finalizes_waiter_after_finalizer_failure(monkeypatch, terminated):
    async with asyncio.timeout(3):
        monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "public-demo")
        module = _load_application(monkeypatch)
        budget = module.gateway.admission
        stopped = asyncio.Event()

        async def chunks():
            try:
                yield "first"
            finally:
                try:
                    if not terminated:
                        yield "private illegal finalizer yield"
                finally:
                    stopped.set()
                raise RuntimeError("private finalizer detail")

        iterator = chunks()
        stream = AdmittedStream(iterator, budget)
        monkeypatch.setattr(module.gateway, "open_stream", lambda req: stream)
        response = await module.create_streaming_completion(CompletionRequest(prompt="test"))
        messages = []

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.body" and b'"text"' in message["body"]:
                stream._expire()  # deterministic deadline after the first emitted chunk

        async def receive():
            await asyncio.Event().wait()

        result = (
            await asyncio.gather(
                response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send),
                return_exceptions=True,
            )
        )[0]
        state = snapshot(stream, budget)
        try:
            assert stopped.is_set() == terminated
            assert state == {
                "closed": terminated,
                "active": 0 if terminated else 1,
                "requests": 0,
                "tracked": not terminated,
                "timer_cancelled": True,
                "closing": not terminated,
            }, state
            assert result is None
            body = b"".join(message.get("body", b"") for message in messages)
            assert messages[0]["status"] == 200 and b'"text": "first"' in body
            assert b"event: error" in body and b"inference_deadline_exceeded" in body
            assert b"private" not in body and b"[DONE]" not in body
            assert messages[-1]["more_body"] is False
            if not terminated:
                with pytest.raises(AdmissionRejected):
                    budget.reserve_request()
        finally:
            # Terminate the invalid generator before explicit pre-fix leak teardown.
            if not terminated:
                await asyncio.gather(anext(iterator), return_exceptions=True)
                await iterator.aclose()
            stream._timer.cancel()
            stream.work.release()
            stream.finish_response()
            budget._streams.discard(stream)
            await module.gateway.aclose()


async def test_unterminated_generator_fails_closed_without_releasing_execution():
    async with asyncio.timeout(3):
        stopped = asyncio.Event()

        async def chunks():
            try:
                yield "first"
            finally:
                try:
                    yield "illegal finalizer yield"  # aclose cannot establish termination
                finally:
                    stopped.set()

        iterator = chunks()
        budget = AdmissionBudget(max_active=2)
        stream = AdmittedStream(iterator, budget)
        assert await asyncio.create_task(anext(stream)) == "first"
        results = await asyncio.gather(stream.aclose(), stream.aclose(), return_exceptions=True)
        state = snapshot(stream, budget)
        try:
            assert not stopped.is_set() and iterator.ag_frame is not None
            assert state == {
                "closed": False,
                "active": 1,
                "requests": 0,
                "tracked": True,
                "timer_cancelled": True,
                "closing": True,
            }, state
            assert all(isinstance(result, ShutdownIncomplete) for result in results)
            with pytest.raises(AdmissionRejected):
                budget.reserve_work(1, queue=False)  # reject even with remaining capacity
            with pytest.raises(AdmissionRejected):
                budget.reserve_request()
            with pytest.raises(ShutdownIncomplete):
                await stream.aclose()
            assert snapshot(stream, budget) == state
            gateway = Gateway([MockProvider(0, 0)], admission=budget)
            shutdown = await asyncio.gather(
                gateway.aclose(), gateway.aclose(), return_exceptions=True
            )
            assert all(isinstance(error, ShutdownIncomplete) for error in shutdown)
            assert snapshot(stream, budget) == state
        finally:
            # Explicitly terminate the deliberately invalid generator for test teardown.
            await asyncio.gather(anext(iterator), return_exceptions=True)
            await iterator.aclose()
            stream._timer.cancel()
        assert stopped.is_set()
        assert budget.closing and budget.active == 1  # never silently reopen after failure


async def test_active_finalizer_retains_capacity_until_it_stops_without_reopening():
    async with asyncio.timeout(3):
        finalizing = asyncio.Event()
        release = asyncio.Event()
        stopped = asyncio.Event()

        async def chunks():
            try:
                yield "first"
            finally:
                finalizing.set()
                await release.wait()
                stopped.set()
                raise RuntimeError("private finalizer detail")

        budget = AdmissionBudget(max_active=2, cancel_grace_s=0.05)
        iterator = chunks()
        stream = AdmittedStream(iterator, budget)
        assert await anext(stream) == "first"
        closing = asyncio.create_task(stream.aclose())
        try:
            await finalizing.wait()
            assert not stopped.is_set() and budget.active == 1
            with pytest.raises(ShutdownIncomplete):
                await closing
            assert not stream._cleanup_task.done() and iterator.ag_frame is not None
            assert budget.closing and budget.active == 1 and stream in budget._streams
            assert stream._timer.cancelled()
            with pytest.raises(AdmissionRejected):
                budget.reserve_work(1, queue=False)
        finally:
            release.set()
            await asyncio.gather(closing, return_exceptions=True)
            await asyncio.gather(stream._cleanup_task, return_exceptions=True)
        assert stopped.is_set() and iterator.ag_frame is None
        assert stream.closed and budget.active == 0 and budget.requests == 0
        assert budget._streams == set() and budget.closing
        with pytest.raises(StreamFinalizationError):
            await stream.aclose()
        with pytest.raises(AdmissionRejected):
            budget.reserve_request()


async def test_unclassifiable_iterator_close_error_fails_closed():
    async with asyncio.timeout(3):

        class OpaqueIterator:
            async def __anext__(self):
                return "first"

            async def aclose(self):
                raise RuntimeError("private opaque finalizer detail")

        budget = AdmissionBudget(max_active=2)
        stream = AdmittedStream(OpaqueIterator(), budget)
        assert await anext(stream) == "first"
        with pytest.raises(ShutdownIncomplete) as error:
            await stream.aclose()
        assert "private" not in str(error.value)
        assert snapshot(stream, budget) == {
            "closed": False,
            "active": 1,
            "requests": 0,
            "tracked": True,
            "timer_cancelled": True,
            "closing": True,
        }
        with pytest.raises(AdmissionRejected):
            budget.reserve_work(1, queue=False)


@pytest.mark.parametrize("wrapper_already_closed", [False, True])
async def test_closed_gateway_wrapper_cannot_hide_unterminated_provider_iterator(
    wrapper_already_closed,
):
    async with asyncio.timeout(3):
        stopped = asyncio.Event()

        class InvalidFinalizerProvider(MockProvider):
            async def stream_complete(self, *args, **kwargs):
                try:
                    yield "first"
                finally:
                    try:
                        yield "illegal provider finalizer yield"
                    finally:
                        stopped.set()

        budget = AdmissionBudget(max_active=2)
        provider = InvalidFinalizerProvider(0, 0)
        iterator = provider.stream_complete()
        provider.stream_complete = lambda *args, **kwargs: iterator
        gateway = Gateway([provider], admission=budget)
        stream = gateway.open_stream(CompletionRequest(prompt="test"))
        assert await asyncio.create_task(anext(stream)) == "first"
        if wrapper_already_closed:
            # A wrapper error may already have reached the iteration owner;
            # a subsequent wrapper aclose then returns successfully.
            await asyncio.gather(stream.iterator.aclose(), return_exceptions=True)
        outcome = (await asyncio.gather(stream.aclose(), return_exceptions=True))[0]
        try:
            assert stream.iterator.ag_frame is None  # the wrapper alone is insufficient
            assert iterator.ag_frame is not None and not stopped.is_set()
            assert isinstance(outcome, ShutdownIncomplete)
            assert snapshot(stream, budget) == {
                "closed": False,
                "active": 1,
                "requests": 0,
                "tracked": True,
                "timer_cancelled": True,
                "closing": True,
            }
            with pytest.raises(AdmissionRejected):
                gateway.open_stream(CompletionRequest(prompt="must reject"))
        finally:
            await asyncio.gather(anext(iterator), return_exceptions=True)
            await iterator.aclose()
        assert stopped.is_set() and budget.closing


async def test_concurrent_shutdown_closes_other_streams_after_finalizer_failure():
    async with asyncio.timeout(3):
        closed = []

        async def chunks(name, failing):
            try:
                yield name
            finally:
                closed.append(name)
                if failing:
                    raise RuntimeError("private finalizer detail")

        budget = AdmissionBudget(shutdown_grace_s=0.1, cancel_grace_s=0.1)
        gateway = Gateway([MockProvider(0, 0)], admission=budget)
        streams = [
            AdmittedStream(chunks(name, failing), budget)
            for name, failing in (("failed", True), ("normal", False))
        ]
        # Separate owner tasks finish before shutdown, so shutdown cannot cancel this test.
        await asyncio.gather(*(anext(stream) for stream in streams))
        outcomes = await asyncio.gather(gateway.aclose(), gateway.aclose(), return_exceptions=True)
        try:
            assert outcomes == [None, None]
            assert sorted(closed) == ["failed", "normal"]
            assert (budget.active, budget.requests, budget.pending) == (0, 0, 0)
            assert budget._streams == set() and budget.closing
            assert all(stream.closed and stream._timer.cancelled() for stream in streams)
            await gateway.aclose()
        finally:
            for stream in streams:
                await asyncio.gather(stream.aclose(), return_exceptions=True)
                stream._timer.cancel()
                stream.work.release()
                stream.finish_response()


async def test_developer_close_error_keeps_original_exception_contract():
    async with asyncio.timeout(3):

        async def chunks():
            try:
                yield "first"
            finally:
                raise RuntimeError("developer finalizer error")

        stream = AdmittedStream(chunks(), None)
        assert await anext(stream) == "first"
        with pytest.raises(RuntimeError, match="developer finalizer error"):
            await stream.aclose()
        await stream.aclose()
