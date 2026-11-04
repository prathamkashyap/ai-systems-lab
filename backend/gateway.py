"""The inference gateway: a single entry point that routes a completion
request to an appropriate provider, transparently applying caching,
micro-batching, in-flight deduplication (SingleFlight), circuit breaking,
and metrics collection.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing

from backend.admission import AdmissionBudget, AdmittedStream
from backend.batching import MicroBatcher
from backend.cache import TTLLRUCache
from backend.circuit_breaker import CircuitBreaker, CircuitOpenError
from backend.deadlines import InferenceDeadlineExceeded, ShutdownIncomplete, observe, settled
from backend.metrics import MetricsRegistry
from backend.models import CompletionRequest, CompletionResponse, Message
from backend.providers.base import Provider
from backend.singleflight import SingleFlight


class NoProviderAvailable(RuntimeError):
    pass


class Gateway:
    def __init__(
        self,
        providers: list[Provider],
        cache: TTLLRUCache | None = None,
        metrics: MetricsRegistry | None = None,
        batch_window_ms: float = 10.0,
        max_batch_size: int = 16,
        circuit_failure_threshold: int = 3,
        circuit_recovery_timeout: float = 10.0,
        admission: AdmissionBudget | None = None,
    ) -> None:
        if not providers:
            raise ValueError("Gateway requires at least one provider")
        self.providers = providers
        self.cache = cache or TTLLRUCache()
        self.metrics = metrics or MetricsRegistry()
        self.admission = admission
        self._uncached_keys = itertools.count()
        self._shutdown_task: asyncio.Task | None = None
        self._singleflight: SingleFlight[CompletionResponse] = SingleFlight()
        self._circuit_breakers: dict[str, CircuitBreaker] = {
            p.name: CircuitBreaker(
                name=p.name,
                failure_threshold=circuit_failure_threshold,
                recovery_timeout=circuit_recovery_timeout,
            )
            for p in providers
        }
        self._batcher: MicroBatcher[CompletionRequest, CompletionResponse] = MicroBatcher(
            batch_fn=self._execute_batch,
            max_batch_size=max_batch_size,
            window_ms=batch_window_ms,
            cancel_grace_s=admission.cancel_grace_s if admission is not None else 1,
            on_stuck=admission.fail_closed if admission is not None else None,
        )

    def _providers_for(self, model: str) -> list[Provider]:
        matching = [p for p in self.providers if p.supports(model)]
        return matching or self.providers

    def _cache_key(self, req: CompletionRequest) -> str:
        # Hash one versioned envelope, never delimiter-flattened user fields.
        # Use effective provider inputs so prompt shorthand and user messages
        # remain equivalent, while float precision and message boundaries survive.
        payload = json.dumps(
            {
                "format": "ai-systems-lab.inference.v1",
                "model": req.model,
                "temperature": req.temperature,
                "max_tokens": req.max_tokens,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in req.as_messages()
                ],
            },
            sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
        )
        return self.cache.make_key(payload)

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        """Public entry point: checks cache and in-flight deduplicator before
        submitting into the micro-batcher so concurrent identical requests
        coalesce without duplicate downstream work.
        """
        if self.admission is not None:
            return (await self.complete_many([req]))[0]
        start = time.perf_counter()
        if req.cache:
            key = self._cache_key(req)
            cached = self.cache.get(key)
            if cached is not None:
                elapsed_ms = (time.perf_counter() - start) * 1000
                response = cached.model_copy(
                    update={"cached": True, "deduplicated": False, "latency_ms": elapsed_ms}
                )
                self.metrics.record_request(
                    response.provider, elapsed_ms, cached=True, deduplicated=False
                )
                return response

            # In-flight deduplication to eliminate cache stampedes
            resp, is_shared = await self._singleflight.execute(
                key, lambda: self._batcher.submit(req)
            )
            if is_shared:
                elapsed_ms = (time.perf_counter() - start) * 1000
                shared_resp = resp.model_copy(
                    update={"cached": True, "deduplicated": True, "latency_ms": elapsed_ms}
                )
                self.metrics.record_request(
                    shared_resp.provider, elapsed_ms, cached=True, deduplicated=True
                )
                return shared_resp
            return resp

        return await self._batcher.submit(req)

    async def complete_many(self, requests: list[CompletionRequest]) -> list[CompletionResponse]:
        """Atomically admit a public HTTP batch's new downstream work."""
        if self.admission is None:
            return list(await asyncio.gather(*(self.complete(req) for req in requests)))

        request_permit = self.admission.reserve_request()
        try:
            start = time.perf_counter()
            cached = [
                self.cache.get(self._cache_key(req), record_stats=False) if req.cache else None
                for req in requests
            ]
            permits = iter(())
            reserved = []

            def reserve(count: int) -> None:
                nonlocal permits, reserved
                reserved = self.admission.reserve_work(count)
                permits = iter(reserved)

            async def work(req: CompletionRequest) -> CompletionResponse:
                permit = next(permits)
                try:
                    try:
                        async with asyncio.timeout_at(permit.deadline):
                            await asyncio.shield(permit.ready)
                    except TimeoutError:
                        raise InferenceDeadlineExceeded() from None
                    # The batcher now owns release, including on submit failure.
                    # A failed/timed-out result is not evidence execution stopped.
                    permit.transferred = True
                    return await self._batcher.submit(
                        req, deadline=permit.deadline, on_finish=permit.release,
                        on_result=lambda response: self._record_result(req, response),
                    )
                finally:
                    if not permit.transferred:
                        permit.release()

            calls = [
                (
                    self._cache_key(req) if req.cache else f"uncached:{next(self._uncached_keys)}",
                    lambda req=req: work(req),
                )
                for req, hit in zip(requests, cached, strict=True) if hit is None
            ]
            prepared = await self._singleflight.prepare_many(calls, before_start=reserve)
            # Also release reservations if a newly created worker is cancelled
            # before its coroutine has entered the try/finally above.
            for (task, shared), permit in zip(
                [(task, shared) for task, shared in prepared if not shared], reserved, strict=True
            ):
                task.add_done_callback(
                    lambda task, permit=permit: None if permit.transferred else permit.release()
                )
            for req, hit in zip(requests, cached, strict=True):
                if req.cache:
                    self.cache.record_lookup(hit is not None)
            # Shield each flight: cancellation ends this waiter, not accepted
            # background work or its execution reservation.
            results = iter(await asyncio.gather(*(asyncio.shield(task) for task, _ in prepared)))
            flags = iter(shared for _, shared in prepared)
            responses = []
            for req, hit in zip(requests, cached, strict=True):
                shared = False if hit is not None else next(flags)
                response = hit if hit is not None else next(results)
                if hit is not None or (req.cache and shared):
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    response = response.model_copy(update={
                        "cached": True, "deduplicated": shared, "latency_ms": elapsed_ms,
                    })
                    self.metrics.record_request(
                        response.provider, elapsed_ms, cached=True, deduplicated=shared
                    )
                responses.append(response)
            return responses
        finally:
            request_permit.release()


    async def _execute_batch(self, requests: list[CompletionRequest]) -> list[CompletionResponse]:
        results: list[CompletionResponse] = []
        for req in requests:
            # A provider may swallow cancellation and return late. Do not let
            # that start later members of an already terminated public flush.
            if self.admission is not None and asyncio.current_task().cancelling():
                raise InferenceDeadlineExceeded()
            results.append(await self._complete_one(req))
        return results

    async def _complete_one(self, req: CompletionRequest) -> CompletionResponse:
        messages: list[Message] = req.as_messages()
        start = time.perf_counter()
        last_error: Exception | None = None

        for provider in self._providers_for(req.model):
            cb = self._circuit_breakers.get(provider.name)
            if cb and not cb.is_available:
                continue

            try:
                if cb:
                    text, in_tok, out_tok = await cb.call(
                        lambda p=provider: p.complete(
                            messages,
                            model=req.model,
                            max_tokens=req.max_tokens,
                            temperature=req.temperature,
                        )
                    )
                else:
                    text, in_tok, out_tok = await provider.complete(
                        messages,
                        model=req.model,
                        max_tokens=req.max_tokens,
                        temperature=req.temperature,
                    )

                latency_ms = (time.perf_counter() - start) * 1000
                response = CompletionResponse(
                    text=text,
                    model=req.model,
                    provider=provider.name,
                    cached=False,
                    deduplicated=False,
                    latency_ms=latency_ms,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                )
                if self.admission is None:
                    self._record_result(req, response)
                return response
            except CircuitOpenError as exc:
                self.metrics.record_circuit_trip()
                last_error = exc
                continue
            except Exception as exc:  # noqa: BLE001 - try next provider
                last_error = exc
                continue

        self.metrics.record_error()
        raise NoProviderAvailable(
            f"No provider could serve model={req.model!r}: {last_error}"
        )

    def _record_result(self, req: CompletionRequest, response: CompletionResponse) -> None:
        """Commit public success only after the whole flush completes in time."""
        self.metrics.record_request(
            response.provider, response.latency_ms, cached=False, deduplicated=False
        )
        if req.cache:
            self.cache.set(self._cache_key(req), response)

    async def stream_complete(self, req: CompletionRequest) -> AsyncIterator[str]:
        stream = self.open_stream(req)
        try:
            async for chunk in stream:
                yield chunk
        finally:
            await stream.aclose()

    def open_stream(self, req: CompletionRequest) -> AdmittedStream:
        """Reserve before headers; the caller must close the returned stream."""
        owned_iterators = [] if self.admission is not None else None
        return AdmittedStream(
            self._stream_complete(req, owned_iterators), self.admission,
            owned_iterators=owned_iterators,
        )

    async def _stream_complete(
        self, req: CompletionRequest, owned_iterators: list[AsyncIterator[str]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream token chunks across providers with fallback."""
        messages = req.as_messages()
        last_error: Exception | None = None

        for provider in self._providers_for(req.model):
            cb = self._circuit_breakers.get(provider.name)
            if cb and not cb.is_available:
                continue

            try:
                chunks = provider.stream_complete(
                    messages,
                    model=req.model,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                )
                if owned_iterators is not None:
                    owned_iterators.append(chunks)
                async with aclosing(chunks):
                    async for chunk in chunks:
                        yield chunk
                return
            except Exception as exc:  # noqa: BLE001 - try next provider
                last_error = exc
                continue

        self.metrics.record_error()
        raise NoProviderAvailable(
            f"No streaming provider could serve model={req.model!r}: {last_error}"
        )

    async def health(self) -> dict[str, bool]:
        results = {}
        for provider in self.providers:
            try:
                results[provider.name] = await provider.health()
            except Exception:  # noqa: BLE001
                results[provider.name] = False
        return results

    def circuit_breaker_states(self) -> dict[str, str]:
        return {name: cb.state.value for name, cb in self._circuit_breakers.items()}

    async def aclose(self) -> None:
        """Close the request batcher after flushing its accepted work."""
        if self.admission is not None:
            self.admission.closing = True
            if self._shutdown_task is None:
                self._shutdown_task = asyncio.create_task(self._close_public())
                self._shutdown_task.add_done_callback(observe)
            await asyncio.shield(self._shutdown_task)
        else:
            await self._batcher.aclose()

    async def _close_public(self) -> None:
        async def drain() -> None:
            await self.admission.close_streams()
            await self._singleflight.wait_idle()
            await self._batcher.aclose()

        drainer = asyncio.create_task(drain())
        drainer.add_done_callback(observe)
        if (await settled({drainer}, self.admission.shutdown_grace_s)
                and not drainer.cancelled() and drainer.exception() is None):
            return
        self.admission.expire_pending()
        self._batcher.abort()
        tasks = {drainer} | self._singleflight.owned_tasks() | self._batcher._batch_tasks | self._batcher._flush_tasks
        tasks |= {stream._cleanup_task for stream in self.admission._streams if stream._cleanup_task is not None}
        tasks |= self.admission._response_tasks
        if not await settled(tasks, self.admission.cancel_grace_s):
            self.admission.fail_closed()
            drainer.cancel()
            raise ShutdownIncomplete()
        drainer.result()
