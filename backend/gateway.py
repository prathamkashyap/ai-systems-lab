"""The inference gateway: a single entry point that routes a completion
request to an appropriate provider, transparently applying caching,
micro-batching, in-flight deduplication (SingleFlight), circuit breaking,
and metrics collection.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from backend.batching import MicroBatcher
from backend.cache import TTLLRUCache
from backend.circuit_breaker import CircuitBreaker, CircuitOpenError
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
    ) -> None:
        if not providers:
            raise ValueError("Gateway requires at least one provider")
        self.providers = providers
        self.cache = cache or TTLLRUCache()
        self.metrics = metrics or MetricsRegistry()
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
        )

    def _providers_for(self, model: str) -> list[Provider]:
        matching = [p for p in self.providers if p.supports(model)]
        return matching or self.providers

    def _cache_key(self, req: CompletionRequest) -> str:
        messages = req.as_messages()
        parts = [req.model, f"{req.temperature:.3f}", f"{req.max_tokens}"]
        parts += [f"{m.role}:{m.content}" for m in messages]
        return self.cache.make_key(*parts)

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        """Public entry point: checks cache and in-flight deduplicator before
        submitting into the micro-batcher so concurrent identical requests
        coalesce without duplicate downstream work.
        """
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


    async def _execute_batch(self, requests: list[CompletionRequest]) -> list[CompletionResponse]:
        results: list[CompletionResponse] = []
        for req in requests:
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
                self.metrics.record_request(
                    provider.name, latency_ms, cached=False, deduplicated=False
                )
                if req.cache:
                    self.cache.set(self._cache_key(req), response)
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

    async def stream_complete(self, req: CompletionRequest) -> AsyncIterator[str]:
        """Stream token chunks across providers with fallback."""
        messages = req.as_messages()
        last_error: Exception | None = None

        for provider in self._providers_for(req.model):
            cb = self._circuit_breakers.get(provider.name)
            if cb and not cb.is_available:
                continue

            try:
                async for chunk in provider.stream_complete(
                    messages,
                    model=req.model,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                ):
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
