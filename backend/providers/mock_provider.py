"""A deterministic, zero-dependency provider used for local development,
tests, benchmarks, and CI — so the whole system runs end to end with no API
key and no network access.

It is not meant to produce "smart" text: it produces stable, inspectable
output so callers can verify plumbing (routing, batching, caching, metrics)
independently of any real model's behaviour.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from collections.abc import AsyncIterator

from backend.models import Message
from backend.providers.base import Provider


class MockProvider(Provider):
    name = "mock"

    def __init__(self, base_latency_ms: float = 15.0, jitter_ms: float = 10.0) -> None:
        self.base_latency_ms = base_latency_ms
        self.jitter_ms = jitter_ms

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, int]:
        # Simulate realistic-but-fast network/inference latency so the
        # batching and caching layers have something meaningful to do.
        delay = (self.base_latency_ms + random.uniform(0, self.jitter_ms)) / 1000
        await asyncio.sleep(delay)

        joined = "\n".join(f"{m.role}: {m.content}" for m in messages)
        digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:8]
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        input_tokens = max(1, len(joined) // 4)
        reply = (
            f"[mock:{model}#{digest}] echoing {len(last_user.split())} word(s), "
            f"temperature={temperature:.2f}: {last_user[: max_tokens]}"
        )
        output_tokens = max(1, len(reply) // 4)
        return reply, input_tokens, output_tokens

    async def stream_complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        full_text, _, _ = await self.complete(
            messages, model=model, max_tokens=max_tokens, temperature=temperature
        )
        tokens = full_text.split(" ")
        for i, token in enumerate(tokens):
            yield token if i == len(tokens) - 1 else token + " "
            await asyncio.sleep(0.005)

    async def health(self) -> bool:
        return True

    def supports(self, model: str) -> bool:
        return model.startswith("mock")
