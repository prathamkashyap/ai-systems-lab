"""Provider interface. Every model backend (a real API, a local model, a
mock/offline stub) implements this so the gateway can route to any of them
uniformly. See system-design/adr/0001-provider-abstraction.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from backend.models import Message


class Provider(ABC):
    name: str

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens)."""

    @abstractmethod
    async def health(self) -> bool:
        """Cheap liveness check used by /health."""

    def supports(self, model: str) -> bool:
        """Whether this provider can serve the given logical model name."""
        return True

    async def stream_complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Stream completion tokens asynchronously.

        Default implementation falls back to non-streaming complete() and yields the full text.
        """
        text, _, _ = await self.complete(
            messages, model=model, max_tokens=max_tokens, temperature=temperature
        )
        yield text
