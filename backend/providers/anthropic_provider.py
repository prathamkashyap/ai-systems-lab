"""Provider backed by the real Anthropic Messages API.

Only activates if ANTHROPIC_API_KEY is set in the environment — otherwise
`health()` reports False and the gateway simply routes around it, falling
back to other providers. This means the service is honest about what it can
actually serve rather than crashing at import time when no key is present.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx

from backend.models import Message
from backend.providers.base import Provider

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None, timeout_s: float = 30.0) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.timeout_s = timeout_s

    def _build_payload(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        stream: bool = False,
    ) -> tuple[dict, dict]:
        system_prompt = "\n".join(m.content for m in messages if m.role == "system")
        conversation = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]
        resolved_model = model if model.startswith("claude") else DEFAULT_MODEL

        payload: dict = {
            "model": resolved_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conversation,
            "stream": stream,
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        return payload, headers

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, int]:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")

        payload, headers = self._build_payload(
            messages, model=model, max_tokens=max_tokens, temperature=temperature, stream=False
        )

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(ANTHROPIC_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        text = "".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        return (
            text,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )

    async def stream_complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")

        payload, headers = self._build_payload(
            messages, model=model, max_tokens=max_tokens, temperature=temperature, stream=True
        )

        async with (
            httpx.AsyncClient(timeout=self.timeout_s) as client,
            client.stream("POST", ANTHROPIC_API_URL, json=payload, headers=headers) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:]
                try:
                    event = json.loads(raw)
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {}).get("text", "")
                        if delta:
                            yield delta
                except json.JSONDecodeError:
                    continue


    async def health(self) -> bool:
        return bool(self.api_key)

    def supports(self, model: str) -> bool:
        return model.startswith("claude") or model == "auto"
