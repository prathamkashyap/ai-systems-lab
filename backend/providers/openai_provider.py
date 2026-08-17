"""Provider backed by OpenAI and OpenAI-compatible Chat Completions endpoints.

Compatible with:
- OpenAI API (GPT-4o, GPT-4o-mini, o1, o3-mini)
- Local Ollama (e.g. http://localhost:11434/v1)
- Local vLLM / SGLang / LM Studio instances
- Cloud inference providers: DeepSeek, Groq, Together, Mistral, Fireworks

Only activates if OPENAI_API_KEY is set or a custom OPENAI_BASE_URL is provided.
Otherwise health() reports False and the gateway falls back gracefully.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx

from backend.models import Message
from backend.providers.base import Provider

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s

    def supports(self, model: str) -> bool:
        lowered = model.lower()
        openai_prefixes = (
            "gpt",
            "o1",
            "o3",
            "deepseek",
            "llama",
            "mistral",
            "qwen",
            "meta-llama",
            "phi",
        )
        return any(lowered.startswith(p) for p in openai_prefixes) or model == "auto"

    async def health(self) -> bool:
        # If running against local vLLM/Ollama, API key may not be required
        if self.base_url != DEFAULT_OPENAI_BASE_URL:
            return True
        return bool(self.api_key)

    def _build_headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def _resolve_model(self, model: str) -> str:
        if model == "auto" or not self.supports(model):
            return DEFAULT_OPENAI_MODEL
        return model

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, int]:
        if not await self.health():
            raise RuntimeError("OpenAIProvider is not configured (missing OPENAI_API_KEY or custom base_url)")

        payload = {
            "model": self._resolve_model(model),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(url, json=payload, headers=self._build_headers())
            resp.raise_for_status()
            data = resp.json()

        choice = data.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "") or ""
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        return text, input_tokens, output_tokens

    async def stream_complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        if not await self.health():
            raise RuntimeError("OpenAIProvider is not configured")

        payload = {
            "model": self._resolve_model(model),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        url = f"{self.base_url}/chat/completions"
        async with (
            httpx.AsyncClient(timeout=self.timeout_s) as client,
            client.stream("POST", url, json=payload, headers=self._build_headers()) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or line.startswith(":") or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    raw_json = line[6:]
                    try:
                        chunk = json.loads(raw_json)
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                        if delta:
                            yield delta
                    except json.JSONDecodeError:
                        continue

