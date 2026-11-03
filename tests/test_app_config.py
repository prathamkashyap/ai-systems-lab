import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

from backend.config import build_providers
from backend.providers import AnthropicProvider, MockProvider, OpenAIProvider

_PROVIDER_ENV = {
    "ANTHROPIC_API_KEY": "dummy-anthropic-secret",
    "OPENAI_API_KEY": "dummy-openai-secret",
    "OPENAI_BASE_URL": "https://must-not-contact.invalid/v1",
}


def _provider_environment(monkeypatch: pytest.MonkeyPatch, configured: bool) -> None:
    for name, value in _PROVIDER_ENV.items():
        if configured:
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)


def _load_application(monkeypatch: pytest.MonkeyPatch):
    """Execute the real composition root with fresh state, without reloading main."""
    name = "backend._config_test_main"
    path = Path(__file__).resolve().parents[1] / "backend" / "main.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("configured", [False, True], ids=["no-credentials", "credentials"])
async def test_public_demo_enforces_mock_only_http(
    monkeypatch: pytest.MonkeyPatch, configured: bool
) -> None:
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "public-demo")
    _provider_environment(monkeypatch, configured)
    constructed_live: list[str] = []
    outbound: list[str] = []

    def forbid_live_constructor(self, *args, **kwargs):
        constructed_live.append(type(self).__name__)
        raise AssertionError("Public demo constructed a live provider")

    async def forbid_async_http(self, request):
        outbound.append("async")
        raise AssertionError("Public demo attempted outbound HTTP")

    def forbid_sync_http(self, request):
        outbound.append("sync")
        raise AssertionError("Public demo attempted outbound HTTP")

    monkeypatch.setattr(AnthropicProvider, "__init__", forbid_live_constructor)
    monkeypatch.setattr(OpenAIProvider, "__init__", forbid_live_constructor)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbid_async_http)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbid_sync_http)

    module = _load_application(monkeypatch)
    assert [type(p) for p in module.gateway.providers] == [MockProvider]

    async with asyncio.timeout(5):
        async with (
            module.app.router.lifespan_context(module.app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=module.app), base_url="http://test"
            ) as client,
        ):
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["providers"] == {"mock": True}
            assert health.json()["circuit_breakers"] == {"mock": "closed"}

            # Real and unknown model names cannot escape the provider allowlist.
            for model in ("mock-small", "claude-sonnet-4-6", "gpt-4o-mini", "auto", "custom"):
                payload = {"prompt": "mode boundary", "model": model, "cache": False}
                response = await client.post("/v1/completions", json=payload)
                assert response.status_code == 200
                data = response.json()
                assert data["provider"] == "mock"
                assert data["model"] == model
                assert data["text"].startswith(f"[mock:{model}#")
                for value in _PROVIDER_ENV.values():
                    assert value not in response.text

                stream = await client.post("/v1/completions/stream", json=payload)
                assert stream.status_code == 200
                events = [
                    line.removeprefix("data: ")
                    for line in stream.text.splitlines()
                    if line.startswith("data: ")
                ]
                assert events[-1] == "[DONE]"
                text = "".join(json.loads(event)["text"] for event in events[:-1])
                assert text.startswith(f"[mock:{model}#")
                for value in _PROVIDER_ENV.values():
                    assert value not in stream.text

            batch = await client.post(
                "/v1/completions/batch",
                json={"requests": [{"prompt": "batch", "model": "gpt-4o-mini"}]},
            )
            assert batch.status_code == 200
            assert batch.json()["responses"][0]["provider"] == "mock"

    assert constructed_live == []
    assert outbound == []


@pytest.mark.parametrize("mode", [None, "developer"], ids=["unset", "explicit"])
@pytest.mark.parametrize("configured", [False, True], ids=["no-credentials", "credentials"])
async def test_developer_mode_preserves_provider_configuration(
    monkeypatch: pytest.MonkeyPatch, mode: str | None, configured: bool
) -> None:
    if mode is None:
        monkeypatch.delenv("AI_SYSTEMS_LAB_MODE", raising=False)
    else:
        monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", mode)
    _provider_environment(monkeypatch, configured)

    module = _load_application(monkeypatch)
    providers = module.gateway.providers
    assert [type(p) for p in providers] == [AnthropicProvider, OpenAIProvider, MockProvider]
    assert providers[0].api_key == (_PROVIDER_ENV["ANTHROPIC_API_KEY"] if configured else None)
    assert providers[1].api_key == (_PROVIDER_ENV["OPENAI_API_KEY"] if configured else None)
    assert providers[1].base_url == (
        _PROVIDER_ENV["OPENAI_BASE_URL"] if configured else "https://api.openai.com/v1"
    )
    async with module.app.router.lifespan_context(module.app):
        assert await module.gateway.health() == {
            "anthropic": configured,
            "openai": configured,
            "mock": True,
        }


@pytest.mark.parametrize("mode", ["", "production", "PUBLIC-DEMO", "dummy-secret-invalid-mode"])
def test_invalid_mode_fails_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", mode)
    _provider_environment(monkeypatch, True)

    def forbid_construction(self, *args, **kwargs):
        raise AssertionError("Invalid mode constructed a provider")

    for provider in (AnthropicProvider, OpenAIProvider, MockProvider):
        monkeypatch.setattr(provider, "__init__", forbid_construction)

    with pytest.raises(ValueError) as error:
        _load_application(monkeypatch)
    assert str(error.value) == "AI_SYSTEMS_LAB_MODE must be 'developer' or 'public-demo'"
    assert "dummy-secret-invalid-mode" not in str(error.value)


def test_provider_factory_reads_current_environment_and_returns_fresh_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _provider_environment(monkeypatch, False)
    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "public-demo")
    first = build_providers()
    second = build_providers()
    assert first[0] is not second[0]

    monkeypatch.setenv("AI_SYSTEMS_LAB_MODE", "developer")
    assert [type(p) for p in build_providers()] == [
        AnthropicProvider,
        OpenAIProvider,
        MockProvider,
    ]
