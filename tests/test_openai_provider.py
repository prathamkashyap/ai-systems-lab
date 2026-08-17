from backend.providers.openai_provider import OpenAIProvider


def test_openai_provider_model_support():
    p = OpenAIProvider()
    assert p.supports("gpt-4o") is True
    assert p.supports("gpt-4o-mini") is True
    assert p.supports("o1-preview") is True
    assert p.supports("o3-mini") is True
    assert p.supports("deepseek-chat") is True
    assert p.supports("llama-3.3-70b") is True
    assert p.supports("auto") is True
    assert p.supports("unknown-custom-model") is False


async def test_openai_provider_health():
    # When no key is set and default url is used -> health False
    p_no_key = OpenAIProvider(api_key=None, base_url="https://api.openai.com/v1")
    assert await p_no_key.health() is False

    # When key is present -> health True
    p_with_key = OpenAIProvider(api_key="sk-test-key")
    assert await p_with_key.health() is True

    # When custom local base URL is provided (e.g. Ollama/vLLM) -> health True
    p_local = OpenAIProvider(api_key=None, base_url="http://localhost:11434/v1")
    assert await p_local.health() is True
