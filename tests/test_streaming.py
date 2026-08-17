from backend.gateway import Gateway
from backend.models import CompletionRequest
from backend.providers.mock_provider import MockProvider


async def test_streaming_completion():
    gw = Gateway(providers=[MockProvider(base_latency_ms=1, jitter_ms=1)])
    chunks = []
    async for chunk in gw.stream_complete(
        CompletionRequest(prompt="test stream", model="mock-small", stream=True)
    ):
        chunks.append(chunk)

    assert len(chunks) > 1
    joined = "".join(chunks)
    assert "[mock:mock-small" in joined
    assert "echoing" in joined
