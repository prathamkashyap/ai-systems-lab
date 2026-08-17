"""Request/response schemas for the inference gateway."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    """A single completion request accepted by the gateway."""

    prompt: str | None = None
    messages: list[Message] | None = None
    model: str = Field(default="mock-small", description="Logical model name")
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    cache: bool = Field(default=True, description="Allow serving from / writing to cache")
    stream: bool = Field(default=False, description="Stream response tokens via SSE")

    def as_messages(self) -> list[Message]:
        if self.messages:
            return self.messages
        if self.prompt is not None:
            return [Message(role="user", content=self.prompt)]
        raise ValueError("Either `prompt` or `messages` must be provided")


class CompletionResponse(BaseModel):
    text: str
    model: str
    provider: str
    cached: bool = False
    deduplicated: bool = False
    latency_ms: float
    input_tokens: int
    output_tokens: int


class StreamChunk(BaseModel):
    """A single token delta chunk in a streaming response."""

    text: str
    model: str
    provider: str
    finish_reason: str | None = None


class BatchCompletionRequest(BaseModel):
    requests: list[CompletionRequest]


class BatchCompletionResponse(BaseModel):
    responses: list[CompletionResponse]
    total_latency_ms: float
    batched: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    providers: dict[str, bool]
    circuit_breakers: dict[str, str] = Field(default_factory=dict)
    uptime_s: float


class MetricsSnapshot(BaseModel):
    total_requests: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    deduplicated_requests: int = 0
    deduplication_rate: float = 0.0
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_by_provider: dict[str, int]
    circuit_breaker_trips: int = 0
    errors: int
