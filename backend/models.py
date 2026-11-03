"""Request/response schemas for the inference gateway."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)

    @field_validator("content")
    @classmethod
    def require_meaningful_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message content must contain non-whitespace text")
        return value


class CompletionRequest(BaseModel):
    """A single completion request accepted by the gateway."""

    prompt: str | None = Field(default=None, min_length=1, max_length=8000)
    messages: list[Message] | None = Field(default=None, min_length=1, max_length=32)
    model: str = Field(default="mock-small", description="Logical model name")
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    cache: bool = Field(default=True, description="Allow serving from / writing to cache")
    stream: bool = Field(default=False, description="Stream response tokens via SSE")

    @field_validator("prompt")
    @classmethod
    def require_meaningful_prompt(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Prompt must contain non-whitespace text")
        return value

    @model_validator(mode="after")
    def require_one_input(self) -> Self:
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("Provide exactly one of prompt or messages")
        return self

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
    requests: list[CompletionRequest] = Field(min_length=1, max_length=16)


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
