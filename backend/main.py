"""FastAPI application: the HTTP surface of the AI systems lab.

Run locally:
    uvicorn backend.main:app --reload

Endpoints:
    POST /v1/completions         - single completion (routed, cached, batched, deduplicated)
    POST /v1/completions/stream  - streaming completion tokens via Server-Sent Events (SSE)
    POST /v1/completions/batch   - explicit batch of completions
    GET  /health                 - liveness, per-provider health & circuit breaker states
    GET  /metrics                - JSON latency percentiles, hit rate, dedup stats
    GET  /metrics/prometheus     - standard Prometheus text format scrape endpoint
    GET  /v1/rag/search          - hybrid/dense/BM25 retrieval search endpoint
    GET  /                       - interactive AI Systems Lab Playground UI
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse

from backend.gateway import Gateway, NoProviderAvailable
from backend.models import (
    BatchCompletionRequest,
    BatchCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    HealthResponse,
    MetricsSnapshot,
)
from backend.providers import AnthropicProvider, MockProvider, OpenAIProvider
from data.pipeline import Document, RetrievalPipeline

_START_TIME = time.monotonic()
_STATIC_DIR = Path(__file__).parent / "static"

# Register providers in priority order: Anthropic -> OpenAI -> Mock fallback
gateway = Gateway(providers=[AnthropicProvider(), OpenAIProvider(), MockProvider()])

# Pre-seeded RAG knowledge base for interactive exploration
rag_pipeline = RetrievalPipeline(dim=256, chunk_size=80)
rag_pipeline.ingest([
    Document(
        id="sys-caching",
        text=(
            "Inference gateways use TTL and LRU caching to reduce redundant compute. "
            "Under high concurrency, cache stampedes (thundering herd) occur when identical "
            "requests simultaneously miss the cache. SingleFlight request deduplication coalesces "
            "in-flight executions so only one request hits the model provider."
        ),
        metadata={"category": "caching"},
    ),
    Document(
        id="sys-batching",
        text=(
            "Micro-batching accumulates concurrent requests across a brief time window (e.g. 10ms-50ms) "
            "and passes them to the GPU or provider in a single tensor operation or batch request. "
            "This significantly improves throughput while slightly trading off tail latency."
        ),
        metadata={"category": "batching"},
    ),
    Document(
        id="sys-hybrid-search",
        text=(
            "Hybrid search combines dense vector similarity (cosine distance on neural embeddings) "
            "with sparse lexical matching (BM25 term frequencies). Reciprocal Rank Fusion (RRF) "
            "robustly fuses the two ranking distributions without requiring score normalization."
        ),
        metadata={"category": "retrieval"},
    ),
    Document(
        id="sys-circuit-breaker",
        text=(
            "Circuit breakers monitor provider error rates. When consecutive failures cross a threshold, "
            "the circuit opens to fast-fail subsequent calls immediately. After a cooldown, half-open probe "
            "calls test provider recovery before fully closing the circuit."
        ),
        metadata={"category": "resilience"},
    ),
])

app = FastAPI(
    title="AI Systems Lab — Inference Gateway",
    version="1.1.0",
    description=(
        "An enterprise-grade, routed, cached, batched, and deduplicated inference gateway "
        "with circuit breaking, hybrid RAG, and live Prometheus observability."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(req: CompletionRequest) -> CompletionResponse:
    try:
        return await gateway.complete(req)
    except NoProviderAvailable as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/completions/stream")
async def create_streaming_completion(req: CompletionRequest) -> StreamingResponse:
    async def event_generator():
        try:
            async for token in gateway.stream_complete(req):
                chunk_data = json.dumps({"text": token, "model": req.model})
                yield f"data: {chunk_data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001
            err_data = json.dumps({"error": str(exc)})
            yield f"data: {err_data}\n\n"


    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/v1/completions/batch", response_model=BatchCompletionResponse)
async def create_batch_completion(req: BatchCompletionRequest) -> BatchCompletionResponse:
    start = time.perf_counter()
    try:
        responses = await asyncio.gather(*(gateway.complete(r) for r in req.requests))
    except NoProviderAvailable as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    total_ms = (time.perf_counter() - start) * 1000
    return BatchCompletionResponse(
        responses=list(responses), total_latency_ms=total_ms, batched=True
    )


@app.get("/v1/rag/search")
async def rag_search(
    query: str = Query(..., description="Search query string"),
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
    top_k: int = 5,
) -> list[dict]:
    results = rag_pipeline.query(query, top_k=top_k, mode=mode)
    return [{"text": r.text, "score": round(r.score, 4), "metadata": r.metadata} for r in results]


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    provider_health = await gateway.health()
    status = "ok" if any(provider_health.values()) else "degraded"
    return HealthResponse(
        status=status,
        providers=provider_health,
        circuit_breakers=gateway.circuit_breaker_states(),
        uptime_s=time.monotonic() - _START_TIME,
    )


@app.get("/metrics", response_model=MetricsSnapshot)
async def metrics() -> MetricsSnapshot:
    return MetricsSnapshot(**gateway.metrics.snapshot())


@app.get("/metrics/prometheus", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    return gateway.metrics.to_prometheus_text()


@app.get("/", response_class=HTMLResponse)
@app.get("/playground", response_class=HTMLResponse)
async def playground() -> HTMLResponse:
    index_file = _STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>AI Systems Lab Playground</h1><p>UI loading error</p>")
