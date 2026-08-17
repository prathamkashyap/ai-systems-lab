"""Runs the core components of the AI systems lab end to end with zero setup:
no API key, no network, no external services.

    python -m examples.quickstart
"""

from __future__ import annotations

import asyncio

from backend.gateway import Gateway
from backend.models import CompletionRequest
from backend.providers.mock_provider import MockProvider
from data.pipeline import Document, RetrievalPipeline


async def gateway_demo() -> None:
    print("--- 1. Inference Gateway: Caching & SingleFlight Deduplication ---")
    gateway = Gateway(providers=[MockProvider(base_latency_ms=10, jitter_ms=2)])

    # Initial completion request (Cache Miss)
    resp = await gateway.complete(
        CompletionRequest(prompt="Summarize what a vector database is for.", model="mock-small")
    )
    print(f"[{resp.provider}] {resp.text}")
    print(f"Initial call: latency={resp.latency_ms:.1f}ms cached={resp.cached}\n")

    # Sequential repeat -> Instant Cache Hit
    resp2 = await gateway.complete(
        CompletionRequest(prompt="Summarize what a vector database is for.", model="mock-small")
    )
    print(f"Sequential call: cached={resp2.cached} latency={resp2.latency_ms:.2f}ms\n")

    # Concurrent identical requests -> In-Flight SingleFlight Deduplication
    print("Firing 5 concurrent identical requests simultaneously...")
    tasks = [
        gateway.complete(CompletionRequest(prompt="Concurrent stampede test prompt", model="mock-small", cache=True))
        for _ in range(5)
    ]
    responses = await asyncio.gather(*tasks)
    dedup_count = sum(1 for r in responses if r.deduplicated)
    print(f"Received 5 responses. Deduplicated in-flight: {dedup_count}/5 (Provider called once)\n")


def retrieval_demo() -> None:
    print("--- 2. Hybrid Retrieval Pipeline (BM25 + Dense Vector RRF) ---")
    docs = [
        Document(
            id="vector-dbs",
            text=(
                "Vector databases store high-dimensional embeddings and support "
                "approximate nearest-neighbour search. They power retrieval-augmented generation (RAG), "
                "where relevant text chunks are fetched and inserted into a model prompt."
            ),
        ),
        Document(
            id="containers",
            text=(
                "Containers package an application with its dependencies into a "
                "portable unit. Docker is the most widely used container runtime, "
                "and Kubernetes orchestrates containerized workloads at scale."
            ),
        ),
    ]

    pipeline = RetrievalPipeline(dim=256, chunk_size=80)
    n_chunks = pipeline.ingest(docs)
    print(f"Indexed {n_chunks} chunks from {len(docs)} documents.\n")

    query = "How does RAG use vector databases to fetch context?"
    results = pipeline.query(query, mode="hybrid", top_k=1)
    print(f"Query: {query}")
    for r in results:
        print(f"  score={r.score:.4f} source={r.metadata['source_id']!r}")
        print(f"  text: {r.text[:120]}...\n")


async def main() -> None:
    await gateway_demo()
    retrieval_demo()


if __name__ == "__main__":
    asyncio.run(main())
