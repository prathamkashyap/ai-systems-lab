# Architecture

## System overview

```mermaid
flowchart LR
    Client[Client] -->|HTTP / SSE| API[FastAPI app<br/>backend/main.py]
    API --> GW[Gateway<br/>backend/gateway.py]
    GW --> Cache[(TTL+LRU Cache<br/>backend/cache.py)]
    GW --> SF[SingleFlight Deduplicator<br/>backend/singleflight.py]
    SF --> Batcher[MicroBatcher<br/>backend/batching.py]
    Batcher --> Router{Provider router & circuit breakers}
    Router -->|Guarded| P1[AnthropicProvider]
    Router -->|Guarded| P2[OpenAIProvider]
    Router -->|Guarded| P3[MockProvider]
    GW --> Metrics[(MetricsRegistry & Prometheus<br/>backend/metrics.py)]

    subgraph Hybrid Retrieval Pipeline
        Docs[Documents] --> Chunk[chunking.py]
        Chunk --> Embed[embeddings.py<br/>hashed TF-IDF]
        Chunk --> BM25[hybrid.py<br/>BM25 sparse index]
        Embed --> Store[(VectorStore<br/>dense cosine index)]
        Query[Query text] --> Embed
        Query --> BM25
        Store --> RRF[hybrid.py<br/>Reciprocal Rank Fusion]
        BM25 --> RRF
        RRF --> Results[Top-k fused chunks]
    end
```

## Request lifecycle with SingleFlight Deduplication

```mermaid
sequenceDiagram
    participant C1 as Client 1
    participant C2 as Client 2 (Concurrent Duplicate)
    participant G as Gateway
    participant Ca as Cache
    participant SF as SingleFlight
    participant B as MicroBatcher
    participant P as Provider

    C1->>G: POST /v1/completions (req A)
    C2->>G: POST /v1/completions (req A)
    G->>Ca: lookup(key A) [miss]
    G->>SF: execute(key A)
    Note over SF: C1 becomes Initiator<br/>C2 joins existing in-flight future
    SF->>B: submit(req A) [Initiator only]
    Note over B: wait up to window_ms<br/>or until batch full
    B->>P: complete(batch)
    P-->>B: response A
    B-->>SF: response A
    SF-->>G: C1 receives (deduplicated=false)
    SF-->>G: C2 receives (deduplicated=true)
    G->>Ca: store(key A, response A)
    G-->>C1: Response A (cached=false)
    G-->>C2: Response A (cached=true, deduplicated=true)
```

## Design principles

1. **Providers are interchangeable and resilient.**
   Everything implements the `Provider` interface (`backend/providers/base.py`) and is automatically guarded by an active `CircuitBreaker` (`backend/circuit_breaker.py`). Adding or swapping backends (Anthropic, OpenAI, local Ollama/vLLM) never requires modifications to gateway routing or batching logic. See [ADR 0001](adr/0001-provider-abstraction.md) and [ADR 0004](adr/0004-resilience-circuit-breaker.md).

2. **In-Flight Request Deduplication prevents cache stampedes.**
   The gateway uses `SingleFlight` (`backend/singleflight.py`) to coalesce identical concurrent requests into a single execution, raising burst throughput to thousands of requests per second with 99.5% deduplication rates. See [ADR 0003](adr/0003-singleflight-deduplication.md).

3. **Hybrid Retrieval with Reciprocal Rank Fusion.**
   The retrieval pipeline combines dense vector embeddings with BM25 sparse term frequency indexing via Reciprocal Rank Fusion (`data/hybrid.py`), delivering superior recall across lexical and semantic queries. See [ADR 0005](adr/0005-hybrid-retrieval-rrf.md).

4. **Zero external dependencies for offline reproducibility.**
   `MockProvider` requires no API key, no network, and no downloaded weights. Unit tests, benchmarks, and local development run in seconds. Real providers activate seamlessly when credentials (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) are present.
