# Current State Baseline

Measurement performed 2026-10-26 on branch `docs/asl-baseline`, starting from
`5b48fcd20998a4736aabe4f5a3bc92a238c9de67` (`main`). This records the behavior
observed in this checkout; no behavior fixes were made.

## Environment

| Item | Observed value | Evidence |
|---|---|---|
| Python | `Python 3.15.0` | `/tmp/ai-systems-lab-asl-baseline-venv/bin/python --version` |
| OS | macOS `27.0.1`, build `26A434`; Darwin kernel `27.0.0`; `arm64` | `sw_vers`; `uname -a`; `uname -m` |
| Environment | Fresh venv outside the repository | `python3 -m venv /tmp/ai-systems-lab-asl-baseline-venv` |
| Install | Editable project plus `.[dev]` installed successfully | `/tmp/ai-systems-lab-asl-baseline-venv/bin/python -m pip install -e '.[dev]'` |
| Key packages | FastAPI `0.162.0`; uvicorn `0.54.0`; httpx `0.28.1`; Pydantic `2.14.1`; NumPy `2.5.4`; pytest `9.1.1`; pytest-asyncio `1.4.1`; Ruff `0.17.2` | `python -m pip freeze` in the fresh venv |

Exact environment command output:

```text
$ /tmp/ai-systems-lab-asl-baseline-venv/bin/python --version
Python 3.15.0
$ sw_vers
ProductName: macOS
ProductVersion: 27.0.1
BuildVersion: 26A434
$ uname -m
arm64
```

The local Python version is outside the workflow's listed Python `3.11` and
`3.12` matrix; the local checks below do not replace those CI environments.

The install initially retried interrupted downloads, then completed successfully.
The full installed package list was: `annotated-doc==0.0.5`,
`annotated-types==0.8.0`, `anyio==4.15.1`, `certifi==2026.10.23`,
`click==8.5.0`, `fastapi==0.162.0`, `h11==0.16.0`, `httpcore==1.0.9`,
`httptools==0.9.0`, `httpx==0.28.1`, `idna==3.20`, `iniconfig==2.3.1`,
`numpy==2.5.4`, `opentelemetry-api==1.45.1`, `packaging==26.3`,
`pluggy==1.7.0`, `pydantic==2.14.1`, `pydantic_core==2.50.1`,
`Pygments==2.21.0`, `pytest==9.1.1`, `pytest-asyncio==1.4.1`,
`python-dotenv==1.2.4`, `PyYAML==6.0.3`, `ruff==0.17.2`,
`starlette==1.10.0`, `typing-inspection==0.4.4`,
`typing_extensions==4.16.0`, `uvicorn==0.54.0`, `uvloop==0.23.0`,
`watchfiles==1.3.0`, and `websockets==17.3`.

## Documented Commands and Results

Statuses describe this execution only. Output is included as the command's
observed terminal result or its relevant unedited summary; variable latency and
timestamps are left as printed by the command.

| Documented command / operation | Status | Observed output or evidence |
|---|---|---|
| `git clone https://github.com/prathamkashyap/ai-systems-lab.git` | not verified | Not run; this existing checkout was used. |
| `pip install -e ".[dev]"` | passed | Command run with the fresh venv's Python; ended `Successfully installed ... fastapi-0.162.0 ... numpy-2.5.4 ... pytest-9.1.1 ... ruff-0.17.2 ...`. |
| `python -m examples.quickstart` | passed | Exit `0`; produced a MockProvider completion, a cache hit, `Received 5 responses. Deduplicated in-flight: 4/5 (Provider called once)`, indexed 2 chunks from 2 documents, and returned a RAG result. Also passed with the three provider environment variables explicitly unset; see command/output below. |
| `pytest -v` | passed | Three runs, each collected 40 items and ended `40 passed`; runtimes were `1.77s`, `0.71s`, and `0.85s`. No failures or skips. |
| `python -m benchmarks.latency_bench` | passed | Three successful runs; full raw outputs below. |
| `uvicorn backend.main:app --reload` (README) | passed | Server started without `--reload` on `127.0.0.1:8000`; requested routes returned responses. Both server runs were stopped with Ctrl-C and logged clean shutdown. Reload mode itself was not run. |
| README Anthropic `curl` completion example | not verified | No real provider credentials/call used. |
| README OpenAI-compatible `curl` completion example | not verified | No real provider credentials/call used. |
| README Ollama/vLLM `OPENAI_BASE_URL` and `curl` example | not verified | No local external model endpoint was configured or called. |
| `docker compose -f infrastructure/docker-compose.yml up --build` | blocked | Docker/Compose CLIs are installed, but Compose could not connect to `unix:///Users/prathamkashyap/.docker/run/docker.sock`; no daemon was running. |
| CI `ruff check .` | passed | `All checks passed!` (exit `0`). |
| CI `pytest -v` | passed | Same three local runs above; all 40 passed each time. |
| CI `python -m benchmarks.latency_bench` | passed | Same three local runs above; exit `0` each. Workflow marks this step non-blocking. |
| CI `docker build -f infrastructure/Dockerfile -t ai-systems-lab:ci .` | blocked | Docker engine unavailable at the socket above; image build not completed. |

Actual measurement commands used the explicit venv executable paths shown here:

```text
python3 -m venv /tmp/ai-systems-lab-asl-baseline-venv
/tmp/ai-systems-lab-asl-baseline-venv/bin/python -m pip install -e '.[dev]'
/tmp/ai-systems-lab-asl-baseline-venv/bin/python --version
sw_vers
uname -a
uname -m
/tmp/ai-systems-lab-asl-baseline-venv/bin/python -m pip freeze
/tmp/ai-systems-lab-asl-baseline-venv/bin/pytest -v
/tmp/ai-systems-lab-asl-baseline-venv/bin/python -m examples.quickstart
env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u OPENAI_BASE_URL /tmp/ai-systems-lab-asl-baseline-venv/bin/python -m examples.quickstart
/tmp/ai-systems-lab-asl-baseline-venv/bin/python -m benchmarks.latency_bench
/tmp/ai-systems-lab-asl-baseline-venv/bin/ruff check .
gh run list --repo prathamkashyap/ai-systems-lab --branch main --limit 1 --json databaseId,headSha,status,conclusion,createdAt,displayTitle,url,event
env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u OPENAI_BASE_URL docker compose -f infrastructure/docker-compose.yml up --build -d
```

The final Compose command's actual result was:

```text
failed to connect to the docker API at unix:///Users/prathamkashyap/.docker/run/docker.sock; check if the path is correct and if the daemon is running: dial unix /Users/prathamkashyap/.docker/run/docker.sock: connect: no such file or directory
```

The zero-key quickstart command exited `0` and output:

```text
--- 1. Inference Gateway: Caching & SingleFlight Deduplication ---
[mock] [mock:mock-small#8b25cbda] echoing 7 word(s), temperature=0.70: Summarize what a vector database is for.
Initial call: latency=12.9ms cached=False

Sequential call: cached=True latency=0.02ms

Firing 5 concurrent identical requests simultaneously...
Received 5 responses. Deduplicated in-flight: 4/5 (Provider called once)

--- 2. Hybrid Retrieval Pipeline (BM25 + Dense Vector RRF) ---
Indexed 2 chunks from 2 documents.

Query: How does RAG use vector databases to fetch context?
  score=0.0328 source='vector-dbs'
  text: Vector databases store high-dimensional embeddings and support approximate nearest-neighbour search. They power retrieva...
```

## Test Inventory

Each `pytest -v` run collected 40 tests. Results were 40 passed, 0 failed,
0 skipped in `1.77s`, `0.71s`, and `0.85s` for runs 1, 2, and 3. No flaky test
was observed in these three runs; that does not establish absence of rare flakiness.

| Module | Count | Coverage inventory |
|---|---:|---|
| `tests/test_cache.py` | 6 | Cache round trips, misses, TTL expiry, LRU eviction, hit rate, deterministic keys. |
| `tests/test_chunking.py` | 6 | Word and sentence chunking, overlap, invalid overlap, empty input, source IDs. |
| `tests/test_circuit_breaker.py` | 3 | Closed operation, opening after failures, half-open recovery. |
| `tests/test_gateway.py` | 8 | Completion, cache, fallback/no provider, batching, SingleFlight, metrics. |
| `tests/test_hybrid_search.py` | 3 | BM25 matching, reciprocal rank fusion, hybrid retrieval. |
| `tests/test_openai_provider.py` | 2 | Model support and provider health. |
| `tests/test_rag_eval.py` | 4 | Context precision/recall, keyword coverage, faithfulness, summary. |
| `tests/test_singleflight.py` | 3 | Same-key coalescing, different-key independence, exception propagation. |
| `tests/test_streaming.py` | 1 | Streaming completion. |
| `tests/test_vector_store.py` | 4 | Best-match ordering, length, end-to-end and empty-ingest retrieval. |
| **Total** | **40** | |

Run summaries as printed:

```text
Run 1: ============================== 40 passed in 1.77s ==============================
Run 2: ============================== 40 passed in 0.71s ==============================
Run 3: ============================== 40 passed in 0.85s ==============================
```

The README's 40-test count is accurate for this checkout.

## Quickstart

Command:

```text
/tmp/ai-systems-lab-asl-baseline-venv/bin/python -m examples.quickstart
```

Exit `0`. Output included a mock completion (`latency=13.1ms`), sequential
cache hit (`latency=0.02ms`), 4 of 5 concurrent identical calls deduplicated,
2 chunks indexed from 2 documents, and a RAG result with score `0.0328`.

Observed output:

```text
--- 1. Inference Gateway: Caching & SingleFlight Deduplication ---
[mock] [mock:mock-small#8b25cbda] echoing 7 word(s), temperature=0.70: Summarize what a vector database is for.
Initial call: latency=13.1ms cached=False

Sequential call: cached=True latency=0.02ms

Firing 5 concurrent identical requests simultaneously...
Received 5 responses. Deduplicated in-flight: 4/5 (Provider called once)

--- 2. Hybrid Retrieval Pipeline (BM25 + Dense Vector RRF) ---
Indexed 2 chunks from 2 documents.

Query: How does RAG use vector databases to fetch context?
  score=0.0328 source='vector-dbs'
  text: Vector databases store high-dimensional embeddings and support approximate nearest-neighbour search. They power retrieva...
```

## Raw Benchmark Output

Command for each run:

```text
/tmp/ai-systems-lab-asl-baseline-venv/bin/python -m benchmarks.latency_bench
```

Run 1:

```text
scenario                                            | n   | wall_ms | throughput_rps | p50_ms  | p95_ms  | mean_ms
------------------------------------------------------------------------------------------------------------------
baseline (unique prompts, no cache)                 | 200 | 310.37  | 644.38         | 290.94  | 306.32  | 286.61
concurrent identical prompts (SingleFlight enabled) | 200 | 58.38   | 3425.64        | 50.29   | 51.10   | 49.34
unique prompts (wide batch window 50ms)             | 200 | 1216.20 | 164.45         | 1211.84 | 1213.26 | 1166.72
repeated prompt (sequential cache hits)             | 200 | 36.18   | 5528.39        | 0.02    | 0.07    | 0.18
  baseline (unique prompts, no cache): {'batches_executed': 13, 'items_processed': 200, 'avg_batch_size': 15.384615384615385}
  concurrent identical prompts (SingleFlight enabled): {'deduplicated_requests': 199, 'deduplication_rate': '99.5%', 'batches_executed': 1}
  unique prompts (wide batch window 50ms): {'batches_executed': 4, 'items_processed': 200, 'avg_batch_size': 50.0}
  repeated prompt (sequential cache hits): {'cache_hit_rate': '99.5%'}
```

Run 2:

```text
scenario                                            | n   | wall_ms | throughput_rps | p50_ms  | p95_ms  | mean_ms
------------------------------------------------------------------------------------------------------------------
baseline (unique prompts, no cache)                 | 200 | 314.29  | 636.35         | 299.37  | 306.64  | 290.37
concurrent identical prompts (SingleFlight enabled) | 200 | 44.50   | 4494.05        | 39.22   | 40.01   | 38.61
unique prompts (wide batch window 50ms)             | 200 | 1173.37 | 170.45         | 1166.70 | 1167.52 | 1122.39
repeated prompt (sequential cache hits)             | 200 | 34.71   | 5762.09        | 0.02    | 0.03    | 0.17
  baseline (unique prompts, no cache): {'batches_executed': 13, 'items_processed': 200, 'avg_batch_size': 15.384615384615385}
  concurrent identical prompts (SingleFlight enabled): {'deduplicated_requests': 199, 'deduplication_rate': '99.5%', 'batches_executed': 1}
  unique prompts (wide batch window 50ms): {'batches_executed': 4, 'items_processed': 200, 'avg_batch_size': 50.0}
  repeated prompt (sequential cache hits): {'cache_hit_rate': '99.5%'}
```

Run 3:

```text
scenario                                            | n   | wall_ms | throughput_rps | p50_ms  | p95_ms  | mean_ms
------------------------------------------------------------------------------------------------------------------
baseline (unique prompts, no cache)                 | 200 | 294.89  | 678.22         | 280.95  | 293.34  | 277.69
concurrent identical prompts (SingleFlight enabled) | 200 | 34.51   | 5795.26        | 32.79   | 33.01   | 32.49
unique prompts (wide batch window 50ms)             | 200 | 1179.34 | 169.59         | 1162.04 | 1178.37 | 1122.53
repeated prompt (sequential cache hits)             | 200 | 32.42   | 6169.32        | 0.02    | 0.03    | 0.16
  baseline (unique prompts, no cache): {'batches_executed': 13, 'items_processed': 200, 'avg_batch_size': 15.384615384615385}
  concurrent identical prompts (SingleFlight enabled): {'deduplicated_requests': 199, 'deduplication_rate': '99.5%', 'batches_executed': 1}
  unique prompts (wide batch window 50ms): {'batches_executed': 4, 'items_processed': 200, 'avg_batch_size': 50.0}
  repeated prompt (sequential cache hits): {'cache_hit_rate': '99.5%'}
```

The following are max-minus-min range / sample standard deviation across these
three runs, computed from the raw command output above:

| Scenario | Wall ms | Throughput req/s | p50 ms | p95 ms | Mean ms |
|---|---:|---:|---:|---:|---:|
| Baseline unique | 19.40 / 10.26 | 41.87 / 22.22 | 18.42 / 9.22 | 13.30 / 7.59 | 12.68 / 6.51 |
| Concurrent identical | 23.87 / 11.99 | 2369.62 / 1186.71 | 17.50 / 8.85 | 18.09 / 9.12 | 16.85 / 8.53 |
| Wide window | 42.83 / 23.20 | 6.00 / 3.24 | 49.80 / 27.51 | 45.74 / 23.90 | 44.33 / 25.55 |
| Sequential cache | 3.76 / 1.89 | 640.93 / 324.36 | 0.00 / 0.00 | 0.04 / 0.02 | 0.02 / 0.01 |

The three runs reported 13, 13, and 13 micro-batcher batches for the default
10 ms window and 4, 4, and 4 for the 50 ms window. The exact average batch
sizes printed by the command were `15.384615384615385` and `50.0` respectively.

## Provider Invocation Count

The temporary script was kept at `/tmp/asl_provider_call_count.py`, outside
the repository. Its exact contents are reproduced here; it was not committed.
It wraps the real `MockProvider.complete` coroutine and counts invocations.
Both scenarios use `Gateway` defaults and default `CompletionRequest` settings.

```python
import asyncio

from backend.gateway import Gateway
from backend.models import CompletionRequest
from backend.providers.mock_provider import MockProvider


async def measure(identical: bool) -> tuple[int, int]:
    provider = MockProvider()
    actual_complete = provider.complete
    calls = 0

    async def counted_complete(*args, **kwargs):
        nonlocal calls
        calls += 1
        return await actual_complete(*args, **kwargs)

    provider.complete = counted_complete
    gateway = Gateway(providers=[provider])

    async def one(i: int) -> None:
        prompt = "same prompt" if identical else f"unique prompt {i}"
        await gateway.complete(CompletionRequest(prompt=prompt))

    await asyncio.gather(*(one(i) for i in range(200)))
    return calls, gateway._batcher.stats["batches_executed"]


async def main() -> None:
    unique_calls, unique_batches = await measure(identical=False)
    identical_calls, identical_batches = await measure(identical=True)
    print(f"unique: provider.complete invocations={unique_calls}; batches={unique_batches}")
    print(f"identical: provider.complete invocations={identical_calls}; batches={identical_batches}")


asyncio.run(main())
```

Exact command and output:

```text
/tmp/ai-systems-lab-asl-baseline-venv/bin/python /tmp/asl_provider_call_count.py
unique: provider.complete invocations=200; batches=13
identical: provider.complete invocations=1; batches=1
```

Thus the benchmark note's “~13 provider calls” describes batcher flushes in
this implementation. Actual provider calls for 200 unique prompts were 200;
identical concurrent prompts made one provider call.

## HTTP Smoke Test

The server command was run with provider credentials and the provider base URL
unset:

```text
env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u OPENAI_BASE_URL /tmp/ai-systems-lab-asl-baseline-venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Uvicorn printed `Application startup complete.` and `Uvicorn running on
http://127.0.0.1:8000`. It was stopped with Ctrl-C; the log ended with
`Application shutdown complete.` and `Finished server process`.

The smoke command used httpx requests and abbreviated each body to the first
260 characters:

```python
import httpx

base = "http://127.0.0.1:8000"
with httpx.Client(timeout=20) as client:
    cases = [
        ("GET", "/health", None),
        ("POST", "/v1/completions", {"prompt": "smoke completion", "model": "mock-small"}),
        ("POST", "/v1/completions/stream", {"prompt": "smoke stream", "model": "mock-small"}),
        ("POST", "/v1/completions/batch", {"requests": [{"prompt": "batch one", "model": "mock-small"}, {"prompt": "batch two", "model": "mock-small"}]}),
        ("GET", "/v1/rag/search?query=cache stampede&mode=hybrid", None),
        ("GET", "/v1/rag/search?query=dense vector search&mode=vector", None),
        ("GET", "/v1/rag/search?query=BM25 lexical search&mode=bm25", None),
        ("GET", "/metrics", None),
        ("GET", "/metrics/prometheus", None),
    ]
    for method, path, payload in cases:
        response = client.request(method, base + path, json=payload)
        body = response.text.replace("\n", " ")
        print(f"{method} {path} -> {response.status_code}: {body[:260]}")
```

Observed output (latency and uptime fields are as printed during the request):

```text
GET /health -> 200: {"status":"ok","providers":{"anthropic":false,"openai":false,"mock":true},"circuit_breakers":{"anthropic":"closed","openai":"closed","mock":"closed"},"uptime_s":13.933349333936349}
POST /v1/completions -> 200: {"text":"[mock:mock-small#b9c18191] echoing 2 word(s), temperature=0.70: smoke completion","model":"mock-small","provider":"mock","cached":false,"deduplicated":false,"latency_ms":17.193458043038845,"input_tokens":5,"output_tokens":20}
POST /v1/completions/stream -> 200: data: {"text": "[mock:mock-small#47d4388e] ", "model": "mock-small"}  data: {"text": "echoing ", "model": "mock-small"}  data: {"text": "2 ", "model": "mock-small"}  data: {"text": "word(s), ", "model": "mock-small"}  data: {"text": "temperature=0.70: ", "mode
POST /v1/completions/batch -> 200: {"responses":[{"text":"[mock:mock-small#80743e8c] echoing 2 word(s), temperature=0.70: batch one","model":"mock-small","provider":"mock","cached":false,"deduplicated":false,"latency_ms":19.134083064273,"input_tokens":3,"output_tokens":18},{"text":"[mock:mock-s
GET /v1/rag/search?query=cache stampede&mode=hybrid -> 200: [{"text":"Inference gateways use TTL and LRU caching to reduce redundant compute. Under high concurrency, cache stampedes (thundering herd) occur when identical requests simultaneously miss the cache. SingleFlight request deduplication coalesces in-flight exec
GET /v1/rag/search?query=dense vector search&mode=vector -> 200: [{"text":"Hybrid search combines dense vector similarity (cosine distance on neural embeddings) with sparse lexical matching (BM25 term frequencies). Reciprocal Rank Fusion (RRF) robustly fuses the two ranking distributions without requiring score normalizatio
GET /v1/rag/search?query=BM25 lexical search&mode=bm25 -> 200: [{"text":"Hybrid search combines dense vector similarity (cosine distance on neural embeddings) with sparse lexical matching (BM25 term frequencies). Reciprocal Rank Fusion (RRF) robustly fuses the two ranking distributions without requiring score normalizatio
GET /metrics -> 200: {"total_requests":3,"cache_hits":0,"cache_misses":3,"cache_hit_rate":0.0,"deduplicated_requests":0,"deduplication_rate":0.0,"avg_latency_ms":19.143819343298674,"p50_latency_ms":19.134083064273,"p95_latency_ms":20.90693353675306,"p99_latency_ms":21.064520245417
GET /metrics/prometheus -> 200: # HELP ai_gateway_requests_total Total number of inference requests processed # TYPE ai_gateway_requests_total counter ai_gateway_requests_total 3  # HELP ai_gateway_cache_hits_total Total cache hits # TYPE ai_gateway_cache_
```

Additional README endpoint checks were:

```text
GET / -> 200: <!DOCTYPE html> <html lang="en"> ...
GET /playground -> 200: <!DOCTYPE html> <html lang="en"> ...
GET /docs -> 200: <!DOCTYPE html> ...
GET /v1/rag/search?query=dense%20search&mode=dense -> 422: {"detail":[{"type":"literal_error","loc":["query","mode"],"msg":"Input should be 'hybrid', 'vector' or 'bm25'","input":"dense","ctx":{"expected":"'hybrid', 'vector' or 'bm25'"}}]}
```

The supported dense retrieval parameter is `mode=vector`; `mode=dense` is not
accepted. All endpoint smoke requests were stopped before leaving this task;
both Uvicorn processes logged successful shutdown.

## README Claim Check

“Contradicted” for a measured value means the listed README value did not
reproduce in these runs; it is not a claim that all hardware must produce the
same timing. Timing evidence is from the raw outputs above.

| README claim | Status | Evidence |
|---|---|---|
| 40-test suite | verified | Three pytest runs collected and passed exactly 40 tests each. |
| 99.5% concurrent deduplication | verified | All three benchmark runs reported 199 deduplicated requests and `deduplication_rate: '99.5%'`; direct provider count was one invocation for identical prompts. |
| “~13 provider calls” for 200 unique prompts | contradicted | Three benchmark runs report 13 batches, but direct counting measured 200 `provider.complete` invocations across 13 batches. |
| Baseline unique row: 297.7 ms, 671.9 req/s, p50 282.5 ms, p95 296.4 ms | contradicted | Three measured wall times were 310.37, 314.29, and 294.89 ms; full run output has all columns. None exactly reproduces the README row. |
| Concurrent duplicate row: 45.5 ms, 4397.1 req/s, p50 41.3 ms, p95 41.8 ms, 99.5% | contradicted | Measured wall times were 58.38, 44.50, and 34.51 ms, with variable p50/p95 and throughput. The 99.5% dedup claim itself is verified above. |
| Wide batch row: 1166.5 ms, 171.5 req/s, p50 1164.0 ms, p95 1164.3 ms | contradicted | Measured wall times were 1216.20, 1173.37, and 1179.34 ms; none exactly reproduces the README row. |
| Sequential repeated row: 37.9 ms, 5277.9 req/s, p50 0.03 ms, p95 0.05 ms, 99.5% hit rate | contradicted | Measured wall times were 36.18, 34.71, and 32.42 ms; hit rate was 99.5% in each run, but timing values differ. |
| Offline quickstart with zero API keys | verified | Quickstart exited `0` after `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `OPENAI_BASE_URL` were unset; output used MockProvider and local retrieval. |
| README endpoint list | contradicted | `/`, `/playground`, `/docs`, completion, stream, batch, RAG, health, and metrics paths returned 200. Dense retrieval works as `mode=vector`; README's Dense Vector label does not correspond to accepted `mode=dense`, which returned 422. |
| Prometheus scrape endpoint at `/metrics/prometheus` | verified | GET returned `200` with Prometheus text beginning `# HELP ai_gateway_requests_total` and `# TYPE ai_gateway_requests_total counter`. |

## CI and Latest Actions Run

The workflow's lint command `/tmp/ai-systems-lab-asl-baseline-venv/bin/ruff
check .` printed `All checks passed!`. The workflow has no separate formatter
command. Its `pytest -v` and non-blocking benchmark commands passed as recorded
above.

Command:

```text
gh run list --repo prathamkashyap/ai-systems-lab --branch main --limit 1 --json databaseId,headSha,status,conclusion,createdAt,displayTitle,url,event
```

Observed output:

```json
[{"conclusion":"success","createdAt":"2026-08-17T12:53:56Z","databaseId":32032170000,"displayTitle":"feat: complete AI systems lab with gateway, SingleFlight dedup, circu…","event":"push","headSha":"5b48fcd20998a4736aabe4f5a3bc92a238c9de67","status":"completed","url":"https://github.com/prathamkashyap/ai-systems-lab/actions/runs/32032170000"}]
```

The latest reported run was successful and ran on the baseline commit.

## Observed Problems

- `GET /v1/rag/search?mode=dense` returns 422. The endpoint accepts `hybrid`,
  `vector`, or `bm25`; callers must use `mode=vector` for dense retrieval.
- “~13 provider calls” is inaccurate for the observed implementation. The
  batcher invokes its batch callback 13 times, while the callback loops over
  requests and calls `provider.complete` once per request. The direct count
  observed 200 provider invocations for 200 unique prompts.
- The benchmark timing rows do not reproduce exactly in this environment;
  measured ranges are provided above. The duplicate scenario varied most in
  wall time and throughput across the three observed runs.

No fixes were made.

## Not Verified

- Real Anthropic, OpenAI-compatible, Ollama, or vLLM API requests.
- Docker Compose startup or Docker image build, because the Docker daemon was
  unavailable.
- Reload-mode Uvicorn (`--reload`); the server was run without reload.
- Absence of rare test flakiness; only the three recorded pytest runs were
  observed.
