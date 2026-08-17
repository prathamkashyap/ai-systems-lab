# Development Workflow

This describes the day-to-day loop for working in this codebase — as
opposed to `CONTRIBUTING.md`, which covers the PR/review process.

## Local loop

```bash
make install     # editable install + dev deps
make test         # pytest -v
make lint         # ruff check .
make run          # uvicorn backend.main:app --reload  ->  localhost:8000/docs
```

Everything runs with **no API key and no network access** by default —
`MockProvider` and the hashed-TF-IDF embedder mean the full test suite,
the quickstart, and the benchmark harness are all self-contained. You only
need `ANTHROPIC_API_KEY` set if you want to exercise `AnthropicProvider`
against the real API.

## Adding a new provider

1. Implement `backend/providers/base.py`'s `Provider` interface in a new
   file under `backend/providers/`.
2. Register it in the `providers=[...]` list in `backend/main.py`
   (order matters — it's fallback order).
3. Add it to `backend/providers/__init__.py`'s exports.
4. Add a test in `tests/test_gateway.py` following the pattern used for
   `FailingProvider` (a fake provider is usually enough — you don't need
   a real API to test routing/fallback logic).

No changes to `gateway.py`, `cache.py`, `batching.py`, or the HTTP routes
should be necessary. If you find yourself needing one, that's a signal the
`Provider` interface is missing something — raise it in the PR rather than
routing around it.

## Adding a new chunking/embedding strategy

Both are structural seams, not hard-coded implementations:
- Chunkers are plain functions returning `list[Chunk]`
  (`data/chunking.py`) — add a new function, don't modify the existing
  ones' signatures.
- Embedders follow the `Embedder` protocol in `data/embeddings.py`
  (`fit(corpus)`, `embed(texts) -> np.ndarray`) — implement a new class
  against that protocol and it drops into `RetrievalPipeline` unchanged.

## Working on the benchmark harness

`benchmarks/latency_bench.py` uses `MockProvider` specifically so results
are reproducible without network variance or API cost. If you add a new
scenario, run it a few times locally to sanity check variance before
committing numbers to `benchmarks/results/README.md` — CI runs the
benchmark as a non-blocking smoke check, not a performance gate.

## Releasing

This is a lab, not a versioned library consumers pin to — there's no
formal release process. `pyproject.toml`'s `version` is bumped on
meaningful milestones (a new component landing, a breaking API change to
the gateway's request/response schema) rather than on every merge.
