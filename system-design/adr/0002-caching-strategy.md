# ADR 0002: Caching strategy

## Status
Accepted (with a known follow-up)

## Context
Repeated identical completion requests (same prompt/messages, model,
temperature, max_tokens) are wasted work if re-sent to a provider — slower
for the caller and, for paid APIs, an unnecessary cost. The gateway needed a
caching layer that's simple to reason about and doesn't require standing up
external infrastructure (Redis, etc.) just to run the service locally or in
CI.

## Decision
Implement `TTLLRUCache` (`backend/cache.py`) as a single in-process
`OrderedDict`:
- **Key:** SHA-256 of `(model, temperature, max_tokens, role:content...)`,
  so any change to the request that could change the output changes the key.
- **TTL:** entries expire after a configurable window (default 5 minutes),
  checked lazily on read.
- **LRU:** once `max_size` is reached, the least-recently-used entry is
  evicted on write.
- Cache lookup/write is opt-out per-request (`CompletionRequest.cache:
  bool`), not opt-in, since caching identical completions is almost always
  desirable, and callers that need guaranteed-fresh output can disable it.

## Consequences
- **Positive:** Zero external dependencies; the whole service — cache
  included — runs from a single process, which is what makes the test suite
  and benchmarks fast and free to run in CI.
- **Positive:** Measured effect is large and easy to demonstrate: a
  sequential run of 200 identical requests drops from ~350ms to ~35ms wall
  time once cached (see `benchmarks/results/README.md`).
- **Negative (accepted, documented):** the cache does not deduplicate
  *in-flight* requests. If N identical requests arrive concurrently before
  any of them completes, all N miss the cache and all N hit the provider —
  measured directly in the benchmark ("repeated prompt (cache enabled)"
  scenario shows a 0% hit rate under full concurrency vs. 99.5% sequentially).
  This is a real limitation of any bare TTL/LRU cache, not specific to this
  implementation.
- **Follow-up:** add a "singleflight" layer — when a cache miss occurs,
  register an in-flight future keyed the same way; concurrent identical
  requests await that future instead of independently calling the provider.
  This is the standard fix for the cache-stampede pattern and is the
  natural next PR against `backend/cache.py` / `backend/gateway.py`.
- **Negative (accepted):** cache is per-process and in-memory, so it doesn't
  survive a restart and doesn't share state across multiple gateway
  instances. A multi-instance deployment would need Redis or similar behind
  the same `get`/`set` interface.
