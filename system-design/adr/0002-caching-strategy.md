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
- **Key:** SHA-256 of one versioned JSON envelope of effective provider inputs:
  model, full validated temperature, max_tokens and ordered role/content messages.
  See the key-encoding correction below; user fields are not delimiter-flattened.
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

## Key-encoding correction

The original gateway rounded temperatures to three decimal places and flattened
messages with the cache utility's unit separator. Distinct valid provider inputs
could therefore share both a response-cache entry and a SingleFlight execution.
These were serialization collisions, not SHA-256 cryptographic collisions.

`Gateway._cache_key` now hashes a single JSON string with explicit fields
`format="ai-systems-lab.inference.v1"`, `model`, `temperature`, `max_tokens` and
`messages`. Each message has its own `role` and `content`, in original order.
Python's JSON encoder uses `sort_keys=True`, compact separators, `ensure_ascii=True`
and `allow_nan=False`. Validated floats are serialized without decimal rounding,
including their signed-zero representation. Unicode/control characters are escaped
reversibly; content is never stripped or Unicode-normalized. This is a defined
Python JSON format, not a claim of RFC 8785 cross-language canonicalization.

Inputs pass through `as_messages()`, so prompt shorthand and an equivalent user
message share a key. Cache/stream flags do not change the arguments passed to
`Provider.complete()` and are excluded. Cache-disabled calls still bypass cache
reads/writes and do not join cached flights. Model, message boundaries/order/roles,
exact contents, token limit and full temperature remain distinct key inputs.

A length-prefixed binary encoding could also remove ambiguity, but requires a
custom format without improving this small process-local use case. JSON provides
explicit, inspectable structure with the standard library. `TTLLRUCache.make_key`
remains unchanged: the gateway supplies only one complete payload, avoiding any
effect on unrelated multipart callers. Raw content is hashed, not used as a
dictionary key or logged. Existing field/body limits remain unchanged.

TTL/LRU, accounting, response flags, cancellation isolation and admission ownership
are unchanged. Entries are process-local and disappear on restart; the new encoding
requires no persistent-cache migration. Future changes to effective provider inputs
must update this envelope and its format version as appropriate.
