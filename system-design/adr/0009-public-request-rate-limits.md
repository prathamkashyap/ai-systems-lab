# ADR 0009: Bound public HTTP frequency and identity state

## Status
Proposed; pending independent review.

## Context
Body limits bound individual payloads and admission bounds concurrent inference.
Neither bounds request frequency. The playground sends individual requests and a
50-request SingleFlight demonstration; its health poll runs every 15 seconds.

## Decision
Public startup constructs one per-process token-bucket limiter. Defaults: burst 60,
refill 2 tokens/second, registry 1024. Startup ranges are burst 1–120, refill finite
0.01–10, registry 1–4096. Developer mode ignores all three settings. The default
supports one 50-request demonstration and ordinary UI interactions; retries are
not automatic, and rapid repeat demonstrations may receive 429.

Charge one token per HTTP request before reading the body. Exempt only exact
GET /health, whose public MockProvider health path cannot invoke inference or RAG.
All other paths/methods, cache hits, duplicates, errors, and batches count. A batch
is one HTTP envelope, not one provider call; batch-member/admission bounds remain.
After frequency admission, enforce body bytes, model validation, then existing
Gateway/cache/SingleFlight/batcher/provider logic. SSE reserves its HTTP token at
entry, not per chunk. Frequency rejection never alters already accepted work.

Starlette inserts user middleware at index zero: register the rate middleware
after the body middleware so rate checks execute first. Tests inspect the actual
composition and use a forbidden receive callable to prove rejection reads no body.

Use canonical socket IP bytes (IPv4/IPv6, mapped IPv4 normalized), ignoring ports.
Missing/malformed clients share a fallback key within the same registry cap. Never
store arbitrary header strings, hostnames, or IPv6 zone identifiers as keys.

The installed uvicorn enables proxy headers by default and trusts loopback peers
unless configured otherwise. Its middleware overwrites the ASGI client from XFF
and sets port zero. Merely ignoring headers after this rewrite is insufficient.
Public servers must disable that processing (`--no-proxy-headers`, or CLI env
`UVICORN_PROXY_HEADERS=false`; programmatic `proxy_headers=False`). As a failsafe,
XFF-present port-zero scopes are rejected rather than assigned spoofable buckets.
Other forwarding-header values are ignored. This does not establish a generic
trust policy. Proxy/NAT peers may share a quota; future deployment must verify the
actual connection identity and forwarded scheme/redirect behavior. No end-user
fairness claim is made for an unverified proxy setup.

## Bounded state and rejection
An ordered dictionary tracks last activity, including denied attempts. No eviction
is allowed until a bucket is full and idle for max(60s, burst/refill). That interval
is at least the time needed to fill any bucket; if the oldest entry cannot safely
be reclaimed, conservatively reject a new identity. Existing active clients keep
their quotas. Reclamation happens on demand, without polling or background tasks.
Keys are at most 16 canonical IP bytes, or one empty fallback key; entry count
never exceeds the configured cap. Check/update has no await, so concurrent tasks
on the supported single event loop need no lock or unbounded acquisition queue.

Return sanitized HTTP 429 rate_limited. Token deficit yields Retry-After equal to
ceil((1-tokens)/refill), minimum one second. This predicts availability only absent
competing requests. Registry-full or unsafe-proxy-identity rejections omit the hint.
No request body, application parsing, inference reservations, providers, or RAG are
touched on rejection; response begins with 429 JSON even for the SSE endpoint.

## Limits
Each worker/instance owns independent state. Multiple IPs, shared proxies, and NAT
affect fairness; this is not authentication, distributed quotas, or anti-DDoS.
GET /health remains unthrottled but body-bounded. Slow upload concurrency, upload
deadlines, headers/sockets, verified proxy infrastructure, UI error presentation,
telemetry redesign, and deployment remain outside this issue. Accepted inference
ownership, deadlines, cancellation isolation, and bounded shutdown are unchanged.
