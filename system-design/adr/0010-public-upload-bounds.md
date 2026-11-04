# ADR 0010: Bound public body readers and total upload duration

## Status
Accepted for implementation; deployment remains separately gated.

## Context
Per-body byte limits and HTTP token buckets do not bound how many accepted
uploads accumulate concurrently or how long each reader waits. An exempt health
request can also carry a slow body. The existing ingress component pre-reads
bounded bodies in the ASGI request task, then replays them to FastAPI.

## Decision
Extend that same buffer, only in public-demo mode:

- `AI_SYSTEMS_LAB_MAX_BODY_READERS=8`, integer range 1–64.
- `AI_SYSTEMS_LAB_BODY_TIMEOUT_S=5`, finite range 0.1–30 seconds.
- Validate at application import without echoing raw configuration. Developer
  mode ignores both settings and retains existing CORS/body behavior.

Eight readers permit overlapping small playground requests with limited upload
accumulation. Five seconds allows small JSON uploads without becoming an
indefinite slow-reader reservation. These are conservative demo defaults, not
deployment throughput or latency claims. Compose forwards both settings.

After declared-length validation, atomically check/increment one per-process
counter, without an await. Reject immediately at capacity; never enqueue waiting
readers. All HTTP paths/methods, even empty bodies and health, reserve a slot.
Rate middleware remains outermost. Non-HTTP scopes pass through.

Start one absolute `loop.time()` deadline at reservation and use
`asyncio.timeout_at` only around upload reading and completed-length validation.
Explicit checks around receives reject late results even if a receive suppresses
the timeout. Yield each iteration so ready empty/tiny frames cannot starve timers
or other tasks. Progress does not renew the deadline. Completion observed exactly
at the deadline expires; strictly earlier completion can proceed.

The request task owns receive, cancellation and cleanup. A synchronous `finally`
releases exactly once after the original upload receive has unwound. No detached
reader task, permit waiter, second body buffer or inference ownership change is
introduced. A validated bounded body transfers to downstream replay; the upload
slot and timeout end before inference or response streaming. Later replay calls
retain the original disconnect channel, which is not another upload phase.

## Responses and precedence

1. Rate exhaustion: existing 429, before header/body inspection.
2. Invalid/oversized declared length: existing 400/413, before reader admission.
3. Reader saturation: 503 `ingress_capacity_unavailable` / `Request ingress is busy`.
4. Upload expiry: 408 `request_timeout` / `Request body timed out`.
5. Within the read phase, deadline checks precede chunk processing, byte overflow
   precedes completed-length mismatch; existing 413/400 contracts remain.
6. Normal downstream validation and inference retain 422, inference 503/504 and
   SSE lifetime/disconnect behavior.

Use the existing nested `error.code`/`error.message` JSON shape. No retry interval
is predictable for ingress saturation/expiry. Rejection never enters FastAPI,
retrieval, SingleFlight, micro-batching or inference metrics. Disconnect/cancellation
may produce no response. Errors raised while sending a rejection propagate rather
than attempting a second HTTP response.

## Consequences and limits
Upload accumulation is bounded by reader count and each body's byte budget, plus
allocation overhead/copies. Completed replay bodies, server/socket buffers and
downstream allocations are separate; this is not a total memory bound. Budgets
multiply per worker/instance. Health can report 503 when all readers are busy.

Cooperative cancellation stops a stalled receive. If cleanup resists cancellation,
retain its slot until it actually unwinds; a late result cannot invoke downstream.
Middleware cannot force-stop arbitrary code or run deadlines while the event loop
is blocked. Supervisor process termination remains the fallback. Header parsing,
socket acceptance/concurrency, proxy buffering and production client/disconnect
behavior require separate server/proxy verification. No deployment is authorized.
