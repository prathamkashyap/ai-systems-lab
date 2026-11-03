# ADR 0006: Bound public-demo inference admission

## Status
Proposed; pending independent PR review.

## Context
Micro-batcher flush size does not bound queued requests or simultaneous flushes.
SingleFlight workers outlive individual callers, and streaming bypasses the batcher.
An HTTP-handler semaphore would release too early for both detached work and SSE.

## Decision
Only the public-demo composition root installs a single-event-loop admission budget.
Default to four active work reservations, sixteen FIFO pending completion items,
and sixty-four inference request waiters. Operators may configure these per process.

Reserve a whole completion group's new SingleFlight keys synchronously under the
existing flight lock, before starting any worker. Non-cached members receive unique
keys so they never deduplicate. Cache hits and existing flight joins need no new
work reservation; cache lookup statistics are recorded only after successful admission.
All callers shield accepted workers, including uncached public work. Cancellation
releases a caller's waiter slot but never falsely releases still-running work capacity.

Pending work waits on at most the configured number of readiness futures. Active
completion reservations include batcher residence, serial per-item provider calls,
and batch-wide result delivery. This is conservative: actual provider concurrency
may be lower than the active reservation count. No provider batch API is introduced.

HTTP batches reserve atomically based on their members' new work; one waiter slot
represents the whole validated HTTP batch. Individual HTTP joiners use separate waiter
slots. Streams share the same active-work budget, cannot queue, and reserve before
headers. A response-owned wrapper closes the stream and releases capacity on all
response exit paths, including cancellation before iteration. Gateway callers have
the same ownership via the existing stream iterator or an explicitly closed open stream.

Shutdown rejects new reservations, closes/cancels streams, awaits accepted flight
workers, then closes the batcher. Capacity release is synchronous and idempotent,
so it needs no cancellable lock acquisition. Existing circuit-breaker behavior is unchanged.

Overload is HTTP 503 with the stable `capacity_unavailable` JSON error. Rejection
is not provider failure or successful execution. Do not guess a Retry-After interval
without a deadline/completion estimate. Accepted streams are not shed on overload;
disconnect and application shutdown terminate their execution and release capacity.

## Consequences
- Explicit upper bounds on admitted inference work and waiting inference requests.
- Developer mode and directly constructed gateways remain unbounded unless opted in.
- Atomic reservation requires a narrow SingleFlight group-preparation operation.
- Slow streams can occupy all active slots; bounded pending completions then fill
  and later callers receive 503. This task does not add deadlines or rate limits.
- Input/body buffering, slow socket clients, retrieval work, distributed budgets,
  metrics redesign, and deployment remain separate work. Multiple processes multiply
  the budgets; these limits do not establish complete public deployment security.
