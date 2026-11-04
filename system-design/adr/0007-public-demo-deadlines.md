# ADR 0007: Own public inference deadlines through actual execution

## Status
Proposed; pending independent review.

## Context and ownership
HTTP callers await shielded SingleFlight workers. A worker awaits a MicroBatcher
item future; a separate batch task executes sequential provider calls. Cancelling
the item future cannot establish that provider work stopped. Releasing its permit
at that point would violate ADR 0006's active-work bound.

## Decision
Use monotonic absolute deadlines assigned with new admission reservations: work
10s, stream 15s. Joiners share the flight deadline. Pending admission uses the same
deadline as batcher residence and execution. Developer mode has no new deadlines.

Transfer each completion's permit-release callback to the batcher at submission.
Queued expiry removes/fails the item and releases capacity. Active expiry errors
all unresolved futures in the flush and cancels its owning execution task. Only
actual task termination invokes release callbacks. Never infer termination from
a completed or cancelled item future. Future exceptions and detached task outcomes
are observed even when all callers have cancelled. Shared waiters remain shielded.

Batch-wide timeout discards results and suppresses public cache/success-metric
commit until the whole flush delivers in time. This is a targeted placement change,
not per-item provider failure isolation or a metrics redesign.

Streams own a tracked per-step provider task and an independent cleanup task.
The total response deadline also covers header/body sending and pauses between
chunks. Deadline or disconnect cleanup cancels/joins actual provider work before
releasing its work permit. Observers use bounded waits; unfinished cleanup retains
its reservation. Public HTTP timeout is sanitized 504 before headers, or one terminal
SSE error after headers with no subsequent success marker.

An `aclose()` exception is not proof of termination. After a finalization failure,
release execution capacity only if the owned provider step has finished and all
owned native async generators have no execution frames. Gateway records provider
iterators as well as its wrapper: a closed wrapper can hide an unterminated provider
iterator after a nested close error. Finalize bookkeeping and report a
sanitized `StreamFinalizationError`; shutdown continues closing other streams.
If generator termination cannot be established (including non-native iterators),
retain the work reservation and stream ownership, fail admission closed, and report
`ShutdownIncomplete`. Cleanup cancels its obsolete deadline timer when it starts.
The response owner finalizes its waiter reservation even if cleanup fails. Repeated
close calls observe the same cleanup result and never reopen failed admission.

Stop admission synchronously at shutdown. One shielded shutdown task closes streams
and drains work for 5s. On grace expiry, error queued work, cancel running flushes,
and observe all owned tasks for 1s. A nonterminating task causes fail-closed admission
and a sanitized incomplete-shutdown exception. Never reset its active counter to zero.

## Process boundary and consequences
No in-process scheme safely kills arbitrary cancellation-resistant Python or an
event loop blocked in synchronous code. Public operation requires a supervisor
SIGKILL budget, not just a restart policy. With defaults, configure uvicorn request
drain to 6s and a supervisor hard stop 15s after SIGTERM. Compose records the 15s
stop grace and forwards an opt-in uvicorn timeout; future hosting must verify it.
Larger duration settings require a larger supervisor stop budget.

Timeout errors can reach callers before cooperative cancellation cleanup completes;
capacity stays occupied until actual execution stops. Grace-exceeding cleanup fails
admission closed even if it eventually finishes, requiring an operator restart.
Deadlines are cooperative bounds, not real-time guarantees under event-loop starvation.
Per-client rate/body limits, other correctness fixes, metrics architecture, streaming
fallback redesign, UI, and deployment remain separate work.
