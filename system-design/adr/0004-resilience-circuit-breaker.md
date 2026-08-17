# ADR 0004: Circuit breaker pattern for provider fault tolerance

## Status
Accepted

## Context
External model APIs (OpenAI, Anthropic) or self-hosted LLM endpoints (vLLM, Ollama) can experience
transient degradation, network timeouts, or quota exhaustion. If the inference gateway repeatedly
sends requests to a failing provider, callers experience prolonged latency timeouts (e.g. 30s per request)
and resource exhaustion.

## Decision
Introduce a `CircuitBreaker` (`backend/circuit_breaker.py`) per provider in the `Gateway`:
- **States**: `CLOSED` (normal routing), `OPEN` (tripped, fast-fail), and `HALF_OPEN` (probe requests).
- **Thresholds**: Configurable failure threshold (default: 3 consecutive failures) trips the circuit to `OPEN`.
- **Fast-fail**: When `OPEN`, requests bypass the failing provider immediately in sub-millisecond time and attempt the next provider in the fallback chain.
- **Recovery**: After a cooldown window (default: 10s), the state transitions to `HALF_OPEN`. If consecutive probe requests succeed, the circuit closes; if a probe fails, the cooldown resets.
- State is reported via `/health` and Prometheus metrics.

## Consequences
- **Positive:** Protects gateway tail latency during third-party provider outages; requests failover immediately rather than waiting for socket timeouts.
- **Positive:** Protects upstream APIs from repeated hammering during partial outages.
- **Negative:** Provider must experience successful probe calls in `HALF_OPEN` before normal traffic resumes.
