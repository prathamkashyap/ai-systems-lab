# ADR 0001: Provider abstraction for model backends

## Status
Accepted

## Context
The gateway needs to serve completions from more than one kind of backend:
a real hosted API (Anthropic), and a deterministic offline stand-in used for
tests/CI/benchmarks (`MockProvider`). Over time this list will grow (other
vendor APIs, locally hosted models). The rest of the system (routing,
caching, batching, metrics, the HTTP layer) should not need to change every
time a backend is added, removed, or temporarily unavailable.

## Decision
Define a minimal `Provider` interface (`backend/providers/base.py`):

```python
class Provider(ABC):
    name: str
    async def complete(self, messages, *, model, max_tokens, temperature) -> tuple[str, int, int]: ...
    async def health(self) -> bool: ...
    def supports(self, model: str) -> bool: ...
```

The `Gateway` holds a list of providers and:
1. Filters to providers whose `supports(model)` returns True for the
   requested logical model name (falls back to trying all providers if none
   explicitly claim support).
2. Tries them in order, catching exceptions and falling through to the next
   provider on failure.
3. Never imports a concrete provider class directly outside of
   `backend/main.py`'s composition root.

## Consequences
- **Positive:** Adding a provider is a single new file implementing three
  methods; no changes to `gateway.py`, `main.py`'s routes, caching, or
  batching.
- **Positive:** `AnthropicProvider.health()` returns `False` (rather than
  raising) when no API key is configured, so the service degrades
  gracefully to `MockProvider` instead of crashing at import time.
- **Negative:** Ordered fallback is simple but not cost- or
  latency-aware — a provider that's slow-but-working is preferred over a
  fast-but-unsupported one purely by list position. Acceptable for now;
  flagged in `system-design/architecture.md` as a next step (weighted /
  cost-aware routing).
