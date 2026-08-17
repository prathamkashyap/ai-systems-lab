"""Minimal in-process metrics collection with Prometheus exposition support.

Kept dependency-free (no prometheus_client requirement) so the service starts
with zero external state, while supporting standard Prometheus scrape endpoints.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


class MetricsRegistry:
    def __init__(self, max_samples: int = 5000) -> None:
        self._lock = Lock()
        self._latencies: list[float] = []
        self.max_samples = max_samples
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.deduplicated_requests = 0
        self.circuit_breaker_trips = 0
        self.errors = 0
        self.requests_by_provider: dict[str, int] = defaultdict(int)
        self._start = time.monotonic()

    def record_request(
        self,
        provider: str,
        latency_ms: float,
        cached: bool = False,
        deduplicated: bool = False,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.requests_by_provider[provider] += 1
            if cached:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            if deduplicated:
                self.deduplicated_requests += 1
            self._latencies.append(latency_ms)
            if len(self._latencies) > self.max_samples:
                self._latencies.pop(0)

    def record_circuit_trip(self) -> None:
        with self._lock:
            self.circuit_breaker_trips += 1

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def uptime_s(self) -> float:
        return time.monotonic() - self._start

    def snapshot(self) -> dict:
        with self._lock:
            latencies = sorted(self._latencies)
            avg = sum(latencies) / len(latencies) if latencies else 0.0
            total_cache_ops = self.cache_hits + self.cache_misses
            cache_rate = (self.cache_hits / total_cache_ops) if total_cache_ops else 0.0
            dedup_rate = (self.deduplicated_requests / self.total_requests) if self.total_requests else 0.0
            return {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": cache_rate,
                "deduplicated_requests": self.deduplicated_requests,
                "deduplication_rate": dedup_rate,
                "avg_latency_ms": avg,
                "p50_latency_ms": _percentile(latencies, 0.50),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "p99_latency_ms": _percentile(latencies, 0.99),
                "requests_by_provider": dict(self.requests_by_provider),
                "circuit_breaker_trips": self.circuit_breaker_trips,
                "errors": self.errors,
            }

    def to_prometheus_text(self) -> str:
        """Format metrics in standard Prometheus text exposition format."""
        snap = self.snapshot()
        lines = [
            "# HELP ai_gateway_requests_total Total number of inference requests processed",
            "# TYPE ai_gateway_requests_total counter",
            f"ai_gateway_requests_total {snap['total_requests']}",
            "",
            "# HELP ai_gateway_cache_hits_total Total cache hits",
            "# TYPE ai_gateway_cache_hits_total counter",
            f"ai_gateway_cache_hits_total {snap['cache_hits']}",
            "",
            "# HELP ai_gateway_cache_misses_total Total cache misses",
            "# TYPE ai_gateway_cache_misses_total counter",
            f"ai_gateway_cache_misses_total {snap['cache_misses']}",
            "",
            "# HELP ai_gateway_deduplicated_requests_total In-flight deduplicated requests",
            "# TYPE ai_gateway_deduplicated_requests_total counter",
            f"ai_gateway_deduplicated_requests_total {snap['deduplicated_requests']}",
            "",
            "# HELP ai_gateway_errors_total Total gateway errors",
            "# TYPE ai_gateway_errors_total counter",
            f"ai_gateway_errors_total {snap['errors']}",
            "",
            "# HELP ai_gateway_latency_ms Inference gateway latency percentiles in ms",
            "# TYPE ai_gateway_latency_ms gauge",
            f'ai_gateway_latency_ms{{quantile="0.50"}} {snap["p50_latency_ms"]:.2f}',
            f'ai_gateway_latency_ms{{quantile="0.95"}} {snap["p95_latency_ms"]:.2f}',
            f'ai_gateway_latency_ms{{quantile="0.99"}} {snap["p99_latency_ms"]:.2f}',
            f'ai_gateway_latency_ms{{quantile="mean"}} {snap["avg_latency_ms"]:.2f}',
            "",
            "# HELP ai_gateway_provider_requests_total Requests served by provider",
            "# TYPE ai_gateway_provider_requests_total counter",
        ]
        for provider, count in snap["requests_by_provider"].items():
            lines.append(f'ai_gateway_provider_requests_total{{provider="{provider}"}} {count}')
        lines.append("")
        return "\n".join(lines)
