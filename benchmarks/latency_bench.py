"""Benchmarks the gateway's latency/throughput characteristics with and
without caching, micro-batching, and SingleFlight in-flight deduplication,
using the deterministic MockProvider so results are reproducible without
network access or API costs.

Run:
    python -m benchmarks.latency_bench
"""

from __future__ import annotations

import asyncio
import statistics
import time

from backend.gateway import Gateway
from backend.models import CompletionRequest
from backend.providers.mock_provider import MockProvider


async def _fire(gateway: Gateway, n: int, unique_prompts: bool, cache: bool) -> dict:
    async def one(i: int) -> float:
        prompt = f"prompt-{i}" if unique_prompts else "prompt-shared"
        start = time.perf_counter()
        await gateway.complete(CompletionRequest(prompt=prompt, model="mock-small", cache=cache))
        return (time.perf_counter() - start) * 1000

    wall_start = time.perf_counter()
    latencies = await asyncio.gather(*(one(i) for i in range(n)))
    wall_ms = (time.perf_counter() - wall_start) * 1000
    return {
        "n": n,
        "wall_ms": wall_ms,
        "throughput_rps": n / (wall_ms / 1000),
        "p50_ms": statistics.median(latencies),
        "p95_ms": statistics.quantiles(latencies, n=20)[18] if n >= 20 else max(latencies),
        "mean_ms": statistics.mean(latencies),
    }


async def run_all() -> list[dict]:
    scenarios = []

    n = 200

    # 1. Baseline: No caching, unique prompts (independent batches)
    gw = Gateway(providers=[MockProvider(base_latency_ms=15, jitter_ms=5)], batch_window_ms=10, max_batch_size=16)
    r = await _fire(gw, n, unique_prompts=True, cache=False)
    scenarios.append({"scenario": "baseline (unique prompts, no cache)", **r, **gw._batcher.stats})

    # 2. Concurrent identical prompts WITH SingleFlight request deduplication
    gw = Gateway(providers=[MockProvider(base_latency_ms=15, jitter_ms=5)], batch_window_ms=10, max_batch_size=16)
    r = await _fire(gw, n, unique_prompts=False, cache=True)
    snap = gw.metrics.snapshot()
    scenarios.append(
        {
            "scenario": "concurrent identical prompts (SingleFlight enabled)",
            **r,
            "deduplicated_requests": snap["deduplicated_requests"],
            "deduplication_rate": f"{snap['deduplication_rate'] * 100:.1f}%",
            "batches_executed": gw._batcher.stats["batches_executed"],
        }
    )

    # 3. Unique prompts, large batch window -> batch coalescing demonstration
    gw = Gateway(providers=[MockProvider(base_latency_ms=15, jitter_ms=5)], batch_window_ms=50, max_batch_size=64)
    r = await _fire(gw, n, unique_prompts=True, cache=False)
    scenarios.append({"scenario": "unique prompts (wide batch window 50ms)", **r, **gw._batcher.stats})

    # 4. Sequential repeated prompt -> sub-millisecond memory cache speedup
    gw = Gateway(providers=[MockProvider(base_latency_ms=15, jitter_ms=5)], batch_window_ms=10, max_batch_size=16)
    latencies = []
    seq_start = time.perf_counter()
    for _ in range(n):
        start = time.perf_counter()
        await gw.complete(CompletionRequest(prompt="prompt-shared", model="mock-small", cache=True))
        latencies.append((time.perf_counter() - start) * 1000)
    wall_ms = (time.perf_counter() - seq_start) * 1000
    scenarios.append(
        {
            "scenario": "repeated prompt (sequential cache hits)",
            "n": n,
            "wall_ms": wall_ms,
            "throughput_rps": n / (wall_ms / 1000),
            "p50_ms": statistics.median(latencies),
            "p95_ms": statistics.quantiles(latencies, n=20)[18] if n >= 20 else max(latencies),
            "mean_ms": statistics.mean(latencies),
            "cache_hit_rate": f"{gw.cache.hit_rate * 100:.1f}%",
        }
    )

    return scenarios


def print_table(rows: list[dict]) -> None:
    cols = ["scenario", "n", "wall_ms", "throughput_rps", "p50_ms", "p95_ms", "mean_ms"]
    widths = {
        c: max(
            len(c),
            *(
                len(f"{row.get(c, ''):.2f}" if isinstance(row.get(c), float) else str(row.get(c, "")))
                for row in rows
            ),
        )
        for c in cols
    }

    def fmt(v) -> str:
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(fmt(row.get(c, "")).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    results = asyncio.run(run_all())
    print_table(results)
    for row in results:
        extra = {
            k: v
            for k, v in row.items()
            if k not in {"scenario", "n", "wall_ms", "throughput_rps", "p50_ms", "p95_ms", "mean_ms"}
        }
        if extra:
            print(f"  {row['scenario']}: {extra}")
