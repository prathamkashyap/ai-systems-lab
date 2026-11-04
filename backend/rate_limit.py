"""Single-event-loop, bounded HTTP request-frequency budgets."""

from __future__ import annotations

import ipaddress
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

MAX_BURST = 120
MIN_REFILL_PER_S = 0.01
MAX_REFILL_PER_S = 10
MAX_CLIENTS = 4096
_RATE_LIMITED = {"error": {"code": "rate_limited", "message": "Too many requests"}}


@dataclass(frozen=True, slots=True)
class RateDecision:
    allowed: bool
    retry_after: int | None = None


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated: float  # also last activity, including rejected attempts


class TokenBucketLimiter:
    """Atomic synchronous checks on one event loop; no tasks, locks, or waiters."""

    def __init__(
        self,
        burst: int = 60,
        refill_per_s: float = 2,
        max_clients: int = 1024,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            type(burst) is not int
            or not 1 <= burst <= MAX_BURST
            or not math.isfinite(refill_per_s)
            or not MIN_REFILL_PER_S <= refill_per_s <= MAX_REFILL_PER_S
            or type(max_clients) is not int
            or not 1 <= max_clients <= MAX_CLIENTS
        ):
            raise ValueError("Invalid rate-limit settings")
        self.burst = burst
        self.refill_per_s = refill_per_s
        self.max_clients = max_clients
        self.reclaim_after_s = max(60.0, burst / refill_per_s)
        self._clock = clock
        self._last_time = -math.inf
        self._buckets: OrderedDict[bytes, _Bucket] = OrderedDict()

    def _available(self, bucket: _Bucket, now: float) -> float:
        return min(self.burst, bucket.tokens + (now - bucket.updated) * self.refill_per_s)

    def check(self, key: bytes) -> RateDecision:
        now = max(self._last_time, self._clock())
        self._last_time = now
        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= self.max_clients:
                # Entries are ordered by activity. Younger entries cannot be
                # idle long enough if the oldest one is not safely reclaimable.
                oldest = next(iter(self._buckets.values()))
                if (
                    now - oldest.updated < self.reclaim_after_s
                    or self._available(oldest, now) < self.burst
                ):
                    return RateDecision(False)  # registry full; no reliable retry prediction
                self._buckets.popitem(last=False)
            bucket = _Bucket(float(self.burst), now)
            self._buckets[key] = bucket
        else:
            bucket.tokens = self._available(bucket, now)
            bucket.updated = now
            self._buckets.move_to_end(key)
        if bucket.tokens >= 1:
            bucket.tokens -= 1
            return RateDecision(True)
        delay = math.ceil((1 - bucket.tokens) / self.refill_per_s)
        return RateDecision(False, max(1, delay))


def _client_key(scope: Scope) -> bytes | None:
    """Canonical socket IP, shared unknown bucket, or reject unsafe proxy identity."""
    client = scope.get("client")
    if not isinstance(client, (tuple, list)) or len(client) != 2:
        return b""  # one fallback bucket, counted within max_clients
    host, port = client
    if port == 0 and any(name.lower() == b"x-forwarded-for" for name, _ in scope["headers"]):
        # Uvicorn's ProxyHeadersMiddleware rewrites client=(forwarded_host, 0).
        # The original peer cannot be recovered here. Never mint buckets from it.
        return None
    if (
        not isinstance(host, str)
        or len(host) > 45
        or "%" in host
        or type(port) is not int
        or not 1 <= port <= 65535
    ):
        return b""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return b""
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.packed  # fixed-size keys; no ports or header contents stored


class RateLimitMiddleware:
    def __init__(self, app: ASGIApp, limiter: TokenBucketLimiter) -> None:
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or (scope["method"] == "GET" and scope["path"] == "/health"):
            await self.app(scope, receive, send)
            return
        key = _client_key(scope)
        decision = RateDecision(False) if key is None else self.limiter.check(key)
        if not decision.allowed:
            headers = (
                {} if decision.retry_after is None else {"Retry-After": str(decision.retry_after)}
            )
            await JSONResponse(_RATE_LIMITED, status_code=429, headers=headers)(
                scope, receive, send
            )
            return
        await self.app(scope, receive, send)
