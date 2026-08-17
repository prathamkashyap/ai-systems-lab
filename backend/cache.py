"""A small, dependency-free TTL + LRU cache used to avoid re-hitting model
providers for repeated requests.

Design notes live in system-design/adr/0002-caching-strategy.md.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


class TTLLRUCache:
    """Thread-safe cache combining a TTL (time-based expiry) with an LRU
    eviction policy (size-based expiry).

    - O(1) get/set via OrderedDict.
    - Expired entries are evicted lazily on access, plus swept opportunistically.
    """

    def __init__(self, max_size: int = 1024, ttl_seconds: float = 300.0) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(*parts: str) -> str:
        raw = "\x1f".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expires_at < time.monotonic():
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return entry.value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = CacheEntry(
                value=value, expires_at=time.monotonic() + self.ttl_seconds
            )
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
