import time

from backend.cache import TTLLRUCache


def test_set_get_roundtrip():
    cache = TTLLRUCache(max_size=10, ttl_seconds=60)
    key = cache.make_key("a", "b")
    cache.set(key, "value")
    assert cache.get(key) == "value"


def test_miss_on_unknown_key():
    cache = TTLLRUCache()
    assert cache.get("nope") is None
    assert cache.misses == 1


def test_ttl_expiry():
    cache = TTLLRUCache(ttl_seconds=0.05)
    key = cache.make_key("x")
    cache.set(key, 1)
    assert cache.get(key) == 1
    time.sleep(0.08)
    assert cache.get(key) is None


def test_lru_eviction():
    cache = TTLLRUCache(max_size=2, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # evicts "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_hit_rate():
    cache = TTLLRUCache()
    cache.set("k", "v")
    cache.get("k")
    cache.get("missing")
    assert cache.hit_rate == 0.5


def test_make_key_deterministic():
    a = TTLLRUCache.make_key("model", "temp:0.7", "user:hi")
    b = TTLLRUCache.make_key("model", "temp:0.7", "user:hi")
    c = TTLLRUCache.make_key("model", "temp:0.8", "user:hi")
    assert a == b
    assert a != c
