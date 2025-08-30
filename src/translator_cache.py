# translator_cache.py
import functools
import hashlib
import os
import threading
import time
from typing import Optional, cast

from cachetools import LRUCache

try:
    import redis
    from redis.client import Redis

    REDIS_URL = os.getenv("REDIS_URL") or "redis://localhost:6379/0"
    _r = redis.from_url(REDIS_URL, decode_responses=True)
    _r.ping()
    _redis: Optional[Redis] = cast(Redis, _r)
except Exception:
    _redis = None
_L1 = LRUCache(maxsize=1000)
_L1_lock = threading.Lock()
_L1_TTL = 60 * 60


def _digest(txt: str) -> str:
    return hashlib.sha256(txt.encode()).hexdigest()[:32]


def cache_translate(func):

    @functools.wraps(func)
    def wrapper(text: str) -> str:
        text = text.strip()
        key = f"{func.__name__}:{_digest(text)}"

        now = time.time()
        with _L1_lock:
            if key in _L1:
                val, ts = _L1[key]
                if now - ts < _L1_TTL:
                    return val
                else:
                    del _L1[key]

        if _redis is not None:
            try:
                r_val = cast(Optional[str], _redis.get(key))
            except Exception:
                r_val = None
            if r_val is not None:
                with _L1_lock:
                    _L1[key] = (r_val, now)
                return r_val

        val: str = func(text)
        with _L1_lock:
            _L1[key] = (val, now)
        if _redis:
            _redis.set(key, val, ex=60 * 60 * 24 * 30)
        return val

    return wrapper
