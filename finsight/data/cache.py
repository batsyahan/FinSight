"""Caching layer for market data, so repeated analyses do not refetch from Yahoo.

Backends: Redis when FINSIGHT_REDIS_URL is set (shared across API workers / containers),
otherwise JSON files under .finsight_cache/data. Values are stored as JSON (no pickle), with
per-kind TTLs. Disable with FINSIGHT_CACHE=off.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

TTL_SECONDS = {"prices": 6 * 3600, "fundamentals": 24 * 3600, "profile": 24 * 3600, "news": 3600,
               "history": 24 * 3600}


def _encode(value: Any) -> str:
    if isinstance(value, pd.Series):
        return json.dumps({"__series__": {"name": value.name, "index": [str(i) for i in value.index],
                                          "values": [None if pd.isna(v) else float(v) for v in value.values]}})
    return json.dumps(value, default=str)


def _decode(raw: str) -> Any:
    obj = json.loads(raw)
    if isinstance(obj, dict) and "__series__" in obj:
        s = obj["__series__"]
        return pd.Series(s["values"], index=pd.to_datetime(s["index"]), name=s["name"], dtype=float)
    return obj


class FileBackend:
    def __init__(self, root: str | Path = ".finsight_cache/data") -> None:
        self.root = Path(root)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.root / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"

    def get(self, key: str) -> str | None:
        p = self._path(key)
        try:
            env = json.loads(p.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        if env["expires"] < time.time():
            return None
        return env["value"]

    def set(self, key: str, value: str, ttl: int) -> None:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self._path(key).with_suffix(".tmp")
            tmp.write_text(json.dumps({"expires": time.time() + ttl, "value": value}), encoding="utf-8")
            tmp.replace(self._path(key))  # atomic on the same filesystem


class RedisBackend:  # pragma: no cover - exercised in docker-compose
    def __init__(self, url: str) -> None:
        import redis

        self.r = redis.Redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> str | None:
        return self.r.get(f"finsight:{key}")

    def set(self, key: str, value: str, ttl: int) -> None:
        self.r.set(f"finsight:{key}", value, ex=ttl)


def default_backend() -> Any:
    url = os.getenv("FINSIGHT_REDIS_URL", "").strip()
    if url:
        try:
            return RedisBackend(url)
        except Exception:  # fall back rather than fail the analysis
            pass
    return FileBackend(os.getenv("FINSIGHT_CACHE_DIR", ".finsight_cache/data"))


class CachedProvider:
    """Wraps any MarketDataProvider with read-through caching."""

    def __init__(self, inner: Any, backend: Any = None, ttl: dict[str, int] | None = None) -> None:
        self.inner, self.backend = inner, backend or default_backend()
        self.ttl = {**TTL_SECONDS, **(ttl or {})}
        self.name = inner.name
        self.hits = self.misses = 0

    def _cached(self, kind: str, key: str, fetch: Callable[[], Any]) -> Any:
        full = f"{self.name}:{kind}:{key}"
        raw = self.backend.get(full)
        if raw is not None:
            self.hits += 1
            return _decode(raw)
        self.misses += 1
        value = fetch()
        try:
            self.backend.set(full, _encode(value), self.ttl[kind])
        except Exception:  # a cache write failure must never break an analysis
            pass
        return value

    def profile(self, ticker: str) -> dict[str, Any]:
        return self._cached("profile", ticker, lambda: self.inner.profile(ticker))

    def prices(self, ticker: str, period: str = "2y") -> pd.Series:
        return self._cached("prices", f"{ticker}:{period}", lambda: self.inner.prices(ticker, period))

    def fundamentals(self, ticker: str) -> dict[str, Any]:
        return self._cached("fundamentals", ticker, lambda: self.inner.fundamentals(ticker))

    def news(self, ticker: str, limit: int = 8) -> list[dict[str, Any]]:
        return self._cached("news", f"{ticker}:{limit}", lambda: self.inner.news(ticker, limit))

    def financial_history(self, ticker: str, years: int = 4) -> list[dict[str, Any]]:
        return self._cached("history", f"{ticker}:{years}", lambda: self.inner.financial_history(ticker, years))
