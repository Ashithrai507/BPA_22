from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import requests

from protein_engine.config import RATE_LIMIT_DEFAULT, RETRY_BACKOFF


class ApiError(RuntimeError):
    def __init__(self, status_code: int, url: str) -> None:
        super().__init__(f"HTTP {status_code} for {url}")
        self.status_code = status_code
        self.url = url


class CacheMissError(RuntimeError):
    pass


class ApiCache:
    def __init__(
        self,
        path: Path,
        transport: object | None = None,
        rate_limit: float = RATE_LIMIT_DEFAULT,
        backoff: Sequence[float] = RETRY_BACKOFF,
    ) -> None:
        self._transport = transport if transport is not None else requests
        self._rate_limit = rate_limit
        self._backoff = list(backoff)
        self._last_request = 0.0
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS http_cache ("
            "key TEXT PRIMARY KEY, body TEXT NOT NULL, fetched_at TEXT NOT NULL)"
        )
        self._db.commit()

    @staticmethod
    def _key(url: str, params: dict | None) -> str:
        canonical = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256((url + "\x00" + canonical).encode()).hexdigest()

    def get(self, url: str, params: dict | None = None) -> str | None:
        row = self._db.execute("SELECT body FROM http_cache WHERE key = ?", (self._key(url, params),)).fetchone()
        return row[0] if row else None

    def set(self, url: str, params: dict | None, body: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO http_cache (key, body, fetched_at) VALUES (?, ?, ?)",
            (self._key(url, params), body, datetime.now(UTC).isoformat()),
        )
        self._db.commit()

    def _throttle(self) -> None:
        if self._rate_limit <= 0:
            return
        delay = (1.0 / self._rate_limit) - (time.monotonic() - self._last_request)
        if delay > 0:
            time.sleep(delay)
        self._last_request = time.monotonic()

    def fetch(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        allow_network: bool = True,
    ) -> str:
        cached = self.get(url, params)
        if cached is not None:
            return cached
        if not allow_network:
            raise CacheMissError(f"no cached response for {url} {params}")
        attempts = self._backoff or [0.0]
        last_err: ApiError | None = None
        for delay in attempts:
            self._throttle()
            resp = self._transport.get(url, params=params, headers=headers)
            if resp.status_code == 200:
                self.set(url, params, resp.text)
                return resp.text
            if resp.status_code not in (429, 500, 502, 503, 504):
                raise ApiError(resp.status_code, url)
            last_err = ApiError(resp.status_code, url)
            if delay:
                time.sleep(delay)
        raise last_err if last_err is not None else ApiError(0, url)
