"""Shared SQLite-backed cache for cross-worker reuse."""

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


CACHE_DB_MAX_RETRIES = int(os.getenv("VIDEO_AGENT_CACHE_DB_MAX_RETRIES", "5"))
CACHE_DB_RETRY_BASE_SECONDS = float(
    os.getenv("VIDEO_AGENT_CACHE_DB_RETRY_BASE_SECONDS", "0.05")
)


class SharedTTLCache:
    """A small SQLite-backed TTL cache that can be shared across workers."""

    def __init__(self, path: str):
        self.path = str(Path(path).resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path,
            timeout=30.0,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                expires_at REAL,
                PRIMARY KEY(namespace, key)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_entries_expiry ON cache_entries(expires_at)"
        )
        self._conn.commit()

    @staticmethod
    def _is_retryable_db_error(error: Exception) -> bool:
        """Retry SQLite lock/busy errors caused by cross-worker contention."""
        if not isinstance(error, sqlite3.OperationalError):
            return False
        message = str(error).lower()
        return "locked" in message or "busy" in message

    def _run_with_retry(self, operation_name: str, fn):
        """Run a DB operation with short retries for lock contention."""
        for attempt in range(CACHE_DB_MAX_RETRIES + 1):
            try:
                return fn()
            except Exception as e:
                if attempt >= CACHE_DB_MAX_RETRIES or not self._is_retryable_db_error(
                    e
                ):
                    raise
                delay = min(CACHE_DB_RETRY_BASE_SECONDS * (2**attempt), 1.0)
                logger.warning(
                    "Retrying shared cache %s after attempt %d/%d failed: %s",
                    operation_name,
                    attempt + 1,
                    CACHE_DB_MAX_RETRIES + 1,
                    e,
                )
                time.sleep(delay)

    def get_json(self, namespace: str, key: str) -> Optional[Any]:
        """Return a cached JSON value when present and not expired."""
        now = time.time()
        try:

            def _read():
                with self._lock:
                    row = self._conn.execute(
                        "SELECT value, expires_at FROM cache_entries WHERE namespace = ? AND key = ?",
                        (namespace, key),
                    ).fetchone()
                    if row is None:
                        return None

                    value, expires_at = row
                    if expires_at is not None and expires_at <= now:
                        self._conn.execute(
                            "DELETE FROM cache_entries WHERE namespace = ? AND key = ?",
                            (namespace, key),
                        )
                        self._conn.commit()
                        return None
                    return value

            value = self._run_with_retry("read", _read)
            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            logger.warning("Shared cache read failed: %s", e)
            return None

    def set_json(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        """Store a JSON-serializable value with an optional TTL."""
        expires_at = None if ttl_seconds is None else time.time() + ttl_seconds
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        try:

            def _write():
                with self._lock:
                    self._conn.execute(
                        """
                        INSERT INTO cache_entries(namespace, key, value, expires_at)
                        VALUES(?, ?, ?, ?)
                        ON CONFLICT(namespace, key) DO UPDATE SET
                            value = excluded.value,
                            expires_at = excluded.expires_at
                        """,
                        (namespace, key, payload, expires_at),
                    )
                    self._conn.commit()

            self._run_with_retry("write", _write)
        except Exception as e:
            logger.warning("Shared cache write failed: %s", e)


_shared_cache = None
_shared_cache_lock = threading.Lock()


def get_shared_cache() -> Optional[SharedTTLCache]:
    """Get or create the process-local handle to the shared cache DB."""
    if os.getenv("VIDEO_AGENT_DISABLE_SHARED_CACHE") == "1":
        return None

    global _shared_cache
    with _shared_cache_lock:
        if _shared_cache is None:
            cache_path = os.getenv(
                "VIDEO_AGENT_SHARED_CACHE_PATH",
                "./cache/shared_agent_cache.sqlite3",
            )
            _shared_cache = SharedTTLCache(cache_path)
        return _shared_cache
