"""Unified cache for MCP server results.

Single FanoutCache instance shared across all modules. Cross-process safe
(multiple uvicorn workers share via SQLite), bounded (LRU eviction at
configurable size limit), with per-key TTL.

Resilience guarantees:
- If cache initialization fails (bad dir, bad env, corrupt DB), the server
  still starts with a no-op cache that always misses.
- Uses JSON serialization (not pickle) to eliminate deserialization attacks.
- Cache directory is created with 0o700 permissions (owner-only).

Usage::

    from indra_agent.mcp_server.cache import cache, make_key, DEFAULT_TTL

    key = make_key("endpoint", endpoint_name, kwargs_dict)
    cached = cache.get(key)
    if cached is None:
        result = expensive_query()
        cache.set(key, result, expire=DEFAULT_TTL)
"""

import atexit
import hashlib
import json
import logging
import os
import sqlite3
import stat
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLite cross-thread safety patch
# ---------------------------------------------------------------------------
# Third-party libraries (GILDA's SqliteEntries, INDRA's SqliteOntology) store
# SQLite connections as plain instance attributes — NOT thread-local.  When
# asyncio.to_thread() dispatches work to different thread-pool workers, those
# connections are accessed from a thread other than the one that created them,
# raising: "SQLite objects created in a thread can only be used in that same
# thread."
#
# Setting check_same_thread=False disables the Python-level thread check.
# This is safe here because:
#   1. The affected libraries only SELECT (read-only).
#   2. SQLite itself is compiled with SQLITE_THREADSAFE=1 (serialized mode)
#      on all standard distributions, so the C layer handles concurrent access.
#
# Applied once at import time — before diskcache or any other module creates
# SQLite connections.
# ---------------------------------------------------------------------------
_original_sqlite3_connect = sqlite3.connect


def _sqlite3_connect_thread_safe(*args, **kwargs):
    kwargs.setdefault("check_same_thread", False)
    return _original_sqlite3_connect(*args, **kwargs)


sqlite3.connect = _sqlite3_connect_thread_safe


# ---------------------------------------------------------------------------
# Configuration (environment overrides with validation)
# ---------------------------------------------------------------------------

def _parse_env_int(name: str, default: int, min_val: int = 1) -> int:
    """Parse an environment variable as a bounded positive integer."""
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        value = int(raw)
    except (ValueError, TypeError):
        logger.warning(
            "%s=%r is not a valid integer, using default %d", name, raw, default
        )
        return default
    if value < min_val:
        logger.warning(
            "%s=%d is below minimum %d, using default %d",
            name, value, min_val, default,
        )
        return default
    return value


_CACHE_DIR = os.environ.get(
    "INDRA_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "indra_cogex_mcp"),
)
_SIZE_LIMIT_MB = _parse_env_int("INDRA_CACHE_SIZE_MB", default=2048, min_val=1)
_SIZE_LIMIT = _SIZE_LIMIT_MB * 1024 * 1024
_SHARDS = _parse_env_int("INDRA_CACHE_SHARDS", default=4, min_val=1)

DEFAULT_TTL = _parse_env_int("INDRA_CACHE_TTL", default=3600, min_val=1)


# ---------------------------------------------------------------------------
# No-op fallback cache (dict-like interface that always misses)
# ---------------------------------------------------------------------------

class _NullCache:
    """Fallback cache that silently discards all writes and returns misses.

    Implements the subset of the diskcache.FanoutCache API used by this
    codebase so callers don't need to know whether the real cache is alive.
    """

    def get(self, key: str, default: Any = None, **kw: Any) -> Any:
        return default

    def set(self, key: str, value: Any, expire: int | None = None, **kw: Any) -> bool:
        return True

    def close(self) -> None:
        pass

    def volume(self) -> int:
        return 0

    def __len__(self) -> int:
        return 0

    def __contains__(self, key: str) -> bool:
        return False


# ---------------------------------------------------------------------------
# Singleton cache (safe init with fallback)
# ---------------------------------------------------------------------------

def _init_cache() -> "_NullCache | Any":
    """Create the FanoutCache, falling back to NullCache on any failure."""
    try:
        from diskcache import FanoutCache
        from diskcache import JSONDisk

        instance = FanoutCache(
            directory=_CACHE_DIR,
            shards=_SHARDS,
            size_limit=_SIZE_LIMIT,
            eviction_policy="least-recently-used",
            timeout=1.0,
            disk=JSONDisk,
            tag_index=True,
        )

        # Restrict cache directory to owner-only (F8: directory permissions)
        try:
            os.chmod(_CACHE_DIR, stat.S_IRWXU)  # 0o700
        except OSError as exc:
            logger.warning("Could not set cache dir permissions: %s", exc)

        logger.info(
            "Cache initialized: dir=%s, size_limit=%dMB, ttl=%ds, shards=%d, disk=JSONDisk",
            _CACHE_DIR, _SIZE_LIMIT_MB, DEFAULT_TTL, _SHARDS,
        )

        # Clean up SQLite connections on interpreter exit
        atexit.register(instance.close)

        return instance

    except Exception as exc:
        logger.error(
            "Cache initialization failed (%s). Running with no-op cache — "
            "all queries will bypass cache. Fix the underlying issue for "
            "production use. Cache dir: %s",
            exc, _CACHE_DIR,
        )
        return _NullCache()


cache = _init_cache()


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

def make_key(namespace: str, *parts: Any) -> str:
    """Deterministic cache key from a namespace and arbitrary parts.

    Examples::

        make_key("endpoint", "get_diseases_for_gene", {"gene": ["HGNC", "6407"]})
        make_key("cypher", query_string, {"param": "value"}, 100)

    Returns a namespaced key like ``endpoint:a3f8b2c1d9e0``.
    """
    payload = json.dumps(parts, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{namespace}:{digest}"


def cache_stats() -> dict:
    """Return cache statistics for observability."""
    try:
        return {
            "total_entries": len(cache),
            "total_volume_bytes": cache.volume(),
            "cache_dir": _CACHE_DIR,
            "size_limit_bytes": _SIZE_LIMIT,
            "is_null_cache": isinstance(cache, _NullCache),
        }
    except Exception:
        return {"error": "stats unavailable", "is_null_cache": isinstance(cache, _NullCache)}
