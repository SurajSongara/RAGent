"""asyncpg connection pool.

No ORM. `infra/postgres/init.sql` is the source of truth for the schema, and an
ORM layer restating it in Python is one more place for the two to drift apart.
The queries here are short enough to read directly, which matters when the whole
point of the project is that the mechanics stay visible.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from ragent.config import get_settings

__all__ = ["get_pool", "close_pool", "acquire"]

_pool: asyncpg.Pool | None = None


def _dsn() -> str:
    """asyncpg wants a bare postgres:// DSN, not the SQLAlchemy-style URL."""
    return get_settings().database_url.replace("postgresql+asyncpg://", "postgresql://")


async def _init_connection(conn: asyncpg.Connection) -> None:
    # Without this, jsonb round-trips as a string and every caller has to
    # remember to decode it.
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            _dsn(),
            min_size=2,
            max_size=10,
            init=_init_connection,
            # Stage handlers can be slow; a hung query should surface as an
            # error the retry policy can act on, not as a wedged worker.
            command_timeout=60,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def acquire() -> Any:
    """`async with acquire() as conn:` — the common case."""

    class _Acquirer:
        async def __aenter__(self) -> asyncpg.Connection:
            self._pool = await get_pool()
            self._conn = await self._pool.acquire()
            return self._conn

        async def __aexit__(self, *exc: object) -> None:
            await self._pool.release(self._conn)

    return _Acquirer()
