"""Live pipeline progress over Valkey pub/sub.

Workers publish; the API subscribes and relays to the browser over SSE. Going
through Valkey rather than having the API poll Postgres means several API
replicas can each serve the same document's progress without any of them
hammering the database, and the UI updates the instant a stage finishes rather
than on the next poll tick.

Progress is deliberately fire-and-forget. A dropped event is a slightly stale
progress bar; blocking a stage handler on the UI transport would be a far worse
trade.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import redis.asyncio as redis

from ragent.config import get_settings

__all__ = ["publish", "subscribe", "channel_for"]

log = logging.getLogger(__name__)

_client: redis.Redis | None = None


def channel_for(document_id: str) -> str:
    return f"ragent:progress:{document_id}"


def get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(get_settings().valkey_url, decode_responses=True)
    return _client


async def publish(document_id: str, event: dict[str, Any]) -> None:
    try:
        await get_client().publish(channel_for(document_id), json.dumps(event))
    except Exception as exc:  # noqa: BLE001 - progress must never fail a stage
        log.warning("progress publish failed for %s: %s", document_id, exc)


async def subscribe(document_id: str) -> AsyncIterator[dict[str, Any]]:
    """Yield progress events for one document until the caller stops listening."""
    pubsub = get_client().pubsub()
    await pubsub.subscribe(channel_for(document_id))
    try:
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            try:
                yield json.loads(message["data"])
            except json.JSONDecodeError:
                continue
    finally:
        await pubsub.unsubscribe(channel_for(document_id))
        await pubsub.aclose()
