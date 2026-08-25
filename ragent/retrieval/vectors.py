"""Qdrant: dense vectors with the payload filters that justify the dependency.

One collection per chunking strategy. That is what makes the Phase 2 bake-off a
real experiment — four indexes over the same corpus, queried identically, scored
against one golden set — instead of a reindex-and-hope each time.

Payload carries the filing metadata so "compare FY24 against FY25 for this CIK"
is a *pre-filtered* ANN query. Fetching 100 unfiltered neighbours and hoping the
right year survives reranking is the thing this avoids.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from ragent.config import get_settings

__all__ = [
    "collection_for",
    "ensure_collection",
    "upsert_chunks",
    "search",
    "delete_document",
    "DimensionMismatch",
]


def collection_for(strategy: str) -> str:
    return f"ragent_{strategy}"


_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=get_settings().qdrant_url)
    return _client


class DimensionMismatch(RuntimeError):
    """Existing collection was built for a different embedding model."""


async def ensure_collection(strategy: str, dims: int) -> None:
    """Create the collection if missing. Safe to call concurrently.

    check-then-create is a race: several embed workers finish their first
    document at roughly the same time, all see no collection, and all try to
    create it. Losing that race is not an error — the collection exists, which
    is the postcondition — so a 409 is swallowed rather than retried.

    A size mismatch, on the other hand, is fatal and worth saying plainly.
    Switching embedding model changes the vector width, and Qdrant would
    otherwise reject every upsert with a message that says nothing about the
    actual cause.
    """
    client = get_client()
    name = collection_for(strategy)

    if await client.collection_exists(name):
        info = await client.get_collection(name)
        existing = info.config.params.vectors.size  # type: ignore[union-attr]
        if existing != dims:
            raise DimensionMismatch(
                f"collection {name!r} holds {existing}-dim vectors but the current "
                f"embedding model produces {dims}. Re-index with `make reset`, or "
                f"point EMBEDDING_BACKEND back at the model that built it."
            )
        return

    try:
        await client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dims, distance=models.Distance.COSINE),
        )
    except UnexpectedResponse as exc:
        if exc.status_code != HTTPStatus.CONFLICT:
            raise
        return  # another worker got there first


async def upsert_chunks(
    strategy: str,
    chunk_ids: list[str],
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
) -> None:
    if not chunk_ids:
        return
    await get_client().upsert(
        collection_name=collection_for(strategy),
        points=[
            models.PointStruct(id=cid, vector=vec, payload=payload)
            for cid, vec, payload in zip(chunk_ids, vectors, payloads, strict=True)
        ],
        wait=True,
    )


def _build_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    if not filters:
        return None
    conditions: list[models.FieldCondition] = []
    for key, value in filters.items():
        if value is None:
            continue
        match = (
            models.MatchAny(any=list(value))
            if isinstance(value, (list, tuple, set))
            else models.MatchValue(value=value)
        )
        conditions.append(models.FieldCondition(key=key, match=match))
    return models.Filter(must=conditions) if conditions else None


async def search(
    strategy: str,
    vector: list[float],
    *,
    limit: int = 50,
    filters: dict[str, Any] | None = None,
) -> list[tuple[str, float]]:
    result = await get_client().query_points(
        collection_name=collection_for(strategy),
        query=vector,
        limit=limit,
        query_filter=_build_filter(filters),
        with_payload=False,
    )
    return [(str(point.id), float(point.score)) for point in result.points]


async def delete_document(strategy: str, document_id: str) -> None:
    """Drop a document's points so a re-ingest cannot leave orphans behind."""
    name = collection_for(strategy)
    client = get_client()
    if not await client.collection_exists(name):
        return
    await client.delete(
        collection_name=name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            )
        ),
        wait=True,
    )
