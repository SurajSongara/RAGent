"""Qdrant: dense vectors with the payload filters that justify the dependency.

One collection per chunking strategy. That is what makes the Phase 2 bake-off a
real experiment — four indexes over the same corpus, queried identically, scored
against one golden set — instead of a reindex-and-hope each time.

Payload carries the filing metadata so "compare FY24 against FY25 for this CIK"
is a *pre-filtered* ANN query. Fetching 100 unfiltered neighbours and hoping the
right year survives reranking is the thing this avoids.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient, models

from ragent.config import get_settings

__all__ = ["collection_for", "ensure_collection", "upsert_chunks", "search", "delete_document"]


def collection_for(strategy: str) -> str:
    return f"ragent_{strategy}"


_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=get_settings().qdrant_url)
    return _client


async def ensure_collection(strategy: str, dims: int) -> None:
    client = get_client()
    name = collection_for(strategy)
    if await client.collection_exists(name):
        return

    await client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dims, distance=models.Distance.COSINE),
    )
    # Unindexed payload fields still filter correctly but scan; these four are
    # the ones every comparison query touches.
    for field, schema in (
        ("tenant_id", models.PayloadSchemaType.KEYWORD),
        ("document_id", models.PayloadSchemaType.KEYWORD),
        ("cik", models.PayloadSchemaType.KEYWORD),
        ("fiscal_year", models.PayloadSchemaType.INTEGER),
    ):
        await client.create_payload_index(
            collection_name=name, field_name=field, field_schema=schema
        )


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
