"""Hybrid retrieval: dense and lexical, fused on rank.

Both retrievers run concurrently — they hit different systems and there is no
reason to pay for them serially. Their scores are never compared directly; RRF
fuses on rank alone, which is what makes cosine similarity and `ts_rank_cd`
combinable without per-query calibration.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from ragent.config import get_settings
from ragent.db.pool import acquire
from ragent.db.repo import chunk_citation_targets, lexical_search
from ragent.providers.embeddings import get_embedder
from ragent.retrieval import vectors
from ragent.retrieval.fusion import reciprocal_rank_fusion

__all__ = ["Passage", "hybrid_search"]

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Passage:
    chunk_id: str
    text: str
    score: float
    document_id: str
    document_name: str
    provenance: str
    section_path: list[str] = field(default_factory=list)
    #: Paged provenance: one region per page the chunk touches.
    regions: list[dict[str, Any]] = field(default_factory=list)
    #: Flow provenance: character range into the source text.
    char_start: int | None = None
    char_end: int | None = None
    #: Which retrievers found it, and at what rank. Powers the debug view.
    retrievers: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(self.score, 6),
            "document_id": self.document_id,
            "document_name": self.document_name,
            "provenance": self.provenance,
            "section_path": self.section_path,
            "regions": self.regions,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "retrievers": self.retrievers,
        }


async def _dense(query: str, strategy: str, limit: int, filters: dict[str, Any]) -> list[str]:
    try:
        embedder = get_embedder()
        vector = await embedder.embed_query(query)
        hits = await vectors.search(strategy, vector, limit=limit, filters=filters)
        return [chunk_id for chunk_id, _ in hits]
    except Exception as exc:  # noqa: BLE001
        # Degrade to lexical-only rather than failing the query outright. A
        # partial answer beats an error page when one backend is down.
        log.warning("dense retrieval failed, continuing lexical-only: %s", exc)
        return []


async def _lexical(query: str, strategy: str, limit: int, tenant_id: str) -> list[str]:
    try:
        async with acquire() as conn:
            hits = await lexical_search(
                conn, query, strategy=strategy, limit=limit, tenant_id=tenant_id
            )
        return [chunk_id for chunk_id, _ in hits]
    except Exception as exc:  # noqa: BLE001
        log.warning("lexical retrieval failed, continuing dense-only: %s", exc)
        return []


async def hybrid_search(
    query: str,
    *,
    strategy: str | None = None,
    limit: int | None = None,
    tenant_id: str = "default",
    document_ids: list[str] | None = None,
    filters: dict[str, Any] | None = None,
) -> list[Passage]:
    """Retrieve, fuse and resolve citation targets in one pass."""
    settings = get_settings()
    strategy = strategy or settings.chunk_strategies[0]
    top_k = limit or settings.retrieval_top_k

    payload_filters: dict[str, Any] = {"tenant_id": tenant_id, **(filters or {})}
    if document_ids:
        payload_filters["document_id"] = document_ids

    dense_ids, lexical_ids = await asyncio.gather(
        _dense(query, strategy, top_k, payload_filters),
        _lexical(query, strategy, top_k, tenant_id),
    )
    if not dense_ids and not lexical_ids:
        return []

    fused = reciprocal_rank_fusion(
        {"dense": dense_ids, "lexical": lexical_ids},
        k=settings.rrf_k,
        limit=settings.rerank_top_n,
    )

    # Citation targets are resolved here, once, so the UI never re-derives a
    # highlight and every passage arrives ready to render.
    async with acquire() as conn:
        targets = await chunk_citation_targets(conn, [hit.doc_id for hit in fused])

    passages: list[Passage] = []
    for hit in fused:
        target = targets.get(hit.doc_id)
        if target is None:
            continue  # chunk deleted between retrieval and resolution
        passages.append(
            Passage(
                chunk_id=hit.doc_id,
                text=target["text"],
                score=hit.score,
                document_id=target["document_id"],
                document_name=target["document_name"],
                provenance=target["provenance"] or "paged",
                section_path=target["section_path"],
                regions=target["regions"],
                char_start=target["char_start"],
                char_end=target["char_end"],
                retrievers=hit.ranks,
            )
        )
    return passages
