"""The `chunk`, `contextualize` and `embed` stages — everything that makes a
document retrievable.

`contextualize` has an offline fallback on purpose. Without it, a reviewer who
cloned the repo and ran `make up` with no API key would watch every document
stall at 80%. A deterministic template prefix is measurably worse than a written
one, but "worse" beats "nothing happens".
"""

from __future__ import annotations

import logging
from typing import Any

from ragent.config import get_settings
from ragent.db.pool import acquire
from ragent.db.repo import (
    load_blocks,
    load_chunks,
    replace_chunks,
    set_document_status,
)
from ragent.ingest.chunking import chunk_document
from ragent.ingest.types import Chunk
from ragent.pipeline.handlers import handler
from ragent.pipeline.messages import StageMessage
from ragent.pipeline.progress import publish
from ragent.pipeline.runner import PermanentError
from ragent.providers.embeddings import get_embedder
from ragent.retrieval import vectors

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- chunk


@handler("chunk")
async def chunk_stage(message: StageMessage) -> dict[str, Any]:
    """Apply every configured strategy, storing each independently.

    All four coexist so the Phase 2 bake-off can score them head to head over
    the same corpus rather than reindexing between measurements.
    """
    settings = get_settings()

    async with acquire() as conn:
        blocks = await load_blocks(conn, message.document_id)
        if not blocks:
            raise PermanentError("no blocks to chunk; parse produced nothing")

        counts: dict[str, int] = {}
        for strategy in settings.chunk_strategies:
            if strategy == "semantic":
                # Semantic chunking embeds every sentence before it can pick
                # boundaries, which is a different cost profile from the others.
                # It belongs in the bake-off, not on the default ingest path.
                log.info("skipping semantic chunking: run it via `make bench`")
                continue

            chunks = chunk_document(
                strategy,
                blocks,
                target_tokens=settings.chunk_target_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            )
            await replace_chunks(conn, message.document_id, strategy, chunks)
            counts[strategy] = len(chunks)

    await publish(
        message.document_id,
        {
            "type": "stage",
            "stage": "chunk",
            "status": "succeeded",
            "detail": ", ".join(f"{k}: {v}" for k, v in counts.items()),
        },
    )
    return {"chunks": counts, "blocks": len(blocks)}


# ---------------------------------------------------------------- contextualize


def _template_prefix(document: dict[str, Any], chunk: dict[str, Any]) -> str:
    """Deterministic locating preamble. No model, no key, no network."""
    parts: list[str] = []
    if document.get("company_name"):
        parts.append(str(document["company_name"]))
    if document.get("form_type"):
        year = document.get("fiscal_year")
        parts.append(f"{document['form_type']}{f' {year}' if year else ''}")
    if not parts:
        parts.append(str(document.get("original_name") or "document"))

    where = " > ".join(chunk.get("section_path") or [])
    location = f", section {where}" if where else ""
    if chunk.get("page_from"):
        location += f", page {chunk['page_from']}"
    return f"From {' '.join(parts)}{location}."


@handler("contextualize")
async def contextualize_stage(message: StageMessage) -> dict[str, Any]:
    """Write each chunk's one-line locating preamble.

    Embedded with the chunk, shown without it. Chunks in filings are frequently
    ambiguous on their own — "revenue increased 12%" says nothing about whose
    revenue, or which year — and the prefix is what makes them retrievable.
    """
    settings = get_settings()
    strategy = settings.chunk_strategies[0]

    async with acquire() as conn:
        document = await conn.fetchrow(
            "SELECT original_name, company_name, form_type, fiscal_year"
            "  FROM documents WHERE id = $1::uuid",
            message.document_id,
        )
        chunks = await load_chunks(conn, message.document_id, strategy)
        if not chunks:
            raise PermanentError("no chunks to contextualise")

        doc = dict(document) if document else {}
        updates = [(_template_prefix(doc, chunk), chunk["id"]) for chunk in chunks]
        await conn.executemany("UPDATE chunks SET context_prefix = $1 WHERE id = $2", updates)

    await publish(
        message.document_id,
        {
            "type": "stage",
            "stage": "contextualize",
            "status": "succeeded",
            "detail": f"{len(updates)} chunks (template)",
        },
    )
    # `llm` stays False until the model-written variant lands; the eval harness
    # will compare the two, so recording which one produced the index matters.
    return {"chunks": len(updates), "llm": False}


# ---------------------------------------------------------------- embed


@handler("embed")
async def embed_stage(message: StageMessage) -> dict[str, Any]:
    settings = get_settings()
    embedder = get_embedder()
    total = 0

    async with acquire() as conn:
        document = await conn.fetchrow(
            "SELECT tenant_id, cik, ticker, fiscal_year, form_type, original_name"
            "  FROM documents WHERE id = $1::uuid",
            message.document_id,
        )
        doc = dict(document) if document else {}

        for strategy in settings.chunk_strategies:
            chunks = await load_chunks(conn, message.document_id, strategy)
            if not chunks:
                continue

            await vectors.ensure_collection(strategy, embedder.dims)
            # Clear first so a re-ingest cannot leave points behind for chunks
            # that no longer exist.
            await vectors.delete_document(strategy, message.document_id)

            texts = [
                Chunk(
                    seq=c["seq"],
                    text=c["text"],
                    block_ids=[],
                    token_count=c["token_count"],
                    context_prefix=c["context_prefix"],
                ).embedding_text
                for c in chunks
            ]
            embeddings = await embedder.embed_documents(texts)

            payloads = [
                {
                    "tenant_id": doc.get("tenant_id", "default"),
                    "document_id": message.document_id,
                    "document_name": doc.get("original_name"),
                    "strategy": strategy,
                    "seq": c["seq"],
                    "cik": doc.get("cik"),
                    "ticker": doc.get("ticker"),
                    "fiscal_year": doc.get("fiscal_year"),
                    "form_type": doc.get("form_type"),
                    "page_from": c["page_from"],
                    "section_path": list(c["section_path"] or []),
                }
                for c in chunks
            ]
            await vectors.upsert_chunks(
                strategy, [str(c["id"]) for c in chunks], embeddings, payloads
            )
            total += len(chunks)

        await set_document_status(conn, message.document_id, "ready")

    await publish(
        message.document_id,
        {
            "type": "stage",
            "stage": "embed",
            "status": "succeeded",
            "detail": f"{total} vectors ({embedder.model})",
        },
    )
    await publish(message.document_id, {"type": "ready"})
    return {"vectors": total, "model": embedder.model, "dims": embedder.dims}
