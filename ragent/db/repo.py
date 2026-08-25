"""Data access for the ingest pipeline and the query path.

Every function takes an explicit connection so callers control transactions.
That matters most in `complete_stage`, where recording a completion and reading
back the run's progress has to be one atomic step.
"""

from __future__ import annotations

import uuid
from typing import Any

import asyncpg

from ragent.ingest.formats import DetectedFormat
from ragent.ingest.types import Chunk, SourceBlock

__all__ = [
    "create_document",
    "get_document",
    "list_documents",
    "set_document_format",
    "set_document_status",
    "ensure_run",
    "insert_pages",
    "insert_blocks",
    "load_blocks",
    "replace_chunks",
    "load_chunks",
    "chunk_citation_targets",
    "lexical_search",
]


# ---------------------------------------------------------------- documents


async def create_document(
    conn: asyncpg.Connection,
    *,
    sha256: str,
    source_uri: str,
    original_name: str,
    mime_type: str,
    byte_size: int,
    tenant_id: str = "default",
) -> tuple[str, bool]:
    """Insert, or return the existing document with the same content hash.

    Content-hash dedupe means re-uploading the same filing is a no-op rather
    than a second copy with a second set of embeddings.

    Returns (document_id, created).
    """
    row = await conn.fetchrow(
        """
        INSERT INTO documents (tenant_id, sha256, source_uri, original_name,
                               mime_type, byte_size, status)
        VALUES ($1, $2, $3, $4, $5, $6, 'pending')
        ON CONFLICT (tenant_id, sha256) DO NOTHING
        RETURNING id
        """,
        tenant_id,
        sha256,
        source_uri,
        original_name,
        mime_type,
        byte_size,
    )
    if row is not None:
        return str(row["id"]), True

    existing = await conn.fetchval(
        "SELECT id FROM documents WHERE tenant_id = $1 AND sha256 = $2", tenant_id, sha256
    )
    return str(existing), False


async def set_document_format(
    conn: asyncpg.Connection, document_id: str, fmt: DetectedFormat, page_count: int | None = None
) -> None:
    await conn.execute(
        """
        UPDATE documents
           SET format_family = $2::format_family,
               provenance    = $3::provenance_mode,
               mime_type     = $4,
               page_count    = COALESCE($5, page_count),
               status        = 'processing'
         WHERE id = $1
        """,
        uuid.UUID(document_id),
        str(fmt.family),
        str(fmt.provenance),
        fmt.mime,
        page_count,
    )


async def set_document_status(
    conn: asyncpg.Connection, document_id: str, status: str, *, page_count: int | None = None
) -> None:
    await conn.execute(
        "UPDATE documents SET status = $2::doc_status, page_count = COALESCE($3, page_count)"
        " WHERE id = $1",
        uuid.UUID(document_id),
        status,
        page_count,
    )


async def get_document(conn: asyncpg.Connection, document_id: str) -> dict[str, Any] | None:
    row = await conn.fetchrow("SELECT * FROM documents WHERE id = $1", uuid.UUID(document_id))
    return dict(row) if row else None


async def list_documents(
    conn: asyncpg.Connection, tenant_id: str = "default", limit: int = 100
) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT d.*,
               (SELECT count(*) FROM chunks c
                 WHERE c.document_id = d.id AND c.strategy = 'layout') AS chunk_count
          FROM documents d
         WHERE d.tenant_id = $1
         ORDER BY d.created_at DESC
         LIMIT $2
        """,
        tenant_id,
        limit,
    )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- runs


async def ensure_run(conn: asyncpg.Connection, document_id: str, pipeline_version: str) -> str:
    """One run per (document, pipeline version). Re-ingesting on a new version
    is an explicit, comparable run rather than a destructive overwrite."""
    row = await conn.fetchrow(
        """
        INSERT INTO ingest_runs (document_id, pipeline_version, status, started_at)
        VALUES ($1, $2, 'running', now())
        ON CONFLICT (document_id, pipeline_version)
            DO UPDATE SET status = 'running'
        RETURNING id
        """,
        uuid.UUID(document_id),
        pipeline_version,
    )
    return str(row["id"])


# ---------------------------------------------------------------- pages / blocks


async def insert_pages(
    conn: asyncpg.Connection, document_id: str, pages: list[dict]
) -> dict[int, str]:
    """Upsert pages, returning page_no -> page_id."""
    if not pages:
        return {}
    rows = await conn.fetch(
        """
        INSERT INTO pages (document_id, page_no, width_pt, height_pt, rotation,
                           render_uri, text_confidence, needs_ocr)
        SELECT $1, x.page_no, x.width_pt, x.height_pt, x.rotation,
               x.render_uri, x.text_confidence, x.needs_ocr
          FROM jsonb_to_recordset($2::jsonb) AS x(
               page_no int, width_pt real, height_pt real, rotation int,
               render_uri text, text_confidence real, needs_ocr boolean)
        ON CONFLICT (document_id, page_no) DO UPDATE
            SET width_pt = EXCLUDED.width_pt,
                height_pt = EXCLUDED.height_pt,
                rotation = EXCLUDED.rotation,
                text_confidence = EXCLUDED.text_confidence,
                needs_ocr = EXCLUDED.needs_ocr
        RETURNING page_no, id
        """,
        uuid.UUID(document_id),
        pages,
    )
    return {r["page_no"]: str(r["id"]) for r in rows}


async def insert_blocks(
    conn: asyncpg.Connection,
    document_id: str,
    blocks: list[SourceBlock],
    page_ids: dict[int, str] | None = None,
) -> dict[str, str]:
    """Insert blocks, returning the caller's temp id -> database id.

    Ids are generated client-side rather than read back from RETURNING. An
    INSERT..SELECT gives no ordering guarantee between input rows and returned
    ids, so correlating them by position would be relying on undefined
    behaviour — and a mis-correlated block id silently points a citation at the
    wrong region.

    `section_path` is a text[] column, which is why the payload goes over as
    JSON: `unnest` flattens a 2-D array into scalars and cannot feed an array
    column.

    Blocks carry either a bbox (paged) or a char range (flow); the
    `block_is_locatable` constraint refuses anything with neither.
    """
    if not blocks:
        return {}

    page_ids = page_ids or {}
    assigned = {block.id: str(uuid.uuid4()) for block in blocks}

    payload = []
    for block in blocks:
        bbox = block.bbox
        payload.append(
            {
                "id": assigned[block.id],
                "page_id": page_ids.get(block.page_no),
                "page_no": block.page_no if bbox is not None else None,
                "reading_order": block.reading_order,
                "kind": block.kind,
                "origin": block.origin,
                "confidence": block.confidence,
                "x0": bbox.x0 if bbox else None,
                "y0": bbox.y0 if bbox else None,
                "x1": bbox.x1 if bbox else None,
                "y1": bbox.y1 if bbox else None,
                "char_start": block.char_start,
                "char_end": block.char_end,
                "text": block.text,
                "section_path": list(block.section_path),
            }
        )

    await conn.execute(
        """
        INSERT INTO blocks (id, document_id, page_id, page_no, reading_order, kind,
                            origin, confidence, x0, y0, x1, y1, char_start, char_end,
                            text, section_path)
        SELECT x.id, $1, x.page_id, x.page_no, x.reading_order, x.kind::block_kind,
               x.origin::text_origin, x.confidence, x.x0, x.y0, x.x1, x.y1,
               x.char_start, x.char_end, x.text, x.section_path
          FROM jsonb_to_recordset($2::jsonb) AS x(
               id uuid, page_id uuid, page_no int, reading_order int, kind text,
               origin text, confidence real, x0 real, y0 real, x1 real, y1 real,
               char_start int, char_end int, text text, section_path text[])
        """,
        uuid.UUID(document_id),
        payload,
    )
    return assigned


async def load_blocks(conn: asyncpg.Connection, document_id: str) -> list[SourceBlock]:
    """Read blocks back as the in-flight type the chunkers expect."""
    from ragent.ingest.bbox import BBox

    rows = await conn.fetch(
        """
        SELECT id, page_no, reading_order, kind, origin, confidence,
               x0, y0, x1, y1, char_start, char_end, text, section_path
          FROM blocks
         WHERE document_id = $1
         ORDER BY COALESCE(page_no, 0), reading_order
        """,
        uuid.UUID(document_id),
    )
    out = []
    for r in rows:
        bbox = BBox(r["x0"], r["y0"], r["x1"], r["y1"]) if r["x0"] is not None else None
        out.append(
            SourceBlock(
                id=str(r["id"]),
                page_no=r["page_no"] or 0,
                reading_order=r["reading_order"],
                kind=r["kind"],
                text=r["text"],
                bbox=bbox,
                char_start=r["char_start"],
                char_end=r["char_end"],
                section_path=tuple(r["section_path"] or ()),
                confidence=r["confidence"],
                origin=r["origin"],
            )
        )
    return out


# ---------------------------------------------------------------- chunks


async def replace_chunks(
    conn: asyncpg.Connection, document_id: str, strategy: str, chunks: list[Chunk]
) -> list[str]:
    """Replace this document's chunks for one strategy.

    Scoped by strategy so the four chunkings coexist and the Phase 2 bake-off
    can re-run one of them without disturbing the others.

    Ids are client-generated for the same reason as blocks: `chunk_blocks` has
    to pair each chunk with the exact blocks it came from, and correlating
    RETURNING rows by position is undefined.
    """
    async with conn.transaction():
        await conn.execute(
            "DELETE FROM chunks WHERE document_id = $1 AND strategy = $2",
            uuid.UUID(document_id),
            strategy,
        )
        if not chunks:
            return []

        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        payload = [
            {
                "id": cid,
                "seq": c.seq,
                "text": c.text,
                "context_prefix": c.context_prefix,
                "token_count": c.token_count,
                "section_path": list(c.section_path),
                "page_from": c.page_from,
                "page_to": c.page_to,
                "char_start": c.char_start,
                "char_end": c.char_end,
            }
            for cid, c in zip(chunk_ids, chunks, strict=True)
        ]

        await conn.execute(
            """
            INSERT INTO chunks (id, document_id, strategy, seq, text, context_prefix,
                                token_count, section_path, page_from, page_to,
                                char_start, char_end)
            SELECT x.id, $1, $2, x.seq, x.text, x.context_prefix, x.token_count,
                   x.section_path, x.page_from, x.page_to, x.char_start, x.char_end
              FROM jsonb_to_recordset($3::jsonb) AS x(
                   id uuid, seq int, text text, context_prefix text, token_count int,
                   section_path text[], page_from int, page_to int,
                   char_start int, char_end int)
            """,
            uuid.UUID(document_id),
            strategy,
            payload,
        )

        # The join that keeps citations resolvable. Written in the same
        # transaction as the chunks so a chunk can never exist without it.
        links = [
            (uuid.UUID(cid), uuid.UUID(bid), ordinal)
            for cid, chunk in zip(chunk_ids, chunks, strict=True)
            for ordinal, bid in enumerate(chunk.block_ids)
        ]
        if links:
            await conn.executemany(
                "INSERT INTO chunk_blocks (chunk_id, block_id, ordinal)"
                " VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
                links,
            )
        return chunk_ids


async def load_chunks(
    conn: asyncpg.Connection, document_id: str, strategy: str
) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT id, seq, text, context_prefix, token_count, section_path,
               page_from, page_to, char_start, char_end
          FROM chunks
         WHERE document_id = $1 AND strategy = $2
         ORDER BY seq
        """,
        uuid.UUID(document_id),
        strategy,
    )
    return [dict(r) for r in rows]


async def chunk_citation_targets(
    conn: asyncpg.Connection, chunk_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Everything the viewer needs to draw a highlight, per chunk.

    Resolved here rather than in the UI so a citation never has to re-derive its
    own location. Paged chunks get the union of their blocks' bboxes per page;
    flow chunks get the character range.
    """
    if not chunk_ids:
        return {}

    rows = await conn.fetch(
        """
        SELECT c.id                AS chunk_id,
               c.text,
               c.section_path,
               c.char_start,
               c.char_end,
               d.id                AS document_id,
               d.original_name,
               d.provenance,
               b.page_no,
               min(b.x0) AS x0, min(b.y0) AS y0, max(b.x1) AS x1, max(b.y1) AS y1
          FROM chunks c
          JOIN documents d     ON d.id = c.document_id
          LEFT JOIN chunk_blocks cb ON cb.chunk_id = c.id
          LEFT JOIN blocks b        ON b.id = cb.block_id
         WHERE c.id = ANY($1::uuid[])
         GROUP BY c.id, c.text, c.section_path, c.char_start, c.char_end,
                  d.id, d.original_name, d.provenance, b.page_no
         ORDER BY b.page_no NULLS LAST
        """,
        [uuid.UUID(c) for c in chunk_ids],
    )

    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        cid = str(r["chunk_id"])
        entry = out.setdefault(
            cid,
            {
                "chunk_id": cid,
                "document_id": str(r["document_id"]),
                "document_name": r["original_name"],
                "provenance": r["provenance"],
                "text": r["text"],
                "section_path": list(r["section_path"] or ()),
                "char_start": r["char_start"],
                "char_end": r["char_end"],
                "regions": [],
            },
        )
        if r["page_no"] is not None and r["x0"] is not None:
            entry["regions"].append(
                {
                    "page_no": r["page_no"],
                    "x0": r["x0"],
                    "y0": r["y0"],
                    "x1": r["x1"],
                    "y1": r["y1"],
                }
            )
    return out


# ---------------------------------------------------------------- lexical search


async def lexical_search(
    conn: asyncpg.Connection,
    query: str,
    *,
    strategy: str = "layout",
    limit: int = 50,
    tenant_id: str = "default",
) -> list[tuple[str, float]]:
    """The sparse half of hybrid retrieval.

    `ts_rank_cd` is not true BM25; ParadeDB's pg_search is the drop-in upgrade
    if the eval ever shows lexical recall is the binding constraint.
    """
    rows = await conn.fetch(
        """
        SELECT c.id, ts_rank_cd(c.tsv, websearch_to_tsquery('english', $1)) AS score
          FROM chunks c
          JOIN documents d ON d.id = c.document_id
         WHERE d.tenant_id = $3
           AND c.strategy = $4
           AND c.tsv @@ websearch_to_tsquery('english', $1)
         ORDER BY score DESC
         LIMIT $2
        """,
        query,
        limit,
        tenant_id,
        strategy,
    )
    return [(str(r["id"]), float(r["score"])) for r in rows]
