"""FastAPI application: upload, monitor, search, ask.

Two SSE endpoints rather than polling. `/documents/{id}/events` relays worker
progress from Valkey so the pipeline view updates the instant a stage finishes,
and `/ask` streams tokens so a multi-passage answer does not read as a hang.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import aio_pika
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ragent.agent.answer import answer_stream
from ragent.api.openai_compat import router as openai_router
from ragent.config import get_settings
from ragent.db.pool import acquire, close_pool
from ragent.db.repo import create_document, ensure_run, get_document, list_documents
from ragent.pipeline import progress
from ragent.pipeline.messages import StageMessage
from ragent.pipeline.stages import PIPELINE, STAGES_BY_NAME, stages_for
from ragent.pipeline.topology import INGEST_EXCHANGE, declare
from ragent.retrieval.search import hybrid_search
from ragent.storage import object_uri, presign, put_object, sha256_of

log = logging.getLogger(__name__)
settings = get_settings()

#: Uploads are held in memory to hash them, so this bounds worker RSS.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


class _Broker:
    """Lazily-opened publisher connection, shared by the whole app."""

    def __init__(self) -> None:
        self._connection: Any = None
        self._channel: Any = None
        self._lock = asyncio.Lock()

    async def exchange(self) -> Any:
        async with self._lock:
            if self._channel is None or self._channel.is_closed:
                self._connection = await aio_pika.connect_robust(settings.rabbitmq_url)
                self._channel = await self._connection.channel(publisher_confirms=True)
                # Declaring here means the API can seed work even if no worker
                # has started yet — the queues exist and messages wait.
                await declare(self._channel)
            return await self._channel.get_exchange(INGEST_EXCHANGE)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()


broker = _Broker()


async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await broker.close()
    await close_pool()


app = FastAPI(
    title="RAGent",
    version="0.1.0",
    description="Document intelligence with bbox-grounded citations.",
    lifespan=lifespan,
)

# Point any OpenAI client at /v1 and RAGent answers as if it were a model.
app.include_router(openai_router)

app.add_middleware(
    CORSMiddleware,
    # Local dev only. A hosted deployment must pin this to the real origin.
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- models


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] | None = None
    limit: int | None = Field(default=None, ge=1, le=50)
    strategy: str | None = None


class AskRequest(SearchRequest):
    pass


# ---------------------------------------------------------------- health


@app.get("/health")
async def health() -> dict[str, Any]:
    checks: dict[str, str] = {}
    try:
        async with acquire() as conn:
            await conn.fetchval("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["postgres"] = f"error: {exc}"

    from ragent.providers import llm
    from ragent.providers.embeddings import get_embedder

    try:
        embedder = get_embedder()
        embedding = {"backend": embedder.provider, "model": embedder.model}
    except Exception as exc:  # noqa: BLE001
        embedding = {"backend": settings.embedding_backend, "error": str(exc)}

    return {
        "status": "ok" if all(v == "ok" for v in checks.values()) else "degraded",
        "checks": checks,
        "llm_configured": llm.available(),
        "generation": llm.describe(),
        "embedding": embedding,
        # Kept for the existing UI badge.
        "embedding_backend": settings.embedding_backend,
    }


@app.get("/pipeline")
async def pipeline() -> dict[str, Any]:
    """The DAG, as data. The UI renders its stage list from this."""
    return {
        "stages": [
            {
                "name": s.name,
                "pool": str(s.pool),
                "families": sorted(str(f) for f in s.families),
                "depends_on": sorted(s.depends_on),
                "description": s.description,
            }
            for s in PIPELINE
        ]
    }


# ---------------------------------------------------------------- documents


@app.post("/documents", status_code=201)
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB")

    digest = sha256_of(data)
    name = file.filename or "upload"
    key = f"raw/{digest}{('.' + name.rsplit('.', 1)[-1]) if '.' in name else ''}"

    await put_object(key, data, file.content_type or "application/octet-stream")

    async with acquire() as conn:
        document_id, created = await create_document(
            conn,
            sha256=digest,
            source_uri=object_uri(key),
            original_name=name,
            mime_type=file.content_type or "application/octet-stream",
            byte_size=len(data),
        )
        if not created:
            # Content-hash dedupe: the same bytes are the same document.
            return {"document_id": document_id, "status": "duplicate"}

        run_id = await ensure_run(conn, document_id, settings.pipeline_version)

    # `detect` is the only stage with no dependencies, so it is always the entry
    # point regardless of what the file turns out to be.
    message = StageMessage(
        document_id=document_id,
        run_id=run_id,
        stage="detect",
        # Provisional: the detect stage overwrites this once it reads the bytes.
        family="pdf",  # type: ignore[arg-type]
        pipeline_version=settings.pipeline_version,
    )
    exchange = await broker.exchange()
    await exchange.publish(
        aio_pika.Message(
            body=message.to_bytes(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=STAGES_BY_NAME["detect"].routing_key,
    )

    return {"document_id": document_id, "status": "queued", "name": name}


@app.get("/documents")
async def documents() -> dict[str, Any]:
    async with acquire() as conn:
        rows = await list_documents(conn)
    return {"documents": [_serialise_document(r) for r in rows]}


@app.get("/documents/{document_id}")
async def document_detail(document_id: str) -> dict[str, Any]:
    _validate_uuid(document_id)
    async with acquire() as conn:
        row = await get_document(conn, document_id)
        if row is None:
            raise HTTPException(404, "document not found")
        stages = await conn.fetch(
            """
            SELECT s.stage, s.status, s.attempt, s.error, s.metrics,
                   s.started_at, s.finished_at
              FROM ingest_stages s
              JOIN ingest_runs r ON r.id = s.run_id
             WHERE r.document_id = $1::uuid
            """,
            document_id,
        )

    family = row.get("format_family")
    expected = [s.name for s in stages_for(family)] if family else [s.name for s in PIPELINE]
    by_name = {r["stage"]: dict(r) for r in stages}

    return {
        **_serialise_document(row),
        "stages": [
            {
                "name": name,
                "status": (by_name.get(name) or {}).get("status", "pending"),
                "error": (by_name.get(name) or {}).get("error"),
                "metrics": (by_name.get(name) or {}).get("metrics") or {},
            }
            for name in expected
        ],
    }


@app.get("/documents/{document_id}/text")
async def document_text(document_id: str) -> dict[str, Any]:
    """Source text for the flow-provenance viewer to highlight into."""
    _validate_uuid(document_id)
    async with acquire() as conn:
        row = await get_document(conn, document_id)
        if row is None:
            raise HTTPException(404, "document not found")
        if row["provenance"] != "flow":
            raise HTTPException(400, "document is paged; request its pages instead")

    from ragent.storage import get_object

    data = await get_object(row["source_uri"])
    return {"document_id": document_id, "text": data.decode("utf-8", errors="replace")}


@app.get("/documents/{document_id}/file")
async def document_file(document_id: str) -> dict[str, Any]:
    """Presigned URL for the document itself (the converted PDF when there is one).

    The viewer renders PDFs client-side with pdf.js and draws the citation
    overlay on top. Doing it in the browser means bbox-grounded citations work
    without a server-side rasteriser, and the page image never round-trips.
    """
    _validate_uuid(document_id)
    async with acquire() as conn:
        row = await conn.fetchrow(
            "SELECT source_uri, converted_uri, mime_type FROM documents WHERE id = $1::uuid",
            document_id,
        )
    if row is None:
        raise HTTPException(404, "document not found")

    uri = row["converted_uri"] or row["source_uri"]
    return {
        "url": await presign(uri),
        "mime_type": "application/pdf" if row["converted_uri"] else row["mime_type"],
    }


@app.get("/documents/{document_id}/pages/{page_no}")
async def document_page(document_id: str, page_no: int) -> dict[str, Any]:
    """Presigned URL for a page render, for the bbox viewer."""
    _validate_uuid(document_id)
    async with acquire() as conn:
        row = await conn.fetchrow(
            "SELECT render_uri, width_pt, height_pt, rotation FROM pages"
            " WHERE document_id = $1::uuid AND page_no = $2",
            document_id,
            page_no,
        )
    if row is None:
        raise HTTPException(404, "page not found")
    if not row["render_uri"]:
        # Page rasterising ships with the OCR stage; until then the viewer
        # falls back to rendering the source PDF client-side.
        raise HTTPException(409, "page has not been rendered yet")

    return {
        "url": await presign(row["render_uri"]),
        "width_pt": row["width_pt"],
        "height_pt": row["height_pt"],
        "rotation": row["rotation"],
    }


@app.get("/documents/{document_id}/events")
async def document_events(document_id: str) -> EventSourceResponse:
    """Relay worker progress to the browser."""
    _validate_uuid(document_id)

    async def stream() -> AsyncIterator[dict[str, str]]:
        # Send current state first so a client connecting late is not stuck on
        # an empty pipeline view until the next stage happens to finish.
        with contextlib.suppress(Exception):
            snapshot = await document_detail(document_id)
            yield {"event": "snapshot", "data": json.dumps(snapshot)}

        async for event in progress.subscribe(document_id):
            yield {"event": "progress", "data": json.dumps(event)}

    return EventSourceResponse(stream())


# ---------------------------------------------------------------- query


@app.post("/search")
async def search(request: SearchRequest) -> dict[str, Any]:
    passages = await hybrid_search(
        request.query,
        strategy=request.strategy,
        limit=request.limit,
        document_ids=request.document_ids,
    )
    return {"query": request.query, "passages": [p.to_dict() for p in passages]}


@app.post("/ask")
async def ask(request: AskRequest) -> EventSourceResponse:
    passages = await hybrid_search(
        request.query,
        strategy=request.strategy,
        limit=request.limit,
        document_ids=request.document_ids,
    )

    async def stream() -> AsyncIterator[dict[str, str]]:
        # Passages first: the UI shows retrieved evidence while the answer is
        # still being written, which makes the wait informative.
        yield {
            "event": "passages",
            "data": json.dumps([p.to_dict() for p in passages]),
        }
        async for chunk in answer_stream(request.query, passages):
            yield {
                "event": chunk.type,
                "data": json.dumps(
                    {"text": chunk.text} if chunk.type == "delta" else (chunk.data or {})
                ),
            }

    return EventSourceResponse(stream())


# ---------------------------------------------------------------- helpers


def _validate_uuid(value: str) -> None:
    try:
        uuid.UUID(value)
    except ValueError:
        raise HTTPException(400, "malformed document id") from None


def _serialise_document(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "name": row["original_name"],
        "status": row["status"],
        "mime_type": row["mime_type"],
        "byte_size": row["byte_size"],
        "page_count": row["page_count"],
        "format_family": row.get("format_family"),
        "provenance": row.get("provenance"),
        "chunk_count": row.get("chunk_count"),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
    }


@app.exception_handler(Exception)
async def unhandled(_: Any, exc: Exception) -> JSONResponse:
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
