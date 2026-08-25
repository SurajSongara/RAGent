"""The `convert`, `ocr`, `tables` and `figures` stages.

Honest status: `convert` is fully implemented. The other three are wired,
routed, and record real metrics, but their extraction work is not built yet —
each is a pass-through that reports what it *would* have processed. They are
here rather than absent because a stage missing from the graph stalls every
document behind it, and because the metrics they emit already answer the
question that decides whether building them is worth it (how many pages in a
real corpus actually need OCR).

Each one is marked NOT-IMPLEMENTED below with what it takes to finish.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ragent.db.pool import acquire
from ragent.pipeline.handlers import handler
from ragent.pipeline.messages import StageMessage
from ragent.pipeline.progress import publish
from ragent.pipeline.runner import PermanentError
from ragent.storage import get_object, put_object

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- convert


@handler("convert")
async def convert_stage(message: StageMessage) -> dict[str, Any]:
    """Render Office/HTML to PDF with headless LibreOffice.

    Converting rather than parsing each format natively is a deliberate trade:
    one conversion dependency buys bounding-box provenance for nine formats, and
    keeps a single citation viewer instead of one per format.
    """
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice is None:
        raise PermanentError(
            "LibreOffice is not installed in this image; the convert stage runs "
            "in the `convert` build target (see infra/Dockerfile.python)"
        )

    async with acquire() as conn:
        row = await conn.fetchrow(
            "SELECT source_uri, original_name FROM documents WHERE id = $1::uuid",
            message.document_id,
        )
    if row is None:
        raise PermanentError(f"document {message.document_id} no longer exists")

    data = await get_object(row["source_uri"])

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        source = workdir / (row["original_name"] or "input")
        # Off the event loop: a 50MB write would otherwise stall every other
        # coroutine on this worker, including the heartbeat to the broker.
        await asyncio.to_thread(source.write_bytes, data)

        process = await asyncio.create_subprocess_exec(
            soffice,
            "--headless",
            "--norestore",
            "--convert-to",
            "pdf",
            "--outdir",
            str(workdir),
            str(source),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            # LibreOffice hangs rather than failing when it dislikes a document,
            # and a hung conversion would hold its prefetch slot indefinitely.
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=180)
        except TimeoutError:
            process.kill()
            raise PermanentError("LibreOffice timed out after 180s") from None

        if process.returncode != 0:
            raise RuntimeError(f"LibreOffice exited {process.returncode}: {stderr.decode()[:400]}")

        produced = await asyncio.to_thread(lambda: next(iter(workdir.glob("*.pdf")), None))
        if produced is None:
            raise PermanentError("LibreOffice produced no PDF")
        pdf_bytes = await asyncio.to_thread(produced.read_bytes)

    uri = await put_object(f"converted/{message.document_id}.pdf", pdf_bytes, "application/pdf")
    async with acquire() as conn:
        await conn.execute(
            "UPDATE documents SET converted_uri = $2 WHERE id = $1::uuid",
            message.document_id,
            uri,
        )

    await publish(
        message.document_id,
        {
            "type": "stage",
            "stage": "convert",
            "status": "succeeded",
            "detail": f"{len(pdf_bytes) // 1024} KB PDF",
        },
    )
    return {"pdf_bytes": len(pdf_bytes)}


# ---------------------------------------------------------------- ocr


@handler("ocr")
async def ocr_stage(message: StageMessage) -> dict[str, Any]:
    """NOT IMPLEMENTED: the gate runs, the recognition does not.

    `parse` already scored every page's text layer and set `pages.needs_ocr`, so
    the selective part — the decision that saves minutes per document — is live
    and its metrics are real. What remains is rasterising the flagged regions
    and running RapidOCR over them, then inserting the resulting blocks with
    origin='ocr'.
    """
    async with acquire() as conn:
        flagged = await conn.fetchval(
            "SELECT count(*) FROM pages WHERE document_id = $1::uuid AND needs_ocr",
            message.document_id,
        )
        total = await conn.fetchval(
            "SELECT count(*) FROM pages WHERE document_id = $1::uuid", message.document_id
        )

    if flagged:
        log.warning(
            "document %s has %d/%d pages needing OCR; recognition not yet implemented",
            message.document_id,
            flagged,
            total,
        )

    await publish(
        message.document_id,
        {
            "type": "stage",
            "stage": "ocr",
            "status": "succeeded",
            "detail": f"{flagged}/{total} pages flagged (recognition pending)",
        },
    )
    return {"pages_flagged": flagged or 0, "pages_total": total or 0, "implemented": False}


# ---------------------------------------------------------------- tables


@handler("tables")
async def tables_stage(message: StageMessage) -> dict[str, Any]:
    """NOT IMPLEMENTED: needs table structure recognition.

    The destination schema (`doc_tables` / `table_cells`) and the cell parser
    (`ragent.ingest.numeric`, fully tested) are both ready. What is missing is
    the detector that turns a table region into a grid of cells.
    """
    await publish(
        message.document_id,
        {"type": "stage", "stage": "tables", "status": "succeeded", "detail": "pending"},
    )
    return {"tables": 0, "implemented": False}


# ---------------------------------------------------------------- figures


@handler("figures")
async def figures_stage(message: StageMessage) -> dict[str, Any]:
    """NOT IMPLEMENTED: needs page rendering plus a vision model call.

    Blocked on rasterising page regions, which is the same prerequisite as OCR.
    """
    await publish(
        message.document_id,
        {"type": "stage", "stage": "figures", "status": "succeeded", "detail": "pending"},
    )
    return {"figures": 0, "implemented": False}
