"""The `parse` and `parse_flow` stages: source bytes become citable blocks.

Two handlers because the two provenance modes genuinely differ. `parse` works on
rendered pages and produces bounding boxes; `parse_flow` works on a character
stream and produces offsets. Everything downstream — chunking, embedding,
citation resolution — treats their output identically.
"""

from __future__ import annotations

import logging
from typing import Any

from ragent.db.pool import acquire
from ragent.db.repo import insert_blocks, insert_pages, set_document_status
from ragent.ingest.bbox import to_normalised
from ragent.ingest.classify import assess_text_layer
from ragent.ingest.flowtext import split_flow
from ragent.ingest.formats import detect_format
from ragent.ingest.types import SourceBlock
from ragent.pipeline.handlers import handler
from ragent.pipeline.messages import StageMessage
from ragent.pipeline.progress import publish
from ragent.pipeline.runner import PermanentError
from ragent.storage import get_object

log = logging.getLogger(__name__)


async def _load_source(document_id: str) -> tuple[bytes, str, str]:
    async with acquire() as conn:
        row = await conn.fetchrow(
            "SELECT source_uri, converted_uri, original_name FROM documents WHERE id = $1::uuid",
            document_id,
        )
    if row is None:
        raise PermanentError(f"document {document_id} no longer exists")
    # Office and HTML parse from the PDF the convert stage produced, never from
    # the original bytes.
    uri = row["converted_uri"] or row["source_uri"]
    return await get_object(uri), row["original_name"], uri


# ---------------------------------------------------------------- flow


@handler("parse_flow")
async def parse_flow_stage(message: StageMessage) -> dict[str, Any]:
    data, name, _ = await _load_source(message.document_id)

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        # detect_format already established this decodes; if it does not now,
        # the object changed underneath us and retrying will not help.
        raise PermanentError("flow document is not valid UTF-8") from None

    extension = detect_format(data[: 1 << 20], name).extension
    flow_blocks = split_flow(text, extension)
    if not flow_blocks:
        raise PermanentError("document contains no extractable text")

    blocks = [
        SourceBlock(
            id=f"tmp-{i}",
            page_no=0,
            reading_order=i,
            kind=block.kind,
            text=block.text,
            char_start=block.char_start,
            char_end=block.char_end,
            section_path=block.section_path,
            origin="native",
        )
        for i, block in enumerate(flow_blocks)
    ]

    async with acquire() as conn, conn.transaction():
        await conn.execute("DELETE FROM blocks WHERE document_id = $1::uuid", message.document_id)
        await insert_blocks(conn, message.document_id, blocks)

    await publish(
        message.document_id,
        {
            "type": "stage",
            "stage": "parse_flow",
            "status": "succeeded",
            "detail": f"{len(blocks)} blocks",
        },
    )
    return {"blocks": len(blocks), "chars": len(text), "extension": extension}


# ---------------------------------------------------------------- paged


def _extract_pdf(data: bytes) -> tuple[list[dict], list[SourceBlock]]:
    """Native text extraction with real coordinates, via pypdfium2.

    Deliberately not Docling yet. This gets a genuine bbox for every text
    segment on every page, which is all the citation viewer needs; full layout
    analysis (reading order across columns, table structure, figure regions) is
    the next increment and slots in behind the same interface.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    pages: list[dict] = []
    blocks: list[SourceBlock] = []
    order = 0

    for index in range(len(pdf)):
        page = pdf[index]
        width, height = page.get_width(), page.get_height()
        rotation = int(page.get_rotation() or 0)
        textpage = page.get_textpage()
        page_text = textpage.get_text_bounded()

        assessment = assess_text_layer(page_text, page_width_pt=width, page_height_pt=height)
        pages.append(
            {
                "page_no": index + 1,
                "width_pt": float(width),
                "height_pt": float(height),
                "rotation": rotation,
                "render_uri": None,
                "text_confidence": assessment.confidence,
                "needs_ocr": assessment.needs_ocr,
            }
        )

        # One block per detected rectangle of text. Pages flagged for OCR still
        # emit whatever they have, so a partially broken page degrades rather
        # than disappearing.
        for rect_index in range(textpage.count_rects()):
            left, bottom, right, top = textpage.get_rect(rect_index)
            fragment = textpage.get_text_bounded(left, bottom, right, top).strip()
            if not fragment:
                continue
            try:
                bbox = to_normalised(
                    left,
                    bottom,
                    right,
                    top,
                    page_width=width,
                    page_height=height,
                    rotation=rotation,
                )
            except ValueError:
                continue  # degenerate rect from the extractor; skip it
            order += 1
            blocks.append(
                SourceBlock(
                    id=f"tmp-{order}",
                    page_no=index + 1,
                    reading_order=order,
                    kind="paragraph",
                    text=fragment,
                    bbox=bbox,
                    confidence=assessment.confidence,
                    origin="native",
                )
            )

    pdf.close()
    return pages, blocks


@handler("parse")
async def parse_stage(message: StageMessage) -> dict[str, Any]:
    import asyncio

    data, _, _ = await _load_source(message.document_id)

    # pypdfium2 is synchronous and CPU-bound; a 200-page filing would otherwise
    # block every other consumer sharing this worker's event loop.
    pages, blocks = await asyncio.to_thread(_extract_pdf, data)

    if not blocks:
        # A scan with no text layer at all is not an error — it is exactly what
        # the OCR stage exists for.
        log.info("no native text in %s; leaving it to OCR", message.document_id)

    async with acquire() as conn, conn.transaction():
        await conn.execute("DELETE FROM blocks WHERE document_id = $1::uuid", message.document_id)
        page_ids = await insert_pages(conn, message.document_id, pages)
        await insert_blocks(conn, message.document_id, blocks, page_ids)
        await set_document_status(conn, message.document_id, "processing", page_count=len(pages))

    flagged = sum(1 for p in pages if p["needs_ocr"])
    await publish(
        message.document_id,
        {
            "type": "stage",
            "stage": "parse",
            "status": "succeeded",
            "detail": f"{len(pages)} pages, {len(blocks)} blocks, {flagged} need OCR",
        },
    )
    return {"pages": len(pages), "blocks": len(blocks), "pages_needing_ocr": flagged}
