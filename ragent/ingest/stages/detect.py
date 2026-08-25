"""The `detect` stage: identify the upload and decide its route.

Runs before anything expensive. Its only job is to look at the bytes, decide
what they are, and record that on the document — the format family it writes is
what the scheduler then uses to pick every subsequent stage.

Detection failures are permanent by construction. Retrying cannot change what
the bytes are, so unsupported content is quarantined immediately instead of
burning three attempts and two backoff tiers to reach the same conclusion.
"""

from __future__ import annotations

import logging
from typing import Any

from ragent.db.pool import acquire
from ragent.db.repo import set_document_format, set_document_status
from ragent.ingest.formats import FormatFamily, UnsupportedFormatError, detect_format
from ragent.pipeline.handlers import handler
from ragent.pipeline.messages import StageMessage
from ragent.pipeline.progress import publish
from ragent.storage import get_object

log = logging.getLogger(__name__)

#: Enough bytes for every signature we check, including the ZIP central
#: directory an OOXML file needs. Reading whole documents to sniff a header
#: would pull 200MB filings through memory for no reason.
_SNIFF_BYTES = 1 << 20


@handler("detect")
async def detect_stage(message: StageMessage) -> dict[str, Any]:
    async with acquire() as conn:
        row = await conn.fetchrow(
            "SELECT source_uri, original_name FROM documents WHERE id = $1::uuid",
            message.document_id,
        )
        if row is None:
            raise UnsupportedFormatError(f"document {message.document_id} no longer exists")

        data = await get_object(row["source_uri"])

        try:
            fmt = detect_format(data[:_SNIFF_BYTES], row["original_name"])
        except UnsupportedFormatError:
            # Quarantined rather than failed: the upload is intact, we simply
            # have no route for it. The distinction matters to whoever reviews
            # the DLQ.
            await set_document_status(conn, message.document_id, "quarantined")
            await publish(
                message.document_id,
                {"type": "quarantined", "reason": "unsupported format"},
            )
            raise

        await set_document_format(conn, message.document_id, fmt)

    await publish(
        message.document_id,
        {
            "type": "stage",
            "stage": "detect",
            "status": "succeeded",
            "detail": fmt.label,
            "family": str(fmt.family),
        },
    )
    log.info("detected %s as %s", message.document_id, fmt.label)

    return {
        "family": str(fmt.family),
        "mime": fmt.mime,
        "provenance": str(fmt.provenance),
        "bytes": len(data),
    }


def family_of(value: str) -> FormatFamily:
    return FormatFamily(value)
