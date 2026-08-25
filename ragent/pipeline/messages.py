"""The envelope that moves between stages.

Kept deliberately small: a message names the document and the stage, never the
document's content. Anything a stage needs it reads from Postgres or MinIO, so
a message that sits in a queue across a deploy cannot go stale, and the broker
never becomes a second copy of the data.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from ragent.ingest.formats import FormatFamily

__all__ = ["StageMessage", "MalformedMessageError"]


class MalformedMessageError(ValueError):
    """Message that cannot be decoded. Dead-lettered immediately; retries cannot fix it."""


@dataclass(frozen=True, slots=True)
class StageMessage:
    document_id: str
    run_id: str
    stage: str
    family: FormatFamily
    pipeline_version: str
    tenant_id: str = "default"
    #: 1-based count of attempts already made, including the current one.
    attempt: int = 1
    #: Correlates every stage of one document into a single OTel trace.
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    #: Small stage-specific hints. Not a payload channel — no document content.
    meta: dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        return json.dumps(
            {
                "document_id": self.document_id,
                "run_id": self.run_id,
                "stage": self.stage,
                "family": str(self.family),
                "pipeline_version": self.pipeline_version,
                "tenant_id": self.tenant_id,
                "attempt": self.attempt,
                "trace_id": self.trace_id,
                "meta": self.meta,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, raw: bytes) -> StageMessage:
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MalformedMessageError(f"undecodable message: {exc}") from exc

        if not isinstance(data, dict):
            raise MalformedMessageError(f"expected an object, got {type(data).__name__}")

        missing = {"document_id", "run_id", "stage", "family", "pipeline_version"} - set(data)
        if missing:
            raise MalformedMessageError(f"missing fields {sorted(missing)}")

        try:
            family = FormatFamily(data["family"])
        except ValueError as exc:
            raise MalformedMessageError(f"unknown family {data['family']!r}") from exc

        attempt = data.get("attempt", 1)
        if not isinstance(attempt, int) or attempt < 1:
            raise MalformedMessageError(f"attempt must be a positive int, got {attempt!r}")

        meta = data.get("meta") or {}
        if not isinstance(meta, dict):
            raise MalformedMessageError("meta must be an object")

        return cls(
            document_id=str(data["document_id"]),
            run_id=str(data["run_id"]),
            stage=str(data["stage"]),
            family=family,
            pipeline_version=str(data["pipeline_version"]),
            tenant_id=str(data.get("tenant_id", "default")),
            attempt=attempt,
            trace_id=str(data.get("trace_id") or uuid.uuid4().hex),
            meta=meta,
        )

    def with_family(self, family: FormatFamily) -> StageMessage:
        """Correct the family once `detect` has actually read the bytes.

        The upload endpoint cannot know the format before detection runs, so it
        publishes a provisional one. Everything downstream routes on this field,
        so it has to be replaced with the real value before the first handoff —
        otherwise a Markdown file follows the PDF path and fails in the parser.
        """
        return replace(self, family=family)

    def for_stage(self, stage: str) -> StageMessage:
        """Hand off to the next stage, resetting the attempt counter.

        The trace id survives so one document's whole run stays a single trace.
        """
        return replace(self, stage=stage, attempt=1)

    def retried(self) -> StageMessage:
        return replace(self, attempt=self.attempt + 1)
