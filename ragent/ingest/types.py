"""Shared value objects for the ingest pipeline.

These are the in-flight equivalents of the `blocks` and `chunks` tables. Stages
pass these between each other; only the final stage writes rows.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ragent.ingest.bbox import BBox

__all__ = ["SourceBlock", "Chunk", "TokenCounter", "approx_tokens"]

TokenCounter = Callable[[str], int]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def approx_tokens(text: str) -> int:
    """Dependency-free token estimate.

    The real pipeline injects a tiktoken counter. This default keeps chunking
    unit-testable without loading a model, and keeps the chunkers deterministic
    so the Phase 2 bake-off compares strategies rather than tokeniser noise.
    """
    if not text.strip():
        return 0
    return max(1, round(len(_TOKEN_RE.findall(text)) * 1.3))


@dataclass(frozen=True, slots=True)
class SourceBlock:
    """One block of a document, carrying whatever makes it citable.

    Paged sources (PDF, images, converted Office files) fill `page_no` and
    `bbox`. Flow sources (markdown, csv, plain text) have no geometry, so they
    fill `char_start` / `char_end` instead. Exactly one of the two is required —
    a block that can locate itself neither way is unciteable, and the database
    refuses it via the `block_is_locatable` constraint.
    """

    id: str
    page_no: int
    reading_order: int
    kind: str
    text: str
    bbox: BBox | None = None
    char_start: int | None = None
    char_end: int | None = None
    section_path: tuple[str, ...] = ()
    confidence: float | None = None
    origin: str = "native"

    def __post_init__(self) -> None:
        if self.bbox is None and self.char_start is None:
            raise ValueError(f"block {self.id!r} has neither a bbox nor a char range")

    @property
    def is_paged(self) -> bool:
        return self.bbox is not None

    @property
    def is_atomic(self) -> bool:
        """Blocks that lose their meaning if a chunk boundary cuts through them.

        A table split down the middle produces two chunks that are each worse
        than useless: headers without rows, rows without headers.
        """
        return self.kind in ("table", "figure")

    @property
    def is_boundary(self) -> bool:
        """Blocks that should start a new chunk rather than continue one."""
        return self.kind in ("title", "heading")

    @property
    def is_furniture(self) -> bool:
        """Repeated page furniture, dropped before chunking.

        Running headers and page numbers otherwise appear in every chunk and
        dominate lexical retrieval with noise.
        """
        return self.kind in ("header", "footer", "page_number")


@dataclass(slots=True)
class Chunk:
    """A retrievable unit, still carrying the blocks it was built from."""

    seq: int
    text: str
    block_ids: list[str]
    token_count: int
    #: Paged provenance. None for flow documents.
    page_from: int | None = None
    page_to: int | None = None
    #: Flow provenance. None for paged documents.
    char_start: int | None = None
    char_end: int | None = None
    section_path: tuple[str, ...] = ()
    context_prefix: str | None = None
    strategy: str = "layout"
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        """What gets embedded: the context prefix is included here and only here.

        The user is shown `text`; the index sees the prefixed form. Keeping the
        two apart is what makes contextual retrieval invisible in the UI.
        """
        if self.context_prefix:
            return f"{self.context_prefix}\n\n{self.text}"
        return self.text
