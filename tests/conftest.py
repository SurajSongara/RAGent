"""Shared fixtures. Builders here keep the tests about behaviour, not construction."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import pytest

from ragent.ingest.bbox import BBox
from ragent.ingest.types import SourceBlock

_counter = itertools.count()


def make_block(
    text: str,
    *,
    kind: str = "paragraph",
    page_no: int = 1,
    reading_order: int | None = None,
    section_path: tuple[str, ...] = ("Part I", "Item 1"),
    bbox: BBox | None = None,
) -> SourceBlock:
    """Build a SourceBlock with sane defaults and a unique id."""
    n = next(_counter)
    return SourceBlock(
        id=f"blk-{n}",
        page_no=page_no,
        reading_order=n if reading_order is None else reading_order,
        kind=kind,
        text=text,
        bbox=bbox or BBox(0.1, 0.1, 0.9, 0.2),
        section_path=section_path,
    )


@pytest.fixture
def make_block_factory():
    return make_block


def fake_embedder(topic_of: dict[str, str] | None = None):
    """Deterministic 2-D embedder driven by a keyword in each sentence.

    Sentences containing "revenue" map to one axis and "litigation" to the other,
    so intra-topic cosine is exactly 1.0 and cross-topic exactly 0.0. That gives
    the semantic chunker an unambiguous topic shift to find, without pulling a
    real embedding model into a unit test.
    """

    def embed(sentences: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for sentence in sentences:
            lowered = sentence.lower()
            if "litigation" in lowered:
                out.append([0.0, 1.0])
            else:
                out.append([1.0, 0.0])
        return out

    return embed


@pytest.fixture
def embedder():
    return fake_embedder()


@pytest.fixture
def filing_blocks() -> list[SourceBlock]:
    """A small document shaped like a real filing: headings, prose, a table."""
    md_a = ("Part II", "Item 7")
    legal = ("Part I", "Item 3")
    return [
        make_block("ANNUAL REPORT", kind="header", page_no=1, section_path=()),
        make_block("Item 7. Management Discussion", kind="heading", page_no=1, section_path=md_a),
        make_block(
            "Total revenue increased twelve percent year over year, driven by services. "
            "Revenue growth was broad based across every geography we operate in.",
            page_no=1,
            section_path=md_a,
        ),
        make_block(
            "Revenue from the Americas segment grew to a record level this year.",
            page_no=1,
            section_path=md_a,
        ),
        make_block(
            "Revenue | FY24 | FY25\nServices | 1,234 | 1,500",
            kind="table",
            page_no=2,
            section_path=md_a,
        ),
        make_block("Item 3. Legal Proceedings", kind="heading", page_no=2, section_path=legal),
        make_block(
            "The litigation described below remains pending before the district court. "
            "We believe the litigation claims are without merit and intend to defend them.",
            page_no=2,
            section_path=legal,
        ),
        make_block("12", kind="page_number", page_no=2, section_path=()),
    ]
