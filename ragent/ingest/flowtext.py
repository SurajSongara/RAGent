"""Structure recovery for documents that have no layout.

Markdown, CSV and plain text carry no geometry, so there is nothing to detect
visually — but they are not unstructured. Headings nest, lists group, fenced
code must not be split, and a CSV is a table whose header row belongs with every
group of rows beneath it. Recovering that gives flow documents the same
section-aware chunking paged documents get from layout analysis.

Every block records exact character offsets into the original text. That is the
flow equivalent of a bounding box: it is what lets a citation highlight the
precise span a claim came from, so `.md` files are as citable as PDFs.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass

__all__ = ["FlowBlock", "split_flow", "split_markdown", "split_plain", "split_csv"]

_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_LIST_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
_TABLE_RE = re.compile(r"^\s*\|")
_SETEXT_RE = re.compile(r"^\s*(=+|-{2,})\s*$")


@dataclass(frozen=True, slots=True)
class FlowBlock:
    kind: str
    text: str
    char_start: int
    char_end: int
    section_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.char_end <= self.char_start:
            raise ValueError(f"empty span at {self.char_start}")


def _emit(
    blocks: list[FlowBlock],
    source: str,
    start: int,
    end: int,
    kind: str,
    section: tuple[str, ...],
) -> None:
    """Trim surrounding whitespace but keep offsets pointing at real content."""
    raw = source[start:end]
    lead = len(raw) - len(raw.lstrip())
    trail = len(raw) - len(raw.rstrip())
    if end - trail <= start + lead:
        return
    blocks.append(
        FlowBlock(
            kind=kind,
            text=raw.strip(),
            char_start=start + lead,
            char_end=end - trail,
            section_path=section,
        )
    )


def _line_spans(text: str) -> list[tuple[int, int]]:
    """(start, end_including_newline) for each line."""
    spans = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        spans.append((cursor, cursor + len(line)))
        cursor += len(line)
    return spans


def split_markdown(text: str) -> list[FlowBlock]:
    """Split on headings, fences, lists and tables, tracking the heading stack."""
    blocks: list[FlowBlock] = []
    spans = _line_spans(text)
    lines = text.splitlines()

    # Heading trail by level, so a chunk under "## Risk Factors" inside
    # "# Part I" knows both.
    stack: list[tuple[int, str]] = []
    section: tuple[str, ...] = ()

    i = 0
    pending_start: int | None = None
    pending_kind = "paragraph"

    def flush(end_line: int) -> None:
        nonlocal pending_start
        if pending_start is None:
            return
        _emit(blocks, text, pending_start, spans[end_line - 1][1], pending_kind, section)
        pending_start = None

    while i < len(lines):
        line = lines[i]
        start, end = spans[i]

        heading = _ATX_RE.match(line)
        fence = _FENCE_RE.match(line)

        if fence:
            flush(i)
            marker = fence.group(1)
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(marker):
                j += 1
            close = spans[min(j, len(lines) - 1)][1]
            # Fenced code is atomic: splitting it produces two fragments that
            # are each syntactically broken.
            _emit(blocks, text, start, close, "paragraph", section)
            i = j + 1
            continue

        if heading:
            flush(i)
            level, title = len(heading.group(1)), heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            section = tuple(t for _, t in stack)
            # The heading itself is filed under the section it opens.
            _emit(blocks, text, start, end, "heading", section)
            i += 1
            continue

        if not line.strip():
            flush(i)
            i += 1
            continue

        kind = "table" if _TABLE_RE.match(line) else "list" if _LIST_RE.match(line) else "paragraph"
        if pending_start is None:
            pending_start, pending_kind = start, kind
        elif kind != pending_kind:
            # A table starting directly under a paragraph is a new block.
            flush(i)
            pending_start, pending_kind = start, kind
        i += 1

    flush(len(lines))
    return blocks


def split_plain(text: str) -> list[FlowBlock]:
    """Blank-line separated paragraphs, offsets preserved."""
    blocks: list[FlowBlock] = []
    cursor = 0
    for piece in re.split(r"(\n\s*\n)", text):
        if piece and not re.fullmatch(r"\n\s*\n", piece):
            _emit(blocks, text, cursor, cursor + len(piece), "paragraph", ())
        cursor += len(piece)
    return blocks


def split_csv(text: str, *, rows_per_block: int = 40, delimiter: str = ",") -> list[FlowBlock]:
    """Group rows into table blocks, repeating the header in each.

    Repeating the header is the point. A chunk of bare data rows retrieves badly
    and reads worse — the column names are what make the numbers mean anything,
    and they have to survive into whatever chunk the rows land in.
    """
    spans = _line_spans(text)
    lines = text.splitlines()
    if not lines:
        return []

    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        header = next(reader, None)
    except csv.Error:
        header = None

    header_line = lines[0] if header else ""
    blocks: list[FlowBlock] = []

    body = range(1, len(lines)) if header else range(len(lines))
    group: list[int] = []

    def flush_group() -> None:
        if not group:
            return
        start = spans[group[0]][0]
        end = spans[group[-1]][1]
        body_text = text[start:end].rstrip()
        combined = f"{header_line}\n{body_text}" if header_line else body_text
        blocks.append(
            FlowBlock(
                kind="table",
                text=combined,
                char_start=start,
                char_end=start + len(body_text),
                section_path=(),
            )
        )
        group.clear()

    for idx in body:
        if not lines[idx].strip():
            continue
        group.append(idx)
        if len(group) >= rows_per_block:
            flush_group()
    flush_group()
    return blocks


def split_flow(text: str, extension: str = "txt") -> list[FlowBlock]:
    """Dispatch on extension. `detect_format` has already vetted the content."""
    if not text.strip():
        return []
    if extension in ("md", "markdown"):
        return split_markdown(text)
    if extension == "csv":
        return split_csv(text)
    if extension == "tsv":
        return split_csv(text, delimiter="\t")
    return split_plain(text)
