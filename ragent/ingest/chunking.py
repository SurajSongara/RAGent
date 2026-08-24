"""Four chunking strategies, implemented side by side so they can be measured.

The point of having four is Phase 2: `make bench` indexes the same corpus under
every strategy and scores them against one golden set. The winner is whichever
the data picks, not whichever the README asserts.

    fixed       token windows with overlap, structure-blind. The baseline
                nearly every RAG demo ships, kept honest so the comparison
                has a floor to beat.
    recursive   descend a separator hierarchy until pieces fit. Structure-aware
                about punctuation, not about the document.
    layout      pack whole layout blocks, break on headings, never split a
                table. Uses what the parse stage already recovered.
    semantic    split where the topic actually shifts, detected from embedding
                distance between adjacent sentences.

Every strategy preserves block provenance, including the ones whose boundaries
cut through blocks. A chunk always knows which blocks it drew from, so a
citation can always resolve to a highlight. That invariant is not negotiable —
it is the constraint the whole schema exists to protect.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ragent.ingest.types import Chunk, SourceBlock, TokenCounter, approx_tokens

__all__ = [
    "chunk_document",
    "chunk_fixed",
    "chunk_recursive",
    "chunk_layout",
    "chunk_semantic",
    "STRATEGIES",
]

_BLOCK_JOIN = "\n\n"
_WORD_RE = re.compile(r"\S+")
# Sentence end: terminator, optional closing quote/bracket, whitespace, then a
# capital. Deliberately conservative — over-splitting hurts semantic chunking
# more than under-splitting does. Two fixed-width lookbehinds rather than one
# optional group, because Python requires fixed-width lookbehind.
_SENTENCE_RE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[A-Z(\"'])")
_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " "]


@dataclass(frozen=True, slots=True)
class _Span:
    """Where a block's text landed in the flattened document."""

    start: int
    end: int
    block: SourceBlock


def _flatten(blocks: Sequence[SourceBlock]) -> tuple[str, list[_Span]]:
    parts: list[str] = []
    spans: list[_Span] = []
    cursor = 0
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        if parts:
            cursor += len(_BLOCK_JOIN)
        spans.append(_Span(cursor, cursor + len(text), block))
        parts.append(text)
        cursor += len(text)
    return _BLOCK_JOIN.join(parts), spans


def _blocks_in_range(spans: Sequence[_Span], start: int, end: int) -> list[SourceBlock]:
    """Blocks whose text overlaps [start, end).

    This is what keeps provenance intact when a chunk boundary falls inside a
    block: the chunk still points at every block it touched, so the citation
    highlight covers the right region even though the split ignored layout.
    """
    return [s.block for s in spans if s.start < end and s.end > start]


def _prepare(blocks: Sequence[SourceBlock]) -> list[SourceBlock]:
    """Drop page furniture and anything empty, then fix reading order."""
    kept = [b for b in blocks if b.text.strip() and not b.is_furniture]
    return sorted(kept, key=lambda b: (b.page_no, b.reading_order))


def _assemble(
    seq: int,
    text: str,
    blocks: Sequence[SourceBlock],
    strategy: str,
    count_tokens: TokenCounter,
) -> Chunk | None:
    text = text.strip()
    if not text or not blocks:
        return None
    return Chunk(
        seq=seq,
        text=text,
        block_ids=[b.id for b in blocks],
        page_from=min(b.page_no for b in blocks),
        page_to=max(b.page_no for b in blocks),
        token_count=count_tokens(text),
        # The deepest heading trail present wins, so a chunk spanning a heading
        # boundary is filed under the more specific section.
        section_path=max((b.section_path for b in blocks), key=len, default=()),
        strategy=strategy,
    )


# ---------------------------------------------------------------- fixed


def _fit_window(
    doc: str,
    words: Sequence[tuple[int, int]],
    costs: Sequence[int],
    i: int,
    target: int,
    count_tokens: TokenCounter,
) -> int:
    """Largest j > i whose word span actually fits the budget.

    Summing per-word costs is only an estimate: real tokenisers do not compose
    additively across a whitespace split, so 64 words each costing 1 can measure
    83 tokens once joined. We grow on the cheap estimate, then verify against the
    real counter and scale back proportionally until it fits — which converges in
    a couple of steps rather than walking back one word at a time.
    """
    est = 0
    j = i
    while j < len(words) and est + costs[j] <= target:
        est += costs[j]
        j += 1
    if j == i:  # a single word longer than the whole budget
        return i + 1

    while j > i + 1:
        actual = count_tokens(doc[words[i][0] : words[j - 1][1]])
        if actual <= target:
            break
        span = j - i
        scaled = max(1, int(span * target / actual))
        j = i + (scaled if scaled < span else span - 1)

    return j


def chunk_fixed(
    blocks: Sequence[SourceBlock],
    *,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
    count_tokens: TokenCounter = approx_tokens,
) -> list[Chunk]:
    """Sliding token windows. Ignores every structural signal in the document."""
    prepared = _prepare(blocks)
    doc, spans = _flatten(prepared)
    if not doc:
        return []

    words = [(m.start(), m.end()) for m in _WORD_RE.finditer(doc)]
    if not words:
        return []
    # Per-word costs are the cheap first guess that _fit_window then verifies.
    costs = [count_tokens(doc[s:e]) for s, e in words]

    chunks: list[Chunk] = []
    i = 0
    while i < len(words):
        j = _fit_window(doc, words, costs, i, target_tokens, count_tokens)

        start, end = words[i][0], words[j - 1][1]
        chunk = _assemble(
            len(chunks), doc[start:end], _blocks_in_range(spans, start, end), "fixed", count_tokens
        )
        if chunk:
            chunks.append(chunk)

        if j >= len(words):
            break

        # Walk back to build the overlap; guarantee forward progress.
        back = 0
        k = j
        while k > i + 1 and back + costs[k - 1] <= overlap_tokens:
            k -= 1
            back += costs[k]
        i = k if k > i else j

    return chunks


# ---------------------------------------------------------------- recursive


def _split_recursive(
    doc: str,
    start: int,
    end: int,
    separators: Sequence[str],
    target: int,
    count_tokens: TokenCounter,
) -> list[tuple[int, int]]:
    if count_tokens(doc[start:end]) <= target:
        return [(start, end)]
    if not separators:
        return [(start, end)]  # nothing left to split on; emit oversized

    sep, rest = separators[0], separators[1:]
    pieces: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        found = doc.find(sep, cursor, end)
        if found == -1:
            pieces.append((cursor, end))
            break
        stop = found + len(sep)
        if stop > cursor:
            pieces.append((cursor, stop))
        cursor = stop

    if len(pieces) <= 1:
        return _split_recursive(doc, start, end, rest, target, count_tokens)

    out: list[tuple[int, int]] = []
    for p_start, p_end in pieces:
        out.extend(_split_recursive(doc, p_start, p_end, rest, target, count_tokens))
    return out


def chunk_recursive(
    blocks: Sequence[SourceBlock],
    *,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
    count_tokens: TokenCounter = approx_tokens,
) -> list[Chunk]:
    """Descend paragraph -> line -> sentence -> clause -> word until pieces fit."""
    prepared = _prepare(blocks)
    doc, spans = _flatten(prepared)
    if not doc:
        return []

    pieces = _split_recursive(doc, 0, len(doc), _SEPARATORS, target_tokens, count_tokens)

    # The split leaves many undersized fragments; greedily repack them so chunks
    # land near the target instead of averaging a fraction of it.
    chunks: list[Chunk] = []
    cur_start: int | None = None
    cur_end = 0
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_tokens
        if cur_start is None:
            return
        chunk = _assemble(
            len(chunks),
            doc[cur_start:cur_end],
            _blocks_in_range(spans, cur_start, cur_end),
            "recursive",
            count_tokens,
        )
        if chunk:
            chunks.append(chunk)
        cur_start, cur_end, cur_tokens = None, 0, 0

    for p_start, p_end in pieces:
        cost = count_tokens(doc[p_start:p_end])
        if cur_start is not None and cur_tokens + cost > target_tokens:
            flush()
        if cur_start is None:
            cur_start = p_start
        cur_end = p_end
        cur_tokens += cost
    flush()

    if overlap_tokens > 0:
        chunks = _apply_text_overlap(chunks, doc, spans, overlap_tokens, "recursive", count_tokens)
    return chunks


def _apply_text_overlap(
    chunks: list[Chunk],
    doc: str,
    spans: Sequence[_Span],
    overlap_tokens: int,
    strategy: str,
    count_tokens: TokenCounter,
) -> list[Chunk]:
    """Prepend a tail of the previous chunk to each chunk after the first."""
    if len(chunks) < 2:
        return chunks

    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        tail_words = _WORD_RE.findall(prev.text)
        tail: list[str] = []
        budget = 0
        for word in reversed(tail_words):
            cost = count_tokens(word)
            if budget + cost > overlap_tokens:
                break
            tail.insert(0, word)
            budget += cost
        if not tail:
            out.append(cur)
            continue
        merged_text = " ".join(tail) + " " + cur.text
        # Provenance follows the text: the overlap's blocks join this chunk too.
        block_ids = list(dict.fromkeys(prev.block_ids[-1:] + cur.block_ids))
        out.append(
            Chunk(
                seq=cur.seq,
                text=merged_text,
                block_ids=block_ids,
                page_from=min(prev.page_to, cur.page_from),
                page_to=cur.page_to,
                token_count=count_tokens(merged_text),
                section_path=cur.section_path,
                strategy=strategy,
            )
        )
    return out


# ---------------------------------------------------------------- layout


def chunk_layout(
    blocks: Sequence[SourceBlock],
    *,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
    count_tokens: TokenCounter = approx_tokens,
) -> list[Chunk]:
    """Pack whole blocks, respecting what the parser recovered about the document.

    Rules, in priority order:
      1. Tables and figures are never split and never share a chunk.
      2. A heading starts a new chunk and attaches to the text beneath it.
      3. A section change starts a new chunk.
      4. Otherwise pack until the token budget is reached.
    """
    prepared = _prepare(blocks)
    if not prepared:
        return []

    chunks: list[Chunk] = []
    buffer: list[SourceBlock] = []
    buffered_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffered_tokens
        if not buffer:
            return
        chunk = _assemble(
            len(chunks),
            _BLOCK_JOIN.join(b.text.strip() for b in buffer),
            buffer,
            "layout",
            count_tokens,
        )
        if chunk:
            chunks.append(chunk)
        buffer, buffered_tokens = [], 0

    for block in prepared:
        cost = count_tokens(block.text)

        if block.is_atomic:
            flush()
            chunk = _assemble(len(chunks), block.text, [block], "layout", count_tokens)
            if chunk:
                chunk.meta["atomic"] = block.kind
                chunks.append(chunk)
            continue

        section_changed = bool(buffer) and buffer[-1].section_path != block.section_path
        if block.is_boundary or section_changed or buffered_tokens + cost > target_tokens:
            flush()

        buffer.append(block)
        buffered_tokens += cost

    flush()

    # Overlap is applied block-wise and never across a table, so an atomic chunk
    # stays exactly the table it represents.
    if overlap_tokens > 0:
        chunks = _apply_block_overlap(chunks, prepared, overlap_tokens, count_tokens)
    return chunks


def _apply_block_overlap(
    chunks: list[Chunk],
    blocks: Sequence[SourceBlock],
    overlap_tokens: int,
    count_tokens: TokenCounter,
) -> list[Chunk]:
    by_id = {b.id: b for b in blocks}
    out: list[Chunk] = []
    for idx, cur in enumerate(chunks):
        if idx == 0 or "atomic" in cur.meta or "atomic" in chunks[idx - 1].meta:
            out.append(cur)
            continue
        prev = chunks[idx - 1]
        carry = [
            by_id[bid]
            for bid in prev.block_ids[-1:]
            if bid in by_id and count_tokens(by_id[bid].text) <= overlap_tokens
        ]
        if not carry:
            out.append(cur)
            continue
        merged_blocks = carry + [by_id[b] for b in cur.block_ids if b in by_id]
        merged = _assemble(
            cur.seq,
            _BLOCK_JOIN.join(b.text.strip() for b in merged_blocks),
            merged_blocks,
            "layout",
            count_tokens,
        )
        out.append(merged or cur)
    return out


# ---------------------------------------------------------------- semantic

EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. Avoids a numpy import for one call."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def chunk_semantic(
    blocks: Sequence[SourceBlock],
    *,
    embed: EmbedFn,
    target_tokens: int = 512,
    breakpoint_percentile: float = 0.25,
    count_tokens: TokenCounter = approx_tokens,
    **_ignored: object,
) -> list[Chunk]:
    """Break where the topic shifts, measured by embedding distance.

    Similarity between adjacent sentences is scored, and the lowest
    `breakpoint_percentile` of those gaps become chunk boundaries. A percentile
    rather than a fixed threshold, because absolute cosine values are not
    comparable across embedding models — the shape of the distribution is.
    """
    prepared = _prepare(blocks)
    doc, spans = _flatten(prepared)
    if not doc:
        return []

    bounds: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_RE.finditer(doc):
        if match.start() > cursor:
            bounds.append((cursor, match.start()))
        cursor = match.end()
    if cursor < len(doc):
        bounds.append((cursor, len(doc)))
    if len(bounds) < 2:
        chunk = _assemble(0, doc, prepared, "semantic", count_tokens)
        return [chunk] if chunk else []

    sentences = [doc[s:e] for s, e in bounds]
    vectors = list(embed(sentences))
    if len(vectors) != len(sentences):
        raise ValueError(f"embedder returned {len(vectors)} vectors for {len(sentences)} sentences")

    gaps = [_cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    cut_below = _percentile(gaps, breakpoint_percentile)

    chunks: list[Chunk] = []
    start_idx = 0
    running = count_tokens(sentences[0])

    for i, similarity in enumerate(gaps):
        next_cost = count_tokens(sentences[i + 1])
        # Strictly below, not at-or-below: when every gap is similar the
        # percentile lands *on* the common value, and `<=` would then cut at
        # every sentence instead of recognising there is no topic shift.
        if similarity < cut_below or running + next_cost > target_tokens:
            start, end = bounds[start_idx][0], bounds[i][1]
            chunk = _assemble(
                len(chunks),
                doc[start:end],
                _blocks_in_range(spans, start, end),
                "semantic",
                count_tokens,
            )
            if chunk:
                chunk.meta["cut_similarity"] = round(similarity, 4)
                chunks.append(chunk)
            start_idx = i + 1
            running = next_cost
        else:
            running += next_cost

    start, end = bounds[start_idx][0], bounds[-1][1]
    tail = _assemble(
        len(chunks), doc[start:end], _blocks_in_range(spans, start, end), "semantic", count_tokens
    )
    if tail:
        chunks.append(tail)

    return chunks


# ---------------------------------------------------------------- registry

STRATEGIES: dict[str, Callable[..., list[Chunk]]] = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "layout": chunk_layout,
    "semantic": chunk_semantic,
}


def chunk_document(strategy: str, blocks: Sequence[SourceBlock], **kwargs: object) -> list[Chunk]:
    """Dispatch by name. `strategy` comes straight from CHUNK_STRATEGIES."""
    try:
        fn = STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"unknown chunking strategy {strategy!r}; expected one of {sorted(STRATEGIES)}"
        ) from None
    return fn(blocks, **kwargs)
