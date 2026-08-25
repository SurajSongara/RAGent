"""Answer synthesis with mandatory inline citations.

Two paths, and the fallback is not an afterthought. With an API key, Claude
writes a cited answer. Without one, the system returns the retrieved passages
ranked and labelled — which is a genuinely useful result, not an error page, and
it keeps the citation viewer demonstrable for anyone who has not configured a
key.

Citations are resolved against the passages that were actually retrieved rather
than trusted from the model's output. A marker pointing at a passage that was
never in context is dropped, so a hallucinated `[7]` cannot render as a
highlight over a real document.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ragent.providers import llm
from ragent.retrieval.search import Passage

__all__ = ["AnswerChunk", "answer_stream", "build_prompt", "resolve_citations"]

_MARKER_RE = re.compile(r"\[(\d{1,2})\]")

SYSTEM = """You answer questions about documents using only the passages provided.

Rules:
- Cite every factual claim with a bracketed passage number, like [1] or [2].
- Use only the numbered passages given. Never use outside knowledge.
- If the passages do not contain the answer, say so plainly. Do not guess.
- Quote exact figures rather than paraphrasing them.
- Be concise. No preamble, no restating the question."""


@dataclass(slots=True)
class AnswerChunk:
    """One SSE frame: streamed text, or the final citation set."""

    type: str  # "delta" | "citations" | "usage" | "done"
    text: str = ""
    data: Any = None


def build_prompt(question: str, passages: list[Passage]) -> str:
    """Number passages from 1 so the model's markers map straight to the list."""
    blocks = []
    for i, passage in enumerate(passages, start=1):
        where = " > ".join(passage.section_path) if passage.section_path else ""
        header = f"[{i}] {passage.document_name}"
        if where:
            header += f" — {where}"
        if passage.regions:
            header += f" (page {passage.regions[0]['page_no']})"
        blocks.append(f"{header}\n{passage.text}")

    joined = "\n\n---\n\n".join(blocks)
    return f"Passages:\n\n{joined}\n\n---\n\nQuestion: {question}"


def resolve_citations(text: str, passages: list[Passage]) -> list[dict[str, Any]]:
    """Map the markers the model actually used back to real passages.

    Only markers within range survive. The model citing `[9]` when eight
    passages were supplied is a hallucinated reference, and rendering it as a
    highlight over a real page would be worse than dropping it.
    """
    seen: dict[int, dict[str, Any]] = {}
    for match in _MARKER_RE.finditer(text):
        marker = int(match.group(1))
        if marker < 1 or marker > len(passages) or marker in seen:
            continue
        passage = passages[marker - 1]
        seen[marker] = {"marker": marker, **passage.to_dict()}
    return [seen[k] for k in sorted(seen)]


def _extractive_answer(question: str, passages: list[Passage]) -> str:
    """No-key fallback: the passages themselves, cited and ranked."""
    if not passages:
        return "No passages in the indexed corpus match that question."

    lines = [
        "_No model provider is configured, so this is the retrieved evidence rather "
        "than a written answer. Retrieval, ranking and citations are live. Set "
        "`ANTHROPIC_API_KEY` or `OPENAI_API_KEY` for a written one._",
        "",
    ]
    for i, passage in enumerate(passages, start=1):
        snippet = passage.text.strip()
        if len(snippet) > 600:
            snippet = snippet[:600].rsplit(" ", 1)[0] + "…"
        lines.append(f"**[{i}] {passage.document_name}**")
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines)


async def answer_stream(question: str, passages: list[Passage]) -> AsyncIterator[AnswerChunk]:
    """Stream an answer, then its resolved citations."""
    if not passages:
        yield AnswerChunk("delta", text="I could not find anything relevant in the corpus.")
        yield AnswerChunk("citations", data=[])
        yield AnswerChunk("done")
        return

    if not llm.available():
        text = _extractive_answer(question, passages)
        yield AnswerChunk("delta", text=text)
        yield AnswerChunk("citations", data=resolve_citations(text, passages))
        yield AnswerChunk("done")
        return

    collected: list[str] = []
    usage: llm.Usage | None = None

    async for delta, final_usage in llm.stream_text(
        build_prompt(question, passages), tier=llm.Tier.SYNTHESIS, system=SYSTEM
    ):
        if delta:
            collected.append(delta)
            yield AnswerChunk("delta", text=delta)
        if final_usage is not None:
            usage = final_usage

    text = "".join(collected)
    yield AnswerChunk("citations", data=resolve_citations(text, passages))

    if usage is not None:
        yield AnswerChunk(
            "usage",
            data={
                "model": usage.model,
                "provider": usage.provider,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "latency_ms": usage.latency_ms,
                "cost_usd": round(usage.cost_usd, 6),
                "priced": usage.priced,
            },
        )
    yield AnswerChunk("done")
