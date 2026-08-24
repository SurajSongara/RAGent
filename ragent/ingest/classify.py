"""Decide, per page, whether the embedded text layer can be trusted.

This is the selective-OCR gate and it is the single highest-leverage decision in
the pipeline. The naive approach rasterises and OCRs every page; on a 200-page
10-K that costs minutes per document *and makes accuracy worse*, because a
born-digital filing already has a perfect text layer and OCR can only degrade it.

The opposite failure is just as bad: a scanned exhibit with a broken or absent
text layer silently ingests as garbage, and nobody notices until a citation
points at nonsense.

So we score each page and only OCR what actually needs it. Real filings fail in
four recognisable ways, and each gets its own signal:

    empty            scanned image with no text layer at all
    (cid:NNN) soup   subsetted font with no ToUnicode map
    U+FFFD           mis-decoded bytes
    consonant soup   wrong encoding, extracts as letters that form no words
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["PageAssessment", "assess_text_layer"]

_CID_RE = re.compile(r"\(cid:\d+\)")
_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_VOWELS = frozenset("aeiouyAEIOUY")

# A normal typeset page runs well above this; scanned pages that pick up a stray
# header stamp fall far below it. Units are characters per 1000 pt².
_MIN_CHAR_DENSITY = 0.35


@dataclass(slots=True)
class PageAssessment:
    """Per-page verdict, persisted to `pages.text_confidence` / `pages.needs_ocr`."""

    confidence: float
    needs_ocr: bool
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))


def _word_shape_score(text: str) -> float:
    """Fraction of alphabetic runs that look like real words.

    A broken encoding extracts letters that never form pronounceable words, so
    the share of vowel-bearing runs collapses. English running text sits near
    1.0; consonant soup sits near 0.

    Returns 1.0 when there is too little to judge, so this signal never fires on
    a sparse-but-valid page (a cover sheet, a chart label) on its own.
    """
    words = _WORD_RE.findall(text)
    if len(words) < 8:
        return 1.0
    with_vowel = sum(1 for w in words if _VOWELS & set(w))
    return with_vowel / len(words)


def assess_text_layer(
    text: str,
    *,
    page_width_pt: float,
    page_height_pt: float,
    threshold: float = 0.72,
) -> PageAssessment:
    """Score an extracted text layer in 0..1 and decide whether to OCR the page."""
    if page_width_pt <= 0 or page_height_pt <= 0:
        raise ValueError("page dimensions must be positive")

    reasons: list[str] = []
    stripped = text.strip()

    if not stripped:
        return PageAssessment(0.0, True, ["empty_text_layer"])

    total = len(text)
    confidence = 1.0

    # --- mis-decoded glyphs ---------------------------------------------
    # Both signals are strong evidence on their own, so they scale the score
    # rather than subtracting from it.
    cid_chars = sum(len(m) for m in _CID_RE.findall(text))
    cid_ratio = cid_chars / total
    if cid_ratio > 0.02:
        reasons.append(f"cid_glyphs={cid_ratio:.0%}")
        confidence *= max(0.0, 1.0 - cid_ratio * 3.0)

    replacement_ratio = text.count("�") / total
    if replacement_ratio > 0.005:
        reasons.append(f"replacement_chars={replacement_ratio:.1%}")
        confidence *= max(0.0, 1.0 - replacement_ratio * 10.0)

    # --- density ---------------------------------------------------------
    # Catches the scanned page whose only text layer is a stamped header.
    density = len(stripped) / (page_width_pt * page_height_pt / 1000.0)
    if density < _MIN_CHAR_DENSITY:
        reasons.append(f"sparse_text density={density:.2f}")
        confidence *= max(0.15, density / _MIN_CHAR_DENSITY)

    # --- encoding sanity --------------------------------------------------
    shape = _word_shape_score(text)
    if shape < 0.65:
        reasons.append(f"unwordlike_text={shape:.2f}")
        confidence *= shape

    return PageAssessment(confidence, confidence < threshold, reasons)
