"""The selective-OCR gate.

Two failure modes matter and they pull in opposite directions: OCRing a page that
already has a good text layer makes accuracy *worse*, and trusting a broken text
layer ingests garbage nobody notices until a citation points at nonsense.
"""

from __future__ import annotations

import pytest

from ragent.ingest.classify import assess_text_layer

W, H = 612.0, 792.0

CLEAN_PAGE = (
    "Total net sales increased 12% during fiscal 2025 compared to fiscal 2024. "
    "The growth was driven primarily by higher services revenue, partially offset "
    "by a decline in hardware unit volumes across certain geographic segments. "
    "Gross margin percentage improved by 180 basis points year over year, "
    "reflecting a more favourable mix of higher margin services revenue and "
    "continued operating leverage in our manufacturing organisation."
)


def assess(text: str, **kwargs: object):
    return assess_text_layer(text, page_width_pt=W, page_height_pt=H, **kwargs)


class TestGoodPages:
    def test_clean_text_is_trusted(self) -> None:
        result = assess(CLEAN_PAGE)
        assert result.confidence == pytest.approx(1.0)
        assert result.needs_ocr is False
        assert result.reasons == []

    def test_occasional_cid_artifact_is_tolerated(self) -> None:
        """A stray bad glyph is not worth re-OCRing a whole page over."""
        result = assess(CLEAN_PAGE + "(cid:3)")
        assert result.needs_ocr is False


class TestScannedPages:
    def test_empty_text_layer(self) -> None:
        result = assess("")
        assert result.confidence == 0.0
        assert result.needs_ocr is True
        assert "empty_text_layer" in result.reasons

    def test_whitespace_only_counts_as_empty(self) -> None:
        assert assess("   \n\n \t ").needs_ocr is True

    def test_stamped_header_only(self) -> None:
        """A scan whose only text layer is a header stamp must still be OCRed."""
        result = assess("EXHIBIT 10.1")
        assert result.needs_ocr is True
        assert any(r.startswith("sparse_text") for r in result.reasons)


class TestBrokenEncodings:
    def test_cid_soup(self) -> None:
        """A subsetted font with no ToUnicode map extracts as (cid:NNN) runs."""
        result = assess("(cid:3)(cid:75)(cid:72)(cid:3)(cid:82)(cid:81)" * 40)
        assert result.needs_ocr is True
        assert any(r.startswith("cid_glyphs") for r in result.reasons)

    def test_replacement_characters(self) -> None:
        result = assess("Net sales " + "�" * 120 + " for the period ended")
        assert result.needs_ocr is True
        assert any(r.startswith("replacement_chars") for r in result.reasons)

    def test_consonant_soup(self) -> None:
        """Wrong encoding extracts letters that never form pronounceable words."""
        soup = " ".join(["bcdfg", "hjklm", "npqrs", "tvwxz", "bcdfg", "hjklm"] * 8)
        result = assess(soup)
        assert result.needs_ocr is True
        assert any(r.startswith("unwordlike_text") for r in result.reasons)

    def test_short_but_valid_text_is_not_called_unwordlike(self) -> None:
        """Fewer than 8 words is too little evidence; this signal must not fire."""
        result = assess("Consolidated Balance Sheets")
        assert not any(r.startswith("unwordlike_text") for r in result.reasons)


class TestThreshold:
    def test_threshold_governs_the_decision(self) -> None:
        marginal = "Exhibit 10.1 Employment Agreement dated as of January 1"
        assert assess(marginal, threshold=0.9).needs_ocr is True
        assert assess(marginal, threshold=0.0).needs_ocr is False

    def test_confidence_is_clamped_to_unit_interval(self) -> None:
        for text in ("", CLEAN_PAGE, "(cid:9)" * 500):
            result = assess(text)
            assert 0.0 <= result.confidence <= 1.0


def test_rejects_degenerate_page() -> None:
    with pytest.raises(ValueError, match="page dimensions"):
        assess_text_layer("x", page_width_pt=0, page_height_pt=H)
