"""Financial cell parsing.

Every case here is a shape that actually appears in EDGAR filings. The sign
tests matter most: reading "(1,234)" as positive silently inverts a number in an
answer, which is worse than failing to parse it at all.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from ragent.ingest.numeric import parse_numeric


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1234", Decimal("1234")),
        ("1,234", Decimal("1234")),
        ("1,234.56", Decimal("1234.56")),
        ("0.01", Decimal("0.01")),
        ("$1,234", Decimal("1234")),
        ("$ 1,234", Decimal("1234")),
        ("+42", Decimal("42")),
        ("€1.234", Decimal("1.234")),
    ],
)
def test_parses_plain_values(raw: str, expected: Decimal) -> None:
    parsed = parse_numeric(raw)
    assert parsed is not None
    assert parsed.value == expected
    assert parsed.is_negative is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("(1,234)", Decimal("-1234")),
        ("(0.01)", Decimal("-0.01")),
        ("-1,234", Decimal("-1234")),
        ("−1,234", Decimal("-1234")),  # U+2212 minus, not hyphen
        ("$(1,234)", Decimal("-1234")),
        ("($1,234)", Decimal("-1234")),
        ("$ (1,234 )", Decimal("-1234")),
    ],
)
def test_accounting_parentheses_mean_negative(raw: str, expected: Decimal) -> None:
    parsed = parse_numeric(raw)
    assert parsed is not None
    assert parsed.value == expected
    assert parsed.is_negative is True


@pytest.mark.parametrize(
    "raw", ["", "   ", "—", "–", "-", "N/A", "n/a", "NM", "n.m.", "nil", "*", "None"]
)
def test_null_tokens_yield_nothing(raw: str) -> None:
    """An em dash is a nil marker, not a minus sign."""
    assert parse_numeric(raw) is None


def test_none_input() -> None:
    assert parse_numeric(None) is None


class TestFootnotes:
    """A trailing marker is a reference, not part of the value — unless it is."""

    def test_footnote_stripped_from_positive(self) -> None:
        parsed = parse_numeric("1,234(1)")
        assert parsed is not None
        assert parsed.value == Decimal("1234")
        assert parsed.had_footnote is True
        assert parsed.is_negative is False

    def test_footnote_after_negative(self) -> None:
        parsed = parse_numeric("(1,234)(2)")
        assert parsed is not None
        assert parsed.value == Decimal("-1234")
        assert parsed.had_footnote is True

    def test_bare_parenthesised_number_is_negative_not_footnote(self) -> None:
        """The critical ambiguity: "(12)" alone is negative twelve."""
        parsed = parse_numeric("(12)")
        assert parsed is not None
        assert parsed.value == Decimal("-12")
        assert parsed.had_footnote is False

    def test_bracket_footnote(self) -> None:
        parsed = parse_numeric("987[3]")
        assert parsed is not None
        assert parsed.value == Decimal("987")
        assert parsed.had_footnote is True

    def test_superscript_footnote(self) -> None:
        parsed = parse_numeric("1,234¹")
        assert parsed is not None
        assert parsed.value == Decimal("1234")
        assert parsed.had_footnote is True


class TestPercentages:
    def test_positive(self) -> None:
        parsed = parse_numeric("12.3%")
        assert parsed is not None
        assert parsed.value == Decimal("12.3")
        assert parsed.is_percent is True

    def test_negative(self) -> None:
        parsed = parse_numeric("(12.3)%")
        assert parsed is not None
        assert parsed.value == Decimal("-12.3")
        assert parsed.is_percent is True

    def test_basis_points_are_plain_numbers(self) -> None:
        parsed = parse_numeric("250bps")
        assert parsed is not None
        assert parsed.value == Decimal("250")
        assert parsed.is_percent is False


@pytest.mark.parametrize(
    "raw",
    [
        "Item 7A",
        "Total revenue",
        "abc",
        "Q1",
        "FY2025 highlights",
        "see note 4 below",
    ],
)
def test_prose_is_rejected_rather_than_coerced(raw: str) -> None:
    """Returning None beats inventing a number out of a label."""
    assert parse_numeric(raw) is None


def test_ratio_multiple() -> None:
    parsed = parse_numeric("2.5x")
    assert parsed is not None
    assert parsed.value == Decimal("2.5")


def test_full_width_digits_are_normalised() -> None:
    parsed = parse_numeric("１２３４")
    assert parsed is not None
    assert parsed.value == Decimal("1234")


def test_double_parentheses_do_not_double_negate() -> None:
    """((12)) is malformed; it must not silently come back positive."""
    parsed = parse_numeric("((12))")
    assert parsed is not None
    assert parsed.value == Decimal("12")
