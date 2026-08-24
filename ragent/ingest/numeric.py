"""Parse the numbers out of financial table cells.

This runs once at ingest so `table_cells.numeric_value` holds a typed value, and
a question like "what was FY25 operating income" resolves against real numbers
instead of asking a model to re-read a mangled pipe-table out of a chunk.

Filing tables are hostile in specific, repeatable ways:

    (1,234)      negative, accounting parentheses
    $ (1,234 )   currency symbol and stray internal whitespace
    1,234(1)     trailing footnote marker, NOT a negative
    (1,234)(2)   both at once
    (12.3)%      negative percentage
    —            em dash meaning zero-or-absent, not a minus sign
    NM           "not meaningful"

Getting any one of these wrong flips a sign or invents a value, which is worse
than returning nothing. When in doubt this returns None.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

__all__ = ["ParsedNumber", "parse_numeric"]

# Cells that carry no number. Em/en dashes are the common "nil" glyph in filings
# and must never be read as a minus sign.
_NULL_TOKENS = frozenset(
    {"", "-", "--", "—", "–", "‒", "―", "n/a", "na", "n.a.", "nm", "n.m.", "nil", "none", "*"}
)

_CURRENCY = "$€£¥₹"

# A footnote reference is 1-2 bare digits in brackets at the very end. The digit
# limit is what separates "(1)" the footnote from "(12,345)" the negative, and we
# additionally require something numeric ahead of it before stripping.
_FOOTNOTE_RE = re.compile(r"[\(\[\{]\s*\d{1,2}\s*[\)\]\}]\s*$")

_SUPERSCRIPTS = "¹²³" + "".join(chr(c) for c in range(0x2070, 0x207A))


class ParsedNumber:
    """A parsed cell value plus what the surrounding notation implied."""

    __slots__ = ("value", "is_percent", "is_negative", "had_footnote")

    def __init__(
        self,
        value: Decimal,
        *,
        is_percent: bool = False,
        is_negative: bool = False,
        had_footnote: bool = False,
    ) -> None:
        self.value = value
        self.is_percent = is_percent
        self.is_negative = is_negative
        self.had_footnote = had_footnote

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"ParsedNumber({self.value}, is_percent={self.is_percent}, "
            f"had_footnote={self.had_footnote})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParsedNumber):
            return NotImplemented
        return (
            self.value == other.value
            and self.is_percent == other.is_percent
            and self.is_negative == other.is_negative
            and self.had_footnote == other.had_footnote
        )


def _strip_footnotes(text: str) -> tuple[str, bool]:
    """Remove trailing footnote markers, but never mistake a negative for one.

    We only strip when numeric content remains ahead of the marker, so a lone
    "(12)" stays a negative twelve while "1,234(1)" loses its reference.
    """
    had = False

    while (match := _FOOTNOTE_RE.search(text)) is not None:
        head = text[: match.start()]
        if not any(ch.isdigit() for ch in head):
            break  # nothing before it, so this is the value itself
        had = True
        text = head.rstrip()

    return text, had


def parse_numeric(raw: str | None) -> ParsedNumber | None:
    """Parse one table cell. Returns None when the cell holds no usable number."""
    if raw is None:
        return None

    # Superscript markers come off BEFORE normalisation. NFKC folds "¹" into a
    # plain "1", so normalising first turns "1,234¹" into "1,2341" — a silent
    # order-of-magnitude error, the worst kind this parser can make.
    raw_stripped = raw.strip()
    without_superscript = raw_stripped.rstrip(_SUPERSCRIPTS + " ")
    had_superscript = without_superscript != raw_stripped

    # NFKC then folds full-width digits and unicode minus signs into ASCII.
    text = unicodedata.normalize("NFKC", without_superscript)
    text = text.replace(" ", " ").replace("−", "-").strip()

    if text.lower() in _NULL_TOKENS:
        return None

    text, had_bracket = _strip_footnotes(text)
    had_footnote = had_superscript or had_bracket
    text = text.strip()
    if not text or text.lower() in _NULL_TOKENS:
        return None

    is_percent = False
    if text.endswith("%"):
        is_percent = True
        text = text[:-1].strip()
    elif text.lower().endswith(("bps", "bp")):
        text = re.sub(r"(?i)bps?$", "", text).strip()

    # A currency symbol can sit either side of the parenthesis — "$(1,234)" and
    # "($1,234)" both occur in real filings — so peel a leading one off before
    # testing for accounting parentheses, and the rest during cleanup.
    text = text.lstrip(_CURRENCY + " ")

    is_negative = False
    inner = text
    while len(inner) >= 2 and inner[0] == "(" and inner[-1] == ")":
        is_negative = not is_negative
        inner = inner[1:-1].strip()

    if inner.startswith("-"):
        is_negative = not is_negative
        inner = inner[1:].strip()
    elif inner.startswith("+"):
        inner = inner[1:].strip()

    cleaned = inner.strip(_CURRENCY + " ").replace(",", "").replace(" ", "")
    # A trailing multiple marker ("2.5x") is a ratio, still a real number.
    cleaned = re.sub(r"(?i)x$", "", cleaned)

    if not cleaned or not any(ch.isdigit() for ch in cleaned):
        return None
    # Reject anything that is not purely numeric by this point rather than
    # letting Decimal coerce something unintended.
    if not re.fullmatch(r"\d*\.?\d+|\d+\.", cleaned):
        return None

    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None

    if is_negative:
        value = -value

    return ParsedNumber(
        value,
        is_percent=is_percent,
        is_negative=is_negative,
        had_footnote=had_footnote,
    )
