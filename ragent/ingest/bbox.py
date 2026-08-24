"""Bounding boxes, and the coordinate conversion that keeps citations honest.

PDF user space puts the origin at the bottom-left with y increasing upward. Every
viewer, including the PDF.js canvas this project highlights on, puts the origin at
the top-left with y increasing downward. Pages also carry a /Rotate entry that the
viewer applies but the stored coordinates do not reflect.

Get this wrong and citations land on the wrong part of the page — which looks
exactly like a retrieval bug and is miserable to chase down. So the conversion
happens once, here, at ingest, and everything downstream stores normalised
top-left coordinates in 0..1.

Normalised rather than absolute so the viewer highlights correctly at any zoom
without knowing the scale the page was rendered at.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["BBox", "to_normalised", "union", "iou", "merge_overlapping"]

# Coordinates are floats derived from a rasteriser, so exact 0.0/1.0 comparisons
# lose to rounding. The DB CHECK constraint is strict, hence a small clamp.
_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class BBox:
    """Top-left origin, normalised to 0..1 against the *displayed* page."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"degenerate bbox: {self}")
        if not (self.x0 >= 0.0 and self.y0 >= 0.0 and self.x1 <= 1.0 and self.y1 <= 1.0):
            raise ValueError(f"bbox outside unit square: {self}")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_dict(self) -> dict[str, float]:
        """Shape stored in `citations.bboxes` and consumed by the viewer."""
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


def _clamp_unit(v: float) -> float:
    return 0.0 if v < _EPS else (1.0 if v > 1.0 - _EPS else v)


def to_normalised(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    page_width: float,
    page_height: float,
    rotation: int = 0,
) -> BBox:
    """Convert one PDF-user-space rect to a normalised top-left-origin BBox.

    `rotation` is the page's /Rotate value (0, 90, 180, 270), applied clockwise by
    the viewer. For 90 and 270 the displayed page is landscape, so the box is
    normalised against swapped dimensions.
    """
    if page_width <= 0 or page_height <= 0:
        raise ValueError(f"page dimensions must be positive, got {page_width}x{page_height}")

    rotation %= 360
    if rotation not in (0, 90, 180, 270):
        raise ValueError(f"unsupported page rotation: {rotation}")

    # Callers hand us rects in either corner order; normalise before transforming.
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)

    # Flip the y axis: bottom-left origin -> top-left origin.
    top = page_height - hi_y
    bottom = page_height - lo_y
    left, right = lo_x, hi_x

    w, h = page_width, page_height

    if rotation == 90:
        # Clockwise: (x, y) -> (h - y, x); displayed page is h wide, w tall.
        left, right, top, bottom = h - bottom, h - top, left, right
        w, h = h, w
    elif rotation == 180:
        left, right, top, bottom = w - right, w - left, h - bottom, h - top
    elif rotation == 270:
        # Counter-clockwise: (x, y) -> (y, w - x).
        left, right, top, bottom = top, bottom, w - right, w - left
        w, h = h, w

    return BBox(
        _clamp_unit(left / w),
        _clamp_unit(top / h),
        _clamp_unit(right / w),
        _clamp_unit(bottom / h),
    )


def union(boxes: Iterable[BBox]) -> BBox:
    """Smallest box covering all inputs.

    Used to collapse a chunk's constituent blocks into the single highlight the
    viewer draws for a citation.
    """
    items = list(boxes)
    if not items:
        raise ValueError("union of no boxes is undefined")
    return BBox(
        min(b.x0 for b in items),
        min(b.y0 for b in items),
        max(b.x1 for b in items),
        max(b.y1 for b in items),
    )


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union. Used to dedupe blocks that two extractors both found."""
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    return intersection / (a.area + b.area - intersection)


def merge_overlapping(boxes: Iterable[BBox], *, threshold: float = 0.6) -> list[BBox]:
    """Collapse heavily overlapping boxes.

    The OCR fallback re-detects regions the native parser already found. Without
    this a citation highlight gets drawn two or three times over.
    """
    remaining = sorted(boxes, key=lambda b: b.area, reverse=True)
    merged: list[BBox] = []

    while remaining:
        current = remaining.pop(0)
        group = [current]
        rest: list[BBox] = []
        for other in remaining:
            if iou(current, other) >= threshold:
                group.append(other)
            else:
                rest.append(other)
        remaining = rest
        merged.append(union(group) if len(group) > 1 else current)

    return merged
