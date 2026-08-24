"""Coordinate conversion.

These tests exist because a sign error here does not crash anything — it just
puts citation highlights on the wrong part of the page, which looks like a
retrieval bug and costs hours to trace back to arithmetic.
"""

from __future__ import annotations

import pytest

from ragent.ingest.bbox import BBox, iou, merge_overlapping, to_normalised, union

# Portrait US Letter, in points.
W, H = 612.0, 792.0


class TestConstruction:
    def test_rejects_inverted_box(self) -> None:
        with pytest.raises(ValueError, match="degenerate"):
            BBox(0.5, 0.1, 0.2, 0.4)

    def test_rejects_zero_area(self) -> None:
        with pytest.raises(ValueError, match="degenerate"):
            BBox(0.2, 0.2, 0.2, 0.6)

    def test_rejects_outside_unit_square(self) -> None:
        """The DB CHECK constraint enforces this too; failing early is cheaper."""
        with pytest.raises(ValueError, match="unit square"):
            BBox(0.0, 0.0, 1.5, 0.5)

    def test_geometry(self) -> None:
        box = BBox(0.25, 0.5, 0.75, 1.0)
        assert box.width == pytest.approx(0.5)
        assert box.height == pytest.approx(0.5)
        assert box.area == pytest.approx(0.25)
        assert box.as_dict() == {"x0": 0.25, "y0": 0.5, "x1": 0.75, "y1": 1.0}


class TestFlip:
    """PDF space is bottom-left origin; the viewer is top-left origin."""

    def test_bottom_left_rect_lands_bottom_left(self) -> None:
        box = to_normalised(0, 0, W / 2, H / 2, page_width=W, page_height=H)
        assert box.x0 == pytest.approx(0.0)
        assert box.y0 == pytest.approx(0.5)  # halfway down, not at the top
        assert box.x1 == pytest.approx(0.5)
        assert box.y1 == pytest.approx(1.0)

    def test_top_left_rect_lands_top_left(self) -> None:
        box = to_normalised(0, H / 2, W / 2, H, page_width=W, page_height=H)
        assert box.y0 == pytest.approx(0.0)
        assert box.y1 == pytest.approx(0.5)

    def test_corner_order_does_not_matter(self) -> None:
        """Parsers hand rects over in either corner order."""
        a = to_normalised(10, 20, 110, 220, page_width=W, page_height=H)
        b = to_normalised(110, 220, 10, 20, page_width=W, page_height=H)
        assert a == b


class TestRotation:
    @pytest.mark.parametrize("rotation", [0, 90, 180, 270])
    def test_full_page_always_fills_unit_square(self, rotation: int) -> None:
        box = to_normalised(0, 0, W, H, page_width=W, page_height=H, rotation=rotation)
        assert box.as_dict() == pytest.approx({"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0})

    def test_90_sends_bottom_left_to_top_left(self) -> None:
        """Rotating clockwise, the page's left edge becomes its top edge."""
        box = to_normalised(0, 0, W / 2, H / 2, page_width=W, page_height=H, rotation=90)
        assert box.as_dict() == pytest.approx({"x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 0.5})

    def test_180_sends_bottom_left_to_top_right(self) -> None:
        box = to_normalised(0, 0, W / 2, H / 2, page_width=W, page_height=H, rotation=180)
        assert box.as_dict() == pytest.approx({"x0": 0.5, "y0": 0.0, "x1": 1.0, "y1": 0.5})

    def test_270_sends_bottom_left_to_bottom_right(self) -> None:
        box = to_normalised(0, 0, W / 2, H / 2, page_width=W, page_height=H, rotation=270)
        assert box.as_dict() == pytest.approx({"x0": 0.5, "y0": 0.5, "x1": 1.0, "y1": 1.0})

    def test_90_and_270_are_opposite_corners(self) -> None:
        a = to_normalised(0, 0, W / 2, H / 2, page_width=W, page_height=H, rotation=90)
        b = to_normalised(0, 0, W / 2, H / 2, page_width=W, page_height=H, rotation=270)
        assert a.x0 == pytest.approx(1.0 - b.x1)
        assert a.y0 == pytest.approx(1.0 - b.y1)

    def test_rotation_normalises_past_360(self) -> None:
        a = to_normalised(0, 0, W / 2, H / 2, page_width=W, page_height=H, rotation=450)
        b = to_normalised(0, 0, W / 2, H / 2, page_width=W, page_height=H, rotation=90)
        assert a == b

    def test_rejects_arbitrary_angle(self) -> None:
        with pytest.raises(ValueError, match="rotation"):
            to_normalised(0, 0, 10, 10, page_width=W, page_height=H, rotation=45)


def test_rejects_degenerate_page() -> None:
    with pytest.raises(ValueError, match="page dimensions"):
        to_normalised(0, 0, 10, 10, page_width=0, page_height=H)


class TestUnion:
    def test_covers_all_inputs(self) -> None:
        merged = union([BBox(0.1, 0.1, 0.3, 0.2), BBox(0.5, 0.6, 0.8, 0.9)])
        assert merged.as_dict() == pytest.approx({"x0": 0.1, "y0": 0.1, "x1": 0.8, "y1": 0.9})

    def test_single_box_is_itself(self) -> None:
        box = BBox(0.2, 0.3, 0.4, 0.5)
        assert union([box]) == box

    def test_empty_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            union([])


class TestIou:
    def test_identical_boxes(self) -> None:
        box = BBox(0.1, 0.1, 0.5, 0.5)
        assert iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes(self) -> None:
        assert iou(BBox(0.0, 0.0, 0.2, 0.2), BBox(0.5, 0.5, 0.9, 0.9)) == 0.0

    def test_touching_edges_do_not_overlap(self) -> None:
        assert iou(BBox(0.0, 0.0, 0.5, 0.5), BBox(0.5, 0.0, 1.0, 0.5)) == 0.0

    def test_partial_overlap(self) -> None:
        # Two unit-quarter boxes sharing half their area.
        a = BBox(0.0, 0.0, 0.4, 0.4)
        b = BBox(0.2, 0.0, 0.6, 0.4)
        # intersection 0.2*0.4=0.08, union 0.16+0.16-0.08=0.24
        assert iou(a, b) == pytest.approx(0.08 / 0.24)


class TestMergeOverlapping:
    def test_collapses_duplicate_detections(self) -> None:
        """The OCR fallback re-finds regions the native parser already produced."""
        native = BBox(0.1, 0.1, 0.9, 0.3)
        ocr = BBox(0.105, 0.102, 0.895, 0.298)
        merged = merge_overlapping([native, ocr])
        assert len(merged) == 1

    def test_keeps_genuinely_distinct_regions(self) -> None:
        boxes = [BBox(0.1, 0.1, 0.4, 0.2), BBox(0.6, 0.6, 0.9, 0.8)]
        assert len(merge_overlapping(boxes)) == 2

    def test_empty_input(self) -> None:
        assert merge_overlapping([]) == []
