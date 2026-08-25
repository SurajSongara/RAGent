"""Structure recovery for documents with no layout.

The offset assertions are the important ones. Character offsets are the flow
equivalent of a bounding box — they are what a citation highlights — so an
off-by-one here puts the highlight on the wrong text, which is the exact failure
the paged bbox tests exist to prevent on the other side.
"""

from __future__ import annotations

import pytest

from ragent.ingest.flowtext import FlowBlock, split_csv, split_flow, split_markdown, split_plain

MARKDOWN = """# Annual Report

Some opening prose about the year.

## Item 7. Management Discussion

Revenue increased twelve percent year over year.

### Services

Services revenue grew faster than hardware.

## Item 3. Legal Proceedings

The litigation remains pending.
"""


class TestOffsetsAreExact:
    """Every block must slice back to itself out of the original text."""

    @pytest.mark.parametrize(
        ("text", "extension"),
        [
            (MARKDOWN, "md"),
            ("First paragraph here.\n\nSecond paragraph here.\n", "txt"),
            ("a,b\n1,2\n3,4\n", "csv"),
        ],
    )
    def test_slices_round_trip(self, text: str, extension: str) -> None:
        for block in split_flow(text, extension):
            excerpt = text[block.char_start : block.char_end]
            # CSV blocks prepend the header, so the stored text is a superset;
            # the span itself must still land on real content.
            assert excerpt.strip()
            if block.kind != "table":
                assert excerpt == block.text

    def test_spans_are_ordered_and_non_overlapping(self) -> None:
        blocks = split_markdown(MARKDOWN)
        for previous, current in zip(blocks, blocks[1:], strict=False):
            assert previous.char_end <= current.char_start

    def test_rejects_an_empty_span(self) -> None:
        with pytest.raises(ValueError, match="empty span"):
            FlowBlock(kind="paragraph", text="x", char_start=10, char_end=10)


class TestMarkdown:
    def test_headings_are_detected(self) -> None:
        kinds = [b.kind for b in split_markdown(MARKDOWN)]
        assert kinds.count("heading") == 4

    def test_section_path_nests(self) -> None:
        blocks = split_markdown(MARKDOWN)
        services = next(b for b in blocks if "Services revenue grew" in b.text)
        assert services.section_path == (
            "Annual Report",
            "Item 7. Management Discussion",
            "Services",
        )

    def test_sibling_heading_pops_the_stack(self) -> None:
        """Item 3 is a sibling of Item 7, not a child of Services."""
        blocks = split_markdown(MARKDOWN)
        legal = next(b for b in blocks if "litigation remains pending" in b.text)
        assert legal.section_path == ("Annual Report", "Item 3. Legal Proceedings")

    def test_fenced_code_is_one_atomic_block(self) -> None:
        text = "Intro line.\n\n```python\nx = 1\n\ny = 2\n```\n\nAfter.\n"
        blocks = split_markdown(text)
        fence = next(b for b in blocks if "x = 1" in b.text)
        # The blank line inside the fence must not split it.
        assert "y = 2" in fence.text

    def test_lists_group_together(self) -> None:
        text = "Intro.\n\n- alpha\n- beta\n- gamma\n\nAfter.\n"
        blocks = split_markdown(text)
        listing = next(b for b in blocks if b.kind == "list")
        assert "alpha" in listing.text and "gamma" in listing.text

    def test_markdown_tables_are_tables(self) -> None:
        text = "Intro.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
        assert any(b.kind == "table" for b in split_markdown(text))

    def test_a_table_after_a_paragraph_starts_a_new_block(self) -> None:
        text = "Some prose here.\n| a | b |\n| 1 | 2 |\n"
        kinds = [b.kind for b in split_markdown(text)]
        assert "paragraph" in kinds and "table" in kinds

    def test_setext_underline_is_not_a_list(self) -> None:
        """`---` under a heading must not be read as a bullet."""
        blocks = split_markdown("Title\n---\n\nBody text.\n")
        assert all("Body text." in b.text or b.kind != "list" for b in blocks)


class TestPlain:
    def test_blank_lines_separate_paragraphs(self) -> None:
        blocks = split_plain("One.\n\nTwo.\n\nThree.\n")
        assert len(blocks) == 3
        assert [b.kind for b in blocks] == ["paragraph"] * 3

    def test_single_paragraph(self) -> None:
        assert len(split_plain("Just one paragraph of text.")) == 1

    def test_whitespace_only(self) -> None:
        assert split_plain("   \n\n  \n") == []


class TestCsv:
    def test_header_repeats_in_every_block(self) -> None:
        """Bare data rows retrieve badly; the column names carry the meaning."""
        rows = "\n".join(f"{i},value{i}" for i in range(100))
        blocks = split_csv(f"id,label\n{rows}\n", rows_per_block=40)
        assert len(blocks) == 3
        for block in blocks:
            assert block.text.startswith("id,label")

    def test_rows_are_grouped(self) -> None:
        blocks = split_csv("a,b\n1,2\n3,4\n5,6\n", rows_per_block=2)
        assert len(blocks) == 2

    def test_all_blocks_are_tables(self) -> None:
        blocks = split_csv("a,b\n1,2\n")
        assert all(b.kind == "table" for b in blocks)

    def test_tsv_delimiter(self) -> None:
        blocks = split_csv("a\tb\n1\t2\n", delimiter="\t")
        assert blocks and blocks[0].text.startswith("a\tb")

    def test_header_only_file_yields_nothing(self) -> None:
        assert split_csv("a,b,c\n") == []

    def test_blank_rows_are_skipped(self) -> None:
        blocks = split_csv("a,b\n1,2\n\n3,4\n", rows_per_block=10)
        assert len(blocks) == 1
        assert "1,2" in blocks[0].text and "3,4" in blocks[0].text


class TestDispatch:
    @pytest.mark.parametrize(
        ("extension", "text", "expected_kind"),
        [
            ("md", "# Title\n\nBody.\n", "heading"),
            ("markdown", "# Title\n\nBody.\n", "heading"),
            ("txt", "Body.\n", "paragraph"),
            ("csv", "a,b\n1,2\n", "table"),
            ("tsv", "a\tb\n1\t2\n", "table"),
        ],
    )
    def test_routes_by_extension(self, extension: str, text: str, expected_kind: str) -> None:
        kinds = {b.kind for b in split_flow(text, extension)}
        assert expected_kind in kinds

    def test_unknown_extension_falls_back_to_plain(self) -> None:
        blocks = split_flow("Some text here.", "rst")
        assert [b.kind for b in blocks] == ["paragraph"]

    def test_empty_input(self) -> None:
        assert split_flow("", "md") == []
        assert split_flow("   \n  ", "txt") == []
