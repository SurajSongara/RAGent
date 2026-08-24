"""Chunking strategies.

The parametrised provenance tests are the important ones. Individual strategies
can be tuned or replaced, but a chunk that cannot name the blocks it came from
breaks citations everywhere downstream, so that invariant is asserted against
every strategy rather than trusted per-implementation.
"""

from __future__ import annotations

import pytest

from ragent.ingest.chunking import (
    STRATEGIES,
    chunk_document,
    chunk_fixed,
    chunk_layout,
    chunk_recursive,
    chunk_semantic,
)
from ragent.ingest.types import SourceBlock, approx_tokens

from .conftest import fake_embedder, make_block

ALL_STRATEGIES = ["fixed", "recursive", "layout", "semantic"]


def run(strategy: str, blocks: list[SourceBlock], **kwargs: object):
    """Dispatch, injecting the fake embedder only where it is needed."""
    if strategy == "semantic":
        kwargs["embed"] = fake_embedder()
    return chunk_document(strategy, blocks, **kwargs)


# ---------------------------------------------------------------- invariants


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
class TestProvenanceInvariants:
    """Hold for every strategy, including those that split through blocks."""

    def test_every_chunk_names_its_blocks(
        self, strategy: str, filing_blocks: list[SourceBlock]
    ) -> None:
        chunks = run(strategy, filing_blocks)
        assert chunks
        for chunk in chunks:
            assert chunk.block_ids, f"{strategy} produced a chunk with no provenance"

    def test_block_ids_are_real(self, strategy: str, filing_blocks: list[SourceBlock]) -> None:
        known = {b.id for b in filing_blocks}
        for chunk in run(strategy, filing_blocks):
            assert set(chunk.block_ids) <= known

    def test_all_content_blocks_are_covered(
        self, strategy: str, filing_blocks: list[SourceBlock]
    ) -> None:
        """No content may be silently dropped on the floor."""
        expected = {b.id for b in filing_blocks if not b.is_furniture and b.text.strip()}
        covered = {bid for chunk in run(strategy, filing_blocks) for bid in chunk.block_ids}
        assert expected == covered

    def test_page_furniture_is_excluded(
        self, strategy: str, filing_blocks: list[SourceBlock]
    ) -> None:
        """Running headers would otherwise appear in every chunk and swamp BM25."""
        furniture = {b.id for b in filing_blocks if b.is_furniture}
        covered = {bid for chunk in run(strategy, filing_blocks) for bid in chunk.block_ids}
        assert not (furniture & covered)

    def test_sequence_numbers_are_dense_and_ordered(
        self, strategy: str, filing_blocks: list[SourceBlock]
    ) -> None:
        chunks = run(strategy, filing_blocks)
        assert [c.seq for c in chunks] == list(range(len(chunks)))

    def test_page_range_is_coherent(self, strategy: str, filing_blocks: list[SourceBlock]) -> None:
        for chunk in run(strategy, filing_blocks):
            assert chunk.page_from <= chunk.page_to

    def test_no_empty_text(self, strategy: str, filing_blocks: list[SourceBlock]) -> None:
        for chunk in run(strategy, filing_blocks):
            assert chunk.text.strip()

    def test_empty_input(self, strategy: str) -> None:
        assert run(strategy, []) == []

    def test_furniture_only_input(self, strategy: str) -> None:
        blocks = [make_block("ANNUAL REPORT", kind="header"), make_block("7", kind="page_number")]
        assert run(strategy, blocks) == []

    def test_strategy_is_recorded(self, strategy: str, filing_blocks: list[SourceBlock]) -> None:
        for chunk in run(strategy, filing_blocks):
            assert chunk.strategy == strategy


# ---------------------------------------------------------------- layout


class TestLayout:
    def test_table_is_never_split_or_merged(self, filing_blocks: list[SourceBlock]) -> None:
        """A table cut in half yields headers without rows and rows without headers."""
        table = next(b for b in filing_blocks if b.kind == "table")
        chunks = chunk_layout(filing_blocks)
        owning = [c for c in chunks if table.id in c.block_ids]
        assert len(owning) == 1
        assert owning[0].block_ids == [table.id]
        assert owning[0].meta.get("atomic") == "table"

    def test_heading_starts_a_new_chunk(self, filing_blocks: list[SourceBlock]) -> None:
        heading = next(b for b in filing_blocks if b.text.startswith("Item 7"))
        chunks = chunk_layout(filing_blocks)
        owning = next(c for c in chunks if heading.id in c.block_ids)
        assert owning.block_ids[0] == heading.id

    def test_section_change_forces_a_break(self, filing_blocks: list[SourceBlock]) -> None:
        """Item 7 prose and Item 3 prose must never share a chunk."""
        chunks = chunk_layout(filing_blocks)
        for chunk in chunks:
            sections = {
                b.section_path for b in filing_blocks if b.id in chunk.block_ids and b.section_path
            }
            assert len(sections) <= 1

    def test_section_path_is_carried(self, filing_blocks: list[SourceBlock]) -> None:
        chunks = chunk_layout(filing_blocks)
        assert any(c.section_path == ("Part II", "Item 7") for c in chunks)
        assert any(c.section_path == ("Part I", "Item 3") for c in chunks)

    def test_small_target_forces_more_chunks(self, filing_blocks: list[SourceBlock]) -> None:
        few = chunk_layout(filing_blocks, target_tokens=512)
        many = chunk_layout(filing_blocks, target_tokens=12)
        assert len(many) > len(few)


# ---------------------------------------------------------------- fixed


class TestFixed:
    def test_respects_the_token_budget(self) -> None:
        blocks = [make_block(" ".join(f"word{i}" for i in range(400)))]
        for chunk in chunk_fixed(blocks, target_tokens=64, overlap_tokens=0):
            assert chunk.token_count <= 64

    def test_overlap_repeats_content_across_the_seam(self) -> None:
        """Assert the actual property: adjacent chunks share trailing words."""
        blocks = [make_block(" ".join(f"word{i}" for i in range(200)))]

        with_overlap = chunk_fixed(blocks, target_tokens=64, overlap_tokens=16)
        assert len(with_overlap) >= 2
        shared = set(with_overlap[0].text.split()) & set(with_overlap[1].text.split())
        assert shared, "expected the window to step back and repeat content"

        without = chunk_fixed(blocks, target_tokens=64, overlap_tokens=0)
        assert len(without) >= 2
        assert not (set(without[0].text.split()) & set(without[1].text.split()))

    def test_terminates_on_a_single_oversized_token(self) -> None:
        """A word longer than the whole budget must not spin forever."""
        blocks = [make_block("x" * 500)]
        chunks = chunk_fixed(blocks, target_tokens=2, overlap_tokens=1)
        assert len(chunks) == 1

    def test_splits_through_blocks_but_keeps_provenance(self) -> None:
        """A window that straddles two blocks must claim both.

        This is the case the whole provenance design exists for: `fixed` ignores
        layout, so its boundaries fall mid-block, and the citation still has to
        resolve. One-token-per-word keeps the seam arithmetic exact.
        """
        blocks = [
            make_block(" ".join(f"alpha{i}" for i in range(50))),
            make_block(" ".join(f"beta{i}" for i in range(50))),
        ]
        chunks = chunk_fixed(
            blocks,
            target_tokens=30,
            overlap_tokens=0,
            count_tokens=lambda text: len(text.split()),
        )
        spanning = [c for c in chunks if len(c.block_ids) > 1]
        assert spanning, "expected a window straddling the block boundary"
        straddler = spanning[0]
        assert set(straddler.block_ids) == {blocks[0].id, blocks[1].id}
        assert "alpha49" in straddler.text and "beta0" in straddler.text


# ---------------------------------------------------------------- recursive


class TestRecursive:
    def test_prefers_paragraph_boundaries(self) -> None:
        blocks = [make_block(f"Paragraph number {i} of the filing text." * 3) for i in range(6)]
        chunks = chunk_recursive(blocks, target_tokens=40, overlap_tokens=0)
        assert len(chunks) > 1

    def test_repacks_fragments_toward_the_target(self) -> None:
        """Naive recursive splitting leaves many tiny chunks; these should be merged."""
        blocks = [make_block(". ".join(f"Sentence {i} here" for i in range(40)) + ".")]
        chunks = chunk_recursive(blocks, target_tokens=100, overlap_tokens=0)
        assert chunks
        body = chunks[:-1]  # the tail is legitimately short
        if body:
            assert sum(c.token_count for c in body) / len(body) > 30

    def test_handles_text_with_no_separators(self) -> None:
        chunks = chunk_recursive([make_block("x" * 300)], target_tokens=10)
        assert len(chunks) == 1


# ---------------------------------------------------------------- semantic


class TestSemantic:
    def test_cuts_at_the_topic_shift(self) -> None:
        blocks = [
            make_block("Revenue grew twelve percent. Revenue from services was strong."),
            make_block("Litigation continues today. Litigation costs rose materially."),
        ]
        chunks = chunk_semantic(blocks, embed=fake_embedder())
        assert len(chunks) == 2
        assert "Revenue" in chunks[0].text and "Litigation" not in chunks[0].text
        assert "Litigation" in chunks[1].text

    def test_cuts_inside_a_single_block_and_keeps_provenance(self) -> None:
        """The boundary need not align with a block; the citation still resolves."""
        block = make_block(
            "Revenue grew twelve percent. Revenue was strong. "
            "Litigation continues today. Litigation costs rose."
        )
        chunks = chunk_semantic([block], embed=fake_embedder())
        assert len(chunks) == 2
        for chunk in chunks:
            assert chunk.block_ids == [block.id]

    def test_uniform_topic_is_not_shredded(self) -> None:
        """With no topic shift a percentile cut must not fire at every sentence."""
        block = make_block(" ".join(f"Revenue rose in period {i}." for i in range(8)))
        chunks = chunk_semantic([block], embed=fake_embedder(), target_tokens=512)
        assert len(chunks) == 1

    def test_budget_still_applies(self) -> None:
        block = make_block(" ".join(f"Revenue rose in period {i}." for i in range(30)))
        chunks = chunk_semantic([block], embed=fake_embedder(), target_tokens=20)
        assert len(chunks) > 1

    def test_records_the_cut_similarity(self) -> None:
        blocks = [
            make_block("Revenue grew twelve percent. Revenue from services was strong."),
            make_block("Litigation continues today. Litigation costs rose materially."),
        ]
        chunks = chunk_semantic(blocks, embed=fake_embedder())
        assert "cut_similarity" in chunks[0].meta

    def test_rejects_a_mismatched_embedder(self) -> None:
        block = make_block("Revenue grew. Litigation continues. Costs rose again.")
        with pytest.raises(ValueError, match="vectors for"):
            chunk_semantic([block], embed=lambda sentences: [[1.0, 0.0]])


# ---------------------------------------------------------------- dispatch


class TestDispatch:
    def test_registry_matches_documented_strategies(self) -> None:
        assert sorted(STRATEGIES) == sorted(ALL_STRATEGIES)

    def test_unknown_strategy_names_the_valid_ones(self) -> None:
        with pytest.raises(ValueError, match="unknown chunking strategy"):
            chunk_document("magic", [make_block("hello world")])


# ---------------------------------------------------------------- token counter


class TestTokenCounter:
    def test_blank_text_is_zero(self) -> None:
        assert approx_tokens("   \n ") == 0

    def test_monotonic_in_length(self) -> None:
        assert approx_tokens("one two three") > approx_tokens("one")

    def test_is_injectable(self) -> None:
        """The bake-off must compare strategies, not tokeniser noise."""
        calls: list[str] = []

        def counter(text: str) -> int:
            calls.append(text)
            return len(text.split())

        chunk_layout([make_block("alpha beta gamma")], count_tokens=counter)
        assert calls
