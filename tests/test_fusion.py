"""Reciprocal Rank Fusion.

The determinism tests are not padding: the eval harness compares retrieval
scores across runs, so any tie-breaking left to dict ordering would show up as
phantom regressions in the Phase 2 numbers.
"""

from __future__ import annotations

import pytest

from ragent.retrieval.fusion import reciprocal_rank_fusion as rrf


class TestBasics:
    def test_single_retriever_preserves_order(self) -> None:
        hits = rrf({"dense": ["a", "b", "c"]})
        assert [h.doc_id for h in hits] == ["a", "b", "c"]

    def test_scores_follow_the_formula(self) -> None:
        hits = rrf({"dense": ["a", "b"]}, k=60)
        assert hits[0].score == pytest.approx(1 / 61)
        assert hits[1].score == pytest.approx(1 / 62)

    def test_empty_input(self) -> None:
        assert rrf({}) == []

    def test_empty_ranking_lists(self) -> None:
        assert rrf({"dense": [], "lexical": []}) == []

    def test_limit_truncates(self) -> None:
        assert len(rrf({"dense": list("abcdef")}, limit=3)) == 3


class TestAgreement:
    def test_agreement_beats_a_single_strong_hit(self) -> None:
        """The core reason to use RRF at all."""
        hits = rrf(
            {
                "dense": ["agreed", "solo_dense", "x"],
                "lexical": ["solo_lexical", "agreed", "y"],
            }
        )
        assert hits[0].doc_id == "agreed"

    def test_records_where_each_hit_came_from(self) -> None:
        hits = rrf({"dense": ["a", "b"], "lexical": ["b", "a"]})
        by_id = {h.doc_id: h for h in hits}
        assert by_id["a"].ranks == {"dense": 1, "lexical": 2}
        assert by_id["b"].ranks == {"dense": 2, "lexical": 1}
        assert by_id["a"].retriever_count == 2

    def test_contributions_sum_to_score(self) -> None:
        for hit in rrf({"dense": ["a", "b"], "lexical": ["b", "a"]}):
            assert sum(hit.contributions.values()) == pytest.approx(hit.score)

    def test_document_found_by_one_retriever_still_ranks(self) -> None:
        hits = rrf({"dense": ["a"], "lexical": ["b"]})
        assert {h.doc_id for h in hits} == {"a", "b"}


class TestWeighting:
    def test_weight_shifts_the_winner(self) -> None:
        rankings = {"dense": ["d"], "lexical": ["l"]}
        assert rrf(rankings, weights={"lexical": 5.0})[0].doc_id == "l"
        assert rrf(rankings, weights={"dense": 5.0})[0].doc_id == "d"

    def test_zero_weight_drops_a_retriever_entirely(self) -> None:
        hits = rrf({"dense": ["a"], "lexical": ["b"]}, weights={"lexical": 0.0})
        assert [h.doc_id for h in hits] == ["a"]

    def test_missing_weight_defaults_to_one(self) -> None:
        assert rrf({"dense": ["a"]}, weights={}) == rrf({"dense": ["a"]})


class TestDamping:
    def test_larger_k_flattens_the_curve(self) -> None:
        """Higher k narrows the gap between rank 1 and rank 10."""
        docs = [f"d{i}" for i in range(10)]
        tight = rrf({"r": docs}, k=1)
        loose = rrf({"r": docs}, k=1000)
        assert tight[0].score / tight[-1].score > loose[0].score / loose[-1].score

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            rrf({"dense": ["a"]}, k=0)


class TestDeterminism:
    def test_duplicate_ids_use_the_best_rank_only(self) -> None:
        with_dupe = rrf({"dense": ["a", "b", "a"]})
        without = rrf({"dense": ["a", "b"]})
        assert [h.doc_id for h in with_dupe] == [h.doc_id for h in without]
        assert with_dupe[0].score == pytest.approx(without[0].score)

    def test_ties_break_deterministically(self) -> None:
        """Same inputs, different insertion order, identical output."""
        a = rrf({"r1": ["x"], "r2": ["y"]})
        b = rrf({"r2": ["y"], "r1": ["x"]})
        assert [h.doc_id for h in a] == [h.doc_id for h in b]

    def test_tie_prefers_broader_agreement(self) -> None:
        hits = rrf({"dense": ["both", "solo"], "lexical": ["both"]})
        assert hits[0].doc_id == "both"
        assert hits[0].retriever_count == 2

    def test_scores_are_non_increasing(self) -> None:
        hits = rrf({"dense": list("abcdef"), "lexical": list("fedcba")})
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
