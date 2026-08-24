"""Reciprocal Rank Fusion for combining the dense and lexical result lists.

Dense scores are cosines, lexical scores are `ts_rank_cd` values. They are not on
the same scale, not even the same *kind* of scale, and normalising them against
each other requires per-query calibration that drifts the moment the corpus or
the embedding model changes.

RRF sidesteps that: it throws away the scores and fuses on rank alone. A document
ranked 3rd by both retrievers beats one ranked 1st by a single retriever and
missing from the other, which is exactly the behaviour we want on filings — an
exact match on "Item 7A" and a semantic match on the surrounding discussion should
reinforce each other.

    score(d) = Σ  w_r / (k + rank_r(d))
              r∈R
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["FusedHit", "reciprocal_rank_fusion"]


@dataclass(slots=True)
class FusedHit:
    """One fused result, retaining where it came from.

    `contributions` is what makes the retrieval debug view useful: you can see at
    a glance whether a hit came from both retrievers or was carried by one.
    """

    doc_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def retriever_count(self) -> int:
        return len(self.ranks)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[FusedHit]:
    """Fuse several ranked ID lists into one.

    Args:
        rankings: retriever name -> IDs, best first. Duplicates within one list
            are ignored after their first (best) occurrence.
        k: damping constant. Larger flattens the contribution curve, so deep
            results matter more relative to the top few. 60 is the value from
            the original TREC work and a sane default.
        weights: optional per-retriever multiplier; missing entries default to 1.
        limit: truncate the fused output.

    Returns:
        Hits sorted by score descending. Ties break toward the document more
        retrievers agreed on, then by ID for determinism — without that last
        clause the eval harness produces different numbers run to run.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    weights = weights or {}
    hits: dict[str, FusedHit] = {}

    for retriever, doc_ids in rankings.items():
        weight = weights.get(retriever, 1.0)
        if weight == 0.0:
            continue
        seen: set[str] = set()
        for position, doc_id in enumerate(doc_ids, start=1):
            if doc_id in seen:
                continue  # keep only the best rank a retriever gave it
            seen.add(doc_id)

            contribution = weight / (k + position)
            hit = hits.get(doc_id)
            if hit is None:
                hit = FusedHit(doc_id=doc_id, score=0.0)
                hits[doc_id] = hit
            hit.score += contribution
            hit.ranks[retriever] = position
            hit.contributions[retriever] = contribution

    ordered = sorted(
        hits.values(),
        key=lambda h: (-h.score, -h.retriever_count, h.doc_id),
    )
    return ordered[:limit] if limit is not None else ordered
