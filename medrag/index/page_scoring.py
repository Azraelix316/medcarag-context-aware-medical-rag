"""Page-level scoring: condition a passage on its document's importance.

What this is
------------
The objective lists page-level vectors stored alongside chunk vectors as the first
component of the contextual-contamination system, after Anthropic Contextual
Retrieval and LangChain Contextual Embeddings -- both of which *rewrite* a chunk's
text with its document context and therefore require re-embedding the corpus. The
cost of that approach is the reason it is cited rather than claimed: re-embedding
16.1M chunks costs hours.

This does the same job at query time from vectors that already exist. A chunk's
score is its own text score, discounted by how well the *document* as a whole matches
the query. A passage in a marginally relevant paper is demoted; a passage in a
strongly on-topic paper is promoted. That is the "page" half of Anthropic's framing
without paying to rewrite and re-embed anything.

How it differs from sibling bleed
---------------------------------
Sibling bleed conditions a chunk on the importance of its *siblings*: it is a
within-document signal that says "other chunks of this document also match". Page
scoring conditions on the *document's own vector*: it says "this document as a whole
matches". They are different signals and the ablation keeps them separate --
``bleedctx`` applies both, ``page`` applies only this, and ``pagectx`` only the
combination of page and dependency expansion.

Why the discounting form
------------------------
``score * (1 + alpha * page_score)`` with page_score in [0, 1] can only promote.
A bare multiplicative ``score * page_score`` would let a well-matched document
resurrect a passage whose own text does not match at all, which is the opposite of
the intent: the document should modulate the passage, not override it. The additive
form has the property that at ``alpha=0`` it is exactly the base ranking, so the arm
degrades to ``sparse`` and the ablation measures the mechanism rather than a change
of candidate set.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PageConfig:
    alpha: float = 0.5
    """Weight of the page signal. 0.0 reproduces the base ranking exactly."""


def apply_page_scores(
    hits: Sequence[tuple[int, float]],
    page_scores: dict[str, float],
    doc_of_row: Callable[[int], str | None],
    cfg: PageConfig | None = None,
) -> list[tuple[int, float]]:
    """Re-score chunk hits by the page score of the document each belongs to.

    ``page_scores`` maps doc id -> similarity in [0, 1]. ``doc_of_row`` maps a chunk
    row id to its doc id. Hits whose document is absent from ``page_scores`` keep
    their base score unchanged rather than being dropped: a missing page vector means
    "no information", not "irrelevant", and dropping would silently shrink the
    candidate set and confound the ablation.
    """
    cfg = cfg or PageConfig()
    if cfg.alpha == 0.0 or not page_scores:
        return list(hits)
    out: list[tuple[int, float]] = []
    for row, score in hits:
        doc = doc_of_row(row)
        ps = page_scores.get(doc) if doc else None
        out.append((row, score * (1.0 + cfg.alpha * ps) if ps is not None else score))
    # Re-sort. This was missing, and it is the difference between a scoring function
    # and a ranking function: without it the scores are recomputed correctly and the
    # ORDER is untouched, so the `page` arm returns byte-identical rankings to the
    # baseline on 400 of 400 queries while reporting different score magnitudes.
    # Nothing raises, no value looks wrong, and the arm is inert.
    #
    # Sorted by (-score, row) so that equal scores keep a deterministic order rather
    # than depending on dict or set iteration, which would make a rerun disagree with
    # the previous one.
    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out


def normalise(scores: Sequence[float]) -> list[float]:
    """Scale a score list into [0, 1] so the discount is bounded.

    Inner-product similarities from a normalised embedding space can be negative and
    are not bounded by 1, so clamping is not enough -- an unbounded factor would let
    one document's score reorder the entire list by an arbitrary amount.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]