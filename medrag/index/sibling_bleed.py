"""Sibling-bleed scoring and dependency-context expansion as retrieval signals.

Two distinct mechanisms for the same underlying problem: a passage is not
independently interpretable, so a single retrieved chunk understates what the
document has to say.

Sibling bleed
-------------
A chunk's relevance is conditioned on how relevant its *document's other chunks*
are. If a query matches five separate sections of one paper, that paper is
demonstrably on-topic, and a sixth chunk from it deserves more credit than the
same chunk would if it came from a paper the query touched only once.

This is distinct from contextual retrieval (Anthropic 2024; LangChain
Contextual Embeddings), which injects document context into the *text* before
embedding and therefore requires re-embedding the corpus. Sibling bleed is a
post-hoc re-scoring of an existing index: no re-embedding, no index growth, and it
can be switched off at query time. That is the practical argument for it and also
why it is worth measuring separately.

Formally, with base scores s(c) over a candidate set C:

    bleed(c) = max_{c' in C, doc(c') = doc(c), c' != c} s(c')

The maximum is used rather than the sum so a document with many mediocre matches
does not outrank a document with one excellent match purely for having more
chunks. A sum would reward chunk count, which is a property of the chunker, not
of relevance.

Dependency context
-------------------
Wraps any base ranking so each returned passage is accompanied by the passage
that interprets it, drawn from the section-dependency graph. This is the
retrieval shape the decontextualization failure calls for.

The honest caveat, established by measurement in docs/context_provision.md: most
of the context-provision gain comes from adding *any* second chunk of an already
relevant document, not from the dependency structure specifically. About a quarter
of the gain survives a random same-document control. So the value here is in
making the framing present, and the structure is a modest refinement on top.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ScoredChunk:
    row_id: int
    score: float
    doc_id: str
    section_kind: str = ""
    bleed: float = 0.0


def sibling_bleed(
    candidates: Sequence[ScoredChunk],
    *,
    alpha: float = 0.3,
    mode: str = "max",
) -> list[ScoredChunk]:
    """Re-score candidates by the best sibling in the same document.

    ``alpha`` blends the bleed into the base score:

        s'(c) = (1 - alpha) * s(c) + alpha * bleed(c)

    alpha=0 recovers the input exactly, so a sweep over alpha measures the
    mechanism's contribution rather than a different retriever's.

    ``mode`` is ``max`` (default) or ``sum``. Sum is provided only so the choice is
    testable; it rewards documents with more matching chunks and is expected to
    perform worse, which the tests assert.
    """
    if not candidates or alpha == 0.0:
        return list(candidates)

    # Normalise base scores before bleeding so alpha has a stable meaning across
    # arms whose raw scores live on different scales (BM25 is unbounded, cosine
    # is in [-1,1]).
    raw = np.asarray([c.score for c in candidates], dtype=np.float64)
    lo, hi = float(raw.min()), float(raw.max())
    span = hi - lo
    norm = (raw - lo) / span if span > 0 else np.zeros_like(raw)

    by_doc: dict[str, list[int]] = {}
    for i, c in enumerate(candidates):
        by_doc.setdefault(c.doc_id, []).append(i)

    bleed = np.zeros(len(candidates), dtype=np.float64)
    for members in by_doc.values():
        if len(members) < 2:
            continue
        vals = norm[members]
        for i in members:
            others = [v for j, v in zip(members, vals) if j != i]
            if not others:
                continue
            bleed[i] = max(others) if mode == "max" else sum(others)

    out: list[ScoredChunk] = []
    for i, c in enumerate(candidates):
        out.append(
            ScoredChunk(
                row_id=c.row_id,
                score=float((1.0 - alpha) * norm[i] + alpha * bleed[i]),
                doc_id=c.doc_id,
                section_kind=c.section_kind,
                bleed=float(bleed[i]),
            )
        )
    out.sort(key=lambda c: (-c.score, c.row_id))
    return out


def apply_sibling_bleed(
    hits: Sequence[tuple[int, float]],
    store: Any,
    *,
    alpha: float = 0.3,
    mode: str = "max",
) -> list[tuple[int, float]]:
    """Convenience wrapper over ``(row_id, score)`` pairs and a chunk store."""
    cands: list[ScoredChunk] = []
    for row, score in hits:
        chunk = store.get(int(row))
        cands.append(
            ScoredChunk(
                row_id=int(row),
                score=float(score),
                doc_id=(chunk.doc_id if chunk else ""),
                section_kind=(chunk.section_kind if chunk else ""),
            )
        )
    return [(c.row_id, c.score) for c in sibling_bleed(cands, alpha=alpha, mode=mode)]


def expand_with_context(
    hits: Sequence[tuple[int, float]],
    store: Any,
    dependency: Any,
    *,
    top_k: int = 10,
    seed_count: int = 5,
    context_per_seed: int = 2,
    context_budget: int | None = None,
) -> list[tuple[int, float]]:
    """Return the base ranking plus the interpreting passages of its top seeds.

    Returns ``top_k`` results followed by up to ``context_budget`` framing
    passages, so the list is longer than ``top_k`` by design: the point is to hand
    the generator the interpreting text alongside the finding, and truncating back
    to ``top_k`` would discard exactly what was added.

    Seeds are the top ``seed_count`` hits; their dependency neighbours supply the
    context. Seeding from a prefix rather than the whole candidate pool keeps the
    context tied to the highest-confidence passages, so the added text qualifies
    something actually retrieved instead of being topically adjacent.

    ``context_budget`` defaults to ``top_k``. It must be bounded: one seed in a
    40-chunk Results section can have many equally-weighted neighbours, and an
    unbounded walk turns a 10-result answer into a 60-chunk dump.
    """
    if dependency is None:
        return list(hits[:top_k])
    budget = context_budget if context_budget is not None else top_k
    seeds = [int(row) for row, _ in hits[:seed_count]]
    if not seeds or budget <= 0:
        return list(hits[:top_k])
    pairs = dependency.context_pairs(
        seeds, top_k=max(budget, seed_count), max_per_seed=context_per_seed
    )

    seen = {int(row) for row, _ in hits}
    merged: list[tuple[int, float]] = list(hits[:top_k])
    added = 0
    # Context scores are dependency weights, not comparable to retrieval scores,
    # so context is appended rather than sorted in. Order is retrieval order first,
    # then framing passages -- a caller can therefore split the two by position.
    for _, ctx_row, _w in pairs:
        if added >= budget:
            break
        ctx = int(ctx_row)
        if ctx in seen:
            continue
        seen.add(ctx)
        merged.append((ctx, 0.0))
        added += 1
    return merged


def bleed_report(
    before: Sequence[tuple[int, float]],
    after: Sequence[tuple[int, float]],
    store: Any,
) -> dict[str, Any]:
    """How much sibling bleed actually reordered the candidate set."""
    rank_before = {row: i for i, (row, _) in enumerate(before)}
    rank_after = {row: i for i, (row, _) in enumerate(after)}
    shared = set(rank_before) & set(rank_after)
    if not shared:
        return {
            "n_candidates": 0, "changed": 0, "frac_reordered": 0.0,
            "mean_abs_rank_delta": 0.0, "max_abs_rank_delta": 0,
        }
    deltas = [abs(rank_before[r] - rank_after[r]) for r in shared]
    changed = sum(1 for r in shared if rank_before[r] != rank_after[r])
    return {
        "n_candidates": len(shared),
        "changed": changed,
        "frac_reordered": round(changed / len(shared), 4),
        "mean_abs_rank_delta": round(sum(deltas) / len(deltas), 3),
        "max_abs_rank_delta": max(deltas),
    }