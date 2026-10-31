"""Multi-hop context provision: frame each hop against the whole evidence set.

The problem this fixes
----------------------
Dependency context provisions framing *within* a document. Measured across 2,340
queries, that works everywhere except multi-hop, which has both the lowest
baseline context-provision rate (qar 0.2024 against 0.3654 overall) and the
smallest gain (+0.1881 against +0.2855 overall).

The reason is structural rather than incidental. A multi-hop query's gold set is
the *intersection* of two descriptor sets: documents carrying both A and B, plus
the documents carrying either, as acceptable supporting evidence. The evidence is
therefore spread across several papers by construction, so "the framing of the
document this passage came from" supplies less of what is missing. Provisioning
inside one document cannot fix a question that needs two.

What this does
--------------
Two changes, both cheap and both reusing existing structure:

1. **Cross-hop framing.** Every retrieved passage may be framed by any retrieved
   passage, not only by a sibling in its own document. A Methods passage about
   population can qualify a Results passage in a *different* paper when the query
   genuinely needs both.

2. **Multi-hop-aware seeds.** Seeds are drawn from the top of *each hop's* ranking
   rather than from one merged ranking, so a hop whose best evidence ranks 15th
   overall still contributes its framing. Provisioning from a single merged prefix
   systematically under-serves the second hop.

This is deliberately more permissive than the single-document variant. That
permissiveness is the point, and it is also why the metric that evaluates it
(`readiness`) has to require the framing to be *useful* — a rate and a population
— rather than merely present. Permissive context that is useless would raise a
section-role count without raising readiness, and that difference is measurable.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from .sibling_bleed import ScoredChunk, apply_sibling_bleed, expand_with_context, sibling_bleed

log = logging.getLogger(__name__)


def _per_hop_prefix(
    hop_rankings: Sequence[Sequence[tuple[int, float]]],
    seeds_per_hop: int,
) -> list[tuple[int, float]]:
    """Merge hop rankings, taking each hop's own top-``seeds_per_hop``.

    A single merged prefix under-serves hops whose evidence ranks low overall,
    which is exactly the failure for the second hop of a two-hop question.
    """
    merged: list[tuple[int, float]] = []
    seen: set[int] = set()
    for ranking in hop_rankings:
        for row, score in ranking[:seeds_per_hop]:
            row = int(row)
            if row in seen:
                continue
            seen.add(row)
            merged.append((row, float(score)))
    return merged


def expand_with_cross_hop_context(
    hits: Sequence[tuple[int, float]],
    store: Any,
    dependency: Any,
    *,
    hop_rankings: Sequence[Sequence[tuple[int, float]]] | None = None,
    top_k: int = 10,
    seeds_per_hop: int = 3,
    total_seeds: int = 10,
    context_per_seed: int = 2,
    cross_hop: bool = True,
) -> list[tuple[int, float]]:
    """Provision framing for a multi-hop question.

    ``hop_rankings`` are the per-hop candidate lists. When supplied, seeds come
    from each hop's own prefix via :func:`_per_hop_prefix`; otherwise the merged
    ``hits`` prefix is used, which is the single-hop behaviour.

    ``cross_hop`` allows a retrieved passage to be framed by a passage from a
    different document. Set it False to recover exact single-document behaviour,
    which is what makes the two modes ablatable against each other.
    """
    if dependency is None:
        return list(hits[:top_k])

    if hop_rankings:
        seeds = _per_hop_prefix(hop_rankings, seeds_per_hop)[:total_seeds]
    else:
        seeds = [(int(row), float(score)) for row, score in hits[:total_seeds]]
    if not seeds:
        return list(hits[:top_k])

    seed_rows = [row for row, _ in seeds]
    own_docs = set()
    for row in seed_rows:
        chunk = store.get(row) if store is not None else None
        if chunk is not None:
            own_docs.add(chunk.doc_id)

    pairs = dependency.context_pairs(
        seed_rows, top_k=max(total_seeds * 3, 30), max_per_seed=context_per_seed
    )
    if not cross_hop:
        pairs = [
            (a, b, w) for a, b, w in pairs
            if _doc_of(store, b) in own_docs
        ]

    seen = {int(row) for row, _ in hits}
    merged: list[tuple[int, float]] = list(hits[:top_k])
    added = 0
    for _, ctx_row, _w in pairs:
        if added >= total_seeds:
            break
        ctx = int(ctx_row)
        if ctx in seen:
            continue
        seen.add(ctx)
        merged.append((ctx, 0.0))
        added += 1
    return merged


def _doc_of(store: Any, row_id: int) -> str:
    chunk = store.get(int(row_id)) if store is not None else None
    return chunk.doc_id if chunk is not None else ""


def multi_hop_context_rankings(
    query: str,
    store: Any,
    dependency: Any,
    retrieve_for: Any,
    sub_queries: Sequence[str],
    *,
    top_k: int = 10,
    seeds_per_hop: int = 3,
    total_seeds: int = 10,
    context_per_seed: int = 2,
    cross_hop: bool = True,
    base_rankings: Sequence[Sequence[tuple[int, float]]] | None = None,
) -> tuple[list[tuple[int, float]], dict[str, Any]]:
    """Retrieve per hop, bleed within documents, then provision framing across hops.

    ``retrieve_for(text) -> list[(row, score)]`` is the caller's search function, so
    this works with sparse, dense or hybrid retrieval without importing any of them.

    ``base_rankings`` supplies the ranking whose top ``top_k`` forms the returned
    result set. **This must be passed**, and it must be the ordinary merged
    ranking, because per-hop prefixes are for *choosing seeds*, not for choosing
    results.

    A first version omitted it and rebuilt the result set from
    ``_per_hop_prefix(bled, 3)``, which for a two-hop query yields six rows rather
    than ten and drops the strongest single-hop hits. Measured, that cost more
    than the cross-hop framing gained: on the multi-hop family, readiness fell
    from 0.2881 to 0.1357 against the single-document provisioner, a larger loss
    than the 0.1738 baseline the change was meant to improve on.
    """
    hops = [retrieve_for(q) for q in ([query] + [s for s in sub_queries if s and s != query])]

    # Sibling bleed within each hop's own ranking, so a hop's framing is chosen by
    # that hop's evidence rather than by the merged ranking.
    bled: list[list[tuple[int, float]]] = []
    for ranking in hops:
        bled.append(apply_sibling_bleed(ranking, store, alpha=0.3))

    # Results come from the ordinary merged ranking; per-hop prefixes only seed.
    if base_rankings is None:
        raise ValueError(
            "base_rankings is required: per-hop prefixes select seeds, not results. "
            "Rebuilding the result set from them drops the strongest single-hop hits."
        )
    merged_all: list[tuple[int, float]] = []
    seen_rows: set[int] = set()
    for ranking in base_rankings:
        for row, score in ranking:
            row = int(row)
            if row in seen_rows:
                continue
            seen_rows.add(row)
            merged_all.append((row, float(score)))
    base = merged_all[:top_k]

    merged = expand_with_cross_hop_context(
        base, store, dependency,
        hop_rankings=bled,
        top_k=top_k,
        seeds_per_hop=seeds_per_hop,
        total_seeds=total_seeds,
        context_per_seed=context_per_seed,
        cross_hop=cross_hop,
    )
    info = {
        "n_hops": len(hops),
        "n_seeds": min(total_seeds, len(base)),
        "n_results": len(base),
        "n_returned": len(merged),
        "cross_hop": cross_hop,
    }
    return merged, info