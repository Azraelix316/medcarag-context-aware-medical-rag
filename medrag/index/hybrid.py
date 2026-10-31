"""Hybrid fusion of dense and sparse retrieval.

The two retrievers fail differently: BM25 misses paraphrase ("heart failure"
vs "cardiac decompensation") while dense embeddings miss rare exact tokens
(drug names, dosages, gene symbols). Fusing their rankings recovers both.
Reciprocal Rank Fusion is the default because it needs no score calibration,
which matters here since BM25 scores are unbounded while cosine lives in
[-1, 1].
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    """One retrieved chunk with its per-retriever provenance."""

    chunk_id: str
    score: float = 0.0
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    # Which retrievers surfaced this candidate; used for ablations.
    sources: list[str] = field(default_factory=list)
    rerank_score: float | None = None

    @property
    def fused_score(self) -> float:
        return self.score


def rrf_fuse(
    ranked_lists: dict[str, Sequence[tuple[str, float]]],
    *,
    k: int = 60,
    top_k: int = 100,
    weights: dict[str, float] | None = None,
) -> list[Candidate]:
    """Reciprocal Rank Fusion over named ranked lists.

    ``score(d) = sum_r w_r / (k + rank_r(d))`` with 1-based ranks.
    """
    weights = weights or {name: 1.0 for name in ranked_lists}
    acc: dict[str, Candidate] = {}

    for name, ranked in ranked_lists.items():
        w = weights.get(name, 1.0)
        for rank, (chunk_id, raw_score) in enumerate(ranked, start=1):
            cand = acc.get(chunk_id)
            if cand is None:
                cand = Candidate(chunk_id=chunk_id)
                acc[chunk_id] = cand
            cand.score += w / (k + rank)
            if name == "dense":
                cand.dense_score = raw_score
                cand.dense_rank = rank
            elif name == "sparse":
                cand.sparse_score = raw_score
                cand.sparse_rank = rank
            if name not in cand.sources:
                cand.sources.append(name)

    out = sorted(acc.values(), key=lambda c: (-c.score, c.chunk_id))
    return out[:top_k]


def weighted_fuse(
    dense: Sequence[tuple[str, float]],
    sparse: Sequence[tuple[str, float]],
    *,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    top_k: int = 100,
) -> list[Candidate]:
    """Score-level blend after per-retriever min-max normalisation.

    Min-max normalisation is required because BM25 and cosine scores are not
    comparable without it.
    """
    def normalize(ranked: Sequence[tuple[str, float]]) -> dict[str, float]:
        if not ranked:
            return {}
        scores = [s for _cid, s in ranked]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        return {cid: (s - lo) / span for cid, s in ranked}

    d_norm, s_norm = normalize(dense), normalize(sparse)
    acc: dict[str, Candidate] = {}

    for name, ranked, norms, weight, key in (
        ("dense", dense, d_norm, dense_weight, "dense"),
        ("sparse", sparse, s_norm, sparse_weight, "sparse"),
    ):
        for rank, (chunk_id, raw) in enumerate(ranked, start=1):
            cand = acc.setdefault(chunk_id, Candidate(chunk_id=chunk_id))
            cand.score += weight * norms.get(chunk_id, 0.0)
            setattr(cand, f"{key}_score", raw)
            setattr(cand, f"{key}_rank", rank)
            if name not in cand.sources:
                cand.sources.append(name)

    out = sorted(acc.values(), key=lambda c: (-c.score, c.chunk_id))
    return out[:top_k]


def fuse(
    dense: Sequence[tuple[str, float]],
    sparse: Sequence[tuple[str, float]],
    *,
    method: str = "rrf",
    top_k: int = 100,
    rrf_k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> list[Candidate]:
    """Dispatch to the configured fusion method."""
    if method == "rrf":
        return rrf_fuse(
            {"dense": list(dense), "sparse": list(sparse)},
            k=rrf_k, top_k=top_k,
            weights={"dense": 1.0, "sparse": 1.0},
        )
    if method == "weighted":
        return weighted_fuse(
            dense, sparse, dense_weight=dense_weight,
            sparse_weight=sparse_weight, top_k=top_k,
        )
    raise ValueError(f"unknown fusion method: {method}")


def candidate_stats(candidates: Sequence[Candidate]) -> dict[str, Any]:
    """Summarise overlap between the two retrievers (diagnostics for the paper).

    Note the denominator: this is the whole fused candidate list, which is
    deeper than the top-k actually returned and is computed before reranking.
    Agreement measured on the returned passages is higher.
    """
    if not candidates:
        return {"n": 0}
    both = sum(1 for c in candidates if len(c.sources) > 1)
    return {
        "n": len(candidates),
        "both_retrievers": both,
        "both_fraction": round(both / len(candidates), 4),
        "dense_only": sum(1 for c in candidates if c.sources == ["dense"]),
        "sparse_only": sum(1 for c in candidates if c.sources == ["sparse"]),
        "scope": "fused_candidates_pre_rerank",
    }
