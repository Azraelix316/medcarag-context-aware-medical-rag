"""Retrieval evaluation: recall@k, nDCG@k, MRR, and clinical-QA answer metrics.

The benchmark is built from PubMed's own relevance signal rather than LLM labels:
a passage is relevant to a query if it comes from a document indexed under the
query's MeSH descriptor (or its subtree). That ground truth is free, reproducible,
and independent of any model's output, which is exactly what a retrieval paper
needs to avoid self-referential evaluation.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ qrels I/O
@dataclass
class Query:
    """One evaluation query with its labelled relevant documents."""

    qid: str
    text: str
    relevant_docs: set[str] = field(default_factory=set)
    # Optional answer string for extractive/QA scoring.
    answer: str = ""
    category: str = ""
    # For multi-hop queries: intermediate docs that are acceptable evidence.
    supporting_docs: set[str] = field(default_factory=set)
    difficulty: str = "single-hop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid, "text": self.text,
            "relevant_docs": sorted(self.relevant_docs),
            "answer": self.answer, "category": self.category,
            "supporting_docs": sorted(self.supporting_docs),
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Query":
        return cls(
            qid=str(d["qid"]), text=d["text"],
            relevant_docs=set(d.get("relevant_docs", [])),
            answer=d.get("answer", ""), category=d.get("category", ""),
            supporting_docs=set(d.get("supporting_docs", [])),
            difficulty=d.get("difficulty", "single-hop"),
        )


class Qrels:
    """Query -> relevant document-id map (TREC-style)."""

    def __init__(self, queries: Sequence[Query] | None = None) -> None:
        self.queries: list[Query] = list(queries or [])
        self.by_id: dict[str, Query] = {q.qid: q for q in self.queries}

    def add(self, query: Query) -> None:
        self.queries.append(query)
        self.by_id[query.qid] = query

    def __len__(self) -> int:
        return len(self.queries)

    def sample(self, n: int, seed: int = 13) -> list[Query]:
        """Deterministic subsample; keeps stratification by difficulty."""
        rng = random.Random(seed)
        pool = list(self.queries)
        rng.shuffle(pool)
        return pool[:n]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([q.to_dict() for q in self.queries], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Qrels":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("queries", [])
        return cls([Query.from_dict(d) for d in data])


# -------------------------------------------------------------------- metrics
def recall_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents retrieved in the top k."""
    if not relevant:
        return float("nan")
    top = set(ranked_ids[:k])
    return len(top & relevant) / len(relevant)


def precision_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not ranked_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / k


def hit_rate_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return 1.0 if set(ranked_ids[:k]) & relevant else 0.0


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Standard discounted cumulative gain (binary relevance converted to 1)."""
    return sum((2.0**rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """nDCG@k with binary gains, normalised by the ideal ordering."""
    if not relevant:
        return float("nan")
    gains = [1.0 if cid in relevant else 0.0 for cid in ranked_ids[:k]]
    ideal = [1.0] * min(len(relevant), k)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / idcg if idcg else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], relevant: set[str]) -> float:
    """1/rank of the first relevant document; 0.0 if none is retrieved."""
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(ranked_ids: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    ids = ranked_ids if k is None else ranked_ids[:k]
    return reciprocal_rank(ids, relevant)


def success_at_k(ranked_ids: Sequence[str], relevant: set[str], supporting: set[str], k: int) -> float:
    """Multi-hop success: all required supporting docs present in the top k.

    Stricter than hit_rate and the standard metric for multi-hop RAG, where
    retrieving one hop and missing the other is a failure.
    """
    if not supporting:
        return float("nan")
    return 1.0 if supporting <= set(ranked_ids[:k]) else 0.0


def bpref(
    ranked_ids: Sequence[str],
    relevant: set[str],
    k: int | None = None,
    *,
    judged: set[str] | None = None,
) -> float:
    """Binary preference, robust to incomplete relevance judgments.

    Introduced by Buckley & Voorhees (2004) for the TREC 2005 terabyte track,
    where judgments are known to be far from complete. Their finding is the reason
    this metric exists: average precision, P@k and R-precision all shift when
    judgments are incomplete, so system rankings produced with them do not
    transfer. bpref correlates highly with mean average precision under complete
    judgments and keeps correlating when judgments are incomplete.

    Definition: the fraction of relevant documents that appear *above* a
    judged-irrelevant one.

        bpref = 1/R * sum over relevant r of ( 1 - |{judged-nonrel above r}| / N )

    where R counts relevant documents retrieved and N counts judged-irrelevant
    documents retrieved.

    The ``judged`` argument is what makes this work on incomplete judgments.
    Only documents known to be judged can contribute evidence: a relevant
    document ranks above an *irrelevant* one, not above an *unknown* one. Pass the
    pool of documents that have been assessed; anything retrieved that is in
    neither ``relevant`` nor ``judged`` is ignored entirely.

    Omitting ``judged`` assumes the pool is everything retrieved, which is the
    correct assumption for TREC-style fully-judged collections and the WRONG
    assumption for citation-derived labels -- where equally relevant uncited
    documents are simply unlabelled. Treating those as irrelevant silently
    recreates the instability bpref was invented to avoid, so pass ``judged``
    whenever the labels are incomplete.
    """
    ids = list(ranked_ids) if k is None else list(ranked_ids)[:k]
    if not relevant:
        return float("nan")

    def is_nonrel(cid: str) -> bool:
        if cid in relevant:
            return False
        return judged is None or cid in judged

    judged_nonrel_positions = [i for i, cid in enumerate(ids) if is_nonrel(cid)]
    n_nonrel = len(judged_nonrel_positions)
    if n_nonrel == 0:
        # No judged-irrelevant document was retrieved, so there is no preference
        # relation to be right or wrong about. 0.0 means "no evidence", which
        # keeps the mean honest; 1.0 would flatter a system that retrieved only
        # things it likes.
        return 0.0

    # Positions of relevant documents, restricted to the judged pool.
    rel_positions = [i for i, cid in enumerate(ids) if cid in relevant]
    if not rel_positions:
        return 0.0

    total = 0.0
    for r in rel_positions:
        # Count judged-irrelevant documents strictly above this relevant one.
        # bisect avoids the O(n^2) scan that matters at depth 100 over 76M chunks.
        above = bisect.bisect_left(judged_nonrel_positions, r)
        total += 1.0 - (above / n_nonrel)
    return total / len(rel_positions)


def compute_all_metrics(
    ranked_ids: Sequence[str],
    relevant: set[str],
    k_values: Sequence[int] = (1, 5, 10, 20, 100),
    supporting: set[str] | None = None,
) -> dict[str, float]:
    """Compute the full metric suite for one query."""
    out: dict[str, float] = {}
    max_k = max(k_values) if k_values else len(ranked_ids)
    for k in k_values:
        out[f"recall@{k}"] = recall_at_k(ranked_ids, relevant, k)
        out[f"precision@{k}"] = precision_at_k(ranked_ids, relevant, k)
        out[f"hit@{k}"] = hit_rate_at_k(ranked_ids, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, relevant, k)
        out[f"mrr@{k}"] = mean_reciprocal_rank(ranked_ids, relevant, k)
    # bpref is cut at the deepest k only: it is designed to be read over a
    # generous depth, since it is the incomplete-judgment-robust metric.
    out["bpref"] = bpref(ranked_ids, relevant, max_k)
    if supporting:
        out["success@all"] = success_at_k(ranked_ids, relevant, supporting, max_k)
    return out


# ----------------------------------------------------------------- evaluation
@dataclass
class EvalResult:
    name: str
    n_queries: int
    metrics: dict[str, float]
    # Per-difficulty and per-mode breakdowns for the paper's tables.
    breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    per_query: list[dict[str, Any]] = field(default_factory=list)
    elapsed: float = 0.0

    def to_dict(self, *, include_per_query: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "n_queries": self.n_queries,
            "metrics": {k: (None if math.isnan(v) else round(v, 5)) for k, v in self.metrics.items()},
            "breakdown": {
                key: {k: (None if math.isnan(v) else round(v, 5)) for k, v in vals.items()}
                for key, vals in self.breakdown.items()
            },
            "elapsed_s": round(self.elapsed, 2),
        }
        if include_per_query:
            payload["per_query"] = self.per_query
        return payload

    def format_table(self) -> str:
        """Render as a markdown table for the paper."""
        lines = [f"### {self.name}", "", f"Queries: {self.n_queries} ({self.elapsed:.1f}s)", ""]
        keys = sorted(self.metrics, key=_metric_sort_key)
        header = "| metric | value |"
        lines += [header, "|---|---|"]
        for key in keys:
            value = self.metrics.get(key, float("nan"))
            lines.append(f"| {key} | {'n/a' if math.isnan(value) else f'{value:.4f}'} |")
        if self.breakdown:
            lines += ["", "**Breakdown**", "", "| slice | " + " | ".join(keys[:6]) + " |",
                      "|---" * (min(6, len(keys)) + 1) + "|"]
            for name, vals in self.breakdown.items():
                cells = []
                for key in keys[:6]:
                    v = vals.get(key, float("nan"))
                    cells.append("n/a" if math.isnan(v) else f"{v:.4f}")
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
        return "\n".join(lines)


def _metric_sort_key(key: str) -> tuple[str, int, str]:
    """Order metric names by family then k, so tables read recall@k, nDCG@k, ..."""
    match = re.search(r"@(\d+)", key)
    return (re.sub(r"@\d+", "", key), int(match.group(1)) if match else 0, key)


def _nanmean(values: Iterable[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def evaluate_retriever(
    retriever: Any,
    qrels: Qrels,
    *,
    mode: str = "reranked",
    top_k: int = 100,
    k_values: Sequence[int] = (1, 3, 5, 10, 20, 100),
    name: str | None = None,
    sub_query_fn: Any = None,
    max_k: int = max((20, 100)),
) -> EvalResult:
    """Run every query through the retriever and aggregate metrics.

    ``sub_query_fn`` is an optional ``(query) -> list[str]`` hook used to test
    whether query decomposition improves multi-hop retrieval.
    """
    import time

    t0 = time.time()
    per_query: list[dict[str, Any]] = []
    buckets: dict[str, list[dict[str, float]]] = {}
    # Metric keys are captured from the first query rather than inferred from
    # the record's keys, which also carry qid and text.
    metric_keys: list[str] = []

    for query in qrels.queries:
        sub_queries = sub_query_fn(query) if sub_query_fn else None
        result = retriever.retrieve(query.text, top_k=max_k, mode=mode, sub_queries=sub_queries, include_text=False)
        # Rank documents by their best-ranked chunk.
        doc_order: list[str] = []
        doc_score: dict[str, float] = {}
        for passage in result.passages:
            doc_id = passage.chunk_id.split("#", 1)[0]
            if doc_id not in doc_score:
                doc_order.append(doc_id)
                doc_score[doc_id] = passage.score
            else:
                doc_score[doc_id] = max(doc_score[doc_id], passage.score)
        ranked_docs = sorted(doc_order, key=lambda d: -doc_score[d])

        relevant = query.relevant_docs | query.supporting_docs
        metrics = compute_all_metrics(ranked_docs, relevant, k_values, query.supporting_docs or None)
        # Some metrics only exist for some slices (success@all is multi-hop
        # only), so the reported key set is the union over all queries.
        for key in metrics:
            if key not in metric_keys:
                metric_keys.append(key)
        per_query.append({"qid": query.qid, "text": query.text, **metrics})
        buckets.setdefault(query.difficulty, []).append(metrics)
        buckets.setdefault(query.category or "uncategorized", []).append(metrics)

    metric_keys.sort(key=_metric_sort_key)
    all_metrics = {key: _nanmean(row.get(key, float("nan")) for row in per_query)
                   for key in metric_keys}
    breakdown = {
        name_: {key: _nanmean(row.get(key, float("nan")) for row in rows) for key in metric_keys}
        for name_, rows in buckets.items()
    }
    return EvalResult(
        name=name or f"{mode}",
        n_queries=len(per_query),
        metrics=all_metrics,
        breakdown=breakdown,
        per_query=per_query,
        elapsed=time.time() - t0,
    )


def compare_modes(
    retriever: Any,
    qrels: Qrels,
    modes: Sequence[str] = ("sparse", "dense", "hybrid", "reranked"),
    *,
    top_k: int = 100,
    k_values: Sequence[int] = (1, 5, 10, 20, 100),
    max_k: int = 100,
) -> list[EvalResult]:
    """Evaluate several retrieval modes on the same queries (ablation table)."""
    results: list[EvalResult] = []
    for mode in modes:
        try:
            results.append(
                evaluate_retriever(retriever, qrels, mode=mode, top_k=top_k,
                                   k_values=k_values, name=mode, max_k=max_k)
            )
        except RuntimeError as exc:
            log.warning("skipping mode %s: %s", mode, exc)
    return results


def significance_bootstrap(
    metric_a: Sequence[float],
    metric_b: Sequence[float],
    *,
    n_resamples: int = 1000,
    seed: int = 17,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired bootstrap test on per-query metric differences.

    Reports the observed delta and a p-value so the paper can state whether an
    ablation gain is real rather than noise.
    """
    rng = random.Random(seed)
    a = [v for v in metric_a if not math.isnan(v)]
    b = [v for v in metric_b if not math.isnan(v)]
    n = min(len(a), len(b))
    if n == 0:
        return {"delta": float("nan"), "p_value": float("nan"), "n": 0}
    a, b = a[:n], b[:n]
    observed = sum(x - y for x, y in zip(a, b)) / n

    deltas: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(a[i] - b[i] for i in idx) / n)
    deltas.sort()
    # Two-sided p-value under H0: no mean difference.
    non_positive = sum(1 for d in deltas if d <= 0)
    non_negative = sum(1 for d in deltas if d >= 0)
    p = min(1.0, 2.0 * min(non_positive, non_negative) / n_resamples)
    lo = deltas[int(alpha / 2 * n_resamples)]
    hi = deltas[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return {
        "delta": round(observed, 5),
        "p_value": round(p, 5),
        "ci95": [round(lo, 5), round(hi, 5)],
        "n": n,
        "significant": p < alpha,
    }
