"""Sweep page_alpha: is the page arm's gain near its ceiling, or an artefact of one weight?

``page_alpha`` is 0.5, chosen once and never varied, which makes the ``page`` arm's
+0.0400 readiness vulnerable to the obvious objection: some other weight would have
done better, or worse, and nobody looked.

The sweep is cheap because the page scores do not depend on the weight. They come from
the query and the page matrix; ``alpha`` only scales them afterwards. So the query is
embedded once, page scores computed once, and each alpha is a re-scoring of the same
base ranking. That is what makes this a sweep rather than N runs.

Two properties are asserted rather than assumed, because both failure modes are
silent:

* **alpha=0 must reproduce the sparse baseline exactly.** Not approximately -- the
  same rows in the same order. If it does not, the sweep is measuring something other
  than the mechanism.
* **The sweep must move.** A grid that returns one number for every alpha is either
  not reaching the scoring function or alpha is not wired through. Either way the
  table is meaningless, and it would look like a clean result.

Reported per alpha: readiness, the three ingredient rates, bare rate, and how often the
ranking differs from the baseline at that weight.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.ERROR)

import numpy as np

from medrag.config import PipelineConfig
from medrag.eval.context_provision import (
    aggregate_context_provision,
    context_provision_from_hits,
)
from medrag.eval.metrics import Qrels
from medrag.eval.readiness import aggregate_readiness, readiness_from_rows
from medrag.index.page_scoring import PageConfig, apply_page_scores, normalise
from medrag.index.retriever import load_index


def page_scores_for(r, query: str, docs: list[str]) -> dict[str, float]:
    """Cosine of the query against each candidate document's page vector, in [0,1].

    Computed once per query and reused for every alpha, since none of this depends on
    the weight.
    """
    pos_of = {d: i for i, d in enumerate(r.page.doc_ids)}
    known = [d for d in dict.fromkeys(docs) if d in pos_of]
    if not known:
        return {}
    qv = np.asarray(r.dense_embedder.embed([query], is_query=True), dtype=np.float32)
    qv = qv.reshape(1, -1)
    qv = qv / max(float(np.linalg.norm(qv)), 1e-12)
    mat = np.asarray(r.page.matrix)
    sims = (qv @ mat[[pos_of[d] for d in known]].T)[0]
    return dict(zip(known, normalise([float(s) for s in sims])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--alphas", default="0,0.1,0.25,0.5,1.0,2.0,4.0")
    ap.add_argument("--n-queries", type=int, default=400)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    cfg = PipelineConfig.load(args.config)._finalize()
    print(f"  index dir: {cfg.paths.index_dir}", flush=True)
    r = load_index(cfg, with_reranker=False, with_graph=False, with_dependency=False,
                   with_dense=False)
    if r.page is None or r.page.matrix is None:
        print("  page index missing; run `medrag page-index` first")
        return 1
    if r.dense_embedder is None:
        print("  query embedder not loaded; cannot score page vectors")
        return 1
    print(f"  store {len(r.store):,} chunks | page index {r.page.n_docs:,} documents",
          flush=True)

    bm = Qrels.load(args.qrels)
    qs = [q for q in bm.queries if q.relevant_docs]
    random.Random(7).shuffle(qs)
    qs = qs[: args.n_queries]
    print(f"  {len(qs)} queries, alphas {alphas}", flush=True)

    ready: dict[float, list] = {a: [] for a in alphas}
    rows_by_alpha: dict[float, dict[str, list[int]]] = {a: {} for a in alphas}
    # The true baseline, kept separately: the control has to be the sparse ranking
    # itself, not whichever alpha happened to be first on the command line.
    baseline_rows: dict[str, list[int]] = {}
    t0 = time.time()
    for i, q in enumerate(qs):
        base = [(int(x), s) for x, s in r._sparse_search(q.text, args.pool)]
        docs = []
        doc_of_hit: dict[int, str] = {}
        for row, _ in base:
            c = r.store.get(row)
            did = c.doc_id if c is not None else None
            doc_of_hit[row] = did  # type: ignore[assignment]
            docs.append(did)
        baseline_rows[q.qid] = [row for row, _ in base][: args.top_k]
        scores = page_scores_for(r, q.text, [d for d in docs if d])
        for a in alphas:
            ranked = apply_page_scores(base, scores, doc_of_hit.get,
                                       PageConfig(alpha=a))
            rows = [row for row, _ in ranked][: args.top_k]
            ready[a].append(
                readiness_from_rows(rows, r.store, q.relevant_docs, qid=q.qid))
            rows_by_alpha[a][q.qid] = rows
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(qs)} | {time.time() - t0:.0f}s", flush=True)

    # How far each weight moves the ranking from the sparse baseline it starts from.
    moved = {
        a: sum(1 for q in qs if rows_by_alpha[a][q.qid] != baseline_rows[q.qid])
        for a in alphas
    }

    keys = ("readiness", "quantification_rate", "population_rate", "caveat_rate",
            "benefit_rate", "bare_rate")
    print(f"\n  {'alpha':>7s}" + "".join(f"{k.replace('_rate','').replace('quantification','quant'):>12s}"
                                        for k in keys) + f"{'moved':>9s}")
    table: dict[str, dict] = {}
    for a in alphas:
        s = aggregate_readiness(ready[a]).summary()
        prov = aggregate_context_provision(
            context_provision_from_hits([(row, 0.0) for row in rows_by_alpha[a][q.qid]],
                                        r.store, q.relevant_docs, qid=q.qid)
            for q in qs
        ).summary()
        table[str(a)] = {"readiness": s, "context_provision": prov,
                         "rows_differing_from_sparse": moved[a]}
        print(f"  {a:>7.2f}" + "".join(f"{s[k]:>12.4f}" for k in keys)
              + f"{moved[a]:>6d}/{len(qs)}")

    zero = [a for a in alphas if a == 0.0]
    control_ok = (moved[zero[0]] == 0) if zero else None
    distinct = len({tuple(tuple(rows_by_alpha[a][q.qid]) for q in qs) for a in alphas})
    if control_ok is None:
        print("  no alpha=0 on the grid, so the identity control was not exercised")
    else:
        print(f"  alpha=0 reproduces the sparse ranking on every query: {control_ok}")
    print(f"  distinct rankings across the grid: {distinct} of {len(alphas)}")
    best = max(alphas, key=lambda a: table[str(a)]["readiness"]["readiness"])
    print(f"  best readiness at alpha={best}: "
          f"{table[str(best)]['readiness']['readiness']:.4f} "
          f"(configured {cfg.fusion.page_alpha})")

    out = {
        "n_queries": len(qs),
        "top_k": args.top_k,
        "pool": args.pool,
        "configured_alpha": cfg.fusion.page_alpha,
        "alphas": alphas,
        "alpha_0_reproduces_baseline": control_ok,
        "rows_differing_from_sparse": {str(a): moved[a] for a in alphas},
        "distinct_rankings": distinct,
        "best_alpha": best,
        "table": table,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())