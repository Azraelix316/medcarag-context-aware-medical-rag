"""Full-text evaluation: readiness and context provision on real section structure.

Every measurement so far ran on the abstract-only PubMed corpus, which cannot
support the mechanism under test. An abstract has no Methods section, so the
dependency structure that qualifies a finding does not exist in it. This driver runs
the ablations on the PMC Open Access full-text corpus, where section kinds are
dominated by methods (33,604), results (28,846) and discussion (25,753).

The point of interest is whether the results transfer. Two things could plausibly
differ on real structure:

* the dependency graph is 4.2x denser here (5.01 edges per chunk against 1.19), so
  expansion has more to offer and more to dilute with;
* the passages actually contain qualifying text, so the baseline arm should start
  higher than it did on abstracts -- which would compress the headroom the mechanism
  is trying to demonstrate rather than confirm it.

Arms, all sharing one retrieval pass so the comparison isolates the mechanism:

  sparse   BM25 top-k
  context  dependency-graph expansion over the same seeds (seed_count x context_per_seed)

Reported per arm: readiness, quantification_rate, population_rate, caveat_rate,
bare_rate. readiness requires a quantified finding AND a stated population; caveat
and bare rates are the two halves of the motivating paper's failure mode.

Run after `verify_embeddings.py` passes and `bm25.pkl` exists.
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

from medrag.config import PipelineConfig
from medrag.eval.metrics import Qrels
from medrag.eval.readiness import aggregate_readiness, readiness_from_rows
from medrag.index.retriever import load_index
from medrag.index.sibling_bleed import expand_with_context


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--n-queries", type=int, default=600)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--seed-count", type=int, default=10)
    ap.add_argument("--context-per-seed", type=int, default=1)
    ap.add_argument("--out", default="data/runs/pmc_fulltext_eval.json")
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)._finalize()
    # with_dense=False: loading 16,133,551 x 384 from NFS costs 2,168 s and the
    # sparse baseline arm does not use it.
    r = load_index(cfg, with_reranker=False, with_graph=False, with_dependency=True,
                   with_dense=False)
    print(f"  store {len(r.store):,} chunks, dependency graph "
          f"{'loaded' if r.dependency else 'MISSING'}", flush=True)
    if r.dependency is None:
        print("  dependency graph missing; run `medrag dependency-graph` first")
        return 1

    bm = Qrels.load(args.qrels)
    qs = [q for q in bm.queries if q.relevant_docs]
    random.Random(7).shuffle(qs)
    qs = qs[: args.n_queries]
    print(f"  {len(qs)} queries over {len({d for q in qs for d in q.relevant_docs})} "
          f"gold documents", flush=True)

    # Report what the full-text corpus can and cannot evaluate, before any result.
    #
    # Translating the PubMed gold onto PMC keeps 9.8% of gold documents and 407 of
    # 2,340 queries, and the loss is not uniform -- it is concentrated exactly where
    # multi-hop questions live. Measured retention by slice:
    #
    #     psychiatry-and-psychology  37.5%     multi-hop              0.5%
    #     phenomena-and-processes   25.9%     diseases               18.5%
    #     anatomy                   22.3%     drugs-and-therapeutics 18.9%
    #                                 organisms 18.9%
    #
    # Multi-hop drops to 2 surviving queries of 420. That is not a small sample, it is
    # the absence of the slice, so no multi-hop conclusion can be drawn from this
    # corpus and the earlier cross-hop result stands only on the abstract corpus.
    # Reporting a single pooled number across all 407 would hide that entirely.
    by_cat: dict[str, int] = {}
    by_diff: dict[str, int] = {}
    for q in qs:
        by_cat[q.category] = by_cat.get(q.category, 0) + 1
        by_diff[q.difficulty] = by_diff.get(q.difficulty, 0) + 1
    print("  COMPOSITION of the evaluable set")
    for k in sorted(by_cat):
        print(f"    {k:28s} {by_cat[k]:>5d}")
    print("    -- difficulty --")
    for k in sorted(by_diff):
        print(f"    {k:28s} {by_diff[k]:>5d}")
    if by_diff.get("multi-hop", 0) < 10:
        print(f"    !! only {by_diff.get('multi-hop', 0)} multi-hop queries survive; "
              f"the multi-hop slice is not evaluable on this corpus")

    arms = {"sparse": [], "context": []}
    t0 = time.time()
    for i, q in enumerate(qs):
        pool = r._sparse_search(q.text, args.pool)
        seeds = [(int(x), s) for x, s in pool[: args.top_k]]
        rows = {
            "sparse": [x for x, _ in seeds],
            "context": [int(m[0]) for m in expand_with_context(
                seeds, r.store, r.dependency, top_k=args.top_k,
                seed_count=args.seed_count,
                context_per_seed=args.context_per_seed)[:args.top_k + args.seed_count * args.context_per_seed]],
        }
        for name, rr in rows.items():
            arms[name].append(
                readiness_from_rows(rr, r.store, q.relevant_docs, qid=q.qid))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(qs)} | {time.time() - t0:.0f}s", flush=True)

    summary = {n: aggregate_readiness(v).summary() for n, v in arms.items()}
    print("\n  READINESS on PMC full text")
    keys = ("readiness", "quantification_rate", "population_rate", "caveat_rate",
            "benefit_rate", "bare_rate")
    print(f"    {'arm':10s}" + "".join(f"{k.replace('_rate','').replace('quantification','quant'):>14s}"
                                      for k in keys))
    for n, s in summary.items():
        print(f"    {n:10s}" + "".join(f"{s[k]:>14.4f}" for k in keys))

    delta = {k: round(summary["context"][k] - summary["sparse"][k], 4) for k in keys}
    print("\n  DELTA (context - sparse)")
    for k in keys:
        print(f"    {k:24s} {delta[k]:+.4f}")

    with open(args.out, "w") as fh:
        json.dump({"n_queries": len(qs), "arms": summary, "delta": delta,
                   "composition": {"by_category": by_cat, "by_difficulty": by_diff},
                   "top_k": args.top_k, "seed_count": args.seed_count,
                   "context_per_seed": args.context_per_seed,
                   "elapsed_s": round(time.time() - t0, 1)}, fh, indent=2)
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())