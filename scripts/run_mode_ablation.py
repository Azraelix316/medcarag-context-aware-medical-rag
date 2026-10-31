"""Mode ablation on the full-text corpus, through the shipped retrieval path.

Why this driver exists
-----------------------
``run_fulltext_eval.py`` established that dependency-graph expansion improves
readiness on PMC. But it called ``expand_with_context`` directly rather than going
through ``Retriever.retrieve(mode=...)``, so it measured the mechanism and not the
product. The three components the objective names are *retrieval modes*, and a mode
that is not exercised the way a caller exercises it is not an ablatable mode in any
sense that matters -- it is a function with a mode name attached.

So this runs each arm through ``retrieve()``, which also means a mode that fails to
load its dependency raises here rather than degrading silently in production.

Arms
----
  sparse    BM25 top-k. The baseline every other arm is compared against.
  bleed     sibling-bleed document conditioning
  context   dependency-graph expansion
  bleedctx  both
  page      page-level document conditioning (needs ``page_index.pkl``)
  pagectx   page conditioning plus dependency expansion

Arms are paired against ``sparse`` on the same queries, and readiness is reported
alongside the context-provision measures, because the point of the components is not
relevance but qualification: a mode that returns better passages and no more caveats
has not solved anything.

Reported honestly
-----------------
An arm whose dependency is absent is reported as UNAVAILABLE with the reason, not as
a number. ``page`` in particular degrades to ``sparse`` by design when the page index
is missing, which would otherwise print an identical row and read as "no effect"
rather than "not measured" -- the same conflation that made the circularity control
on PMC print a reversal where it had measured nothing.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import sys
import time
from pathlib import Path

# The repository this script lives in, not a fixed checkout: on worker-A the old fixed
# path held a stale copy without medrag.index.segmented.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.ERROR)

from medrag.config import PipelineConfig
from medrag.eval.context_provision import (
    aggregate_context_provision,
    context_provision_from_hits,
)
from medrag.eval.metrics import Qrels
from medrag.eval.readiness import aggregate_readiness, readiness_from_rows
from medrag.index.retriever import (
    MODE_BLEED,
    MODE_BLEED_CONTEXT,
    MODE_CONTEXT,
    MODE_PAGE,
    MODE_PAGE_CONTEXT,
    MODE_SPARSE,
    load_index,
)

ALL_ARMS = [MODE_SPARSE, MODE_BLEED, MODE_CONTEXT, MODE_BLEED_CONTEXT,
            MODE_PAGE, MODE_PAGE_CONTEXT]


def _sample_queries(qrels_path: str, n: int) -> list:
    bm = Qrels.load(qrels_path)
    qs = [q for q in bm.queries if q.relevant_docs]
    random.Random(7).shuffle(qs)
    return qs[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--n-queries", type=int, default=400)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-queries", metavar="JSONL",
                    help="write the sampled queries as search-segmented input and exit")
    ap.add_argument("--sparse-run", metavar="JSONL",
                    help="search-segmented output to serve sparse hits from, for an index "
                         "with BM25 segments instead of bm25.pkl; needs --sparse-queries")
    ap.add_argument("--sparse-queries", metavar="JSONL",
                    help="the --write-queries file the sparse run was made from")
    ap.add_argument("--dense-run", metavar="JSONL",
                    help="search-segmented --modes dense output, served as the dense hits "
                         "of hybrid3 (an index with no dense.faiss); keyed by --sparse-queries")
    ap.add_argument("--graph-run", metavar="JSONL",
                    help="scripts/precompute_graph_hits.py output, served as the MeSH graph "
                         "hits of graph and hybrid3; keyed by --sparse-queries")
    ap.add_argument("--shard", metavar="K/N",
                    help="retrieve only queries K, K+N, K+2N, ... and write their ranked "
                         "rows to --rows-out instead of reporting; a later run with "
                         "--rows-from over every shard reports on all of them")
    ap.add_argument("--rows-out", metavar="JSON")
    ap.add_argument("--rows-from", metavar="JSON,...",
                    help="shard row files; their rankings are used instead of retrieving")
    ap.add_argument("--per-query-out", metavar="JSON",
                    help="also write every arm's per-query readiness flags, for "
                         "scripts/paired_bootstrap.py")
    args = ap.parse_args()

    if args.write_queries:
        qs = _sample_queries(args.qrels, args.n_queries)
        with open(args.write_queries, "w", encoding="utf-8") as fh:
            for q in qs:
                fh.write(json.dumps({"qid": q.qid, "query": q.text}) + "\n")
        print(f"  {len(qs)} queries -> {args.write_queries}")
        return 0

    cfg = PipelineConfig.load(args.config)._finalize()
    print(f"  index dir: {cfg.paths.index_dir}", flush=True)
    # with_dense=False still loads the page index and the query embedder, because the
    # page arms embed the query to score documents. The 16,133,551-row chunk matrix
    # is what costs 2,168 s and no arm here uses it.
    r = load_index(cfg, with_reranker=False, with_graph=False, with_dependency=True,
                   with_dense=False)
    print(f"  store {len(r.store):,} chunks | dependency "
          f"{'loaded' if r.dependency else 'MISSING'} | page index "
          f"{'loaded' if r.page is not None else 'MISSING'} "
          f"({r.page.n_docs:,} documents)" if r.page is not None else
          f"  store {len(r.store):,} chunks | dependency "
          f"{'loaded' if r.dependency else 'MISSING'} | page index MISSING",
          flush=True)

    if args.sparse_run:
        from medrag.index.segmented import PrecomputedBM25
        r.bm25 = PrecomputedBM25.from_run(args.sparse_run, args.sparse_queries,
                                          n_docs=len(r.store))
        print(f"  sparse hits precomputed: {r.bm25.stats()}", flush=True)
    if args.dense_run:
        from medrag.index.segmented import PrecomputedHits
        dense = PrecomputedHits.from_run(args.dense_run, args.sparse_queries, "dense")
        r.dense = dense
        # retrieve() passes [query] (no sub-queries here) and the fusion depth.
        r._dense_search = lambda queries, top_k: dense.search(queries[0], top_k)
        print(f"  dense hits precomputed: {dense.stats()}", flush=True)
    if args.graph_run:
        from medrag.index.segmented import PrecomputedHits
        r.graph = PrecomputedHits.from_run(args.graph_run, args.sparse_queries, "graph",
                                           as_int=False)
        print(f"  graph hits precomputed: {r.graph.stats()}", flush=True)

    if r.dependency is None:
        print("  dependency graph missing; run `medrag dependency-graph` first")
        return 1

    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    unavailable: dict[str, str] = {}
    if MODE_PAGE in wanted or MODE_PAGE_CONTEXT in wanted:
        if r.page is None or r.page.matrix is None:
            reason = ("page_index.pkl absent or empty -- run `medrag page-index`; "
                      "these arms degrade to sparse by design, so running them "
                      "without it would report 'no effect' for a missing component")
            unavailable[MODE_PAGE] = reason
            unavailable[MODE_PAGE_CONTEXT] = reason

    qs = _sample_queries(args.qrels, args.n_queries)
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        qs = qs[k::n]
    stored: dict[str, dict[str, list[int]]] = {}
    for path in (args.rows_from or "").split(","):
        if path:
            with open(path) as fh:
                for mode, per_q in json.load(fh).items():
                    stored.setdefault(mode, {}).update(per_q)
    print(f"  {len(qs)} queries", flush=True)

    arms: dict[str, list] = {}
    kind_counts: dict[str, collections.Counter] = {}
    gold_kind_counts: dict[str, collections.Counter] = {}
    rows_by_arm: dict[str, dict[str, list[int]]] = {}
    t0 = time.time()
    for i, q in enumerate(qs):
        for mode in wanted:
            if mode in unavailable:
                continue
            if stored:
                rows = stored[mode][q.qid]
            else:
                res = r.retrieve(q.text, top_k=args.top_k, mode=mode, include_text=False)
                # chunk_id is the store row KEY ("pmcid:...#0011"); row_id is the int.
                rows = [int(p.row_id) for p in res.passages]
            rows_by_arm.setdefault(mode, {})[q.qid] = rows
            # Which sections the qualification actually comes from.
            #
            # The objective asks for the metric to be "assessed against section-typed
            # gold". The gold here is document-level and the section typing is applied
            # to the retrieved passages, which is a defensible reading -- but it means
            # the claim that qualification comes from the interpreting sections is
            # assumed rather than shown. Counting the kinds of the passages that do the
            # qualifying settles it: if framing is coming from Discussion and
            # Conclusions, the mechanism is doing what the paper says; if it is coming
            # from Background, the section-typing is decorative.
            for row in rows:
                chunk = r.store.get(row)
                if chunk is None:
                    continue
                kind = chunk.section_kind or "(none)"
                kind_counts.setdefault(mode, collections.Counter())[kind] += 1
                if (chunk.doc_id or "") in set(q.relevant_docs):
                    gold_kind_counts.setdefault(mode, collections.Counter())[kind] += 1
            arms.setdefault(mode, []).append(
                readiness_from_rows(rows, r.store, q.relevant_docs, qid=q.qid))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(qs)} | {time.time() - t0:.0f}s", flush=True)

    if args.shard:
        with open(args.rows_out, "w") as fh:
            json.dump(rows_by_arm, fh)
        print(f"  shard {args.shard}: {len(qs)} queries -> {args.rows_out} "
              f"({time.time() - t0:.0f}s)")
        return 0

    ready = {m: aggregate_readiness(v).summary() for m, v in arms.items()}
    if args.per_query_out:
        with open(args.per_query_out, "w") as fh:
            json.dump({m: [q.to_dict() for q in v] for m, v in arms.items()}, fh)
    provision = {}
    for mode, per_q in rows_by_arm.items():
        try:
            # Needs the gold documents per query, so it is rebuilt here rather than
            # reusing the readiness rows, which do not carry them.
            provision[mode] = aggregate_context_provision(
                context_provision_from_hits(
                    [(row, 0.0) for row in per_q[q.qid]], r.store,
                    q.relevant_docs, qid=q.qid)
                for q in qs if q.qid in per_q
            ).summary()
        except Exception as exc:  # noqa: BLE001
            provision[mode] = {"error": f"{type(exc).__name__}: {exc}"}

    keys = ("readiness", "quantification_rate", "population_rate", "caveat_rate",
            "benefit_rate", "bare_rate")
    print("\n  READINESS by mode (PMC full text)")
    print(f"    {'arm':10s}" + "".join(f"{k.replace('_rate','').replace('quantification','quant'):>13s}"
                                      for k in keys))
    for m in wanted:
        if m in ready:
            print(f"    {m:10s}" + "".join(f"{ready[m][k]:>13.4f}" for k in keys))
    for m, why in unavailable.items():
        print(f"    {m:10s}{'UNAVAILABLE: ' + why}")

    base = ready.get(MODE_SPARSE)
    delta: dict[str, dict[str, float]] = {}
    if base:
        print("\n  DELTA vs sparse")
        print(f"    {'arm':10s}" + "".join(f"{k.replace('_rate','').replace('quantification','quant'):>13s}"
                                          for k in keys))
        for m in wanted:
            if m == MODE_SPARSE or m not in ready:
                continue
            d = {k: round(ready[m][k] - base[k], 4) for k in keys}
            delta[m] = d
            print(f"    {m:10s}" + "".join(f"{d[k]:>+13.4f}" for k in keys))

    # An arm that returns the identical ranking to the baseline has had no effect.
    # Worth stating because `page` degrades to `sparse` when the index is missing,
    # and an identical row is otherwise indistinguishable from a null result.
    print("\n  RANKING CHANGED vs sparse")
    identical: dict[str, bool] = {}
    for m in wanted:
        if m == MODE_SPARSE or m not in rows_by_arm:
            continue
        same = sum(
            1 for q in qs
            if rows_by_arm[m].get(q.qid) == rows_by_arm.get(MODE_SPARSE, {}).get(q.qid)
        )
        identical[m] = same == len(qs)
        print(f"    {m:10s} identical to sparse on {same}/{len(qs)} queries"
              f"{'   <-- no effect' if identical[m] else ''}")

    # Section composition, per arm, over all retrieved passages and over gold-document
    # passages only.
    print("\n  SECTION COMPOSITION of retrieved passages (gold-document passages only)")
    kinds_seen = sorted({k for m in gold_kind_counts.values() for k in m},
                        key=lambda k: -sum(m.get(k, 0) for m in gold_kind_counts.values()))
    print(f"    {'arm':9s}" + "".join(f"{k[:11]:>13s}" for k in kinds_seen))
    section_table: dict[str, dict] = {}
    for m in wanted:
        if m not in gold_kind_counts:
            continue
        tot = sum(gold_kind_counts[m].values()) or 1
        row = {k: round(gold_kind_counts[m].get(k, 0) / tot, 4) for k in kinds_seen}
        section_table[m] = {"share_of_gold_passages_by_kind": row,
                            "counts_by_kind": dict(gold_kind_counts[m])}
        print(f"    {m:9s}" + "".join(f"{row[k]:>13.4f}" for k in kinds_seen))

    out = {
        "n_queries": len(qs),
        "section_composition": section_table,
        "top_k": args.top_k,
        "index_dir": str(cfg.paths.index_dir),
        "page_index_documents": r.page.n_docs if r.page is not None else 0,
        "readiness": ready,
        "delta_vs_sparse": delta,
        "context_provision": provision,
        "identical_to_sparse": identical,
        "unavailable": unavailable,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())