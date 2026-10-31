"""Sweep the dependency-expansion budget: how much context, and when does it stop helping.

The gap
-------
Objective 3 names the third component as "retrieving a much larger candidate set
expanded over the dependency graph so interpreting passages are actually available".
What is measured is that expansion helps: `context` beats `sparse` on readiness 0.0625
to 0.1050 and qar 0.0400 to 0.1700. What is **not** measured anywhere is the size
response. `context_seeds=5` and `context_per_seed=2` are set once in the config, used by
every driver in the repository, and never varied.

That matters because the component has a cost as well as a benefit, and
``expand_with_context`` says so in its own docstring: "It must be bounded: one seed in a
40-chunk Results section can have many equally-weighted neighbours, and an unbounded walk
turns a 10-result answer into a 60-chunk dump." More expansion supplies more framing and
also dilutes the answer with passages that qualify nothing. Where those two cross is the
operating point, and it is currently a guess.

What this measures
------------------
For each ``(seeds, per_seed)`` budget, expansion from the *same* sparse candidate set --
so the comparison isolates the budget and not a different retrieval -- against:

* readiness, caveat, bare rate (what the answer can be built from),
* qar (whether gold documents are qualified), and
* **dilution**: the share of returned passages that come from framing sections, and the
  passages-per-gold-document ratio.

The dilution column is the one a bare "bigger is better" reading would miss.

Controls
--------
``(0, 0)`` must reproduce the sparse baseline exactly, not approximately. If it does
not, the sweep is measuring something other than the budget.

Source: ``data/runs/context_sweep_pmc.json``.
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
from medrag.eval.context_provision import (
    _STRICT_FRAMING,
    aggregate_context_provision,
    context_provision_from_hits,
)
from medrag.eval.metrics import Qrels
from medrag.eval.readiness import aggregate_readiness, readiness_from_rows
from medrag.index.retriever import load_index
from medrag.index.sibling_bleed import expand_with_context

#: (seeds, context_per_seed). The last entry is deliberately large: the point is to find
#: where the curve turns, and a grid that stops at the default would assume the answer.
BUDGETS = [(0, 0), (2, 1), (5, 2), (5, 4), (10, 2), (10, 4), (20, 4), (40, 4)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--n-queries", type=int, default=400)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--budgets", default=None,
                    help="semicolon-separated seeds:per_seed, e.g. '0:0;5:2;20:4'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    budgets = BUDGETS
    if args.budgets:
        budgets = []
        for part in args.budgets.split(";"):
            if not part.strip():
                continue
            seeds, _, per = part.partition(":")
            budgets.append((int(seeds), int(per or 1)))

    cfg = PipelineConfig.load(args.config)._finalize()
    print(f"  index dir: {cfg.paths.index_dir}", flush=True)
    r = load_index(cfg, with_reranker=False, with_graph=False, with_dependency=True,
                   with_dense=False)
    if r.dependency is None:
        print("  dependency graph missing; run `medrag dependency-graph` first")
        return 1
    print(f"  store {len(r.store):,} chunks | configured budget "
          f"{cfg.fusion.context_seeds}x{cfg.fusion.context_per_seed}", flush=True)

    bm = Qrels.load(args.qrels)
    qs = [q for q in bm.queries if q.relevant_docs]
    random.Random(7).shuffle(qs)
    qs = qs[: args.n_queries]
    print(f"  {len(qs)} queries | budgets {budgets}", flush=True)

    ready: dict[tuple[int, int], list] = {b: [] for b in budgets}
    rows_by: dict[tuple[int, int], dict[str, list[int]]] = {b: {} for b in budgets}
    framing_share: dict[tuple[int, int], list[float]] = {b: [] for b in budgets}
    per_gold: dict[tuple[int, int], list[float]] = {b: [] for b in budgets}
    t0 = time.time()
    for i, q in enumerate(qs):
        base = [(int(x), float(s)) for x, s in r._sparse_search(q.text, args.pool)]
        seeds_rows = base[: args.top_k]
        gold = set(q.relevant_docs)
        for b in budgets:
            seed_count, per_seed = b
            rows = [int(m[0]) for m in expand_with_context(
                seeds_rows, r.store, r.dependency, top_k=args.top_k,
                seed_count=seed_count, context_per_seed=per_seed,
                context_budget=(seed_count * per_seed) if seed_count else 0)]
            ready[b].append(readiness_from_rows(rows, r.store, q.relevant_docs,
                                                qid=q.qid))
            rows_by[b][q.qid] = rows
            gold_rows = [x for x in rows if (r.store.get(x) and r.store.get(x).doc_id) in gold]
            if gold_rows:
                framing = 0
                for x in gold_rows:
                    c = r.store.get(x)
                    if c is not None and (c.section_kind or "") in _STRICT_FRAMING:
                        framing += 1
                framing_share[b].append(framing / len(gold_rows))
                per_gold[b].append(len(rows) / max(len(gold_rows), 1))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(qs)} | {time.time() - t0:.0f}s", flush=True)

    keys = ("readiness", "quantification_rate", "population_rate", "caveat_rate",
            "benefit_rate", "bare_rate")
    table: dict[str, dict] = {}
    print(f"\n  {'seeds x per':>12s}{'added':>7s}" +
          "".join(f"{k.replace('_rate','').replace('quantification','quant'):>12s}"
                  for k in keys) + f"{'qar':>8s}{'fram%':>8s}")
    for b in budgets:
        seeds, per_seed = b
        added = seeds * per_seed
        s = aggregate_readiness(ready[b]).summary()
        prov = aggregate_context_provision(
            context_provision_from_hits([(row, 0.0) for row in rows_by[b][q.qid]],
                                        r.store, q.relevant_docs, qid=q.qid)
            for q in qs
        ).summary()
        fs = sum(framing_share[b]) / len(framing_share[b]) if framing_share[b] else 0.0
        pg = sum(per_gold[b]) / len(per_gold[b]) if per_gold[b] else 0.0
        label = f"{seeds} x {per_seed}"
        table[f"{seeds}x{per_seed}"] = {
            "added_budget": added, "readiness": s, "context_provision": prov,
            "gold_framing_share": round(fs, 4),
            "passages_per_gold_document": round(pg, 4),
        }
        print(f"  {label:>12s}{added:>7d}" +
              "".join(f"{s[k]:>12.4f}" for k in keys) +
              f"{prov.get('qar', 0):>8.4f}{fs:>8.4f}")

    base_key = "0x0"
    control = {k: table[base_key]["readiness"][k] == table[base_key]["readiness"][k]
               for k in keys}
    identical = all(table[base_key]["readiness"][k] == v for k, v in
                    {k: table[base_key]["readiness"][k] for k in keys}.items())
    # The control that matters: 0x0 must equal the plain sparse ranking.
    base_rows_same = all(rows_by[(0, 0)][q.qid] ==
                         [int(x) for x, _ in r._sparse_search(q.text, args.pool)][: args.top_k]
                         for q in qs[:25])
    print(f"\n  budget 0x0 reproduces the sparse top-{args.top_k} exactly on 25 sampled "
          f"queries: {base_rows_same}")
    best = max(table.items(), key=lambda kv: kv[1]["readiness"]["readiness"])
    best_qar = max(table.items(), key=lambda kv: kv[1]["context_provision"].get("qar", 0))
    print(f"  best readiness: {best[0]} ({best[1]['readiness']['readiness']:.4f})")
    print(f"  best qar:       {best_qar[0]} "
          f"({best_qar[1]['context_provision'].get('qar', 0):.4f})")
    print(f"  configured:     {cfg.fusion.context_seeds}x{cfg.fusion.context_per_seed}")

    out = {"n_queries": len(qs), "top_k": args.top_k,
           "configured_budget": [cfg.fusion.context_seeds, cfg.fusion.context_per_seed],
           "budgets": [list(b) for b in budgets], "table": table,
           "zero_budget_reproduces_sparse": base_rows_same,
           "best_readiness": best[0], "best_qar": best_qar[0],
           "elapsed_s": round(time.time() - t0, 1)}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())