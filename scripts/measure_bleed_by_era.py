"""Does sibling bleed demote older documents? The era explanation, tested.

The prediction
--------------
Sibling bleed was contradicted on abstracts (MeSH bpref 0.4059 -> 0.3083) and helps on
full text. Era skew is a candidate mechanism: bleed conditions a chunk on the importance
of its document's *other* chunks, and document-level importance depends on how much a
document has to be important about -- 33 references per document in 2019-2026 against 7.3
in 2000-2009. The abstract index is 97.2% 2020s and 1.2% 2000s, so it evaluates bleed
almost entirely on documents of the kind where it helps.

The index coverage measurement makes this testable rather than merely arguable:

    decade    in corpus    indexed   coverage
    1990s        32,273        348       1.1%
    2000s       159,273      1,970       1.2%
    2010s       333,861     78,175      23.4%
    2020s       497,299    483,197      97.2%

What this measures
------------------
Not a retrieval metric. For each query the same candidate set is scored twice -- once by
BM25 alone, once with sibling bleed applied -- and every candidate is bucketed by the
publication year of the document it came from.

**The statistic is rank movement, not score movement.** The first version compared raw
scores and every bucket came out at about -0.99, which is not a finding but a category
error: ``sibling_bleed`` min-max normalises its inputs to [0, 1] before blending, so its
output scores live on that scale while raw BM25 scores are around 10-25. Dividing one by
the other is close to dividing 0.1 by 10 whatever the era, which is exactly what came
back. Retrieval consumes rank, not the score's units, so rank is the invariant to compare
on and the only one that survives the rescaling.

A positive value means bleed promoted that era's documents relative to the others.

That is the mechanism the prediction is about, measured directly on the documents bleed
actually touches. It needs no labels and no gold, so the era sparsity that makes a
labelled per-era comparison impractical does not bite here: the 1990s bucket has 348
indexed documents, which is enough to see a sign and not enough to claim a magnitude, and
the script reports the counts so that limitation is visible rather than assumed.

Interpretation stated in advance
--------------------------------
If bleed systematically demotes older documents, the prediction is supported and the
abstract contradiction is explained in part by which decades the index happens to hold.
If it does not, the prediction is wrong and the era explanation should be dropped rather
than reworded. Either result is recorded.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import statistics
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.ERROR)

from medrag.config import PipelineConfig
from medrag.eval.metrics import Qrels
from medrag.index.retriever import load_index

#: Bucket edges. The 1990s bucket is small on purpose rather than merged away: merging it
#: would hide exactly the sparsity that limits the claim.
EDGES = [(0, 1999, "before 2000"), (2000, 2009, "2000s"),
         (2010, 2019, "2010s"), (2020, 2100, "2020s")]


def bucket(year: int) -> str:
    for lo, hi, name in EDGES:
        if lo <= year <= hi:
            return name
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)._finalize()
    r = load_index(cfg, with_reranker=False, with_graph=False, with_dense=False)
    if r.dependency is None:
        print("  dependency graph missing")
        return 1

    bm = Qrels.load(args.qrels)
    qs = [q for q in bm.queries if q.relevant_docs]
    random.Random(7).shuffle(qs)
    qs = qs[: args.n_queries]
    print(f"  {len(qs)} queries; bleed alpha={cfg.fusion.bleed_alpha} "
          f"mode={cfg.fusion.bleed_mode}", flush=True)

    # Rank movement per candidate, in rank positions. Positive = promoted by bleed.
    delta: dict[str, list[float]] = collections.defaultdict(list)
    rows_by_era: dict[str, set[str]] = collections.defaultdict(set)
    t0 = time.time()
    for i, q in enumerate(qs):
        base = [(int(x), float(s)) for x, s in r._sparse_search(q.text, args.pool)]
        bled = r._apply_bleed(base)
        # Rank by score within each list, best first. Ties broken by row id so the two
        # rankings are deterministic and a tie is not silently order-dependent.
        base_rank = {row: k for k, (row, _) in
                     enumerate(sorted(base, key=lambda kv: (-kv[1], kv[0])))}
        bled_rank = {row: k for k, (row, _) in
                     enumerate(sorted(bled, key=lambda kv: (-kv[1], kv[0])))}
        for row in base_rank:
            c = r.store.get(row)
            if c is None:
                continue
            era = bucket(int(getattr(c, "year", 0) or 0))
            delta[era].append(float(base_rank[row] - bled_rank[row]))
            rows_by_era[era].add(c.doc_id)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(qs)} | {time.time() - t0:.0f}s", flush=True)

    print(f"\n  {'era':<12s}{'candidates':>12s}{'documents':>11s}"
          f"{'mean rank':>11s}{'median rank':>13s}{'se':>9s}")
    table = {}
    for _, _, name in EDGES:            # in era order, not sorted by the statistic
        vals = delta.get(name, [])
        if not vals:
            print(f"  {name:<12s}{0:>12,}{0:>11,}{'--':>11s}{'--':>13s}{'--':>9s}")
            continue
        mean = statistics.fmean(vals)
        med = statistics.median(vals)
        n_docs = len(rows_by_era[name])
        se = (statistics.stdev(vals) / (len(vals) ** 0.5)) if len(vals) > 1 else 0.0
        table[name] = {"n_candidates": len(vals), "n_documents": n_docs,
                       "mean_rank_delta": round(mean, 4),
                       "median_rank_delta": round(med, 4),
                       "standard_error": round(se, 4)}
        print(f"  {name:<12s}{len(vals):>12,}{n_docs:>11,}"
              f"{mean:>+11.4f}{med:>+13.4f}{se:>9.4f}")

    # The extremes are the oldest and newest BUCKETS, named by era. The previous version
    # picked them by sorting the statistic, so it labelled the 2000s bucket "newest"
    # because its mean happened to be highest -- which is a category error of its own.
    ordered = [n for _, _, n in EDGES if n in table]
    MIN_DOCS = 2_000
    verdict = None
    if len(ordered) >= 2:
        oldest_name, newest_name = ordered[0], ordered[-1]
        oldest, newest = table[oldest_name], table[newest_name]
        gap = newest["mean_rank_delta"] - oldest["mean_rank_delta"]
        thin = [n for n in (oldest_name, newest_name)
                if table[n]["n_documents"] < MIN_DOCS]
        print(f"\n  oldest populated bucket {oldest_name}: "
              f"{oldest['mean_rank_delta']:+.4f}")
        print(f"  newest populated bucket {newest_name}: "
              f"{newest['mean_rank_delta']:+.4f}")
        print(f"  gap {gap:+.4f} rank positions")
        if thin:
            # The previous version printed SUPPORTS with a 5-document bucket. A verdict
            # needs a floor on the evidence, not a footnote under it.
            verdict = (f"INSUFFICIENT -- {', '.join(thin)} holds fewer than "
                       f"{MIN_DOCS:,} indexed documents, so the sign is not a finding")
            print(f"  -> {verdict}")
            print(f"     ({', '.join(thin)}: "
                  f"{', '.join(f'{n}={table[n]['n_documents']:,} docs' for n in thin)})")
        else:
            verdict = ("SUPPORTS the era explanation -- bleed is least favourable to the "
                       "oldest documents present"
                       if gap > 0 else
                       "CONTRADICTS it -- bleed does not systematically favour newer "
                       "documents")
            print(f"  -> {verdict}")

    # The comparison that IS supported by the data: the buckets that clear the floor.
    # Comparing 2010s against 2020s is a real test of the prediction with 2,397 and
    # 7,271 documents behind it, where comparing 2000s against 2020s is not.
    solid = [n for n in ordered if table[n]["n_documents"] >= MIN_DOCS]
    if len(solid) >= 2:
        older_name, newer_name = solid[0], solid[-1]
        a, b = table[older_name], table[newer_name]
        diff = b["mean_rank_delta"] - a["mean_rank_delta"]
        se = (a["standard_error"] ** 2 + b["standard_error"] ** 2) ** 0.5
        t = diff / se if se > 0 else 0.0
        print(f"\n  Comparison among buckets that clear {MIN_DOCS:,} documents:")
        print(f"    {older_name}: {a['mean_rank_delta']:+.4f} "
              f"+/- {a['standard_error']:.4f} ({a['n_documents']:,} docs)")
        print(f"    {newer_name}: {b['mean_rank_delta']:+.4f} "
              f"+/- {b['standard_error']:.4f} ({b['n_documents']:,} docs)")
        print(f"    difference {diff:+.4f} +/- {se:.4f}  (t = {t:.2f})")
        # Sign matters and the previous version ignored it, printing "SUPPORTS ... moves
        # newer documents up by -0.241". The prediction is that bleed is LEAST
        # favourable to the oldest documents, so the statistic should RISE with era.
        # Here it falls, which is the opposite.
        rising = diff > 0
        if abs(t) >= 2.0:
            solid_verdict = (
                ("SUPPORTS the era explanation -- the statistic rises with era, so "
                 f"bleed is less favourable to {older_name} documents: {older_name} "
                 f"{a['mean_rank_delta']:+.3f} against {newer_name} "
                 f"{b['mean_rank_delta']:+.3f} (t={t:.2f})")
                if rising else
                ("CONTRADICTS the era explanation -- the statistic FALLS with era, so "
                 f"bleed is if anything slightly MORE favourable to {older_name} "
                 f"documents: {older_name} {a['mean_rank_delta']:+.3f} against "
                 f"{newer_name} {b['mean_rank_delta']:+.3f} (t={t:.2f}). The "
                 "prediction said the opposite and is withdrawn."))
        else:
            solid_verdict = (
                f"INCONCLUSIVE -- the {older_name}/{newer_name} difference of "
                f"{diff:+.3f} rank positions is {abs(t):.2f} sigma, below the 2.0 bar")
        print(f"    -> {solid_verdict}")
        out_solid = {"buckets": [older_name, newer_name], "difference": round(diff, 4),
                     "standard_error": round(se, 4), "t": round(t, 3),
                     "verdict": solid_verdict}
    else:
        out_solid = {"verdict": "INSUFFICIENT -- fewer than two buckets clear the floor",
                     "populated": {n: table[n]["n_documents"] for n in ordered}}

    out = {"n_queries": len(qs), "pool": args.pool,
           "bleed_alpha": cfg.fusion.bleed_alpha, "bleed_mode": cfg.fusion.bleed_mode,
           "by_era": table, "verdict": verdict,
           "comparison_among_populated": out_solid,
           "min_documents_for_verdict": MIN_DOCS,
           "method": "rank movement (base rank - bled rank) per candidate, bucketed by "
                     "the publication year of the candidate's document. Rank rather "
                     "than score because sibling_bleed min-max normalises to [0,1] "
                     "while BM25 scores are ~10-25, so the two are not comparable.",
           "elapsed_s": round(time.time() - t0, 1)}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())