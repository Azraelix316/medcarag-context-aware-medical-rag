"""How much of the benchmark corpus is actually in the index, and which decades.

Why this exists
---------------
Every abstract-corpus measurement in this repository -- the bpref comparisons, the
citation-label agreement, the sibling-bleed contradiction -- was computed against a
chunk index holding 624,394 documents while the corpus beside it holds 1,022,707. The
gap was recorded as one number ("624,489 dense index / 1,118,153 full") and not as what
it actually is: an index that is almost entirely one decade.

Measured over a 180,000-document stratified sample before this script existed:

    decade   in corpus    indexed   coverage
    2000s      29,369         10       0.0%
    2010s      51,683        762       1.5%
    2020s      98,948     91,330      92.3%

That compounds the citation-coverage era bias already measured (6.6% of 1996-2000
abstracts carry a bibliography, 55.1% of 2021-2026 do). Both point the same way, so a
retrieval number from this index is recent-biased twice: old records are absent from
the co-citation graph *and* absent from the index that does the retrieving.

What this does not say
----------------------
It does not say the index is wrong -- it verifies clean, 119 of 119 blocks and every
sampled row retrieving itself at cosine 1.000000. It says the index was built from a
different, narrower corpus than the one now sitting beside it, and that difference is
invisible unless someone compares the two by era.

Method: full pass over every corpus shard, matching on the ``pmid:NNNNNNNNN`` document
identifier the store's row map uses. No sampling for the headline figures; the decade
breakdown is the same full pass, not a subsample of it.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/newcorpus")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--out", default="data/runs/index_coverage.json")
    args = ap.parse_args()

    rowmap = Path(args.index_dir) / "store" / "rowmap.json"
    rows = json.loads(rowmap.read_text())["rows"]
    indexed = {r.rsplit("#", 1)[0] for r in rows}
    n_rows = len(rows)
    print(f"  index: {n_rows:,} chunk rows over {len(indexed):,} documents", flush=True)

    files = sorted(glob.glob(f"{args.corpus}/corpus/pubmed/pubmed-*.jsonl.gz"))
    if not files:
        print(f"  no shards under {args.corpus}")
        return 1
    print(f"  corpus: {len(files)} shards", flush=True)

    per_decade: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    seen_in_corpus: set[str] = set()
    n = 0
    t0 = time.time()
    for fi, path in enumerate(files):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                doc_id = d.get("doc_id") or (f"pmid:{d['pmid']}" if d.get("pmid") else None)
                if not doc_id or doc_id in seen_in_corpus:
                    continue
                seen_in_corpus.add(doc_id)
                n += 1
                decade = (int(d.get("year") or 0) // 10) * 10
                key = str(decade) if decade else "undated"
                per_decade[key][0] += 1
                if doc_id in indexed:
                    per_decade[key][1] += 1
        if (fi + 1) % 15 == 0:
            rate = n / max(time.time() - t0, 1e-9)
            print(f"  {fi + 1}/{len(files)} shards | {n:,} docs | {rate:.0f}/s", flush=True)

    hit = sum(v[1] for v in per_decade.values())
    print(f"\n  corpus: {n:,} unique documents; index covers {hit:,} "
          f"({100 * hit / max(n, 1):.1f}%), missing {n - hit:,}")
    print(f"\n  {'decade':>10s}{'in corpus':>13s}{'indexed':>12s}{'coverage':>11s}")
    table = {}
    for key in sorted(per_decade, key=lambda k: (k == "undated", k)):
        tot, cov = per_decade[key]
        pct = 100 * cov / max(tot, 1)
        table[key] = {"in_corpus": tot, "indexed": cov, "coverage": round(pct, 2)}
        print(f"  {key:>10s}{tot:>13,}{cov:>12,}{pct:>10.1f}%")

    out = {
        "corpus": args.corpus,
        "index_dir": args.index_dir,
        "index_chunk_rows": n_rows,
        "index_documents": len(indexed),
        "corpus_documents": n,
        "covered": hit,
        "missing": n - hit,
        "coverage": round(100 * hit / max(n, 1), 2),
        "by_decade": table,
        "method": "full pass over every corpus shard; documents matched on the "
                  "pmid:NNNNNNNNN identifier the store row map uses",
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())