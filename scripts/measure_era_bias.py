"""Era bias in citation coverage, measured on the corpora themselves.

Why this exists
---------------
The objective requires the era bias to be reported honestly, and the version in
``docs/citation_coverage_era_bias.md`` was measured two ways that both needed
qualifying: 150 randomly sampled live PMC articles per era window, and "~20% / 7.3"
versus "~72% / 33" on abstracts where the numbers were approximate and carried tildes.

Both corpora have since been backfilled -- PMC from 0% to 99.6% of records carrying a
bibliography, and the abstract corpus now holds 960,880 records -- so the measurement
can be taken from the corpus files instead of from samples. Every number below comes
from a full pass over every document, with no sampling and no tildes.

Why the bias is worth measuring at all
--------------------------------------
A co-citation graph is only as representative as its bibliographies. If older records
carry fewer references, the graph's older end is thinner, and any claim resting on it
is a claim about recent literature. This script reports coverage and references-per-
document per era window, per corpus, because the two corpora turn out to have opposite
problems and a single figure would hide that.

Usage::

    scripts/measure_era_bias.py --corpus pmc     --root data/pmcmerged
    scripts/measure_era_bias.py --corpus pubmed  --root data/newcorpus
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from pathlib import Path
import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WINDOWS = [(1996, 2000), (2001, 2005), (2006, 2010), (2011, 2015),
           (2016, 2020), (2021, 2026)]


def window_of(year: int) -> str:
    for lo, hi in WINDOWS:
        if lo <= year <= hi:
            return f"{lo}-{hi}"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=["pmc", "pubmed"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pattern = f"{args.root}/corpus/{args.corpus}/{args.corpus}-*.jsonl.gz"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  no shards matched {pattern}")
        return 1
    print(f"  {len(files)} shards under {args.root}/corpus/{args.corpus}", flush=True)

    n_by_win: dict[str, int] = defaultdict(int)
    cov_by_win: dict[str, int] = defaultdict(int)
    refs_by_win: dict[str, list[int]] = defaultdict(list)
    # Total references per window, over ALL documents including those with none.
    #
    # This is a different number from refs-per-document-with-citations, and conflating
    # them is how "~7.3 versus ~33" can be read as measuring coverage when it is
    # partly measuring how many references an average record carries. Reported
    # separately so the two cannot be confused.
    refs_all_by_win: dict[str, int] = defaultdict(int)
    refs_sum = 0
    n_docs = 0
    n_undated = 0
    t0 = time.time()

    for fi, path in enumerate(files):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                n_docs += 1
                year = int(d.get("year") or 0)
                if not year:
                    n_undated += 1
                w = window_of(year) if year else "other"
                refs = d.get("citations") or ()
                n_by_win[w] += 1
                refs_sum += len(refs)
                refs_all_by_win[w] += len(refs)
                if refs:
                    cov_by_win[w] += 1
                    refs_by_win[w].append(len(refs))
        if (fi + 1) % 20 == 0:
            rate = (n_docs / (time.time() - t0)) if time.time() > t0 else 0
            print(f"  {fi + 1}/{len(files)} shards | {n_docs:,} docs | {rate:.0f}/s",
                  flush=True)

    def median(xs: list[int]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        return float(s[len(s) // 2])

    print(f"\n  {args.corpus.upper()}: {n_docs:,} documents, {n_undated:,} without a year")
    print(f"  {'window':<12s}{'docs':>10s}{'coverage':>11s}"
          f"{'refs/doc*':>11s}{'all-docs':>10s}{'median':>9s}")
    print("  * over documents carrying a bibliography; 'all-docs' is over every "
          "document in the window")
    rows: dict[str, dict] = {}
    for w in [f"{lo}-{hi}" for lo, hi in WINDOWS] + ["other"]:
        n = n_by_win.get(w, 0)
        if not n:
            print(f"  {w:<12s}{0:>10d}{'--':>11s}{'--':>11s}{'--':>9s}")
            continue
        cov = cov_by_win[w] / n
        mean = (sum(refs_by_win[w]) / len(refs_by_win[w])) if refs_by_win[w] else 0.0
        med = median(refs_by_win[w])
        per_all = refs_all_by_win[w] / n
        rows[w] = {"n_documents": n, "coverage": round(cov, 4),
                   "refs_per_doc_with_citations": round(mean, 2),
                   "refs_per_document_all": round(per_all, 2),
                   "median_refs": med}
        print(f"  {w:<12s}{n:>10,}{cov:>10.1%}{mean:>11.1f}{per_all:>10.1f}{med:>9.0f}")

    out = {
        "corpus": args.corpus,
        "root": args.root,
        "n_shards": len(files),
        "n_documents": n_docs,
        "n_undated": n_undated,
        "mean_refs_all_documents": round(refs_sum / max(n_docs, 1), 2),
        "windows": rows,
        "method": "full pass over every shard; no sampling; coverage is the share of "
                  "documents in the window carrying a non-empty bibliography",
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())