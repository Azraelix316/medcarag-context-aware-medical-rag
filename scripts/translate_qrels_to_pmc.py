"""Map PubMed gold document ids onto their PMC ids for the full-text corpus.

The qrels were built against the abstract corpus, whose documents are keyed
``pmid:<n>``. The PMC Open Access corpus is keyed ``pmcid:<PMCnnnnnnn>``. Evaluating
the same queries against the same gold on the full-text corpus therefore needs a
translation, and translating is much better than rebuilding:

* the query set and gold set stay identical, so the two corpora are directly
  comparable -- which is the whole point of running the ablation twice;
* the gold set is not regenerated from the corpus being evaluated, so the
  evaluation is not circular in the way MeSH-derived pseudo-labels already are.

The mapping comes from the PMC records themselves, which carry both ``pmid`` and
``pmcid``. Coverage is partial and must be reported: a gold document is only
evaluable if that paper is in the PMC Open Access subset, and PMC OA is heavily
skewed toward recent RCTs. Queries are kept only while enough of their gold survives,
because a query with three surviving gold documents measures something different from
one with 141.
"""

from __future__ import annotations

import argparse
import gzip
import glob
import json
import sys
from pathlib import Path


def build_map(corpus_glob: str) -> dict[str, str]:
    """pmid:<n> -> pmcid:<PMCnnnnnnn> for every record carrying both."""
    out: dict[str, str] = {}
    for path in sorted(glob.glob(corpus_glob)):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                pmid, pmcid = d.get("pmid"), d.get("pmcid")
                if pmid and pmcid:
                    out[f"pmid:{pmid}"] = f"pmcid:{pmcid}"
    return out


def translate(qrels: list[dict], mapping: dict[str, str], *, min_gold: int = 10
              ) -> tuple[list[dict], dict[str, float]]:
    """Rewrite gold ids onto the PMC namespace, dropping queries left too thin."""
    kept: list[dict] = []
    total_before = total_after = 0
    for q in qrels:
        rel = q.get("relevant_docs") or []
        total_before += len(rel)
        mapped = [mapping[d] for d in rel if d in mapping]
        total_after += len(mapped)
        if len(mapped) < min_gold:
            continue
        out = dict(q)
        out["relevant_docs"] = mapped
        support = q.get("supporting_docs") or []
        out["supporting_docs"] = [mapping[d] for d in support if d in mapping]
        kept.append(out)
    stats = {
        "queries_before": len(qrels),
        "queries_after": len(kept),
        "gold_before": total_before,
        "gold_after": total_after,
        "gold_retained": round(total_after / total_before, 4) if total_before else 0.0,
        "min_gold": min_gold,
    }
    return kept, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--corpus-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-gold", type=int, default=10)
    args = ap.parse_args()

    mapping = build_map(args.corpus_glob)
    print(f"  {len(mapping):,} pmid->pmcid pairs from {args.corpus_glob}")

    qrels = json.load(open(args.qrels))
    kept, stats = translate(qrels, mapping, min_gold=args.min_gold)
    print(json.dumps(stats, indent=2))
    print(f"  gold retention {stats['gold_retained']:.1%} -- the shortfall is papers "
          f"outside the PMC Open Access subset, which is recent-RCT-skewed")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(kept, fh)
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())