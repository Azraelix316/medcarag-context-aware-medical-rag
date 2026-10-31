"""Attach MeSH descriptors to PMC documents by PubMed id, for the circularity control.

Why
---
The citation-derived circularity control (objective 4) cannot be built on PMC at all.
Seed selection requires a MeSH descriptor, because seeds vote *per descriptor*, and PMC
JATS records carry none: 6 descriptors across 555,142 documents. So the control was
reported as structurally inapplicable there.

But PMC and PubMed share identifiers. PMC-OA articles are indexed in PubMed under the
same PMID, and the abstract corpus carries MeSH for the records it crawled. Joining the
two on PMID gives 103,737 of 554,564 PMC PMIDs a MeSH descriptor -- **18.71%**.

That subset is not a random sample of PMC and must not be presented as one. It is
limited by the abstract corpus covering 1,022,706 of roughly 37M PubMed records, and it
was crawled over six publication-type slices, so the overlap is **enriched for RCTs,
systematic reviews, meta-analyses and guidelines** -- the publication types that carry
usable reasoning evidence. A control run on it describes that slice of PMC, not PMC.

The alternative was inventing a topic keying for PMC from its own titles and sections.
That would construct the control without the bias, but it would also break the
agreement half of the measurement, which is specifically a comparison between MeSH-derived
labels and bibliography-derived labels. Borrowing MeSH from PubMed keeps that comparison
honest, at the price of a stated 18.7% coverage.

Output
------
A sidecar, not a corpus rewrite. The PMC corpus stays as crawled; this writes
``pmc_mesh.json.gz`` beside it. Rewriting 103,737 of 555,142 records in place would
change the corpus the retrieval experiments were measured on, which is exactly the kind
of quiet edit that makes an earlier number unreproducible.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pmc-corpus", default="data/pmcmerged")
    ap.add_argument("--abstract-corpus", default="data/newcorpus")
    ap.add_argument("--out", default=None,
                    help="defaults to <pmc-corpus>/pmc_mesh.json.gz")
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(args.pmc_corpus) / "pmc_mesh.json.gz"

    # Pass 1: the PMIDs the PMC corpus needs. Done first so pass 2 keeps MeSH only for
    # those: keyed over the whole abstract corpus, 20M PMIDs of descriptor lists do not
    # fit in memory, while the PMC side is at most a few million PMIDs.
    t0 = time.time()
    pmc_files = sorted(glob.glob(f"{args.pmc_corpus}/corpus/pmc/pmc-*.jsonl.gz"))
    if not pmc_files:
        print(f"  no PMC shards under {args.pmc_corpus}")
        return 1
    doc_by_pmid: dict[str, str] = {}
    n_pmc = 0
    for path in pmc_files:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                n_pmc += 1
                pmid = d.get("pmid")
                if pmid:
                    doc_by_pmid[str(pmid)] = d.get("doc_id") or f"pmid:{pmid}"
    n_pmids = len(doc_by_pmid)
    print(f"  PMC corpus: {n_pmc:,} documents, {n_pmids:,} with a PMID "
          f"({time.time() - t0:.0f}s)", flush=True)

    # Pass 2: MeSH from the abstract corpus, for those PMIDs only.
    t1 = time.time()
    files = sorted(glob.glob(f"{args.abstract_corpus}/corpus/pubmed/pubmed-*.jsonl.gz"))
    if not files:
        print(f"  no abstract shards under {args.abstract_corpus}")
        return 1
    attached: dict[str, list[str]] = {}
    ui_by_descriptor: dict[str, str] = {}
    n_abs = 0
    desc_count = 0
    for path in files:
        try:
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    n_abs += 1
                    doc_id = doc_by_pmid.get(str(d.get("pmid") or ""))
                    if doc_id is None or doc_id in attached:
                        continue
                    descs = set()
                    for m in d.get("mesh_terms") or []:
                        name = (m.get("descriptor") or "").strip()
                        if not name:
                            continue
                        name = sys.intern(name)
                        descs.add(name)
                        if m.get("ui") and name not in ui_by_descriptor:
                            ui_by_descriptor[name] = m["ui"]
                    if descs:
                        attached[doc_id] = sorted(descs)
                        desc_count += len(descs)
        except (EOFError, OSError, json.JSONDecodeError) as exc:
            # A truncated shard costs its tail, not the whole join.
            print(f"  shard {Path(path).name} incomplete ({exc}); using what was read")
    covered = len(attached)
    print(f"  abstract corpus: {n_abs:,} records ({time.time() - t1:.0f}s)")
    print(f"  attached MeSH to {covered:,} documents ({100 * covered / max(n_pmc, 1):.2f}%)")
    if covered:
        print(f"  mean descriptors per attached document: "
              f"{desc_count / covered:.1f}")

    descriptors = sorted({x for v in attached.values() for x in v})
    print(f"  distinct MeSH descriptors now present on PMC: {len(descriptors):,}")

    payload = {
        "pmc_documents_with_mesh": covered,
        "pmc_documents_total": n_pmc,
        "coverage": round(100 * covered / max(n_pmc, 1), 4),
        "n_descriptors": len(descriptors),
        "mean_descriptors": round(desc_count / max(covered, 1), 2),
        "abstract_corpus": args.abstract_corpus,
        "abstract_records": n_abs,
        "caveat": f"MeSH borrowed from {n_abs:,} PubMed records in {args.abstract_corpus}. "
                  "A PMC document is labelled only if its PMID is there; when the abstract "
                  "corpus is a publication-type slice rather than a sweep, the labelled "
                  "subset is enriched for that slice and does not describe PMC as a whole.",
        "ui_by_descriptor": ui_by_descriptor,
        "by_doc_id": attached,
    }
    with gzip.open(out, "wt") as fh:
        json.dump(payload, fh)
    size_mb = out.stat().st_size / 2**20
    print(f"  -> {out} ({size_mb:.1f} MiB compressed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
