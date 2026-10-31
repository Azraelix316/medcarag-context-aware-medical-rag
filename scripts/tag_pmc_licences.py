"""Tag every document of a PMC corpus by reuse licence: commercial, non-commercial or other.

The crawl used PMC's ``"open access"[filter]``, which mixes licences, and the records do
not carry one. NLM's redistribution terms allow distributing only data licensed for it,
so a release has to know each article's licence. PMC exposes licences as E-utilities
filters; the two groups below are taken verbatim from the PMC Article Datasets README
(https://pmc-oa-opendata.s3.amazonaws.com/README.txt, section 5):

  commercial      cc0, cc_by, cc_by-sa, cc_by-nd
  non-commercial  cc_by-nc, cc_by-nc-nd, cc_by-nc-sa

Only PMCID lists are fetched, never article text. A document in neither list is
"other": a custom publisher licence, or TDM-only author-manuscript terms.

Writes, under --out-dir:
  pmc_ids_<group>.txt.gz       the full PMCID list per group, for re-use
  licence_<corpus>.json.gz     {doc_id: group} for each corpus
  licence_counts.json          the counts, and any window NCBI would not page through
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medrag.config import PipelineConfig  # noqa: E402
from medrag.data.eutils import EutilsClient  # noqa: E402

GROUPS = {
    "commercial": "(cc0_license[Filter] OR cc_by_license[Filter] OR "
                  "cc_by-sa_license[Filter] OR cc_by-nd_license[Filter])",
    "noncommercial": "(cc_by-nc_license[Filter] OR cc_by-nc-nd_license[Filter] OR "
                     "cc_by-nc-sa_license[Filter])",
}
SCOPE = "(open_access[Filter] OR author_manuscript[Filter])"


def fetch_group(client: EutilsClient, query: str) -> tuple[set[str], list[str]]:
    """All PMCIDs matching ``query``, plus any day windows NCBI capped at 10,000."""
    ids: set[str] = set()
    per_window: collections.Counter[str] = collections.Counter()
    for label, uid in client.iter_ids_partitioned(query, db="pmc", min_date="1700/01/01",
                                                  max_date="2026/12/31"):
        ids.add(f"PMC{uid}")
        per_window[label] += 1
    # iter_ids_partitioned takes at most 10,000 from a window it cannot split further.
    capped = [w for w, n in per_window.items() if n >= EutilsClient.MAX_RECORDS_PER_QUERY]
    return ids, capped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", default=["configs/default.yaml"])
    ap.add_argument("--corpus", action="append", required=True,
                    help="corpus root with corpus/pmc/pmc-*.jsonl.gz; repeatable")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    acq = PipelineConfig.load(args.config)._finalize().acquire
    client = EutilsClient(api_key=acq.api_key, email=acq.email, tool=acq.tool,
                          rate=acq.max_requests_per_second, timeout=acq.request_timeout,
                          max_retries=acq.max_retries)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    group_ids: dict[str, set[str]] = {}
    report: dict = {"groups": {}, "corpora": {}}
    for name, filt in GROUPS.items():
        t0 = time.time()
        ids, capped = fetch_group(client, f"{filt} AND {SCOPE}")
        group_ids[name] = ids
        with gzip.open(out / f"pmc_ids_{name}.txt.gz", "wt") as fh:
            fh.write("\n".join(sorted(ids)) + "\n")
        report["groups"][name] = {"query": f"{filt} AND {SCOPE}", "pmcids": len(ids),
                                  "capped_windows": capped, "seconds": round(time.time() - t0)}
        print(f"  {name}: {len(ids):,} PMCIDs, {len(capped)} capped windows "
              f"({time.time() - t0:.0f}s)", flush=True)
    both = group_ids["commercial"] & group_ids["noncommercial"]
    report["groups"]["in_both"] = len(both)

    for root in args.corpus:
        name = Path(root).name
        tags: dict[str, str] = {}
        for path in sorted(glob.glob(f"{root}/corpus/pmc/pmc-*.jsonl.gz")):
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    pmcid = d.get("pmcid") or ""
                    if pmcid in group_ids["commercial"]:
                        tags[d["doc_id"]] = "commercial"
                    elif pmcid in group_ids["noncommercial"]:
                        tags[d["doc_id"]] = "noncommercial"
                    else:
                        tags[d["doc_id"]] = "other"
        counts = collections.Counter(tags.values())
        report["corpora"][name] = {"documents": len(tags), **{k: counts.get(k, 0) for k in
                                   ("commercial", "noncommercial", "other")}}
        with gzip.open(out / f"licence_{name}.json.gz", "wt") as fh:
            json.dump(tags, fh)
        print(f"  {name}: {len(tags):,} documents -> " +
              ", ".join(f"{k} {counts.get(k, 0):,} ({counts.get(k, 0) / max(len(tags), 1):.1%})"
                        for k in ("commercial", "noncommercial", "other")), flush=True)

    report["requests"] = client.n_requests
    (out / "licence_counts.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "groups"}, indent=2))
    print(f"  in both lists: {len(both):,}; requests: {client.n_requests:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
