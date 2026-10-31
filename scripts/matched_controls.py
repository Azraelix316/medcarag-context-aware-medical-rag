"""Passage-count controls for the context arms of the mode ablation.

``context``, ``bleedctx`` and ``pagectx`` return their base top-k followed by up to
``context_seeds * context_per_seed`` framing passages, so they hand the readiness
metric more text than ``sparse`` does (about 18 passages against 10). Two controls
separate "more passages" from "the right passages", per query and per arm:

``sparse@<arm>``   the sparse ranking cut at the arm's own passage count for that
                   query: as many passages, chosen by BM25 alone.
``randsec@<arm>``  the arm's own list with every appended framing passage swapped for
                   a uniformly random other chunk *of the same document*: the same
                   documents and the same count, with the section choice randomised.

Both are scored with the same readiness_from_rows as the arms themselves, and written
in run_mode_ablation.py's --per-query-out format so scripts/paired_bootstrap.py
--pairs can compare each arm with its controls.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.config import PipelineConfig  # noqa: E402
from medrag.eval.readiness import aggregate_readiness, readiness_from_rows  # noqa: E402
from medrag.index.retriever import MODE_SPARSE, load_index  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_mode_ablation", Path(__file__).resolve().parent / "run_mode_ablation.py")
_rma = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rma)

CONTEXT_ARMS = ("context", "bleedctx", "pagectx")


class DocRanges:
    """Row -> its document's [start, end) row range, from the dependency graph."""

    def __init__(self, doc_start: np.ndarray, doc_end: np.ndarray) -> None:
        self.order = np.argsort(doc_start, kind="stable")
        self.starts = np.asarray(doc_start)[self.order]
        self.ends = np.asarray(doc_end)[self.order]

    def of(self, row: int) -> tuple[int, int]:
        i = int(np.searchsorted(self.starts, row, side="right")) - 1
        if i < 0 or not (self.starts[i] <= row < self.ends[i]):
            raise KeyError(f"row {row} is in no document range")
        return int(self.starts[i]), int(self.ends[i])


def randomise_sections(rows: list[int], top_k: int, ranges: DocRanges,
                       rng: random.Random) -> tuple[list[int], int]:
    """Keep rows[:top_k]; swap each later row for a random unused row of its document.

    Returns the new list and how many appended rows had no unused alternative and
    were kept as they were.
    """
    out = list(rows[:top_k])
    used = set(rows)
    kept = 0
    for row in rows[top_k:]:
        lo, hi = ranges.of(row)
        free = [r for r in range(lo, hi) if r not in used]
        if not free:
            out.append(row)
            kept += 1
            continue
        pick = rng.choice(free)
        used.add(pick)
        out.append(pick)
    return out, kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--n-queries", type=int, default=2000)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--sparse-run", required=True)
    ap.add_argument("--sparse-queries", required=True)
    ap.add_argument("--rows-from", required=True, metavar="JSON,...")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-query-out", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)._finalize()
    r = load_index(cfg, with_reranker=False, with_graph=False, with_dependency=True,
                   with_dense=False)
    from medrag.index.segmented import PrecomputedBM25
    r.bm25 = PrecomputedBM25.from_run(args.sparse_run, args.sparse_queries, n_docs=len(r.store))
    ranges = DocRanges(r.dependency.doc_start, r.dependency.doc_end)
    stored: dict[str, dict[str, list[int]]] = {}
    for path in args.rows_from.split(","):
        with open(path) as fh:
            for mode, per_q in json.load(fh).items():
                stored.setdefault(mode, {}).update(per_q)
    qs = _rma._sample_queries(args.qrels, args.n_queries)
    arms = [a for a in CONTEXT_ARMS if a in stored]
    rng = random.Random(args.seed)
    per: dict[str, list] = {}
    kept_total = appended_total = 0
    t0 = time.time()
    for i, q in enumerate(qs):
        for arm in arms:
            rows = stored[arm][q.qid]
            n = len(rows)
            sp = r.retrieve(q.text, top_k=n, mode=MODE_SPARSE, include_text=False)
            sp_rows = [int(p.row_id) for p in sp.passages]
            rs_rows, kept = randomise_sections(rows, args.top_k, ranges, rng)
            kept_total += kept
            appended_total += max(0, n - args.top_k)
            for name, rr in ((arm, rows), (f"sparse@{arm}", sp_rows), (f"randsec@{arm}", rs_rows)):
                per.setdefault(name, []).append(
                    readiness_from_rows(rr, r.store, q.relevant_docs, qid=q.qid))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(qs)} | {time.time() - t0:.0f}s", flush=True)
    with open(args.per_query_out, "w") as fh:
        json.dump({m: [x.to_dict() for x in v] for m, v in per.items()}, fh)
    summary = {m: aggregate_readiness(v).summary() for m, v in per.items()}
    out = {"n_queries": len(qs), "top_k": args.top_k, "seed": args.seed,
           "appended_passages": appended_total,
           "randsec_kept_unchanged": kept_total, "readiness": summary}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    for m, s in summary.items():
        print(f"    {m:18s} readiness {s['readiness']:.4f}  caveat {s['caveat_rate']:.4f}  "
              f"bare {s['bare_rate']:.4f}")
    print(f"  randsec kept {kept_total}/{appended_total} appended rows unchanged -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
