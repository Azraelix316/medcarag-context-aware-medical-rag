"""Random-retrieval floor for the mode ablation.

For each query and each named arm, ``random@<arm>`` replaces the arm's retrieved rows
with as many rows drawn uniformly from the whole chunk store. It answers "what does
retrieval itself buy?": the same readiness scoring, the same passage count, no
retrieval. Rows are read from run_mode_ablation.py's stored per-arm rows, so the arm
side is the committed ablation, re-scored.

Output is in run_mode_ablation.py's --per-query-out format, so
scripts/paired_bootstrap.py --pairs sparse:random@sparse,... compares them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.config import PipelineConfig  # noqa: E402
from medrag.eval.readiness import aggregate_readiness, readiness_from_rows  # noqa: E402
from medrag.index.store import ChunkStore  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_mode_ablation", Path(__file__).resolve().parent / "run_mode_ablation.py")
_rma = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rma)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--n-queries", type=int, default=2000)
    ap.add_argument("--rows-from", required=True, metavar="JSON,...")
    ap.add_argument("--arms", default="sparse,pagectx")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-query-out", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)._finalize()
    store = ChunkStore.load(cfg.paths.index_dir / "store")
    n_rows = len(store)
    stored: dict[str, dict[str, list[int]]] = {}
    for path in args.rows_from.split(","):
        with open(path) as fh:
            for mode, per_q in json.load(fh).items():
                stored.setdefault(mode, {}).update(per_q)
    qs = _rma._sample_queries(args.qrels, args.n_queries)
    arms = args.arms.split(",")
    rng = random.Random(args.seed)
    per: dict[str, list] = {}
    t0 = time.time()
    for i, q in enumerate(qs):
        for arm in arms:
            rows = stored[arm][q.qid]
            rand_rows = rng.sample(range(n_rows), len(rows))
            for name, rr in ((arm, rows), (f"random@{arm}", rand_rows)):
                per.setdefault(name, []).append(
                    readiness_from_rows(rr, store, q.relevant_docs, qid=q.qid))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(qs)} | {time.time() - t0:.0f}s", flush=True)
    with open(args.per_query_out, "w") as fh:
        json.dump({m: [x.to_dict() for x in v] for m, v in per.items()}, fh)
    summary = {m: aggregate_readiness(v).summary() for m, v in per.items()}
    with open(args.out, "w") as fh:
        json.dump({"n_queries": len(qs), "seed": args.seed, "store_rows": n_rows,
                   "readiness": summary}, fh, indent=2)
    for m, s in summary.items():
        print(f"    {m:16s} found {s['n_found_gold']:5d}  readiness {s['readiness']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
