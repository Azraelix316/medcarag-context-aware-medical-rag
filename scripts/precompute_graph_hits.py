"""MeSH graph hits for a query file, for run_mode_ablation.py --graph-run.

The retriever's graph and hybrid3 modes call ``graph.search(query, top_k)`` with
``top_k = max(fusion.dense_top_k // 3, 20)`` (MedRAGRetriever._graph_search). At 20M
documents GraphRetriever cannot do that per query, so this runs the identical search
through InvertedGraphSearch once per query and writes the ranked documents.

  python scripts/precompute_graph_hits.py --config configs/default.yaml \
      --config configs/abstracts20m.yaml --queries q.jsonl --out graph.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.config import PipelineConfig  # noqa: E402
from medrag.index.graph_retriever import InvertedGraphSearch, load_graph_index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--queries", required=True, help="JSONL of {qid, query}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)._finalize()
    top_k = max(cfg.fusion.dense_top_k // 3, 20)
    t0 = time.time()
    graph, doc_map = load_graph_index(cfg.paths.index_dir)
    search = InvertedGraphSearch(graph, doc_map)
    del doc_map
    print(f"  graph index loaded in {time.time() - t0:.0f}s; top_k={top_k}", flush=True)
    rows = [json.loads(line) for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    tmp = Path(args.out + ".tmp")
    t0 = time.time()
    with open(tmp, "w", encoding="utf-8") as fh:
        for i, rec in enumerate(rows):
            hits = search.search(rec["query"], top_k=top_k)
            fh.write(json.dumps({"qid": rec["qid"], "graph": hits}) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(rows)} | {time.time() - t0:.0f}s", flush=True)
    tmp.replace(args.out)
    print(f"  {len(rows)} queries -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
