"""Build the section-dependency graph in parts from per-shard chunk files, then merge.

At 118M full-text rows (pmcexpanded) one DependencyGraph.build would hold several
billion directed edges before dedup, beyond any host here. Every edge it adds joins
two chunks of the same document (cross-document edges are only added when the
caller passes them, and cmd_dependency_graph never does), and no document spans two
input shards, so the graph is block-diagonal by shard range: each part is built
independently over its rows and the merge only offsets row ids and edge pointers.

  part   claim a part (a contiguous range of input shards from parallel_store.py's
         chunk directory) and save its graph over local rows to parts/part-NNNN.pkl
  merge  stitch the parts, in order, into <index>/dependency_graph.pkl

The merged graph equals DependencyGraph.build over the assembled store
(tests/test_parallel_graph.py).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.config import PipelineConfig  # noqa: E402
from medrag.index.dependency_graph import DependencyGraph  # noqa: E402


def shard_stems(chunk_dir: Path) -> list[str]:
    return sorted(p.name.removesuffix(".done") for p in chunk_dir.glob("*.done"))


def part_ranges(stems: list[str], n_parts: int) -> list[list[str]]:
    size = -(-len(stems) // n_parts)
    return [stems[i:i + size] for i in range(0, len(stems), size)]


def build_part(chunk_dir: Path, stems: list[str], row0: int) -> dict:
    kinds: list[str] = []
    docs: list[str] = []
    for stem in stems:
        with gzip.open(chunk_dir / f"{stem}.chunks.jsonl.gz", "rb") as fh:
            for line in fh:
                d = json.loads(line)
                kinds.append(sys.intern(d.get("section_kind") or ""))
                docs.append(d.get("doc_id") or "")
    g = DependencyGraph.build(range(len(kinds)), kind_of=kinds.__getitem__,
                              doc_of=docs.__getitem__)
    return {"row0": row0, "n": g.n_nodes, "indptr": g.indptr, "indices": g.indices,
            "weights": g.weights, "section_kind": g.section_kind, "doc_ids": g.doc_ids,
            "doc_start": g.doc_start, "doc_end": g.doc_end}


def merge_parts(parts: list[dict]) -> DependencyGraph:
    n = sum(p["n"] for p in parts)
    indptr = [np.zeros(1, dtype=np.int64)]
    indices, weights, kinds = [], [], []
    docs: list[tuple[str, int, int]] = []
    edges = 0
    for p in parts:
        r0 = p["row0"]
        indptr.append(p["indptr"][1:] + edges)
        indices.append(p["indices"] + np.int32(r0))
        weights.append(p["weights"])
        kinds.extend(sys.intern(k) for k in p["section_kind"])
        docs.extend((d, int(s) + r0, int(e) + r0)
                    for d, s, e in zip(p["doc_ids"], p["doc_start"], p["doc_end"]))
        edges += len(p["indices"])
    docs.sort(key=lambda t: t[0])
    if any(docs[i][0] == docs[i + 1][0] for i in range(len(docs) - 1)):
        raise ValueError("a document appears in two parts; its rows were not contiguous")
    return DependencyGraph(
        n_nodes=n,
        indptr=np.concatenate(indptr),
        indices=np.concatenate(indices) if indices else np.zeros(0, np.int32),
        weights=np.concatenate(weights) if weights else np.zeros(0, np.float32),
        section_kind=kinds,
        doc_start=np.array([s for _, s, _ in docs], dtype=np.int64),
        doc_end=np.array([e for _, _, e in docs], dtype=np.int64),
        doc_ids=[d for d, _, _ in docs],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["part", "merge"])
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--chunks", required=True, help="parallel_store.py chunk directory")
    ap.add_argument("--parts", type=int, default=24)
    ap.add_argument("--host", default="")
    args = ap.parse_args()
    cfg = PipelineConfig.load(args.config)._finalize()
    chunk_dir = Path(args.chunks)
    out = chunk_dir / "graph_parts"
    stems = shard_stems(chunk_dir)
    counts = {s: json.loads((chunk_dir / f"{s}.done").read_text())["n_chunks"] for s in stems}
    ranges = part_ranges(stems, args.parts)
    if args.step == "part":
        (out / "claims").mkdir(parents=True, exist_ok=True)
        row0 = 0
        for k, rng in enumerate(ranges):
            first = row0
            row0 += sum(counts[s] for s in rng)
            if (out / f"part-{k:04d}.pkl").exists():
                continue
            try:
                os.mkdir(out / "claims" / f"p{k}")
            except FileExistsError:
                continue
            t0 = time.time()
            part = build_part(chunk_dir, rng, first)
            tmp = out / f"part-{k:04d}.pkl.tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(part, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(out / f"part-{k:04d}.pkl")
            print(f"{args.host} part {k}: {part['n']} rows, {len(part['indices'])} directed edges, "
                  f"{time.time() - t0:.0f}s", flush=True)
        return 0
    parts = []
    for k in range(len(ranges)):
        with open(out / f"part-{k:04d}.pkl", "rb") as fh:
            parts.append(pickle.load(fh))
    g = merge_parts(parts)
    del parts
    if g.n_nodes != sum(counts.values()):
        raise SystemExit(f"merged {g.n_nodes} rows, chunk files hold {sum(counts.values())}")
    g.save(cfg.paths.index_dir / "dependency_graph.pkl")
    print(json.dumps(g.stats()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
