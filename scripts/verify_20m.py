"""Verify the abstracts20m artifacts that exceed RAM, one part at a time.

Each part prints a JSON report and writes it to ``--out`` (default
``data/logs/embed20m/verify_<part>.json``). The checks are the ones
``verify_embeddings.py`` and the completion audit apply at 16M, adapted to files
that are streamed rather than loaded:

``embeddings``  manifest vs file size; 200 random blocks compared in full against
                ``embeddings.f32``; one streamed pass for zero rows, norms, and
                exact self-retrieval of 64 sampled rows.
``bm25``        segment coverage and contiguity; BM25 self-retrieval of 32 sampled
                rows, each queried with its own indexed text.
``page``        page_index.pkl loads as a memmap; documents and order match the
                page texts; norms; 20 random blocks compared to their npz.
``graph``       node count; CSR consistency; reciprocity of 3,000 sampled nodes.

Results are reported, not judged: a duplicate passage legitimately outranks a
row's own vector or text, so self-retrieval below 1.0 is a number, not a failure.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medrag.index import segmented as sg  # noqa: E402

BLOCK_ROWS = 20_000


def verify_embeddings(index_dir: Path, rng: np.random.Generator) -> dict:
    manifest = json.loads((index_dir / "embeddings.json").read_text())
    n, dim = manifest["shape"]
    path = index_dir / manifest["file"]
    rep = {"shape": [n, dim], "file_bytes": path.stat().st_size,
           "file_bytes_expected": n * dim * 4}
    mat = sg.load_embedding_matrix(index_dir)
    n_blocks = (n + BLOCK_ROWS - 1) // BLOCK_ROWS
    blocks = sorted(rng.choice(n_blocks, size=min(200, n_blocks), replace=False).tolist())
    mismatched = []
    for b in blocks:
        v = np.load(index_dir / f"embeddings-block{b:05d}.npz")["embeddings"]
        if not np.array_equal(np.asarray(mat[b * BLOCK_ROWS: b * BLOCK_ROWS + len(v)]), v):
            mismatched.append(b)
    rep["blocks_compared"] = len(blocks)
    rep["blocks_mismatched"] = mismatched

    sample = np.sort(rng.choice(n, size=64, replace=False))
    q = np.asarray(mat[sample], dtype=np.float32)
    self_score = (q * q).sum(1)
    best = np.full(len(sample), -np.inf, np.float32)
    best_row = np.full(len(sample), -1, np.int64)
    zero_rows = 0
    nmin, nmax = np.inf, -np.inf
    t0 = time.time()
    step = 1 << 18
    for lo in range(0, n, step):
        x = np.asarray(mat[lo:lo + step], dtype=np.float32)
        norms = np.linalg.norm(x, axis=1)
        zero_rows += int((norms == 0).sum())
        nmin, nmax = min(nmin, float(norms.min())), max(nmax, float(norms.max()))
        s = q @ x.T
        i = s.argmax(1)
        better = s[np.arange(len(sample)), i] > best
        best[better] = s[np.arange(len(sample)), i][better]
        best_row[better] = i[better] + lo
    rep.update({
        "zero_rows": zero_rows, "norm_min": nmin, "norm_max": nmax,
        "self_retrieval_rows": len(sample),
        "self_retrieval_rank1_is_self": int((best_row == sample).sum()),
        "self_retrieval_rank1_ties_self_score": int(np.isclose(best, self_score, atol=1e-5).sum()),
        "stream_pass_s": round(time.time() - t0, 1),
    })
    return rep


def verify_bm25(index_dir: Path, rng: np.random.Generator) -> dict:
    seg_dir = index_dir / "bm25_segments"
    stats = sorted(seg_dir.glob("bm25-seg[0-9][0-9][0-9].stats.pkl"))
    n = sg.store_n_rows(index_dir / "store")
    spans, total = [], 0
    for p in stats:
        with open(p, "rb") as fh:
            st = pickle.load(fh)
        spans.append((st["lo"], st["hi"]))
        total += st["n_docs"]
    contiguous = spans == sorted(spans) and all(
        a[1] == b[0] for a, b in zip(spans, spans[1:])) and spans[0][0] == 0 and spans[-1][1] == n
    rows = np.sort(rng.choice(n, size=32, replace=False)).tolist()
    texts = [next(sg.iter_range_texts(index_dir / "store", r, r + 1)) for r in rows]
    t0 = time.time()
    hits = sg.segmented_bm25_search(seg_dir, texts, 10)
    ranks = []
    for r, h in zip(rows, hits):
        ids = [x for x, _ in h]
        ranks.append(ids.index(r) + 1 if r in ids else None)
    return {"segments": len(stats), "rows_covered": total, "store_rows": n,
            "contiguous": contiguous, "self_retrieval_rows": len(rows),
            "self_rank1": sum(1 for x in ranks if x == 1),
            "self_in_top10": sum(1 for x in ranks if x is not None),
            "ranks": ranks, "search_s": round(time.time() - t0, 1)}


def verify_page(index_dir: Path, rng: np.random.Generator) -> dict:
    from medrag.index.page_index import (PageIndex, page_vector_block_path,
                                         read_page_text_block)

    pages = json.loads((index_dir / "page_texts" / "pages.json").read_text())
    idx = PageIndex.load(index_dir / "page_index.pkl")
    rep = {"n_docs": idx.n_docs, "manifest_docs": pages["n_docs"],
           "memmap": isinstance(idx.matrix, np.memmap), "dim": idx.dim}
    blocks = sorted(rng.choice(pages["n_blocks"], size=min(20, pages["n_blocks"]),
                               replace=False).tolist())
    bad = []
    for b in blocks:
        recs = read_page_text_block(index_dir, b)
        lo = b * pages["block_docs"]
        v = np.load(page_vector_block_path(index_dir, b))["embeddings"].astype(np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        if (idx.doc_ids[lo:lo + len(recs)] != [r["doc"] for r in recs]
                or idx.row_ids[lo:lo + len(recs)] != [r["row"] for r in recs]
                or not np.array_equal(np.asarray(idx.matrix[lo:lo + len(recs)]), v / norms)):
            bad.append(b)
    norms = np.linalg.norm(np.asarray(idx.matrix[: min(idx.n_docs, 2_000_000)]), axis=1)
    rep.update({"blocks_compared": len(blocks), "blocks_mismatched": bad,
                "norm_min_first_2M": float(norms.min()), "norm_max_first_2M": float(norms.max()),
                "doc_ids_unique": len(set(idx.doc_ids)) == idx.n_docs})
    return rep


def verify_graph(index_dir: Path, rng: np.random.Generator) -> dict:
    from medrag.index.dependency_graph import DependencyGraph

    g = DependencyGraph.load(index_dir / "dependency_graph.pkl")
    n = sg.store_n_rows(index_dir / "store")
    nodes = rng.choice(g.n_nodes, size=min(3000, g.n_nodes), replace=False)
    unreciprocated = 0
    checked = 0
    for a in nodes.tolist():
        nb, w = g.neighbours(a)
        for b, wt in zip(nb.tolist(), w.tolist()):
            nb2, w2 = g.neighbours(b)
            pos = np.flatnonzero(nb2 == a)
            checked += 1
            if not len(pos) or not np.isclose(w2[pos[0]], wt):
                unreciprocated += 1
    return {"n_nodes": g.n_nodes, "store_rows": n,
            "indptr_monotonic": bool(np.all(np.diff(g.indptr) >= 0)),
            "indptr_end_matches": int(g.indptr[-1]) == len(g.indices),
            "sampled_nodes": len(nodes), "edges_checked": checked,
            "unreciprocated": unreciprocated, **g.stats(),
            "degree_by_kind": {k: round(v, 3) for k, v in g.degree_by_kind().items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["embeddings", "bm25", "page", "graph"])
    ap.add_argument("--index-dir", default="data/abstracts20m/index")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()
    index_dir = Path(args.index_dir)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    rep = {"embeddings": verify_embeddings, "bm25": verify_bm25,
           "page": verify_page, "graph": verify_graph}[args.part](index_dir, rng)
    rep = {"part": args.part, "index_dir": str(index_dir), "seed": args.seed,
           "elapsed_s": round(time.time() - t0, 1), **rep}
    out = Path(args.out or f"data/logs/embed20m/verify_{args.part}.json")
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
