"""End to end: the RAM-free 20M build path on a miniature index, then verify_20m.

Builds a real ChunkStore, writes embedding blocks plus the assembled embeddings.f32
the way build-index --no-faiss does, builds BM25 segments, page texts, page blocks
(with a fake embedder) and the scanned dependency graph, then runs every part of
scripts/verify_20m.py against it. A wiring mistake between these steps -- a path,
a manifest key, an off-by-one block -- shows up here rather than at 48M rows.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from medrag.data.schema import Chunk  # noqa: E402
from medrag.index import dependency_graph as dg  # noqa: E402
from medrag.index import page_index as pi  # noqa: E402
from medrag.index import segmented as sg  # noqa: E402
from medrag.index.store import ChunkStore, build_chunk_store  # noqa: E402


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


remote_embed = _load("remote_embed", "scripts/remote_embed.py")
verify = _load("verify_20m", "scripts/verify_20m.py")

WORDS = "aspirin stroke trial dose risk cohort mortality infection sepsis insulin".split()


def _build(d: Path, block_rows: int) -> int:
    rng = random.Random(1)
    chunks = []
    for i in range(400):
        doc = f"pmid:{5000 + i}"
        for c in range(1 + i % 3):
            body = " ".join(rng.choices(WORDS, k=12)) + f" uniq{i}x{c}"
            chunks.append(Chunk(chunk_id=f"{doc}#{c:04d}", doc_id=doc, text=body,
                                embed_text=f"T{i} | {body}", title=f"T{i}", journal="J",
                                year=2000 + i % 20, section_kind=rng.choice(
                                    ["background", "methods", "results", "conclusions"])))
    build_chunk_store(chunks, dir_path=d / "store")
    n = len(chunks)
    vec = np.random.default_rng(2).standard_normal((n, 8)).astype(np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    for b in range((n + block_rows - 1) // block_rows):
        np.savez(d / f"embeddings-block{b:05d}.npz", embeddings=vec[b * block_rows:(b + 1) * block_rows])
    vec.tofile(d / "embeddings.f32")
    (d / "embeddings.json").write_text(json.dumps({"file": "embeddings.f32", "shape": [n, 8]}))
    for seg, (lo, hi) in enumerate(sg.segment_bounds(n, 300)):
        sg.build_bm25_segment(sg.iter_range_texts(d / "store", lo, hi), d / "bm25_segments",
                              seg, lo, hi, k1=1.2, b=0.75, min_token_length=2)
    pi.write_page_text_blocks(d, block_docs=64)
    args = argparse.Namespace(index_dir=str(d), kind="page", block=None, dim=8, host="t", local=True)
    while True:
        head, texts = remote_embed._claim_local(args)
        if head["block"] is None:
            break
        fake = np.random.default_rng(head["block"]).standard_normal((len(texts), 8))
        assert remote_embed._put_local(args, head["block"], fake.astype(np.float32)) == 0
    pi.assemble_page_blocks(d)
    dg.scan_graph_inputs(d, workers=2, rows_per_task=128)
    store = ChunkStore.load(d / "store")
    dg.build_from_scanned_inputs(d, *store.document_rows()).save(d / "dependency_graph.pkl")
    return n


def test_build_and_verify_every_part() -> None:
    old = verify.BLOCK_ROWS
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        verify.BLOCK_ROWS = 50
        try:
            n = _build(d, 50)
            rng = np.random.default_rng(0)
            e = verify.verify_embeddings(d, rng)
            assert e["blocks_mismatched"] == [] and e["zero_rows"] == 0
            assert e["self_retrieval_rank1_is_self"] == e["self_retrieval_rows"]
            b = verify.verify_bm25(d, rng)
            assert b["contiguous"] and b["rows_covered"] == n and b["self_rank1"] == 32, b
            p = verify.verify_page(d, rng)
            assert p["n_docs"] == 400 and p["memmap"] and p["blocks_mismatched"] == [], p
            assert p["doc_ids_unique"]
            g = verify.verify_graph(d, rng)
            assert g["n_nodes"] == n and g["unreciprocated"] == 0 and g["indptr_end_matches"]
        finally:
            verify.BLOCK_ROWS = old
    print(f"  {n}-row miniature: embeddings, bm25, page and graph all build and verify")


def main() -> int:
    print("20M build path, end to end")
    test_build_and_verify_every_part()
    print("\nall 1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
