"""The streamed dependency-graph build must equal cmd_dependency_graph's.

At 48M rows the graph is fed from a parallel byte scan of the store instead of
store.get() per row. The scan is only acceptable if the graph it produces is the
same graph, array for array, so that is what this checks on a real ChunkStore.
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from medrag.data.schema import Chunk  # noqa: E402
from medrag.index.dependency_graph import (  # noqa: E402
    DependencyGraph,
    build_from_scanned_inputs,
    scan_graph_inputs,
)
from medrag.index.store import ChunkStore, build_chunk_store  # noqa: E402

KINDS = ["background", "methods", "results", "conclusions", "discussion", "other", ""]


def _chunks(n_docs: int, seed: int = 5) -> list[Chunk]:
    rng = random.Random(seed)
    out = []
    for d in range(n_docs):
        prefix = "pmcid:PMC" if d % 3 else "pmid:"
        doc = f"{prefix}{90000 - d * 7}"          # not sorted, so doc order is exercised
        for c in range(rng.randint(1, 9)):
            out.append(Chunk(chunk_id=f"{doc}#{c:04d}", doc_id=doc, text=f"t é {d} {c}",
                             section_kind=rng.choice(KINDS), title=f"T{d}"))
    return out


def _arrays(g: DependencyGraph) -> dict:
    return {"indptr": g.indptr, "indices": g.indices, "weights": g.weights,
            "doc_start": g.doc_start, "doc_end": g.doc_end}


def test_streamed_graph_equals_store_graph() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        build_chunk_store(_chunks(300), dir_path=d / "store")
        store = ChunkStore.load(d / "store")
        ref = DependencyGraph.build(
            store.rows,
            kind_of=lambda r: (store.get(r).section_kind or ""),
            doc_of=lambda r: (store.get(r).doc_id or ""))
        m = scan_graph_inputs(d, workers=3, rows_per_task=97)
        doc_ids, first_rows = store.document_rows()
        got = build_from_scanned_inputs(d, doc_ids, first_rows)
        for k, v in _arrays(ref).items():
            assert np.array_equal(v, _arrays(got)[k]), k
        assert got.section_kind == ref.section_kind
        assert got.doc_ids == ref.doc_ids and got.n_nodes == ref.n_nodes
        assert m["doc_id_chunk_id_mismatches"] == 0
    print(f"  streamed graph == store graph: {ref.n_nodes} nodes, "
          f"{len(ref.indices) // 2} edges, {len(ref.doc_ids)} documents")


def test_split_documents_group_like_the_store_graph() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        chunks = _chunks(20)
        # A document that reappears later in the store, as 472 do in abstracts20m.
        chunks.append(Chunk(chunk_id=chunks[0].doc_id + "#0099", doc_id=chunks[0].doc_id,
                            text="x", section_kind="conclusions"))
        chunks.extend(_chunks(3, seed=9)[:4])
        build_chunk_store(chunks, dir_path=d / "store")
        store = ChunkStore.load(d / "store")
        ref = DependencyGraph.build(
            store.rows,
            kind_of=lambda r: (store.get(r).section_kind or ""),
            doc_of=lambda r: (store.get(r).doc_id or ""))
        m = scan_graph_inputs(d, workers=2, rows_per_task=13)
        got = build_from_scanned_inputs(d, *store.document_rows())
        for k, v in _arrays(ref).items():
            assert np.array_equal(v, _arrays(got)[k]), k
        assert got.doc_ids == ref.doc_ids and got.section_kind == ref.section_kind
        assert m["docs_with_several_runs"] >= 1 and m["n_docs"] == len(ref.doc_ids)
        try:
            build_from_scanned_inputs(d, list(reversed(store.document_rows()[0])))
        except ValueError:
            pass
        else:
            raise AssertionError("a wrong document order must be refused")
    print("  a document split across the store groups exactly as the store graph does")


def _old_dedup(n: int, src: np.ndarray, dst: np.ndarray, wts: np.ndarray):
    """The lexsort dedup _finalize used before the single-key rewrite, verbatim."""
    src, dst = src.astype(np.int64), dst.astype(np.int64)
    order = np.lexsort((dst, src))
    src, dst, wts = src[order], dst[order], wts[order]
    same = (src[1:] == src[:-1]) & (dst[1:] == dst[:-1])
    group = np.concatenate(([0], np.cumsum(~same)))
    n_groups = int(group[-1]) + 1
    g_src = src[np.searchsorted(group, np.arange(n_groups))]
    g_dst = dst[np.searchsorted(group, np.arange(n_groups))]
    g_w = np.zeros(n_groups, dtype=np.float32)
    np.maximum.at(g_w, group, wts)
    degree = np.bincount(g_src, minlength=n).astype(np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(degree, out=indptr[1:])
    return indptr, g_dst.astype(np.int32), g_w


def test_single_key_finalize_equals_the_lexsort_form() -> None:
    from array import array

    rng = np.random.default_rng(11)
    n = 5_000
    src = rng.integers(0, n, 60_000).astype(np.int32)
    dst = rng.integers(0, n, 60_000).astype(np.int32)
    src[:5000], dst[:5000] = src[5000:10000], dst[5000:10000]     # many duplicate pairs
    wts = rng.choice(np.array([0.3, 0.6, 1.0], np.float32), 60_000)
    want = _old_dedup(n, src, dst, wts)
    by_doc = {f"D{i}": list(range(i * 5, i * 5 + 5)) for i in range(n // 5)}
    g = DependencyGraph._finalize(n, [""] * n, [], by_doc, array("i", src.tolist()),
                                  array("i", dst.tolist()), array("f", wts.tolist()))
    assert np.array_equal(g.indptr, want[0])
    assert np.array_equal(g.indices, want[1])
    assert np.array_equal(g.weights, want[2])
    assert g.doc_ids == sorted(f"D{i}" for i in range(n // 5))
    assert int(g.doc_start[g.doc_ids.index("D7")]) == 35 and int(g.doc_end[g.doc_ids.index("D7")]) == 40
    print("  single-key finalize == the lexsort form it replaced, on 60,000 edges with duplicates")


def main() -> int:
    tests = [test_streamed_graph_equals_store_graph, test_split_documents_group_like_the_store_graph,
             test_single_key_finalize_equals_the_lexsort_form]
    print("streamed dependency-graph input tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
