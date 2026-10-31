"""Tests for dense index type selection.

Regression guard. ``index_type="auto"`` used to switch from exact flat search to
approximate IVF-PQ above 200k vectors. On clinical text that silently halved
retrieval quality:

    IndexFlatIP (exact)  nDCG@10 0.542, hit@1 0.644
    IndexIVFPQ nprobe=16 nDCG@10 0.274, hit@1 0.332   (same 500 queries)

The failure was silent: the index built fine, reported a plausible vector count,
and returned plausible-looking results. These tests pin the policy so `auto`
cannot quietly reintroduce approximation, and so the trade-off stays documented
in code where the next person will read it.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np

from medrag.index.dense import IVF_THRESHOLD, DenseIndex


def test_auto_never_selects_approximate_search() -> None:
    """`auto` must resolve to an exact index at realistic corpus sizes."""
    faiss = None
    try:
        import faiss as _faiss

        faiss = _faiss
    except ImportError:  # pragma: no cover
        pass

    # The documented production corpus, which previously tripped the old
    # 200k threshold and got an approximate index.
    production_n = 2_368_622
    assert production_n > 200_000, "the regression case must exceed the old threshold"
    assert IVF_THRESHOLD > production_n, (
        f"auto would switch to approximate search at {production_n} vectors "
        f"(threshold {IVF_THRESHOLD}); that halves dense nDCG@10"
    )

    if faiss is None:
        print("  ok  threshold pinned above production scale (faiss absent, skipped build check)")
        return

    # Build the real thing and assert the concrete index type.
    rng = np.random.default_rng(0)
    n, dim = 250_000, 64   # above the old threshold, small enough to build fast
    matrix = rng.standard_normal((n, dim)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    index = DenseIndex(dim=dim, index_type="auto")
    index.build(matrix, [f"c{i}" for i in range(n)])
    assert index.stats()["type"] == "IndexFlatIP", index.stats()
    print(f"  ok  auto -> {index.stats()['type']} at {n:,} vectors (was IVF-PQ)")


def test_approximate_requires_explicit_opt_in() -> None:
    """IVF-PQ and IVF-Flat must still work, but only when asked for by name."""
    try:
        import faiss  # noqa: F401
    except ImportError:  # pragma: no cover
        print("  ok  faiss absent; explicit index types not exercised")
        return

    rng = np.random.default_rng(1)
    n, dim = 20_000, 32
    matrix = rng.standard_normal((n, dim)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    for kind in ("ivfpq", "ivfflat"):
        index = DenseIndex(dim=dim, index_type=kind)
        index.build(matrix, [f"c{i}" for i in range(n)], batch_size=5000)
        assert index.stats()["type"].startswith("IndexIVF"), index.stats()
        # nprobe must be set or search silently degrades to one list.
        assert index.stats()["nprobe"], f"{kind} left nprobe unset"
    print("  ok  ivfpq/ivfflat available on explicit request, with nprobe set")


def test_unknown_index_type_is_rejected() -> None:
    try:
        import faiss  # noqa: F401
    except ImportError:  # pragma: no cover
        return
    index = DenseIndex(dim=8, index_type="nonsense")
    try:
        index.build(np.zeros((4, 8), dtype=np.float32), ["a", "b", "c", "d"])
    except ValueError:
        print("  ok  an unknown index_type raises instead of silently defaulting")
        return
    raise AssertionError("expected ValueError for an unknown index type")


def test_row_ids_survive_the_index_type_choice() -> None:
    """Row ids are what the retriever uses to resolve payloads; they must not
    be silently re-derived as positions."""
    try:
        import faiss  # noqa: F401
    except ImportError:  # pragma: no cover
        return
    rng = np.random.default_rng(2)
    n, dim = 5_000, 16
    matrix = rng.standard_normal((n, dim)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    shuffled = list(range(n))
    rng.shuffle(shuffled)
    index = DenseIndex(dim=dim, index_type="flat")
    index.build(matrix, [f"c{i}" for i in range(n)], row_ids=shuffled)
    assert index.row_ids == shuffled, "row_ids must be preserved verbatim"
    # FAISS returns a position; the retriever must receive row_ids[position].
    # Probing the vector stored at position p must therefore come back as
    # row_ids[p], which differs from p whenever row ids are shuffled.
    probe = 1438
    hits = index.search_one_rows(matrix[probe], top_k=1)
    assert hits, "no hit returned"
    assert hits[0][0] == shuffled[probe], (
        f"search must resolve through row_ids: got {hits[0][0]}, expected {shuffled[probe]}"
    )
    assert shuffled[probe] != probe, "fixture is not actually shuffled at this index"
    print("  ok  row_ids preserved and used for result resolution (not positions)")


def main() -> int:
    tests = [
        test_auto_never_selects_approximate_search,
        test_approximate_requires_explicit_opt_in,
        test_unknown_index_type_is_rejected,
        test_row_ids_survive_the_index_type_choice,
    ]
    print("Dense index-type policy tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())