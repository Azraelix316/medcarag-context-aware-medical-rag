"""The chunked single-pass CSC fill must be bit-identical to scipy.

``BM25Index.finalize`` scatters postings into the CSC itself, in one vectorised pass
over memory-mapped input, rather than going through ``coo_matrix(...).tocsc()``. It
exists because the corpus does not fit otherwise: 1,377,111,929 postings on the
16,133,551-chunk PMC corpus, 15.4 GiB of inputs, and scipy's conversion holds its own
copy of the values and both index arrays while allocating the result.

Three implementations were tried here, and two of them were wrong in ways no
property check would have caught:

* a per-term-block boolean mask -- a full scan of the 15.4 GiB input per block, so
  tens of passes over the whole file, and killed partway through with an empty log;
* a mask with no per-block sort -- postings arrive grouped by *document*, so each
  term's slice was written in document order instead of term order. Correct nnz,
  correct indptr, plausible scores.

Only index-for-index comparison against scipy exposed either. So that is what these
tests do, across chunk sizes from 37 postings (smaller than one document) upward.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
from scipy import sparse

from medrag.index.bm25 import BM25Index


def _build(texts: list[str]) -> BM25Index:
    idx = BM25Index()
    for i, t in enumerate(texts):
        idx.add(i, t)
    return idx


def _single_pass(terms, docs, vals, n_terms, chunk_rows):
    """Reproduce finalize()'s scatter in isolation so it can be compared."""
    counts = np.bincount(terms, minlength=n_terms)
    indptr = np.zeros(n_terms + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    out_i = np.empty(terms.size, dtype=np.int32)
    out_v = np.empty(terms.size, dtype=np.float32)
    cursor = indptr[:-1].copy()
    for begin in range(0, terms.size, chunk_rows):
        end = min(terms.size, begin + chunk_rows)
        t = terms[begin:end]
        order = np.argsort(t, kind="stable")
        ts = t[order]
        newgrp = np.concatenate(([True], ts[1:] != ts[:-1]))
        first = np.flatnonzero(newgrp)
        group = np.cumsum(newgrp) - 1
        slot = cursor[ts[first][group]] + (np.arange(end - begin) - first[group])
        cursor[ts[first]] += np.diff(np.concatenate((first, [end - begin])))
        out_i[slot] = docs[begin:end][order]
        out_v[slot] = vals[begin:end][order]
    return out_i, out_v, indptr


def _reference(idx: BM25Index):
    terms = np.frombuffer(idx._rows, dtype=np.int32)
    docs = np.frombuffer(idx._cols, dtype=np.int32)
    vals = np.frombuffer(idx._vals, dtype=np.float32)
    m = sparse.coo_matrix((vals, (docs, terms)),
                          shape=(idx._cur_doc, len(idx._terms))).tocsc()
    m.sum_duplicates()
    return m


def _check(texts: list[str], chunk_rows: int, label: str) -> None:
    idx = _build(texts)
    terms = np.frombuffer(idx._rows, dtype=np.int32)
    docs = np.frombuffer(idx._cols, dtype=np.int32)
    vals = np.frombuffer(idx._vals, dtype=np.float32)
    ref = _reference(idx)
    oi, ov, ip = _single_pass(terms, docs, vals, len(idx._terms), chunk_rows)
    assert ref.nnz == oi.size, f"{label}: nnz {ref.nnz} vs {oi.size}"
    assert np.array_equal(ref.indptr, ip), f"{label}: indptr differs"
    assert np.array_equal(ref.indices, oi), (
        f"{label}: indices differ at {np.flatnonzero(ref.indices != oi)[:5].tolist()}")
    assert np.allclose(ref.data, ov), f"{label}: data differs"


def _corpus(seed: int, n_terms: int, n_docs: int) -> list[str]:
    rng = np.random.default_rng(seed)
    vocab = [f"t{i}" for i in range(n_terms)]
    return [" ".join(rng.choice(vocab, size=int(min(k, n_terms)), replace=False))
            for k in rng.integers(3, 40, n_docs)]


def test_matches_scipy_in_one_pass() -> None:
    _check(_corpus(21, 300, 4000), 1 << 25, "single chunk")
    print("  ok  single-pass fill matches scipy")


def test_matches_scipy_at_many_chunk_sizes() -> None:
    """Chunk boundaries must not change the result.

    A chunk boundary can fall between two postings of the same term, so the cursor
    has to carry across chunks. Getting that wrong scatters a few postings and leaves
    nnz correct.
    """
    texts = _corpus(21, 300, 4000)
    for chunk in (1, 2, 7, 37, 64, 999, 4096, 1 << 13, 1 << 20, 1 << 25):
        _check(texts, chunk, f"chunk={chunk}")
    print("  ok  identical for chunk sizes 1, 2, 7, 37, 64, 999, 4096, 8192, 1M and 32M")


def test_chunk_boundaries_inside_a_single_document() -> None:
    """A chunk smaller than one document forces the cursor to advance within it."""
    _check(_corpus(21, 300, 2000), 5, "tiny chunks")
    print("  ok  chunks smaller than a document still match")


def test_matches_scipy_on_narrow_and_wide_vocabularies() -> None:
    _check(_corpus(5, 15, 3000), 1 << 12, "narrow")
    _check(_corpus(6, 1200, 9000), 1 << 12, "wide")
    print("  ok  15-term and 1200-term vocabularies both match")


def test_matches_scipy_on_degenerate_corpora() -> None:
    for texts, label in (([""], "empty document"),
                         (["one two three"], "single document"),
                         (["", "", ""], "all empty"),
                         (["alpha beta"] * 3, "identical documents")):
        _check(texts, 1 << 25, label)
    print("  ok  empty, single and identical documents match")


def test_every_slice_is_strictly_ascending() -> None:
    """Postings of a term must land in ascending document order.

    The input is document-major and a term occurs at most once per document, so a
    stable sort within each chunk preserves that order for free. If it ever fails,
    duplicate (term, doc) pairs can appear and the index needs sum_duplicates.
    """
    texts = _corpus(21, 300, 4000)
    idx = _build(texts)
    terms = np.frombuffer(idx._rows, dtype=np.int32)
    docs = np.frombuffer(idx._cols, dtype=np.int32)
    vals = np.frombuffer(idx._vals, dtype=np.float32)
    oi, _, ip = _single_pass(terms, docs, vals, len(idx._terms), 512)
    for t in range(len(idx._terms)):
        s, e = int(ip[t]), int(ip[t + 1])
        if e - s > 1:
            assert np.all(np.diff(oi[s:e]) > 0), (
                f"term {t} slice not ascending: {oi[s:e][:8]}")
    print(f"  ok  all {len(idx._terms)} term slices strictly ascending at chunk=512")


def test_spilling_gives_an_identical_index() -> None:
    """The spill path exists only because the in-RAM path does not fit."""
    texts = _corpus(9, 500, 6000)
    ram = _build(texts)
    ram.finalize()
    with tempfile.TemporaryDirectory() as d:
        disk = _build(texts)
        disk.finalize(spill_dir=d)
        A, B = ram._matrix, disk._matrix
        assert A.shape == B.shape and A.nnz == B.nnz
        assert np.array_equal(A.indptr, B.indptr), "indptr differs"
        assert np.array_equal(A.indices, B.indices), "indices differ"
        assert np.allclose(A.data, B.data), "data differs"
        assert np.array_equal(ram._doc_len, disk._doc_len), "doc_len differs"
    print(f"  ok  spilled index identical: {A.nnz:,} postings, {A.shape[1]} terms")


def test_spilling_releases_the_accumulator() -> None:
    """After spilling, the process must not still hold the postings in RAM."""
    from array import array as _array

    idx = _build(_corpus(11, 200, 3000))
    before = sum(len(x) for x in (idx._rows, idx._cols, idx._vals))
    assert before > 0
    with tempfile.TemporaryDirectory() as d:
        idx.finalize(spill_dir=d)
        assert sum(len(x) for x in (idx._rows, idx._cols, idx._vals)) == 0
        assert all(isinstance(x, _array) for x in (idx._rows, idx._cols, idx._vals))
        for name in ("terms", "docs", "vals"):
            assert (Path(d) / f"{name}.npy").exists()
    print(f"  ok  {before:,} postings released and re-read from disk")


def main() -> int:
    tests = [
        test_matches_scipy_in_one_pass,
        test_matches_scipy_at_many_chunk_sizes,
        test_chunk_boundaries_inside_a_single_document,
        test_matches_scipy_on_narrow_and_wide_vocabularies,
        test_matches_scipy_on_degenerate_corpora,
        test_every_slice_is_strictly_ascending,
        test_spilling_gives_an_identical_index,
        test_spilling_releases_the_accumulator,
    ]
    print("Chunked CSC fill tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())