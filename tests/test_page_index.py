"""Tests for the document-level (page) vector index.

This is Anthropic Contextual Retrieval (2024) / LangChain Contextual Embeddings
territory, reimplemented post-hoc so it needs no corpus re-embedding. These tests
are about the index: that it holds one vector per document, that page scores are
cosines comparable with chunk scores, and that it stays aligned with the chunk
store's row ids.

The scoring formula itself lives in ``page_scoring`` and is tested in
``test_page_scoring.py``. It used to have a second, row-keyed implementation here
-- a convex blend, ``(1-a)*base + a*page`` -- that nothing called. Two different
formulas under one name is how a result gets reported against the wrong one, so the
dead copy is gone rather than kept as an alternative.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from medrag.index.page_index import PageIndex, page_text


class _Chunk:
    def __init__(self, row_id, doc_id, title="T", text="body", cats=()):
        self.doc_id, self.title, self.text, self.categories = doc_id, title, text, cats
        self.journal, self.year = "J", 2020
        self.row_id = row_id


class _Store:
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def get(self, row_id):
        return self.chunks.get(int(row_id))


def test_page_text_carries_title_and_sections() -> None:
    """A document's frame lives in its structure, which chunk text cannot see."""

    class Sec:
        def __init__(self, title, text=""):
            self.title, self.text = title, text

    class Doc:
        title = "Left atrial appendage closure"
        journal, year = "NEJM", 2024
        sections = [Sec("Background"), Sec("Methods", "1600 patients"), Sec("Limitations")]

    text = page_text(Doc())
    assert "Left atrial appendage closure" in text
    assert "Methods" in text and "Limitations" in text
    print("  ok  page text carries title, venue and section headings")


def test_page_text_falls_back_to_abstract() -> None:
    class Doc:
        title, journal, year = "T", "J", 2020
        sections = ()
        abstract = "abstract body"

    assert "abstract body" in page_text(Doc())
    print("  ok  abstract is used when there are no sections")


def test_search_returns_rows_and_is_normalised() -> None:
    idx = PageIndex(
        dim=3,
        matrix=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        row_ids=[10, 20, 30],
        doc_ids=["A", "B", "C"],
    )
    hits = idx.search_rows(np.asarray([0.0, 2.0, 0.0]), top_k=2)
    assert [r for r, _ in hits] == [20, 10], hits
    assert abs(hits[0][1] - 1.0) < 1e-5, hits
    assert idx.search_rows(np.asarray([1, 1, 1]), top_k=99) and len(
        idx.search_rows(np.asarray([1, 1, 1]), top_k=99)
    ) == 3
    print("  ok  page search returns aligned rows with cosine scores")


def test_empty_index_is_safe() -> None:
    idx = PageIndex()
    assert idx.n_docs == 0
    assert idx.search_rows(np.asarray([1.0, 0.0]), top_k=5) == []
    assert idx.stats()["n_documents"] == 0
    print("  ok  empty page index returns no results instead of raising")


def test_save_load_preserves_alignment() -> None:
    idx = PageIndex(
        dim=2,
        matrix=np.asarray([[0.6, 0.8], [1.0, 0.0]], dtype=np.float32),
        row_ids=[7, 9],
        doc_ids=["X", "Y"],
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = idx.save(Path(tmp) / "page.pkl")
        back = PageIndex.load(p)
    assert back.row_ids == [7, 9]
    assert back.doc_ids == ["X", "Y"]
    assert np.allclose(back.matrix, idx.matrix)
    assert back.search_rows(np.asarray([1.0, 0.0]), 2) == idx.search_rows(np.asarray([1.0, 0.0]), 2)
    print("  ok  page index round-trips with row alignment preserved")


def test_build_enumerates_one_row_per_document() -> None:
    """The whole point: one vector per document, not per chunk."""
    chunks = {}
    for i, (doc, txt) in enumerate([
        ("A", "alpha"), ("A", "alpha again"), ("B", "beta"), ("B", "beta again"),
        ("C", "gamma"),
    ]):
        chunks[i] = _Chunk(i, doc, title=txt.upper(), text=txt)
    store = _Store(chunks)
    rows = [str(i) for i in range(len(chunks))]
    # Build without an embedder by checking enumeration directly.
    seen, rows_out = set(), []
    for row_id in range(len(rows)):
        c = store.get(row_id)
        if c is None or c.doc_id in seen:
            continue
        seen.add(c.doc_id)
        rows_out.append((c.doc_id, row_id))
    assert len(rows_out) == 3, rows_out
    assert [r for _, r in rows_out] == [0, 2, 4], rows_out
    print("  ok  one row per document, anchored at its first chunk")


def test_document_rows_matches_a_brute_force_walk() -> None:
    """The fast path replaced a walk over every row; it must agree with it exactly.

    PageIndex.build used to call store.get() on all 16,133,551 rows purely to find
    which row opened each document -- 46.56 GiB of blob, ~2.2 hours over NFS. It now
    asks the store for the opening rows instead. If the two ever disagreed the page
    vectors would silently attach to the wrong documents, and nothing downstream
    would notice: a wrong page vector still produces plausible cosines.
    """
    from medrag.index.store import ChunkStore

    with tempfile.TemporaryDirectory() as tmp:
        store = ChunkStore(dir_path=Path(tmp), cache_size=2)
        # Deliberately NOT grouped by document, and with a document whose chunks are
        # not adjacent, so an implementation that assumes grouping fails.
        order = ["A", "B", "A", "C", "B", "C", "C"]
        for i, doc in enumerate(order):
            from medrag.data.schema import Chunk

            store.add(Chunk(doc_id=doc, chunk_id=f"{doc}#{i:04d}",
                            text=f"body of {doc} chunk {i}", categories=()))
        store.finalize_build()
        store.close()

        reloaded = ChunkStore.load(Path(tmp), cache_size=2)
        doc_ids, first_rows = reloaded.document_rows()

        brute, seen = [], set()
        for row_id in range(len(reloaded.rows)):
            c = reloaded.get(row_id)
            if c is None or c.doc_id in seen:
                continue
            seen.add(c.doc_id)
            brute.append((c.doc_id, row_id))

        assert doc_ids == [d for d, _ in brute], (doc_ids, brute)
        assert first_rows == [r for _, r in brute], (first_rows, brute)
        assert first_rows[0] == 0 and len(set(first_rows)) == len(first_rows)

        # And the bulk read returns the same chunks the point reads do.
        bulk = reloaded.chunks_bulk(first_rows)
        for row_id in first_rows:
            assert row_id in bulk, row_id
            assert bulk[row_id].text == reloaded.get(row_id).text, row_id
        reloaded.close()
    print(f"  ok  document_rows agrees with a brute-force walk on {len(doc_ids)} "
          f"documents in interleaved row order, and chunks_bulk matches point reads")


def test_page_build_uses_the_fast_path_when_available() -> None:
    """Guard against the fast path silently disappearing.

    If the store lacks document_rows/chunks_bulk the build falls back to the full
    walk, which still produces a correct index -- just 29x slower. A test that only
    checked the result would pass either way, so this asserts the fast methods were
    the ones called.
    """
    import types

    import medrag.index.embeddings as embeddings

    calls: list[str] = []

    class _NoModel:
        def load(self):
            return self

        def embed(self, texts, is_query=False, batch_size=None):
            calls.append("embed")
            return np.ones((len(texts), 4), dtype=np.float32)

    class _FastStore(_Store):
        def document_rows(self):
            calls.append("document_rows")
            return ["A", "B"], [0, 2]

        def chunks_bulk(self, row_ids):
            calls.append("chunks_bulk")
            return {r: self.chunks[r] for r in row_ids if r in self.chunks}

    chunks = {i: _Chunk(i, d, title=d, text=f"body {d} {i}")
              for i, d in enumerate(["A", "A", "B", "B"])}
    store = _FastStore(chunks)
    cfg = types.SimpleNamespace(embed=types.SimpleNamespace(batch_size=8))

    real = embeddings.DenseEmbedder
    embeddings.DenseEmbedder = lambda _cfg: _NoModel()
    try:
        idx = PageIndex.build(cfg, [str(i) for i in range(4)], store=store)
    finally:
        embeddings.DenseEmbedder = real

    assert "document_rows" in calls, f"build did not use the fast path: {calls}"
    assert "chunks_bulk" in calls, f"build did not use the bulk read: {calls}"
    assert idx.doc_ids == ["A", "B"], idx.doc_ids
    assert idx.row_ids == [0, 2], idx.row_ids
    print("  ok  build uses document_rows + chunks_bulk rather than a full walk")


def test_chunks_bulk_returns_every_row_around_a_large_gap() -> None:
    """A document separated from the next by an oversized chunk must still be returned.

    Regression from a clustered-read implementation that merged nearby rows: opening
    chunks sit one per ~84 KB, densely enough that merging is transitive, and capping
    the span at 1 MiB still covered most of the file -- 45x read amplification, 29x
    slower than point reads. The cases below are the ones that version got wrong.
    """
    from medrag.index.store import ChunkStore
    from medrag.data.schema import Chunk

    with tempfile.TemporaryDirectory() as tmp:
        store = ChunkStore(dir_path=Path(tmp), cache_size=1)
        store.add(Chunk(doc_id="A", chunk_id="A#0000", text="A body", categories=()))
        # One chunk far larger than the merge gap, so row 2 starts a new cluster.
        store.add(Chunk(doc_id="F", chunk_id="F#0000", text="x" * (400 * 1024),
                        categories=()))
        store.add(Chunk(doc_id="B", chunk_id="B#0000", text="B body", categories=()))
        store.finalize_build()
        store.close()

        reloaded = ChunkStore.load(Path(tmp), cache_size=1)
        doc_ids, first_rows = reloaded.document_rows()
        assert doc_ids == ["A", "F", "B"], doc_ids
        bulk = reloaded.chunks_bulk(first_rows)
        assert set(bulk) == set(first_rows), sorted(set(first_rows) - set(bulk))
        assert bulk[first_rows[0]].text == "A body"
        assert bulk[first_rows[2]].text == "B body"
        assert len(bulk[first_rows[1]].text) == 400 * 1024
        for row_id in first_rows:
            assert bulk[row_id].text == reloaded.get(row_id).text, row_id
        # Requesting them out of order must give the same mapping, since the read
        # path sorts.
        shuffled = reloaded.chunks_bulk([first_rows[2], first_rows[0]])
        assert set(shuffled) == {first_rows[0], first_rows[2]}, sorted(shuffled)
        reloaded.close()
    print("  ok  bulk read returns every row across a 400 KiB gap, order-independent")


def test_page_text_carries_the_frame_and_not_the_prose() -> None:
    """Page vectors must condition on structure, not on text the chunk vector has.

    The opening chunk's body is already represented by that chunk's own vector.
    Feeding it into the page vector as well makes page-conditioning a second copy of
    the same signal, which is the distinction the component exists to draw.

    It also measured 1,506 characters per document instead of 396, and a 20,000-row
    encode at that length aborts on this host in all three attempts. This is the case
    that decides both the content and the buildability.
    """
    from medrag.index.page_index import DEFAULT_PAGE_CHARS, _text_from_chunk

    class C:
        title = "Aspirin for atrial fibrillation: the AFIRE trial"
        journal, year = "Lancet", 2021
        categories = ("methods", "results", "discussion", "limitations")
        text = "unique-body-marker " * 400

    out = _text_from_chunk(C())
    assert "AFIRE" in out, out
    assert "Lancet 2021" in out, out
    for kind in ("methods", "results", "discussion", "limitations"):
        assert kind in out, out
    assert "unique-body-marker" not in out, "page text leaked the opening chunk's body"
    assert len(out) <= DEFAULT_PAGE_CHARS, (len(out), DEFAULT_PAGE_CHARS)

    # A document with no title or kinds still yields the venue rather than nothing.
    class Bare:
        title, journal, year = "", "", 0
        categories = ()
        text = "body only"
    assert _text_from_chunk(Bare()) == "", "empty framing should not become body text"
    print("  ok  page text is title, venue and section kinds, with no body")


def main() -> int:
    tests = [
        test_page_text_carries_title_and_sections,
        test_page_text_falls_back_to_abstract,
        test_page_text_carries_the_frame_and_not_the_prose,
        test_search_returns_rows_and_is_normalised,
        test_empty_index_is_safe,
        test_save_load_preserves_alignment,
        test_build_enumerates_one_row_per_document,
        test_document_rows_matches_a_brute_force_walk,
        test_chunks_bulk_returns_every_row_around_a_large_gap,
        test_page_build_uses_the_fast_path_when_available,
    ]
    print("Page-level vector index tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
