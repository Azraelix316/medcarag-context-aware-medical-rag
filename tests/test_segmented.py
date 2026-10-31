"""Segmented BM25 and streamed dense search must equal the in-RAM indexes.

Every 20M-corpus number would come from these paths while the 16M numbers came from
BM25Index and FAISS, so the comparison is row for row and score for score, not a
recall overlap.
"""

from __future__ import annotations

import json
import random
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from medrag.index import segmented as sg  # noqa: E402
from medrag.index.bm25 import build_bm25  # noqa: E402

VOCAB = [f"term{i}" for i in range(400)] + ["rare", "aspirin", "stroke", "patients", "treated"]


def _corpus(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        words = rng.choices(VOCAB, weights=[1.0 / (1 + j) for j in range(len(VOCAB))],
                            k=rng.randint(0, 40))
        if i == 1234:
            words.append("singleton")
        text = " ".join(words)
        # Some rows carry embed_text, some fall back to text, as real chunks do.
        recs.append({"row": i, "text": text, "embed_text": ("title " + text) if i % 3 else ""})
    return recs


def _write_store(recs: list[dict], d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "blob.jsonl", "wb") as blob, open(d / "offsets.bin", "wb") as offs:
        pos = 0
        for r in recs:
            raw = json.dumps(r).encode() + b"\n"
            blob.write(raw)
            offs.write(struct.pack("<QQ", pos, len(raw) - 1))
            pos += len(raw)


QUERIES = ["aspirin stroke patients", "term3 term3 term17", "singleton", "rare treated term399",
           "nothing matches zzz", "", "term0 term1 term2 term5 term8"]


def test_segmented_bm25_equals_monolithic() -> None:
    recs = _corpus(3000)
    mono = build_bm25([r["embed_text"] or r["text"] for r in recs])
    with tempfile.TemporaryDirectory() as tmp:
        store, out = Path(tmp) / "store", Path(tmp) / "segs"
        _write_store(recs, store)
        assert sg.store_n_rows(store) == len(recs)
        # Uneven last segment, and a TEXT_READ_ROWS boundary inside a segment.
        bounds = sg.segment_bounds(len(recs), 700)
        old = sg.TEXT_READ_ROWS
        sg.TEXT_READ_ROWS = 256
        try:
            for seg, (lo, hi) in enumerate(bounds):
                sg.build_bm25_segment(sg.iter_range_texts(store, lo, hi), out, seg, lo, hi,
                                      k1=1.2, b=0.75, min_token_length=2,
                                      spill_dir=out / f"spill{seg}")
        finally:
            sg.TEXT_READ_ROWS = old
        for top_k in (5, 100):
            got = sg.segmented_bm25_search(out, QUERIES, top_k)
            for q, hits in zip(QUERIES, got):
                assert hits == mono.search(q, top_k=top_k), (q, top_k)
            # Split across "hosts" in an uneven, out-of-order grouping and merged.
            groups = [[3], [0, 2, 4], [1]]
            parts = [sg.segmented_bm25_search(out, QUERIES, top_k, only=g, partial=True)
                     for g in groups]
            assert sorted(s for g in groups for s in g) == list(range(len(bounds)))
            assert sg.merge_partial_topk(parts, top_k) == got, top_k
    print(f"  segmented BM25 == BM25Index.search over {len(bounds)} segments, "
          f"{len(QUERIES)} queries")


def test_segments_must_be_complete() -> None:
    recs = _corpus(500)
    with tempfile.TemporaryDirectory() as tmp:
        store, out = Path(tmp) / "store", Path(tmp) / "segs"
        _write_store(recs, store)
        for seg, (lo, hi) in enumerate(sg.segment_bounds(len(recs), 200)):
            if seg == 1:
                continue
            sg.build_bm25_segment(sg.iter_range_texts(store, lo, hi), out, seg, lo, hi,
                                  k1=1.2, b=0.75, min_token_length=2)
        try:
            sg.segmented_bm25_search(out, QUERIES, 10)
        except ValueError:
            pass
        else:
            raise AssertionError("a missing segment must not be searched silently")
    print("  a missing segment is refused")


def test_claims_are_exclusive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        got = [sg.claim_segment(tmp, 3) for _ in range(4)]
    assert got == [0, 1, 2, None], got
    print("  segment claims are exclusive")


def test_stream_dense_equals_brute_force() -> None:
    rng = np.random.default_rng(3)
    n, dim = 10_007, 24
    # Small integers, so every dot product is exact whatever the summation order,
    # and ties are common: the ranking, tie order included, must match exactly.
    m = rng.integers(-2, 3, (n, dim)).astype(np.float32)
    m[500] = m[9000]
    q = rng.integers(-2, 3, (9, dim)).astype(np.float32)
    q[0] = m[9000]
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        m.tofile(d / "embeddings.f32")
        (d / "embeddings.json").write_text(json.dumps({"file": "embeddings.f32", "shape": [n, dim]}))
        mm = sg.load_embedding_matrix(d)
        got = sg.stream_dense_search(mm, q, 50, max_slice_rows=1000)
    full = q @ m.T
    for qi, hits in enumerate(got):
        order = np.lexsort((np.arange(n), -full[qi]))[:50]
        assert [r for r, _ in hits] == order.tolist(), qi
        assert [s for _, s in hits] == full[qi, order].tolist()
    print(f"  streamed dense == brute force over {n} rows in 11 slices, ties by row")


def test_precomputed_bm25_serves_the_run() -> None:
    recs = _corpus(800)
    mono = build_bm25([r["embed_text"] or r["text"] for r in recs])
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        qs, run = d / "q.jsonl", d / "run.jsonl"
        live = [q for q in QUERIES if q]
        qs.write_text("".join(json.dumps({"qid": i, "query": q}) + "\n" for i, q in enumerate(live)))
        run.write_text("".join(json.dumps({"qid": str(i), "sparse": mono.search(q, top_k=100)})
                               + "\n" for i, q in enumerate(live)))
        pre = sg.PrecomputedBM25.from_run(run, qs, n_docs=mono.n_docs)
        for q in live:
            assert pre.search(q, top_k=10) == mono.search(q, top_k=10)
        full = max(live, key=lambda q: len(mono.search(q, top_k=100)))
        assert len(mono.search(full, top_k=100)) == 100
        for bad in (lambda: pre.search("never asked"), lambda: pre.search(full, top_k=500)):
            try:
                bad()
            except (KeyError, ValueError):
                pass
            else:
                raise AssertionError("an unanswerable lookup must raise")
    print("  precomputed BM25 serves the run's hits and refuses what it cannot answer")


def main() -> int:
    tests = [
        test_precomputed_bm25_serves_the_run,
        test_segmented_bm25_equals_monolithic,
        test_segments_must_be_complete,
        test_claims_are_exclusive,
        test_stream_dense_equals_brute_force,
    ]
    print("segmented search tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
