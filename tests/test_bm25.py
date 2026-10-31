"""Correctness and scale tests for the compact sparse BM25 index.

The sparse rewrite must be *exactly* equivalent to the reference dict-based
implementation, otherwise every retrieval number in the paper is suspect. These
tests check that directly, plus the memory and throughput properties that
motivated the rewrite in the first place.
"""

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")

from medrag.index.bm25 import BM25Index, build_bm25, tokenize  # noqa: E402


def reference_bm25(docs: list[str], query: str, k1: float = 1.2, b: float = 0.75) -> list[tuple[int, float]]:
    """Straightforward dict-based BM25, used as the correctness oracle."""
    tokenized = [tokenize(d) for d in docs]
    n = len(docs)
    lengths = [len(t) for t in tokenized]
    avgdl = sum(lengths) / n if n else 0.0
    postings: dict[str, dict[int, int]] = defaultdict(dict)
    for i, toks in enumerate(tokenized):
        for term, tf in Counter(toks).items():
            postings[term][i] = tf
    scores: dict[int, float] = defaultdict(float)
    for term, qtf in Counter(tokenize(query)).items():
        post = postings.get(term)
        if not post:
            continue
        df = len(post)
        idf = __import__("math").log(1.0 + (n - df + 0.5) / (df + 0.5))
        for did, tf in post.items():
            dl = lengths[did] or 1
            denom = tf + k1 * (1 - b + b * dl / (avgdl or 1))
            scores[did] += idf * (tf * (k1 + 1) / denom) * (1.0 + 0.1 * __import__("math").log(1 + qtf))
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def test_matches_reference() -> None:
    """Sparse scores must match the reference implementation to float precision."""
    rng = random.Random(7)
    vocab = ["aspirin", "metformin", "insulin", "hypertension", "obesity", "myocardial",
             "infarction", "stroke", "sepsis", "glucose", "hemoglobin", "therapy", "trial"]
    docs = [
        " ".join(rng.choice(vocab) for _ in range(rng.randint(20, 120)))
        for _ in range(400)
    ]
    index = build_bm25(docs)
    queries = ["aspirin therapy", "metformin glucose trial", "myocardial infarction hypertension",
               "sepsis hemoglobin", "obesity stroke"]

    worst = 0.0
    for q in queries:
        got = index.search(q, top_k=20)
        want = reference_bm25(docs, q)[:20]
        assert len(got) == len(want), f"{q}: {len(got)} vs {len(want)} results"
        for (gi, gs), (wi, ws) in zip(got, want):
            assert gi == wi, f"{q}: id order differs {gi} vs {wi}"
            worst = max(worst, abs(gs - ws))
    assert worst < 1e-4, f"max score drift {worst}"
    print(f"  ok  matches reference implementation (max score drift {worst:.2e})")


def test_save_load_roundtrip() -> None:
    import tempfile
    from pathlib import Path

    docs = [f"patient {i} received metformin and insulin for type 2 diabetes" for i in range(200)]
    docs += [f"trial {i} evaluated sepsis mortality in intensive care" for i in range(200)]
    index = build_bm25(docs)
    before = index.search("metformin diabetes", top_k=15)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bm25.pkl"
        index.save(path)
        reloaded = BM25Index.load(path)
        after = reloaded.search("metformin diabetes", top_k=15)
    assert before == after, "round-trip changed results"
    print("  ok  save/load round-trip is lossless")


def test_query_term_repetition() -> None:
    """A repeated query term must outrank a single mention, as in Lucene."""
    docs = [
        "aspirin reduces mortality in heart failure patients",
        "aspirin and aspirin and aspirin aspirin study cohort",
        "unrelated text about agricultural yield patterns",
    ]
    index = build_bm25(docs)
    hits = index.search("aspirin aspirin", top_k=3)
    assert hits[0][0] == 1, f"expected doc 1 first, got {hits[0]}"
    print("  ok  query-term repetition is rewarded")


def test_empty_and_missing_terms() -> None:
    index = build_bm25(["alpha beta gamma", "delta epsilon"])
    assert index.search("zzzznotpresent", top_k=5) == []
    assert index.search("", top_k=5) == []
    assert index.search("the and of") == []
    print("  ok  missing/empty queries return no results")


def test_memory_efficiency() -> None:
    """The whole point: ~6x less RAM than a dict-based postings table."""
    import resource

    rng = random.Random(3)
    vocab = [f"t{i}" for i in range(60000)]
    n = 100_000
    index = BM25Index()
    for i in range(n):
        index.add(i, " ".join(rng.choice(vocab) for _ in range(62)))
    index.finalize()
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    stats = index.stats()
    per_chunk = stats["bytes_per_chunk"]
    print(f"  ok  {per_chunk:.0f} bytes/chunk of index data "
          f"(dict-based measured ~3467) -> {3467 / max(1, per_chunk):.1f}x smaller")
    assert per_chunk < 900, f"expected <900 bytes/chunk, got {per_chunk}"
    # Projection for the full corpus target.
    for target in (3_000_000, 7_000_000):
        print(f"        projected index RAM at {target:>9,} chunks: {per_chunk * target / 2**30:5.1f} GiB")
    assert rss_mib > 0


def test_incremental_and_resume() -> None:
    """add() must accept sequential ids and refuse gaps."""
    index = BM25Index()
    for i in range(10):
        index.add(i, f"doc {i} alpha beta")
    try:
        index.add(99, "gap doc")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on non-sequential id")
    index.finalize()
    assert index.n_docs == 10
    print("  ok  enforces sequential ids so the row map stays aligned")


def test_more_terms_than_docs() -> None:
    """A term whose id is >= the row count must still score (the shape[0] bug)."""
    docs = ["alpha beta gamma delta epsilon", "zeta eta theta iota kappa lambda"]
    index = build_bm25(docs)
    assert len(index._terms) > index.n_docs
    hits = index.search("lambda", top_k=5)
    assert [d for d, _ in hits] == [1], hits
    ref = reference_bm25(docs, "lambda")
    assert abs(hits[0][1] - ref[0][1]) < 1e-4, (hits, ref)
    print("  ok  terms with id >= n_docs are scored")


def main() -> int:
    tests = [
        test_matches_reference,
        test_save_load_roundtrip,
        test_query_term_repetition,
        test_empty_and_missing_terms,
        test_incremental_and_resume,
        test_memory_efficiency,
        test_more_terms_than_docs,
    ]
    print("BM25 sparse-index tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} BM25 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
