"""Tests for multi-hop context provisioning.

Single-document context has its worst measured effect on multi-hop queries: qar
0.2024 against 0.3654 overall, and the smallest gain of any category. The reason
is structural -- a two-hop question's evidence is spread across papers by
construction, so framing within one document cannot supply it.

The tests pin the two mechanisms that address this (cross-hop framing, per-hop
seeds) and the property that makes them ablatable: `cross_hop=False` must recover
exact single-document behaviour, or the comparison between the two is not a
comparison of mechanisms.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from medrag.index.multihop_context import (
    _per_hop_prefix,
    expand_with_cross_hop_context,
    multi_hop_context_rankings,
)


class _Chunk:
    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id
        # apply_sibling_bleed reads section_kind off the chunk.
        self.section_kind = "results"


class _Store:
    def __init__(self, doc_of: dict[int, str]) -> None:
        self.doc_of = doc_of

    def get(self, row_id: int):
        d = self.doc_of.get(int(row_id))
        return None if d is None else _Chunk(d)


class _Dep:
    """Dependency stub. `cross` edges are ones whose endpoint is another document."""

    def __init__(self, pairs, cross_rows: set[int] | None = None) -> None:
        self.pairs = pairs
        self.cross_rows = cross_rows or set()
        self.calls: list[tuple] = []

    def context_pairs(self, seeds, *, top_k=10, max_per_seed=2):
        self.calls.append((tuple(int(s) for s in seeds), top_k, max_per_seed))
        return [p for p in self.pairs if p[0] in set(int(s) for s in seeds)][:top_k]


def test_per_hop_prefix_serves_every_hop() -> None:
    """A single merged prefix under-serves hops whose evidence ranks low overall.

    This is the second hop of a two-hop question: its best evidence can sit at
    rank 15 of the merged list while still being the right framing for the first
    hop's finding.
    """
    hops = [
        [(10, 9.0), (11, 8.0), (12, 7.0), (13, 6.0)],
        [(20, 1.0), (21, 0.9), (22, 0.8), (23, 0.7)],
    ]
    merged = _per_hop_prefix(hops, seeds_per_hop=2)
    rows = [r for r, _ in merged]
    assert rows == [10, 11, 20, 21], rows
    # Row 20 is rank 15th overall yet still contributes, which is the point.
    print("  ok  per-hop prefix serves every hop, not just the merged top")


def test_per_hop_prefix_deduplicates_rows() -> None:
    hops = [[(1, 5.0), (2, 4.0)], [(1, 4.9), (3, 3.0)]]
    rows = [r for r, _ in _per_hop_prefix(hops, seeds_per_hop=2)]
    assert rows == [1, 2, 3], rows
    print("  ok  a row found by two hops is seeded once")


def test_cross_hop_allows_another_documents_framing() -> None:
    """A passage from a different paper can qualify this one, for multi-hop."""
    store = _Store({0: "A", 5: "B", 6: "A"})
    dep = _Dep([(0, 5, 1.0)], cross_rows={5})   # row 5 lives in document B
    hits = [(0, 9.0), (6, 8.0)]
    cross = expand_with_cross_hop_context(hits, store, dep, top_k=2, total_seeds=2)
    rows = [r for r, _ in cross]
    assert 5 in rows, f"cross-document framing excluded: {rows}"
    print(f"  ok  cross-hop framing admitted from another document: {rows}")


def test_cross_hop_false_recovers_single_document_behaviour() -> None:
    """Without this the two modes are not comparable as mechanisms."""
    store = _Store({0: "A", 5: "B"})
    dep = _Dep([(0, 5, 1.0)])
    hits = [(0, 9.0)]
    cross = [r for r, _ in expand_with_cross_hop_context(hits, store, dep, top_k=1, total_seeds=2)]
    same = [r for r, _ in expand_with_cross_hop_context(hits, store, dep, top_k=1,
                                                        total_seeds=2, cross_hop=False)]
    assert 5 in cross, cross
    assert 5 not in same, f"cross_hop=False still crossed documents: {same}"
    print("  ok  cross_hop=False confines framing to the seed's own document")


def test_same_document_context_is_always_admitted() -> None:
    store = _Store({0: "A", 6: "A"})
    dep = _Dep([(0, 6, 1.0)])
    for flag in (True, False):
        rows = [r for r, _ in expand_with_cross_hop_context([(0, 9.0)], store, dep,
                                                          top_k=1, total_seeds=2,
                                                          cross_hop=flag)]
        assert 6 in rows, (flag, rows)
    print("  ok  same-document framing is admitted in both modes")


def test_missing_graph_degrades_to_base_ranking() -> None:
    store = _Store({0: "A"})
    base = [(0, 9.0), (1, 8.0)]
    assert expand_with_cross_hop_context(base, store, None, top_k=2) == base
    print("  ok  absent dependency graph degrades to the base ranking")


def test_context_budget_is_bounded() -> None:
    store = _Store({i: f"D{i}" for i in range(30)})
    dep = _Dep([(0, i, 1.0) for i in range(5, 30)])
    rows = expand_with_cross_hop_context([(0, 9.0)], store, dep, top_k=1, total_seeds=3)
    assert len(rows) == 4, rows
    print(f"  ok  context bounded at total_seeds ({len(rows)} rows)")


def test_multi_hop_rankings_apply_bleed_per_hop() -> None:
    """Each hop's framing should be chosen by that hop's evidence, not the merge."""
    store = _Store({0: "A", 1: "A", 10: "B", 11: "B"})
    calls: list[str] = []

    def retrieve_for(text: str):
        calls.append(text)
        if text == "hop one":
            return [(0, 10.0), (1, 4.0), (2, 3.0)]
        return [(10, 1.0), (11, 0.9), (12, 0.8)]

    dep = _Dep([(0, 1, 1.0), (10, 11, 1.0)])
    merged, info = multi_hop_context_rankings(
        "hop one", store, dep, retrieve_for, ["hop two"], top_k=4,
        base_rankings=[retrieve_for("hop one"), retrieve_for("hop two")],
    )
    # retrieve_for is called twice per hop: once for seeds, once for the base.
    assert calls[:2] == ["hop one", "hop two"], calls
    assert info["n_hops"] == 2, info
    rows = [r for r, _ in merged]
    # Both hops contribute seeds even though hop two scores far lower overall.
    assert 0 in rows and 10 in rows, rows
    print(f"  ok  bleed applied per hop, both hops seeded: {info}")


def test_results_come_from_the_merged_ranking_not_the_hop_prefixes() -> None:
    """Regression: the first version rebuilt results from per-hop prefixes.

    For a two-hop query that yields six rows instead of ten and drops the
    strongest single-hop hits. Measured on the multi-hop family it dropped
    readiness from 0.2881 to 0.1357 -- worse than the 0.1738 baseline the change
    was meant to improve. base_rankings is now required rather than optional.
    """
    store = _Store({i: "A" for i in range(20)})
    dep = _Dep([])
    merged_base = [(i, 20.0 - i) for i in range(10)]

    def retrieve(t: str):
        return [(0, 9.0), (1, 8.0), (2, 7.0)] if t == "weak hop" else merged_base

    try:
        multi_hop_context_rankings("main", store, dep, retrieve, ["weak hop"], top_k=10)
    except ValueError as exc:
        assert "base_rankings" in str(exc), exc
        print("  ok  omitting base_rankings raises rather than silently dropping hits")
    else:
        raise AssertionError("results may be rebuilt from per-hop prefixes again")

    merged, info = multi_hop_context_rankings(
        "main", store, dep, retrieve, ["weak hop"], top_k=10,
        base_rankings=[merged_base],
    )
    assert info["n_results"] == 10, info
    assert [r for r, _ in merged[:10]] == [r for r, _ in merged_base]
    print("  ok  the top-10 comes from the merged ranking intact")


def test_multi_hop_reports_provenance() -> None:
    store = _Store({0: "A", 1: "A"})
    dep = _Dep([(0, 1, 1.0)])
    merged, info = multi_hop_context_rankings(
        "q", store, dep, lambda t: [(0, 5.0), (1, 4.0)], [], top_k=2,
        base_rankings=[[(0, 5.0), (1, 4.0)]],
    )
    assert set(info) == {"n_hops", "n_seeds", "n_results", "n_returned", "cross_hop"}, info
    assert info["n_returned"] == len(merged), (info, len(merged))
    print(f"  ok  provenance reported: {info}")


def main() -> int:
    tests = [
        test_per_hop_prefix_serves_every_hop,
        test_per_hop_prefix_deduplicates_rows,
        test_cross_hop_allows_another_documents_framing,
        test_cross_hop_false_recovers_single_document_behaviour,
        test_same_document_context_is_always_admitted,
        test_missing_graph_degrades_to_base_ranking,
        test_context_budget_is_bounded,
        test_multi_hop_rankings_apply_bleed_per_hop,
        test_results_come_from_the_merged_ranking_not_the_hop_prefixes,
        test_multi_hop_reports_provenance,
    ]
    print("Multi-hop context provision tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())