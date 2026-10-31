"""Tests for sibling-bleed scoring and dependency-context expansion.

Both mechanisms exist to fix one thing: a passage is not independently
interpretable, so a single retrieved chunk understates what its document has to
say. The tests focus on the properties that distinguish them from padding, since
padding a top-k with loosely related chunks is the obvious way to raise a
context-provision score without improving anything.

Sibling bleed is also distinguished from contextual retrieval (Anthropic 2024),
which rewrites chunk text with document context and therefore requires
re-embedding the corpus. The distinction that matters operationally is that this
runs at query time against an existing index, so the tests pin alpha=0 as an exact
identity.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from medrag.index.sibling_bleed import (
    ScoredChunk,
    apply_sibling_bleed,
    bleed_report,
    expand_with_context,
    sibling_bleed,
)


class _Store:
    """Minimal chunk store stub: row_id -> (doc_id, section_kind)."""

    def __init__(self, meta: dict[int, tuple[str, str]]) -> None:
        self.meta = meta

    def get(self, row_id: int):
        doc, kind = self.meta.get(int(row_id), ("", ""))
        return type("C", (), {"doc_id": doc, "section_kind": kind})()


def test_alpha_zero_is_exact_identity() -> None:
    """The property that lets a sweep isolate the mechanism.

    Contextual retrieval needs a corpus re-embed to be evaluated at all. If
    alpha=0 does not reproduce the base ranking exactly, every measurement of
    sibling bleed is confounded with some other change.
    """
    cands = [
        ScoredChunk(0, 9.0, "A", "results"),
        ScoredChunk(1, 8.0, "A", "methods"),
        ScoredChunk(2, 1.0, "B", "results"),
    ]
    out = sibling_bleed(cands, alpha=0.0)
    assert [c.row_id for c in out] == [0, 1, 2], [c.row_id for c in out]
    print("  ok  alpha=0 reproduces the base ranking exactly")


def test_bleed_promotes_a_sibling_of_a_strong_match() -> None:
    """A chunk from a document the query clearly matched should rise.

    This is the mechanism: c2 is a weak match on its own, but its document holds
    c0, the strongest match, so c2 is more likely relevant than an equally weak
    match from a document the query never touched.
    """
    cands = [
        ScoredChunk(0, 10.0, "A", "results"),   # strong
        ScoredChunk(1, 5.0, "A", "methods"),    # moderate sibling
        ScoredChunk(2, 4.9, "B", "results"),    # weak, alone in its document
        ScoredChunk(3, 4.8, "C", "results"),    # weaker, alone
    ]
    out = sibling_bleed(cands, alpha=0.5)
    order = [c.row_id for c in out]
    assert order.index(1) < order.index(2), f"sibling not promoted: {order}"
    assert out[0].row_id == 0, "the strongest match must stay first"
    print(f"  ok  sibling promoted: {order}")


def test_bleed_uses_max_not_sum() -> None:
    """Chunk count must not be rewarded.

    With `sum`, a document holding five mediocre matches would outrank a document
    with one excellent match purely because the chunker split it into five. That
    is a property of the chunker, not of relevance.
    """
    many = [ScoredChunk(i, 5.0 + i * 0.01, "MANY", "results") for i in range(5)]
    one = [ScoredChunk(99, 6.0, "ONE", "results")]
    cands = many + one
    maxed = sibling_bleed(cands, alpha=0.5, mode="max")
    summed = sibling_bleed(cands, alpha=0.5, mode="sum")
    many_max = [c for c in maxed if c.doc_id == "MANY"]
    one_max = [c for c in maxed if c.doc_id == "ONE"][0]
    assert one_max.score > many_max[0].score, (
        f"single strong match lost to five mediocre ones under max: "
        f"{one_max.score} vs {many_max[0].score}"
    )
    # The modes must actually differ, or `mode` is not doing anything.
    assert [c.row_id for c in maxed] != [c.row_id for c in summed]
    print("  ok  max does not reward chunk count; sum differs measurably")


def test_bleed_leaves_singletons_alone() -> None:
    """A document with one chunk has no sibling to bleed from."""
    cands = [ScoredChunk(0, 5.0, "SOLO", "results"), ScoredChunk(1, 4.0, "SOLO2", "results")]
    out = sibling_bleed(cands, alpha=0.5)
    for c in out:
        assert c.bleed == 0.0, f"singleton bled: {c}"
    # Scores are normalised but with no bleed the order is preserved.
    assert [c.row_id for c in out] == [0, 1]
    print("  ok  singleton documents are unaffected")


def test_bleed_is_scale_invariant() -> None:
    """alpha must mean the same thing for BM25 and cosine, which differ hugely.

    Scores are min-max normalised before bleeding, so a rescaled input produces an
    equivalent ordering rather than a wildly different one.
    """
    cands = [ScoredChunk(0, 100.0, "A", "results"), ScoredChunk(1, 50.0, "A", "methods"),
             ScoredChunk(2, 49.0, "B", "results")]
    scaled = [ScoredChunk(c.row_id, c.score * 1000, c.doc_id, c.section_kind) for c in cands]
    o1 = sibling_bleed(cands, alpha=0.3)
    o2 = sibling_bleed(scaled, alpha=0.3)
    assert [c.row_id for c in o1] == [c.row_id for c in o2], (o1, o2)
    print("  ok  bleed ordering is invariant to score scale")


def test_apply_bleed_over_a_store() -> None:
    store = _Store({0: ("A", "results"), 1: ("A", "methods"), 2: ("B", "results")})
    out = apply_sibling_bleed([(0, 10.0), (1, 4.0), (2, 4.5)], store, alpha=0.5)
    assert [r for r, _ in out] == [0, 1, 2], out
    print("  ok  apply_sibling_bleed ranks correctly through the store")


def test_bleed_report_detects_reordering() -> None:
    store = _Store({0: ("A", "results"), 1: ("A", "methods"), 2: ("B", "results")})
    before = [(0, 10.0), (1, 4.0), (2, 4.5)]
    after = apply_sibling_bleed(before, store, alpha=0.9)
    rep = bleed_report(before, after, store)
    assert rep["n_candidates"] == 3
    assert rep["changed"] >= 1, rep
    assert rep["frac_reordered"] > 0
    print(f"  ok  bleed_report sees the reordering: {rep['changed']}/3 moved")


# ------------------------------------------------------------ context expansion


class _Dep:
    """Dependency-graph stub returning fixed context pairs."""

    def __init__(self, pairs):
        self.pairs = pairs
        self.calls: list[tuple] = []

    def context_pairs(self, seeds, *, top_k=10, max_per_seed=2):
        self.calls.append((tuple(seeds), top_k, max_per_seed))
        return [p for p in self.pairs if p[0] in set(seeds)][:top_k]


def test_context_appends_framing_passages_without_duplicates() -> None:
    store = _Store({i: (f"D{i}", "results") for i in range(10)})
    # Seed 0 has a conclusions neighbour at row 5; seed 1 has one at row 6.
    dep = _Dep([(0, 5, 1.0), (1, 6, 1.0), (0, 5, 1.0)])
    base = [(0, 9.0), (1, 8.0), (2, 7.0)]
    out = expand_with_context(base, store, dep, top_k=3, seed_count=2, context_per_seed=1)
    rows = [r for r, _ in out]
    assert rows[:3] == [0, 1, 2], "retrieval order must be preserved"
    assert 5 in rows and 6 in rows, f"context missing: {rows}"
    assert len(rows) == len(set(rows)), f"duplicate rows: {rows}"
    print(f"  ok  context appended without disturbing or duplicating: {rows}")


def test_context_respects_the_growing_list_bound() -> None:
    """One seed in a long section has many equally-weighted neighbours.

    Without an explicit budget a single seed can turn a 2-result answer into a
    15-chunk dump, which is not context provision, it is flooding.
    """
    store = _Store({i: (f"D{i}", "results") for i in range(20)})
    dep = _Dep([(0, i, 1.0) for i in range(5, 20)])
    out = expand_with_context([(0, 9.0), (1, 8.0)], store, dep,
                              top_k=2, seed_count=1, context_per_seed=15)
    assert len(out) == 2 + 2, f"budget not respected: {len(out)}"
    tight = expand_with_context([(0, 9.0), (1, 8.0)], store, dep,
                                top_k=2, seed_count=1, context_per_seed=15,
                                context_budget=1)
    assert len(tight) == 3, len(tight)
    zero = expand_with_context([(0, 9.0), (1, 8.0)], store, dep,
                               top_k=2, seed_count=1, context_budget=0)
    assert len(zero) == 2, zero
    print(f"  ok  context bounded: default {len(out)}, budget=1 {len(tight)}, budget=0 {len(zero)}")


def test_context_returns_more_than_top_k_by_design() -> None:
    """Truncating back to top_k would discard the framing passages just added.

    The budget defaults to top_k, so a top_k=1 request adds 1 context passage; a
    top_k=10 request may add up to 10. Callers wanting a specific number pass
    context_budget.
    """
    store = _Store({i: (f"D{i}", "results") for i in range(10)})
    dep = _Dep([(0, 5, 1.0), (0, 6, 1.0), (0, 7, 1.0)])
    one = expand_with_context([(0, 9.0)], store, dep, top_k=1, seed_count=1, context_per_seed=3)
    assert len(one) == 2, one
    two = expand_with_context([(0, 9.0), (1, 8.0)], store, dep, top_k=2, seed_count=1,
                               context_per_seed=3, context_budget=2)
    assert len(two) == 4, two
    assert [r for r, _ in two] == [0, 1, 5, 6], two
    print(f"  ok  top_k=1 -> {len(one)} rows, top_k=2 budget=2 -> {len(two)} rows")


def test_context_falls_back_when_no_graph() -> None:
    """Absent graph must degrade to the base ranking, not raise."""
    store = _Store({0: ("A", "results")})
    base = [(0, 9.0), (1, 8.0)]
    assert expand_with_context(base, store, None, top_k=2) == base
    print("  ok  missing dependency graph degrades to the base ranking")


def test_context_seeds_from_a_prefix_not_the_whole_pool() -> None:
    """Seeding from everything would attach framing to weakly-matched passages."""
    store = _Store({i: (f"D{i}", "results") for i in range(20)})
    dep = _Dep([(i, 19, 1.0) for i in range(20)])
    expand_with_context([(i, 20.0 - i) for i in range(10)], store, dep,
                        top_k=10, seed_count=3, context_per_seed=1)
    seeds_seen, _, _ = dep.calls[0]
    assert len(seeds_seen) == 3, seeds_seen
    print(f"  ok  only the top {len(seeds_seen)} passages acted as seeds")


def main() -> int:
    tests = [
        test_alpha_zero_is_exact_identity,
        test_bleed_promotes_a_sibling_of_a_strong_match,
        test_bleed_uses_max_not_sum,
        test_bleed_leaves_singletons_alone,
        test_bleed_is_scale_invariant,
        test_apply_bleed_over_a_store,
        test_bleed_report_detects_reordering,
        test_context_appends_framing_passages_without_duplicates,
        test_context_respects_the_growing_list_bound,
        test_context_returns_more_than_top_k_by_design,
        test_context_falls_back_when_no_graph,
        test_context_seeds_from_a_prefix_not_the_whole_pool,
    ]
    print("Sibling-bleed and context-expansion tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())