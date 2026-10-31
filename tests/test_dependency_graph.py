"""Tests for the section-dependency graph over chunks.

The graph's value is a claim about medical argument structure: a Results passage
is not interpretable without the Methods population that produced it, and a
stated complication rate is not trustworthy without the Conclusions that qualify
it. These tests check that claim holds on documents whose correct answers are
known by construction.

Two bugs found while writing these are pinned here:

1. ``indptr`` was assigned the per-node degree instead of a running total, giving
   a non-monotonic array (``[0,2,3,2,3]`` for a 4-node document). Every slice
   after the first isolated node pointed at the wrong rows, so nodes that had
   edges reported none. Silently wrong, not a crash.

2. Pairing to "the nearest chunk of the paired role" was implemented with a
   ``break`` inside the inner loop, which is correct, but the qualification edge
   for a conclusions chunk had no such bound and linked every results and methods
   chunk in the document -- a 40-chunk Results section became a hub.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np

from medrag.index.dependency_graph import (
    INTERPRETIVE_PAIRS,
    SECTION_ORDER,
    W_INTERPRETIVE,
    DependencyGraph,
    section_rank,
)


def canonical_doc(prefix: str = "c", n: int = 1) -> DependencyGraph:
    """One document per call with the standard five-section argument order."""
    kinds = ["background", "methods", "results", "conclusions"]
    rows, ks, docs = [], [], []
    for d in range(n):
        for k in kinds:
            rows.append(f"{prefix}{len(rows)}")
            ks.append(k)
            docs.append(f"D{d}")
    return DependencyGraph.build(rows, kind_of=lambda r: ks[r], doc_of=lambda r: docs[r])


def test_indptr_is_monotonic_and_slices_correctly() -> None:
    """CSR invariants. A non-monotonic indptr silently truncates every node."""
    g = canonical_doc(n=6)
    ptr = g.indptr
    assert all(ptr[i] <= ptr[i + 1] for i in range(len(ptr) - 1)), (
        f"indptr must be non-decreasing, got {ptr}"
    )
    assert ptr[-1] == len(g.indices) == len(g.weights), (
        f"indptr tail {ptr[-1]} != indices {len(g.indices)} / weights {len(g.weights)}"
    )
    # Every node's slice must decode to valid neighbour ids.
    for node in range(g.n_nodes):
        idx, w = g.neighbours(node)
        assert len(idx) == len(w)
        for j in idx:
            assert 0 <= int(j) < g.n_nodes, f"neighbour {j} out of range"
    print(f"  ok  indptr monotonic, {g.stats()['n_edges']} edges over {g.n_nodes} nodes")


def test_results_reaches_methods_and_conclusions() -> None:
    """The central claim: an outcome passage is linked to both anchors."""
    g = canonical_doc()
    i_bg, i_me, i_re, i_co = 0, 1, 2, 3
    for seed, want in ((i_re, {i_me, i_co}), (i_me, {i_re, i_co}), (i_co, {i_re, i_me})):
        idx, _ = g.neighbours(seed)
        got = set(int(j) for j in idx)
        missing = want - got
        assert not missing, f"node {g.section_kind[seed]} missing links to {missing}"
    print("  ok  methods/results/conclusions are mutually linked")


def test_interpretive_edges_outrank_narrative() -> None:
    """Interpretive edges must be the strongest, since they are the ones that
    prevent a fact being read as unconditional."""
    g = canonical_doc()
    idx, w = g.neighbours(2)  # results
    order = {int(j): float(x) for j, x in zip(idx, w)}
    assert order[1] == W_INTERPRETIVE and order[3] == W_INTERPRETIVE, order
    # A background edge would be narrative-only and strictly weaker.
    g2 = DependencyGraph.build(
        ["a", "b", "c"], kind_of=lambda r: ["background", "methods", "results"][r],
        doc_of=lambda r: "D",
    )
    idx2, w2 = g2.neighbours(1)
    strengths = {float(x) for x in w2}
    assert max(strengths) == W_INTERPRETIVE, strengths
    assert len(strengths) > 1, "expected both interpretive and narrative weights"
    print(f"  ok  interpretive {W_INTERPRETIVE} > narrative edges present")


def test_background_does_not_skip_methods_to_reach_results() -> None:
    """Direct Background->Results is wrong; Methods is what sits between them."""
    g = DependencyGraph.build(
        ["a", "b", "c"], kind_of=lambda r: ["background", "methods", "results"][r],
        doc_of=lambda r: "D",
    )
    idx, _ = g.neighbours(0)
    # Background may reach the conclusions (a qualification edge) but must not
    # jump straight to results.
    assert 2 not in {int(j) for j in idx}, (
        "background links directly to results, skipping methods"
    )
    print("  ok  background does not skip over methods")


def test_section_pairs_are_all_reachable() -> None:
    """Every configured interpretive pair present in the document must produce
    an edge. Pairs whose sections are absent are skipped, since a document with
    no Discussion cannot link one."""
    g = canonical_doc()
    present = set()
    for node in range(g.n_nodes):
        idx, w = g.neighbours(node)
        for j, x in zip(idx, w):
            if float(x) == W_INTERPRETIVE:
                present.add(tuple(sorted((node, int(j)))))
    checked = 0
    for a_kind, b_kind in INTERPRETIVE_PAIRS:
        if a_kind not in g.section_kind or b_kind not in g.section_kind:
            continue
        a = g.section_kind.index(a_kind)
        b = g.section_kind.index(b_kind)
        assert tuple(sorted((a, b))) in present, f"no interpretive edge for {a_kind}<->{b_kind}"
        checked += 1
    assert checked >= 2, f"only {checked} pairs testable; fixture lacks sections"
    print(f"  ok  all {checked} applicable interpretive pairs produce edges")

    # And a document that *does* have a Discussion must link it to Conclusions.
    g2 = DependencyGraph.build(
        ["a", "b"], kind_of=lambda r: ["discussion", "conclusions"][r],
        doc_of=lambda r: "D",
    )
    idx2, w2 = g2.neighbours(0)
    assert 1 in {int(j) for j in idx2}, "discussion did not reach conclusions"
    print("  ok  discussion <-> conclusions edge exists when both sections present")


def test_expansion_is_damped_and_bounded() -> None:
    """A two-hop dependency must not outrank a one-hop interpretation."""
    g = canonical_doc()
    hop1 = dict(g.expand([2], depth=1, top_k=10))
    hop2 = dict(g.expand([2], depth=2, top_k=10))
    assert hop1, "depth=1 found nothing"
    for node, s1 in hop1.items():
        if node in hop2:
            assert hop2[node] <= s1 + 1e-6, (
                f"node {node} gained score with more hops: {s1} -> {hop2[node]}"
            )
    print(f"  ok  depth-2 expansion adds {len(set(hop2) - set(hop1))} nodes, none stronger")


def test_expansion_never_returns_the_seed() -> None:
    g = canonical_doc(n=3)
    seeds = [0, 4]
    out = [n for n, _ in g.expand(seeds, depth=2, top_k=50)]
    assert not (set(out) & set(seeds)), f"seeds re-returned as expansions: {out}"
    print("  ok  seeds are excluded from their own expansion")


def test_context_pairs_pair_a_seed_with_its_qualifier() -> None:
    """The retrieval shape the decontextualization failure calls for."""
    g = canonical_doc()
    pairs = g.context_pairs([2], top_k=10)  # seed = results
    assert pairs, "no context pairs for a results seed"
    qualifiers = {g.section_kind[b] for _, b, _ in pairs}
    assert qualifiers & {"methods", "conclusions"}, qualifiers
    for a, _b, _w in pairs:
        assert a == 2, "context pair must be anchored on the seed"
    # Both orientations of the same edge should not appear as duplicate pairs.
    assert len({(a, b) for a, b, _ in pairs}) == len(pairs), "duplicate context pairs"
    print(f"  ok  results seed yields {sorted(qualifiers)} as context")


def test_context_pairs_exclude_other_seeds() -> None:
    """A context chunk that is itself a seed carries no information."""
    g = canonical_doc()
    pairs = g.context_pairs([1, 2], top_k=10)
    for a, b, _ in pairs:
        assert b not in (1, 2), f"seed {b} returned as context for itself"
    print("  ok  seeds are excluded from the context set")


def test_documents_do_not_leak_into_each_other() -> None:
    """Edges must be intra-document unless explicitly given cross-doc edges."""
    kinds_cycle = ["background", "methods", "results", "conclusions"]
    rows, ks, docs = [], [], []
    for d in range(4):
        for k in kinds_cycle:
            rows.append(f"r{len(rows)}")
            ks.append(k)
            docs.append(f"D{d}")
    g = DependencyGraph.build(rows, kind_of=lambda r: ks[r], doc_of=lambda r: docs[r])
    doc_of_node = [docs[r] for r in range(len(rows))]
    for node in range(g.n_nodes):
        idx, _ = g.neighbours(node)
        for j in idx:
            assert doc_of_node[int(j)] == doc_of_node[node], (
                f"edge {node}->{j} crosses documents {doc_of_node[node]}/{doc_of_node[int(j)]}"
            )
    print("  ok  no cross-document edges without explicit cross_doc_edges")


def test_isolated_and_single_section_documents_are_safe() -> None:
    """A document with only an abstract must not crash or invent edges."""
    g = DependencyGraph.build(
        ["a", "b"], kind_of=lambda r: ["background", ""][r], doc_of=lambda r: "D"
    )
    assert g.stats()["n_nodes"] == 2
    assert g.expand([0], depth=2, top_k=5) == [] or all(
        n != 0 for n, _ in g.expand([0], depth=2, top_k=5)
    )
    empty = DependencyGraph.build(["a"], kind_of=lambda r: "", doc_of=lambda r: "")
    assert empty.stats()["n_nodes"] == 1
    assert empty.expand([0], depth=1) == []
    print("  ok  single-section and unlabelled documents degrade safely")


def test_unknown_section_kinds_sort_first() -> None:
    assert section_rank("nonsense") == -1
    assert section_rank("results") > section_rank("methods") > section_rank("background")
    assert list(SECTION_ORDER)[0] == "background"
    print("  ok  section ordering is total and unknown kinds sort first")


def test_save_load_is_lossless() -> None:
    import tempfile
    from pathlib import Path

    g = canonical_doc(n=3)
    with tempfile.TemporaryDirectory() as tmp:
        back = DependencyGraph.load(g.save(Path(tmp) / "dep.pkl"))
    assert back.n_nodes == g.n_nodes
    assert back.section_kind == g.section_kind
    assert np.array_equal(back.indptr, g.indptr)
    assert np.array_equal(back.indices, g.indices)
    assert np.allclose(back.weights, g.weights)
    for node in (0, 2, 7):
        assert [int(x) for x in back.neighbours(node)[0]] == [
            int(x) for x in g.neighbours(node)[0]
        ], f"neighbours changed for node {node} after a save/load round trip"
    print("  ok  persistence is lossless, including per-node adjacency")


def test_degree_by_kind_is_reported() -> None:
    g = canonical_doc(n=4)
    by_kind = g.degree_by_kind()
    assert set(by_kind) >= {"background", "methods", "results", "conclusions"}, by_kind
    assert all(v >= 0 for v in by_kind.values())
    print(f"  ok  mean degree by section kind: "
          + ", ".join(f"{k}={v:.1f}" for k, v in sorted(by_kind.items())))


# ---------------------------------------------------------- CSR invariants
#
# The edge set accumulates into array('i')/array('f') rather than a
# dict[int, list[tuple[int, float]]]. The dict form was OOM-killed twice on the
# 16.1M-chunk PMC corpus -- RSS went 6.1 GiB -> 21.9 GiB inside edge construction
# and dmesg recorded anon-rss 21870008 kB -- and every byte of it was discarded by
# _finalize, which converts to CSR numpy anyway. 12 bytes per directed edge instead
# of well over 100.
#
# The rewrite also silently dropped the reverse half of every edge on its first
# pass, leaving the graph one-way. Traversal that only follows edges forward would
# not notice; a reciprocity check does, and caught 250 of 250 pairs unreciprocated.


def _toy_graph(n_docs: int = 50, per_doc: int = 4) -> DependencyGraph:
    kinds = ("background", "methods", "results", "discussion", "conclusions")
    rows = [f"doc{i // per_doc}#{i}" for i in range(n_docs * per_doc)]
    return DependencyGraph.build(
        rows,
        kind_of=lambda r: kinds[r % 5],
        doc_of=lambda r: f"doc{r // per_doc}",
    )


def test_graph_is_symmetric() -> None:
    """d in neighbours(n) must imply n in neighbours(d)."""
    g = _toy_graph()
    adj = {n: set(int(x) for x in g.neighbours(n)[0]) for n in range(g.n_nodes)}
    bad = [(n, d) for n, ns in adj.items() for d in ns if n not in adj[d]]
    assert not bad, f"{len(bad)} one-way edges, e.g. {bad[:5]}"
    total = sum(len(v) for v in adj.values())
    print(f"  ok  all {total} directed edges are reciprocal")


def test_no_duplicate_neighbours() -> None:
    """Strongest-edge-per-neighbour dedup must actually collapse duplicates."""
    g = _toy_graph()
    for n in range(g.n_nodes):
        d = g.neighbours(n)[0]
        assert len(set(d.tolist())) == len(d), f"duplicate neighbour at node {n}"
    print("  ok  no node lists the same neighbour twice")


def test_neighbours_are_sorted() -> None:
    """Deterministic traversal requires a stable order."""
    g = _toy_graph()
    for n in range(g.n_nodes):
        d = g.neighbours(n)[0]
        assert list(d) == sorted(d), f"neighbours of {n} are unsorted"
    print("  ok  every neighbour list is sorted")


def test_csr_layout_is_consistent() -> None:
    """indptr monotone, its final value equals len(indices), indices in range."""
    g = _toy_graph()
    assert (np.diff(g.indptr) >= 0).all(), "indptr must be non-decreasing"
    assert int(g.indptr[-1]) == len(g.indices), "indptr[-1] must equal len(indices)"
    assert int(g.indptr[0]) == 0
    assert g.indices.min() >= 0 and g.indices.max() < g.n_nodes
    deg = g.degree()
    for n in range(0, g.n_nodes, 13):
        assert len(g.neighbours(n)[0]) == deg[n], f"slice length != degree at {n}"
    print("  ok  CSR layout is internally consistent")


def test_dedup_keeps_exactly_one_weight_per_pair() -> None:
    """Two rules can connect the same pair; only one weight must survive."""
    g = DependencyGraph.build(
        ["d#0", "d#1", "d#2"],
        kind_of=lambda r: "methods" if r < 2 else "discussion",
        doc_of=lambda r: "d",
    )
    counts: dict[tuple[int, int], int] = {}
    for n in range(g.n_nodes):
        for d in g.neighbours(n)[0]:
            key = (min(n, int(d)), max(n, int(d)))
            counts[key] = counts.get(key, 0) + 1
    assert all(v == 2 for v in counts.values()), (
        f"each undirected pair should appear once in each direction: {counts}"
    )
    print(f"  ok  {len(counts)} pairs each carry a single deduplicated weight")


def main() -> int:
    tests = [
        test_indptr_is_monotonic_and_slices_correctly,
        test_results_reaches_methods_and_conclusions,
        test_interpretive_edges_outrank_narrative,
        test_background_does_not_skip_methods_to_reach_results,
        test_section_pairs_are_all_reachable,
        test_expansion_is_damped_and_bounded,
        test_expansion_never_returns_the_seed,
        test_context_pairs_pair_a_seed_with_its_qualifier,
        test_context_pairs_exclude_other_seeds,
        test_documents_do_not_leak_into_each_other,
        test_isolated_and_single_section_documents_are_safe,
        test_unknown_section_kinds_sort_first,
        test_save_load_is_lossless,
        test_degree_by_kind_is_reported,
        test_graph_is_symmetric,
        test_no_duplicate_neighbours,
        test_neighbours_are_sorted,
        test_csr_layout_is_consistent,
        test_dedup_keeps_exactly_one_weight_per_pair,
    ]
    print("Section-dependency graph tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())