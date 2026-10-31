"""Tests for the co-citation graph and its corroboration signal.

Co-citation is where the contextual-contamination thesis gets its teeth: if
several documents answering a query all cite the same paper, that paper is
corroborated evidence. This is a different claim from "these documents are on the
same topic" -- it measures attestation, and it only exists where bibliographies do.

The failure mode these tests guard is a hub. One landmark trial is cited by
thousands of papers; a co-citation graph built without a cap collapses into one
blob where every document is a neighbour of every other and traversal carries no
information at all. Both the shared-count cap and the ubiquitous-reference skip
are therefore pinned.
"""

from __future__ import annotations

import sys
import tempfile

sys.path.insert(0, ".")

from medrag.index.cocitation import CoCitationGraph


class D:
    def __init__(self, doc_id: str, refs, year: int = 2022) -> None:
        self.doc_id = doc_id
        self.citations = tuple(refs)
        self.year = year


def test_shared_reference_creates_an_edge() -> None:
    g = CoCitationGraph.build([D("A", [1, 2]), D("B", [1, 3])], min_shared=1)
    assert ("A", 1) in g.neighbours("B") and ("B", 1) in g.neighbours("A")
    assert g.neighbours("B")[0] == ("A", 1)
    print("  ok  one shared reference creates a symmetric edge")


def test_edges_are_symmetric_and_sorted_by_strength() -> None:
    docs = [D("A", [1, 2, 3]), D("B", [1, 2]), D("C", [1, 2, 3]), D("E", [1, 2, 3, 4])]
    g = CoCitationGraph.build(docs, min_shared=2)
    a = dict(g.neighbours("A"))
    assert all("A" in dict(g.neighbours(b)) for b in a), "asymmetric edges"
    order = [b for b, _ in g.neighbours("E")]
    counts = dict(g.neighbours("E"))
    assert counts[order[0]] >= counts[order[-1]], order
    print(f"  ok  symmetric and strength-ordered: {order}")


def test_indptr_is_monotonic() -> None:
    """CSR invariant; a non-monotonic indptr silently truncates neighbours."""
    docs = [D(f"D{i}", [1, 2, i]) for i in range(20)]
    g = CoCitationGraph.build(docs, min_shared=2)
    assert all(g.indptr[i] <= g.indptr[i + 1] for i in range(len(g.indptr) - 1))
    assert g.indptr[-1] == len(g.indices) == len(g.shared)
    print(f"  ok  indptr monotonic, {g.stats()['n_pairs']} pairs")


def test_min_shared_prunes_one_off_coincidence() -> None:
    docs = [D("A", [1, 2, 3]), D("B", [1, 4, 5]), D("C", [1, 2, 3])]
    loose = CoCitationGraph.build(docs, min_shared=1)
    strict = CoCitationGraph.build(docs, min_shared=2)
    assert strict.neighbours("B") == [], (
        f"B shares only ref 1 with A/C; min_shared=2 must drop it: {strict.neighbours('B')}"
    )
    assert loose.neighbours("B"), "min_shared=1 should admit the single shared ref"
    print("  ok  min_shared prunes single-reference coincidences")


def test_ubiquitous_reference_does_not_create_a_hub() -> None:
    """A reference cited by hundreds of papers carries no information.

    Without the skip, one landmark trial makes every document a neighbour of
    every other and traversal stops discriminating.
    """
    hub = [D(f"H{i}", list(range(100))) for i in range(300)]      # all cite 0..99
    g = CoCitationGraph.build(hub, min_shared=2)
    deg = g.indptr[1] if len(g.doc_ids) else 0
    assert len(g.doc_ids) == 0, (
        f"{len(g.doc_ids)} documents entered the graph via a ubiquitous reference; "
        f"expected none"
    )
    print(f"  ok  a reference cited by 300 papers creates no edges ({deg} neighbours)")


def test_max_shared_cap_limits_strong_pairs() -> None:
    """Without a cap, a pair sharing 40 references dominates traversal."""
    docs = [D("A", list(range(40))), D("B", list(range(40)))]
    capped = CoCitationGraph.build(docs, min_shared=2, max_shared=25)
    assert len(capped.doc_ids) == 0, "pair above max_shared should be dropped"
    uncapped = CoCitationGraph.build(docs, min_shared=2, max_shared=1000)
    assert len(uncapped.doc_ids) == 2, uncapped.stats()
    print("  ok  max_shared drops over-strong pairs")


def test_corroboration_counts_attesting_documents() -> None:
    """The signal the mechanism rests on: attestation, not topical similarity."""
    docs = [D("A", [1, 2]), D("B", [1, 2]), D("C", [1, 2]), D("D", [9])]
    g = CoCitationGraph.build(docs, min_shared=2)
    corr = g.corroboration(["A", "B", "C"], min_shared=2)
    counts = {ref: n for n, ref, _ in corr}
    assert counts.get(1) == 3 and counts.get(2) == 3, counts
    # A reference cited once is not corroborated.
    once = g.corroboration(["A", "D"], min_shared=2)
    assert once == [], once
    print(f"  ok  corroboration counts attesters: {counts}")


def test_corroboration_reports_which_documents_attest() -> None:
    docs = [D("A", [7]), D("B", [7]), D("C", [8])]
    g = CoCitationGraph.build(docs, min_shared=2)
    corr = g.corroboration(["A", "B", "C"], min_shared=2)
    assert corr and corr[0][1] == 7
    assert corr[0][2] == {"A", "B"}, corr[0][2]
    print("  ok  corroboration names the attesting documents")


def test_documents_without_bibliographies_are_excluded() -> None:
    g = CoCitationGraph.build([D("A", []), D("B", ()), D("C", [1], 2020)])
    assert g.doc_ids == [], g.stats()
    assert g.stats()["n_docs_with_bibliography"] == 1
    assert g.corroboration(["A"]) == []
    print("  ok  no bibliography means no co-citation evidence, not a crash")


def test_era_coverage_is_recorded() -> None:
    """Era bias must be visible in the artefact, not reconstructed later."""
    docs = [D("A", [1, 2], 2003), D("B", [1, 2], 2005), D("C", [1, 2], 2023)]
    g = CoCitationGraph.build(docs, min_shared=2)
    assert g.stats()["era_coverage"] == {2000: 2, 2020: 1}, g.stats()["era_coverage"]
    print(f"  ok  era coverage recorded: {g.stats()['era_coverage']}")


def test_dependency_edges_are_emitted_once_per_pair() -> None:
    docs = [D("A", [1, 2]), D("B", [1, 2]), D("C", [1, 2])]
    g = CoCitationGraph.build(docs, min_shared=2)
    edges = g.to_dependency_edges()
    assert len(edges) == 3, edges   # 3 pairs, not 6
    assert all(a != b for a, b, _ in edges), edges
    assert all(w == 0.5 for _, _, w in edges), edges
    print(f"  ok  {len(edges)} undirected pairs emitted once each")


def test_save_load_roundtrip_preserves_traversal() -> None:
    import tempfile
    from pathlib import Path

    docs = [D(f"D{i}", [1, 2, i]) for i in range(15)]
    g = CoCitationGraph.build(docs, min_shared=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "co.pkl"
        g.save(str(path))
        back = CoCitationGraph.load(str(path))
    assert back.doc_ids == g.doc_ids
    assert back.neighbours(g.doc_ids[0]) == g.neighbours(g.doc_ids[0])
    assert back.corroboration(g.doc_ids) == g.corroboration(g.doc_ids)
    print("  ok  co-citation graph round-trips without changing traversal")


def test_max_citers_bounds_pair_enumeration() -> None:
    """Pair enumeration is quadratic in citing-list length, so this is a memory bound.

    Measured: with the bound unset, a reference cited by 5,000 documents alone
    yields 12.5M pairs and the build was OOM-killed on a 930,000-document corpus.
    """
    docs = [D(f"L{i}", [1, 2]) for i in range(300)]
    bounded = CoCitationGraph.build(docs, min_shared=2, max_citers=50)
    assert len(bounded.doc_ids) == 0, (
        f"{len(bounded.doc_ids)} documents entered despite max_citers=50"
    )
    loose = CoCitationGraph.build(docs, min_shared=2, max_citers=1000)
    assert len(loose.doc_ids) == 300, loose.stats()
    print(f"  ok  max_citers bounds enumeration ({len(bounded.doc_ids)} vs {len(loose.doc_ids)})")





def test_save_returns_a_path_not_the_graph() -> None:
    """save() must return a Path, like every other graph in the package.

    It used to return ``self`` for chaining, and the CLI does
    ``path = g.save(...)`` followed by ``"path": str(path)`` -- so the entire graph,
    every document id and every reference tuple, was stringified into a single
    324,133,380-character log line. The build was correct; only the report was
    absurd, and it filled a log file that then had to be read to find the stats.
    """
    from pathlib import Path as _P

    from medrag.index.cocitation import CoCitationGraph as _G

    class _Doc:
        def __init__(self, doc_id, citations, year=2020):
            self.doc_id = doc_id
            self.citations = citations
            self.year = year

    g = _G.build([_Doc("a", (1, 2)), _Doc("b", (2, 3)), _Doc("c", (1, 2))],
                 min_shared=1)
    with tempfile.TemporaryDirectory() as d:
        out = g.save(_P(d) / "co.pkl")
        assert isinstance(out, _P), f"save() returned {type(out).__name__}, not a Path"
        assert out.exists()
        assert len(str(out)) < 200, f"str(path) is {len(str(out))} characters"
    print("  ok  save() returns a Path whose repr is short")


def main() -> int:
    tests = [
        test_save_returns_a_path_not_the_graph,
        test_shared_reference_creates_an_edge,
        test_edges_are_symmetric_and_sorted_by_strength,
        test_indptr_is_monotonic,
        test_min_shared_prunes_one_off_coincidence,
        test_ubiquitous_reference_does_not_create_a_hub,
        test_max_shared_cap_limits_strong_pairs,
        test_corroboration_counts_attesting_documents,
        test_corroboration_reports_which_documents_attest,
        test_documents_without_bibliographies_are_excluded,
        test_era_coverage_is_recorded,
        test_dependency_edges_are_emitted_once_per_pair,
        test_max_citers_bounds_pair_enumeration,
        test_save_load_roundtrip_preserves_traversal,
    ]
    print("Co-citation graph tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
