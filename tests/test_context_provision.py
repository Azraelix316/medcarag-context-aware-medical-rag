"""Tests for the context-provision metrics.

The motivating failure is invisible to hit@k: a system can list every complication
of a procedure, cite real sources, and score a perfect recall while telling the
reader nothing about how often those complications occur. These tests pin the
behaviour that distinguishes "found the fact" from "found the fact and its
framing", because a metric that collapses the two is worse than no metric.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from medrag.eval.context_provision import (
    RetrievedChunk,
    aggregate_context_provision,
    assess_context_provision,
    compare_provision,
)


def c(row: int, doc: str, kind: str) -> RetrievedChunk:
    return RetrievedChunk(row_id=row, doc_id=doc, section_kind=kind)


GOLD = {"D1", "D2"}


def test_bare_answer_is_classified_bare() -> None:
    """The core failure: gold document found, no framing passage from it."""
    r = assess_context_provision(
        "q1", [c(0, "D1", "results"), c(1, "D1", "results"), c(2, "X9", "conclusions")], GOLD
    )
    assert r.found_gold, "should have found a gold document"
    assert r.bare, "a results-only gold document is a bare answer"
    assert not r.qualified
    # The irrelevant document's conclusions chunk must not rescue it.
    assert r.qualifying_docs == [], r.qualifying_docs
    print("  ok  gold findings without same-document framing are bare")


def test_qualified_answer_needs_same_document_framing() -> None:
    r = assess_context_provision("q1", [c(0, "D1", "results"), c(1, "D1", "conclusions")], GOLD)
    assert r.qualified and not r.bare
    assert r.qualifying_docs == ["D1"]
    print("  ok  finding plus same-document conclusions is qualified")


def test_framing_from_another_gold_document_does_not_qualify() -> None:
    """Framing is document-local. D1's result is not qualified by D2's caveats."""
    r = assess_context_provision(
        "q1", [c(0, "D1", "results"), c(1, "D2", "conclusions")], GOLD
    )
    assert r.found_gold
    assert r.bare, (
        "a conclusions chunk from a different study must not qualify this result; "
        "different population, different rate"
    )
    print("  ok  cross-document framing does not qualify a result")


def test_no_gold_document_is_neither_qualified_nor_bare() -> None:
    """Otherwise the metric double-counts and qar + bar > 1."""
    r = assess_context_provision("q1", [c(0, "X9", "results"), c(1, "X8", "conclusions")], GOLD)
    assert not r.found_gold and not r.bare and not r.qualified
    print("  ok  irrelevant results are neither qualified nor bare")


def test_strict_framing_excludes_methods_by_default() -> None:
    """Methods describes how a study was done; it is not a caveat or a benefit."""
    strict = assess_context_provision("q1", [c(0, "D1", "results"), c(1, "D1", "methods")], GOLD)
    assert strict.bare, "a Methods chunk must not count as qualification by default"
    loose = assess_context_provision(
        "q1", [c(0, "D1", "results"), c(1, "D1", "methods")], GOLD, strict_framing=False
    )
    assert loose.qualified, "Methods should count when strict_framing is disabled"
    print("  ok  strict framing counts only conclusions/discussion")


def test_discussion_counts_as_framing() -> None:
    r = assess_context_provision("q1", [c(0, "D1", "results"), c(1, "D1", "discussion")], GOLD)
    assert r.qualified
    print("  ok  discussion qualifies")


def test_qar_bar_and_sabr_are_consistent() -> None:
    """qar + bar + not-found = 1, and sabr is conditional on finding something."""
    qs = [
        assess_context_provision("a", [c(0, "D1", "results"), c(1, "D1", "conclusions")], GOLD),
        assess_context_provision("b", [c(0, "D1", "results")], GOLD),
        assess_context_provision("c", [c(0, "X1", "results")], GOLD),
        assess_context_provision("d", [c(0, "D2", "results"), c(1, "D2", "discussion")], GOLD),
    ]
    rep = aggregate_context_provision(qs)
    assert rep.n_queries == 4 and rep.n_found == 3
    assert rep.n_qualified == 2 and rep.n_bare == 1
    assert abs(rep.qar - 0.5) < 1e-9, rep.qar
    assert abs(rep.bar - 0.25) < 1e-9, rep.bar
    assert abs(rep.qar + rep.bar + (1 - rep.n_found / 4) - 1.0) < 1e-9, (
        f"qar {rep.qar} + bar {rep.bar} + miss must equal 1"
    )
    assert abs(rep.sabr - 2 / 3) < 1e-9, rep.sabr
    print(f"  ok  qar={rep.qar:.2f} bar={rep.bar:.2f} sabr={rep.sabr:.2f} sum to 1")


def test_qual_ratio_reports_framing_balance() -> None:
    qs = [
        assess_context_provision(
            "a", [c(0, "D1", "results"), c(1, "D1", "conclusions"), c(2, "D1", "discussion")], GOLD
        ),
        assess_context_provision("b", [c(0, "D1", "results")], GOLD),
    ]
    rep = aggregate_context_provision(qs)
    # (2/1 + 0/1) / 2 found queries
    assert abs(rep.qual_ratio - 1.0) < 1e-9, rep.qual_ratio
    print(f"  ok  qual_ratio={rep.qual_ratio:.2f} framing passages per finding")


def test_aggregate_of_nothing_is_zero_not_a_crash() -> None:
    rep = aggregate_context_provision([])
    assert rep.qar == 0.0 and rep.bar == 0.0 and rep.sabr == 0.0
    assert rep.summary()["n_queries"] == 0
    print("  ok  empty input yields zeros")


def test_missing_gold_treats_everything_as_irrelevant() -> None:
    """A goldless query must not be counted as a bare answer."""
    r = assess_context_provision("q1", [c(0, "D1", "results")], set())
    assert not r.bare and not r.qualified
    print("  ok  a query with no gold documents produces no verdict")


def test_compare_provision_flags_direction_per_metric() -> None:
    base = aggregate_context_provision(
        [assess_context_provision("a", [c(0, "D1", "results")], GOLD)]
    )
    cand = aggregate_context_provision(
        [assess_context_provision("a", [c(0, "D1", "results"), c(1, "D1", "conclusions")], GOLD)]
    )
    out = compare_provision(base, cand)
    assert out["qar"]["better"] is True, out["qar"]
    assert out["bar"]["better"] is True, out["bar"]   # bar fell: good
    assert out["sabr"]["better"] is True, out["sabr"]
    # Direction must be per-metric, not a blanket ">".
    worse = compare_provision(cand, base)
    assert worse["qar"]["better"] is False, worse["qar"]
    assert worse["bar"]["better"] is False, worse["bar"]
    print("  ok  improvement direction is correct per metric and reverses when swapped")


def test_report_serialises() -> None:
    rep = aggregate_context_provision(
        [assess_context_provision("a", [c(0, "D1", "results"), c(1, "D1", "conclusions")], GOLD)]
    )
    d = rep.to_dict()
    assert d["summary"]["n_qualified"] == 1
    assert d["per_query"][0]["qid"] == "a"
    assert d["per_query"][0]["qualified"] is True
    print("  ok  report serialises with summary and per-query detail")


def main() -> int:
    tests = [
        test_bare_answer_is_classified_bare,
        test_qualified_answer_needs_same_document_framing,
        test_framing_from_another_gold_document_does_not_qualify,
        test_no_gold_document_is_neither_qualified_nor_bare,
        test_strict_framing_excludes_methods_by_default,
        test_discussion_counts_as_framing,
        test_qar_bar_and_sabr_are_consistent,
        test_qual_ratio_reports_framing_balance,
        test_aggregate_of_nothing_is_zero_not_a_crash,
        test_missing_gold_treats_everything_as_irrelevant,
        test_compare_provision_flags_direction_per_metric,
        test_report_serialises,
    ]
    print("Context-provision metric tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())