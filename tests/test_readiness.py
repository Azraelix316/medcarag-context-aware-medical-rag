"""Tests for the answer-readiness metric.

``context_provision`` counts section roles, which is circular with the mechanism
under test (the mechanism supplies framing passages; the metric counts framing
passages) and is only a proxy for the thing that matters. Readiness looks at the
words instead: does the retrieved text actually contain a rate, a population, a
caveat?

The tests therefore focus on two things: that the detectors fire on realistic
clinical prose, and that they do NOT fire on a section label alone. The second is
the property that makes this metric non-circular.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from medrag.eval.readiness import (
    ReadinessReport,
    RetrievedPassage,
    aggregate_readiness,
    assess_readiness,
    compare_readiness,
    find_caveat,
    find_population,
    find_rate,
)

FINDING = (
    "Major bleeding occurred more frequently in the dual antiplatelet group than "
    "in the single-therapy group during the follow-up period."
)
WITH_RATE = (
    "Major bleeding within 12 months occurred in 4.7% of patients in the dual "
    "group versus 2.1% in the single-therapy group (hazard ratio 2.31, 95% CI "
    "1.44 to 3.71)."
)
WITH_POP = (
    "We randomised 1600 patients with atrial fibrillation aged 65 years or older "
    "1:1 to receive dual antiplatelet therapy or single therapy."
)
WITH_CAVEAT = (
    "This study was not powered to detect differences in rare complications, and "
    "the result should be interpreted with caution given the small sample size."
)
# A rate with no population cue attached. Deliberately avoids "versus", "n =",
# "patients with", "aged" and every other population pattern, because a
# comparator IS population information: "4.7% versus 2.1%" tells the reader which
# groups the percentages describe. Testing isolation with the shared WITH_RATE
# fixture would have passed for the wrong reason.
RATE_ONLY = (
    "Bleeding within 12 months occurred in 4.7% of those allocated to the "
    "combination arm (hazard ratio 2.31, 95% CI 1.44 to 3.71)."
)
BARE = "Several complications were reported during follow-up in both groups."


def _p(text: str, doc: str = "D1", kind: str = "results") -> RetrievedPassage:
    return RetrievedPassage(row_id=0, doc_id=doc, text=text, section_kind=kind)


# ------------------------------------------------------------------ detectors


def test_rate_detector_fires_on_quantified_evidence() -> None:
    assert find_rate(WITH_RATE), "percentages, CIs and effect ratios must be detected"
    assert find_rate("12 of 240 patients experienced the event"), find_rate
    assert find_rate("NNTH was 14 over 5 years")
    assert find_rate("the difference was significant (p = 0.003)")
    print("  ok  rate detector fires on quantified clinical evidence")


def test_rate_detector_ignores_unquantified_claims() -> None:
    """The failure this metric exists to catch: a finding with no number."""
    assert not find_rate(BARE), "a qualitative claim is not a rate"
    assert not find_rate("Bleeding was more common in one group.")
    print("  ok  rate detector ignores unquantified claims")


def test_population_detector_fires_on_denominator_language() -> None:
    assert find_population(WITH_POP)
    assert find_population("Patients with diabetes were enrolled.")
    assert find_population("The mean age was 64 years.")
    print("  ok  population detector fires on denominator language")


def test_caveat_detector_fires_on_limitation_language() -> None:
    assert find_caveat(WITH_CAVEAT)
    assert find_caveat("The trial was underpowered for rare events.")
    assert find_caveat("There was no statistically significant difference.")
    assert find_caveat("Further studies are needed.")
    print("  ok  caveat detector fires on limitation language")


# --------------------------------------------------------------- non-circularity


def test_section_label_alone_does_not_constitute_readiness() -> None:
    """The property that separates this from context_provision.

    A Conclusions chunk containing no rate, population or caveat must NOT be
    ready. Under a section-role metric it would qualify; that is exactly the
    proxy failure this module was written to avoid.
    """
    r = assess_readiness(
        [_p(FINDING, kind="results"), _p("Further research is warranted.", kind="conclusions")],
        {"D1"},
    )
    assert r.found_gold and r.has_finding
    assert not r.has_rate, "a hedge is not a rate"
    assert not r.ready, "section label alone must not produce readiness"
    print("  ok  a section label alone never yields readiness")


def test_results_chunk_can_constitute_readiness() -> None:
    """The converse: role is irrelevant, content is what counts."""
    r = assess_readiness([_p(WITH_RATE, kind="results"), _p(WITH_POP, kind="results")],
                         {"D1"})
    assert r.ready, "a rate plus a population is enough regardless of section"
    print("  ok  content, not section role, determines readiness")


# ------------------------------------------------------------------ verdicts


def test_bare_finding_is_not_quantified() -> None:
    r = assess_readiness([_p(BARE)], {"D1"})
    assert r.found_gold and r.has_finding
    assert not r.quantified and not r.ready
    print("  ok  a bare finding is neither quantified nor ready")


def test_rate_without_population_is_quantified_but_not_ready() -> None:
    """A percentage with nothing to anchor it is the decontextualized answer."""
    assert not find_population(RATE_ONLY), "fixture leaked a population cue"
    r = assess_readiness([_p(FINDING), _p(RATE_ONLY)], {"D1"})
    assert r.quantified, "a rate is present"
    assert not r.ready, "but the population it applies to is missing"
    print("  ok  rate without population is quantified but not ready")


def test_full_set_is_ready() -> None:
    r = assess_readiness([_p(FINDING), _p(WITH_RATE), _p(WITH_POP), _p(WITH_CAVEAT)],
                         {"D1"})
    assert r.ready and r.quantified and r.has_caveat
    print("  ok  finding + rate + population is ready, caveat recorded separately")


def test_non_gold_passages_do_not_count() -> None:
    """A passage from another study cannot make this study's result interpretable."""
    r = assess_readiness([_p(BARE, doc="D1"), _p(WITH_RATE, doc="OTHER"),
                          _p(WITH_POP, doc="OTHER")], {"D1"})
    assert r.found_gold
    assert not r.has_rate, "another study's rate was counted"
    print("  ok  only gold-document passages contribute")


def test_nothing_retrieved_is_neither_found_nor_ready() -> None:
    r = assess_readiness([], {"D1"})
    assert not r.found_gold and not r.ready and not r.quantified
    print("  ok  an empty retrieved set produces no verdict")


# ---------------------------------------------------------------- aggregation


def test_rates_are_consistent_and_bare_complements_found() -> None:
    qs = [
        assess_readiness([_p(FINDING), _p(WITH_RATE), _p(WITH_POP)], {"D1"}, qid="a"),
        assess_readiness([_p(FINDING), _p(RATE_ONLY)], {"D1"}, qid="b"),
        assess_readiness([_p(BARE)], {"D1"}, qid="c"),
        assess_readiness([], {"D1"}, qid="d"),
    ]
    rep = aggregate_readiness(qs)
    assert rep.n_queries == 4 and rep.n_found == 3
    assert rep.n_ready == 1 and rep.n_quantified == 2
    assert abs(rep.readiness - 0.25) < 1e-9
    assert abs(rep.quantification_rate - 0.5) < 1e-9
    # bare_rate must be found-but-unquantified over all queries.
    assert abs(rep.bare_rate - (3 - 2) / 4) < 1e-9, rep.bare_rate
    print(f"  ok  readiness={rep.readiness:.2f} quantified={rep.quantification_rate:.2f} "
          f"bare={rep.bare_rate:.2f}")


def test_compare_flags_direction_per_metric() -> None:
    base = aggregate_readiness([assess_readiness([_p(BARE)], {"D1"})])
    cand = aggregate_readiness([assess_readiness([_p(WITH_RATE), _p(WITH_POP)], {"D1"})])
    out = compare_readiness(base, cand)
    assert out["readiness"]["better"] is True, out["readiness"]
    assert out["bare_rate"]["better"] is True, out["bare_rate"]
    assert compare_readiness(cand, base)["readiness"]["better"] is False
    print("  ok  improvement direction correct per metric and reverses when swapped")


def test_empty_aggregate_is_zero() -> None:
    rep = aggregate_readiness([])
    assert rep.readiness == 0.0 and rep.bare_rate == 0.0
    assert rep.summary()["n_queries"] == 0
    print("  ok  empty input yields zeros")


def main() -> int:
    tests = [
        test_rate_detector_fires_on_quantified_evidence,
        test_rate_detector_ignores_unquantified_claims,
        test_population_detector_fires_on_denominator_language,
        test_caveat_detector_fires_on_limitation_language,
        test_section_label_alone_does_not_constitute_readiness,
        test_results_chunk_can_constitute_readiness,
        test_bare_finding_is_not_quantified,
        test_rate_without_population_is_quantified_but_not_ready,
        test_full_set_is_ready,
        test_non_gold_passages_do_not_count,
        test_nothing_retrieved_is_neither_found_nor_ready,
        test_rates_are_consistent_and_bare_complements_found,
        test_compare_flags_direction_per_metric,
        test_empty_aggregate_is_zero,
    ]
    print("Answer-readiness metric tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())