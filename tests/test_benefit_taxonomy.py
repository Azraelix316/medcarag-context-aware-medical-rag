"""The benefit detector: the third ingredient of the motivating paper's taxonomy.

arXiv 2502.14898 reports three failure modes, not two:

* complication rates in 4-5% of responses,
* **procedure benefits in 6-10%,**
* disputed-diagnosis caveats dropping from 100% to 56-69% on presupposing queries.

Rates and caveats both had detectors here. Benefits had none -- the word appeared only
inside the caveat patterns, in the phrase "risks and benefits must be weighed", which is
the language of declining to take a position. So the taxonomy this evaluation is
supposed to use was two-thirds implemented, and every table reported rates and caveats
without saying that a third of the taxonomy was simply absent.

The detector pairs a direction or an outcome term with what it acts on, because "reduced"
alone is not a finding. The properties pinned here are the ones that would make the
metric quietly wrong:

* **A negated finding is not a benefit.** "We did not observe a reduction in
  readmission" contains every token a benefit pattern needs and states the reverse.
  Counting it would make the metric improve on exactly the passages reporting no effect.
* **Negation attaches to the clause, not the paragraph.** A caveat early in a passage
  must not veto an unqualified result later in the same passage.
* **Weighing risks against benefits is not stating a benefit.** That sentence hedges
  completely and would otherwise register as having said what the procedure is good for.
* **It finds benefit language where it belongs.** Measured over 6,000 real PMC section
  texts: 15.2% overall, and 43.7% in background, 33.4% in discussion, 22.1% in
  conclusions, 11.8% in results, **2.8% in methods**. Methods describe what was done,
  not what was achieved, and a detector that did not respect that would be counting
  keywords rather than claims.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from medrag.eval.readiness import Readiness, aggregate_readiness, find_benefit

POSITIVE = [
    "Treatment reduced mortality compared with placebo (HR 0.71, 95% CI 0.54-0.93).",
    "Aspirin was shown to be effective in preventing recurrent infarction.",
    "Fewer major bleeding events than with dual antiplatelet therapy.",
    "The number needed to treat for one additional adverse event was 45.",
    "Surgery was superior to conservative management at 24 months.",
    "Lower blood pressure was observed in the intervention arm.",
    "Aspirin reduced the risk of stroke by 24% compared with placebo.",
]

NEGATIVE = [
    # Negation: contains every benefit token, states the opposite.
    "We did not observe a reduction in readmission.",
    "The intervention failed to reduce mortality over 5 years.",
    "There was no benefit from adding the second drug.",
    "Treatment provided no benefit over placebo.",
    "No evidence of benefit was found for either intervention.",
    "Benefits were not demonstrated in this trial.",
    # Hedges: the language of not taking a position.
    "The risks and benefits of screening must be weighed against each other.",
    # Not a benefit claim at all.
    "No significant difference in complication rates was observed.",
    "This study was limited by its small sample size.",
    "Results: mean age 62, 48% female.",
    "Methods: patients were randomised 1:1. Primary outcome was mortality.",
]


def test_benefit_detects_claims_of_effect() -> None:
    for text in POSITIVE:
        assert find_benefit(text), f"missed a benefit claim: {text}"
    print(f"  ok  detects all {len(POSITIVE)} benefit claims")


def test_a_negated_finding_is_not_a_benefit() -> None:
    """The failure mode that would invert the metric.

    Without this, a trial reporting no effect would score as having stated a benefit,
    and the metric would improve on precisely the passages that say nothing happened.
    """
    for text in NEGATIVE:
        assert not find_benefit(text), f"counted a non-benefit: {text}"
    print(f"  ok  rejects all {len(NEGATIVE)} non-benefits, including negations")


def test_negation_is_scoped_to_the_clause() -> None:
    """A caveat earlier in the passage must not veto an unqualified result later.

    Scoping negation to the whole text would undercount benefits in exactly the
    passages this project cares about, since a Results chunk that states both an effect
    and a limitation is the ordinary case rather than the exception.
    """
    text = ("The trial was limited by its small sample size. However, the intervention "
            "reduced mortality compared with placebo (HR 0.61).")
    assert find_benefit(text), (
        "a caveat earlier in the passage suppressed an unqualified result later in it")
    print("  ok  negation is clause-scoped, so a caveat does not veto a later result")


def test_weighing_is_not_stating() -> None:
    text = "The risks and benefits of screening must be weighed against each other."
    assert not find_benefit(text), (
        "hedging about benefits was counted as having stated one")
    print("  ok  'risks and benefits must be weighed' is a hedge, not a benefit")


def test_readiness_carries_the_benefit_through_to_the_report() -> None:
    """The detector is worthless if nothing aggregates it.

    Readiness gained a `has_benefit` field, a `n_with_benefit` counter and a
    `benefit_rate`. This checks the whole path, because a detector wired to nothing
    would look identical to one that finds nothing.
    """
    rows = [
        Readiness(qid="a", found_gold=True, has_finding=True, has_rate=True,
                  has_population=True, has_caveat=True, has_benefit=True),
        Readiness(qid="b", found_gold=True, has_finding=True, has_rate=True,
                  has_population=True, has_caveat=True, has_benefit=False),
        Readiness(qid="c", found_gold=False),
    ]
    rep = aggregate_readiness(rows)
    assert rep.n_with_benefit == 1, rep.n_with_benefit
    assert abs(rep.benefit_rate - 1 / 3) < 1e-9, rep.benefit_rate
    s = rep.summary()
    assert s["benefit_rate"] == round(rep.benefit_rate, 4), s
    assert s["n_with_benefit"] == 1, s
    # Per-query serialisation carries the boolean; the rate is a report-level
    # property. Asserting for the rate here would be asserting the wrong level.
    assert rows[0].to_dict()["has_benefit"] is True, rows[0].to_dict()
    assert rows[1].to_dict()["has_benefit"] is False, rows[1].to_dict()
    print("  ok  benefit flows through assess -> aggregate -> summary as a reported rate")


def test_benefit_rate_is_zero_not_absent_when_there_are_no_queries() -> None:
    """Matches the other rates: an empty report reports 0.0, not a missing key.

    A metric that vanishes when there is nothing to measure is indistinguishable from
    one that was never computed.
    """
    assert aggregate_readiness([]).benefit_rate == 0.0
    print("  ok  an empty report gives 0.0 rather than omitting the key")


def main() -> int:
    tests = [
        test_benefit_detects_claims_of_effect,
        test_a_negated_finding_is_not_a_benefit,
        test_negation_is_scoped_to_the_clause,
        test_weighing_is_not_stating,
        test_readiness_carries_the_benefit_through_to_the_report,
        test_benefit_rate_is_zero_not_absent_when_there_are_no_queries,
    ]
    print("Benefit-detector tests (motivating paper taxonomy, third ingredient)")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())