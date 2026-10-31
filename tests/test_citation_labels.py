"""Tests for bpref and for citation-derived relevance labels.

Two things are being guarded here.

First, bpref is the metric that makes citation-derived labels usable at all.
Buckley & Voorhees (2004) showed average precision, P@k and R-precision all shift
system rankings when judgments are incomplete, and that bpref does not. Getting
its formula wrong would silently reintroduce exactly the instability it was
invented to avoid, and it would look like a plausible number.

Second, the label construction is tested against a trap it is easy to fall into:
a seed that cites nothing must not count as evidence, and one-off citations must
not define relevance. The tests pin both, plus the era bias that makes these
labels systematically denser on recent literature.
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, ".")

from medrag.eval.citation_labels import (
    SEED_PUBLICATION_TYPES,
    CitationSeed,
    build_citation_labels,
    is_seed_document,
    label_agreement,
    select_seeds,
)
from medrag.eval.metrics import bpref


class _T:
    def __init__(self, descriptor: str) -> None:
        self.descriptor = descriptor


class _Doc:
    def __init__(self, doc_id, pmid, year, types, citations, mesh):
        self.doc_id, self.pmid, self.year = doc_id, pmid, year
        self.publication_types = tuple(types)
        self.citations = tuple(citations)
        self.mesh_terms = [_T(m) for m in mesh]


def seed(doc_id, mesh, citations, year=2022, ptypes=("Systematic Review",)):
    return CitationSeed(doc_id=doc_id, pmid=doc_id, year=year, mesh=frozenset(mesh),
                        citations=tuple(citations))


# ------------------------------------------------------------------- bpref


def test_bpref_perfect_ordering_is_one() -> None:
    assert bpref(["r1", "r2", "x1"], {"r1", "r2"}) == 1.0
    print("  ok  all relevant above all non-relevant -> bpref 1.0")


def test_bpref_worst_ordering_is_zero() -> None:
    assert bpref(["x1", "x2", "r1"], {"r1"}) == 0.0
    print("  ok  all non-relevant above -> bpref 0.0")


def test_bpref_partial_ordering() -> None:
    # r1 above x1, r2 below x1 -> (1.0 + 0.0)/2 = 0.5
    v = bpref(["r1", "x1", "r2"], {"r1", "r2"})
    assert abs(v - 0.5) < 1e-9, v
    print(f"  ok  interleaved ordering -> bpref {v:.2f}")


def test_bpref_ignores_unjudged_tail() -> None:
    """The property that makes it usable with incomplete judgments.

    A relevant document ranks above an *irrelevant* one, not above an *unknown*
    one. Passing the judged pool is what enforces that; without it, every
    unjudged document is treated as irrelevant and bpref degenerates into a
    precision-like measure with none of the robustness that motivated it.
    """
    judged = {"x1"}
    base = ["r1", "x1", "r2"]
    with_tail = base + ["unjudged1", "unjudged2", "unjudged3"]
    a = bpref(base, {"r1", "r2"}, judged=judged)
    b = bpref(with_tail, {"r1", "r2"}, judged=judged)
    assert a == b, f"unjudged tail changed bpref: {a} vs {b}"
    # r1 sits above the judged-irrelevant x1 and earns full credit; r2 sits
    # below it and earns none. Mean 0.5. The unjudged tail is irrelevant to both.
    assert abs(a - 0.5) < 1e-9, a
    # Without the pool, the three unknowns are all counted as irrelevant, so
    # r2's credit drops from 0.0 to 1 - 1/4 = 0.75 and the mean moves to 0.875.
    # Same ranking, materially different score: this is exactly the instability
    # bpref was designed to avoid being subject to.
    without_pool = bpref(with_tail, {"r1", "r2"})
    assert abs(without_pool - 0.875) < 1e-9, without_pool
    print(f"  ok  unjudged tail inert: {a:.3f} with pool vs {without_pool:.3f} without")


def test_bpref_without_judged_pool_counts_everything_as_irrelevant() -> None:
    """Document the assumption, so the default is never a silent trap.

    Omitting ``judged`` assumes a fully-judged pool, which is right for TREC and
    wrong for citation-derived labels. Both behaviours are pinned so the choice at
    each call site is a deliberate one.
    """
    ranked = ["r1", "unknown1", "r2", "known_irrelevant"]
    judged = {"known_irrelevant"}
    assumed_full = bpref(ranked, {"r1", "r2"})              # pool = everything retrieved
    with_pool = bpref(ranked, {"r1", "r2"}, judged=judged)  # pool = what was assessed
    # With the pool: 'unknown1' is neutral, so both relevant docs outrank the one
    # judged-irrelevant document -> 1.0.
    # Without: 'unknown1' counts as irrelevant and is ranked above r2, halving
    # r2's credit -> 0.75.
    assert with_pool == 1.0, with_pool
    assert abs(assumed_full - 0.75) < 1e-9, assumed_full
    assert assumed_full < with_pool, (assumed_full, with_pool)
    print(f"  ok  full-pool assumption is explicit ({assumed_full:.2f} vs {with_pool:.2f})")


def test_bpref_reports_zero_when_no_evidence_of_ordering() -> None:
    """No judged-irrelevant document retrieved means no preference relation.

    Reporting 1.0 there would reward a system for retrieving only things it likes,
    which is the opposite of what the metric is for.
    """
    v = bpref(["r1", "r2", "u1"], {"r1", "r2"}, judged={"known_irrelevant"})
    assert v == 0.0, v
    print("  ok  no judged-irrelevant retrieved -> 0.0 (no evidence), not 1.0")


def test_bpref_tolerates_missing_relevant_documents() -> None:
    """Contrast with mean average precision, which drops when gold is missing."""
    ranked = ["r1", "r2", "x1"]
    judged = {"x1"}
    gold_present = {"r1", "r2"}
    gold_missing = {"r1", "r2", "r_absent", "r_absent2"}
    a = bpref(ranked, gold_present, judged=judged)
    b = bpref(ranked, gold_missing, judged=judged)
    assert abs(a - b) < 1e-9, f"bpref changed with incomplete gold: {a} vs {b}"
    print("  ok  gold documents that were never retrieved do not penalise bpref")


def test_bpref_is_bounded_and_nan_on_no_relevant() -> None:
    for ranked, gold in [(["a", "b"], {"a"}), (["x"], {"y", "z"}), (["a", "x"], {"a"})]:
        v = bpref(ranked, gold)
        assert 0.0 <= v <= 1.0, (ranked, gold, v)
    assert math.isnan(bpref(["a"], set()))
    print("  ok  bpref in [0,1], NaN when no relevant document is judged")


def test_bpref_rewards_unjudged_relevant_over_nonrelevant() -> None:
    """A retrieved document that *might* be relevant must not be punished."""
    # 'maybe' is unjudged, so it ranks above 'x1' without counting as irrelevant.
    v = bpref(["r1", "maybe", "x1"], {"r1"}, judged={"x1"})
    assert v == 1.0, f"unjudged document counted against the system: {v}"
    print("  ok  unjudged-but-relevant-looking results are neutral")


# --------------------------------------------------------- seed selection


def test_seed_requires_trusted_type_and_a_bibliography() -> None:
    ok = _Doc("d1", "1", 2022, ["Systematic Review"], [10, 11], ["Diabetes Mellitus"])
    no_refs = _Doc("d2", "2", 2022, ["Systematic Review"], [], ["Diabetes Mellitus"])
    wrong_type = _Doc("d3", "3", 2022, ["Journal Article"], [10, 11], ["Diabetes Mellitus"])
    assert is_seed_document(ok)
    assert not is_seed_document(no_refs), "a seed with no bibliography casts no votes"
    assert not is_seed_document(wrong_type), "untrusted publication type used as seed"
    assert "Systematic Review" in SEED_PUBLICATION_TYPES
    print("  ok  seeds need a trusted publication type AND a deposited bibliography")


def test_seed_requires_mesh_terms() -> None:
    d = _Doc("d1", "1", 2022, ["Meta-Analysis"], [10], [])
    seeds = select_seeds([d])
    assert seeds == [], "a seed without descriptors has no query to answer"
    print("  ok  seeds without MeSH descriptors are excluded")


# ------------------------------------------------------------ label building


def test_min_seeds_filters_one_off_citations() -> None:
    s = [
        seed("s1", ["Diabetes Mellitus"], [100, 101]),
        seed("s2", ["Diabetes Mellitus"], [100, 102]),
        seed("s3", ["Diabetes Mellitus"], [999]),
    ]
    loose = build_citation_labels(s, min_seeds=1)
    strict = build_citation_labels(s, min_seeds=2)
    assert "pmid:999" in loose.relevant("Diabetes Mellitus"), "min_seeds=1 should admit it"
    assert "pmid:999" not in strict.relevant("Diabetes Mellitus"), (
        "a document cited by a single seed must not define relevance"
    )
    assert "pmid:100" in strict.relevant("Diabetes Mellitus"), "agreed citations must survive"
    print("  ok  min_seeds filters single-seed citations and keeps corroborated ones")


def test_cited_ids_use_the_corpus_id_convention() -> None:
    """Reference lists store bare PMIDs; documents and gold labels use `pmid:<id>`.

    Getting this wrong is silent and total. Emitting bare PMIDs produced a label
    set sharing zero identifiers with the MeSH labels, so agreement measured as
    Jaccard 0.000 across all 4,466 descriptors -- which reads exactly like "the
    two label regimes disagree completely" and is in fact a string-format bug.
    """
    s = [seed("s1", ["Diabetes Mellitus"], [42826085])]
    labels = build_citation_labels(s, min_seeds=1)
    got = labels.relevant("Diabetes Mellitus")
    assert got == {"pmid:42826085"}, got
    # And it must be comparable with MeSH labels, which are `pmid:`-prefixed.
    mesh = {"Diabetes Mellitus": {"pmid:42826085"}}
    ag = label_agreement(mesh, labels.by_descriptor)
    assert ag["mean_jaccard"] == 1.0, ag
    print("  ok  cited references keyed identically to gold labels (Jaccard 1.0)")


def test_custom_id_of_is_honoured() -> None:
    s = [seed("s1", ["D"], [7])]
    labels = build_citation_labels(s, min_seeds=1, id_of=lambda ref: f"doc:{ref}")
    assert labels.relevant("D") == {"doc:7"}, labels.by_descriptor
    print("  ok  id_of overrides the mapping for corpora with different id spaces")


def test_labels_only_cover_queried_descriptors() -> None:
    s = [seed("s1", ["Diabetes Mellitus", "Asthma"], [100])]
    labels = build_citation_labels(s, min_seeds=1, restrict_descriptors={"Diabetes Mellitus"})
    assert labels.relevant("Diabetes Mellitus") == {"pmid:100"}
    assert labels.relevant("Asthma") == set(), "unqueried descriptor leaked in"
    print("  ok  restrict_descriptors bounds the work and the output")


def test_era_bias_is_visible_not_hidden() -> None:
    s = [
        seed("s1", ["T"], [1], year=2003),
        seed("s2", ["T"], [2], year=2005),
        seed("s3", ["T"], [3], year=2021),
        seed("s4", ["T"], [4], year=2023),
    ]
    labels = build_citation_labels(s, min_seeds=1)
    cov = labels.era_coverage
    assert cov.get(2000, 0) == 2 and cov.get(2020, 0) == 2, cov
    assert "seeds_by_decade" in labels.summary()
    # The bias is not corrected, only measured. That is the honest position:
    # citation coverage is era-dependent (72% of 2019-2026 RCTs vs 20% of
    # 2000-2009), so these labels are denser on recent literature by construction.
    print(f"  ok  era coverage recorded rather than hidden: {cov}")


def test_label_agreement_reports_both_recall_directions() -> None:
    mesh = {"D1": {"a", "b", "c"}, "D2": {"a"}}
    cit = {"D1": {"a", "b"}, "D2": {"z"}}
    ag = label_agreement(mesh, cit)
    assert ag["n_descriptors"] == 2
    row = next(r for r in ag["rows"] if r["descriptor"] == "D1")
    # Values are rounded to 4 places for report readability, so compare against
    # the rounded form rather than the exact fraction.
    assert row["intersection"] == 2 and row["jaccard"] == round(2 / 3, 4)
    assert row["citation_recall_of_mesh"] == round(2 / 3, 4)
    assert row["mesh_recall_of_citation"] == 1.0
    # D2 shares nothing; Jaccard must be 0, not undefined.
    row2 = next(r for r in ag["rows"] if r["descriptor"] == "D2")
    assert row2["jaccard"] == 0.0, row2
    print(f"  ok  agreement reported per descriptor (mean Jaccard {ag['mean_jaccard']})")


def test_label_agreement_with_no_overlap_is_zero_not_a_crash() -> None:
    ag = label_agreement({"D1": {"a"}}, {"D1": {"b"}})
    assert ag["mean_jaccard"] == 0.0
    assert label_agreement({}, {})["n_descriptors"] == 0
    print("  ok  disjoint or empty label sets are handled")


def main() -> int:
    tests = [
        test_bpref_perfect_ordering_is_one,
        test_bpref_worst_ordering_is_zero,
        test_bpref_partial_ordering,
        test_bpref_ignores_unjudged_tail,
        test_bpref_without_judged_pool_counts_everything_as_irrelevant,
        test_bpref_reports_zero_when_no_evidence_of_ordering,
        test_bpref_tolerates_missing_relevant_documents,
        test_bpref_is_bounded_and_nan_on_no_relevant,
        test_bpref_rewards_unjudged_relevant_over_nonrelevant,
        test_seed_requires_trusted_type_and_a_bibliography,
        test_seed_requires_mesh_terms,
        test_min_seeds_filters_one_off_citations,
        test_cited_ids_use_the_corpus_id_convention,
        test_custom_id_of_is_honoured,
        test_labels_only_cover_queried_descriptors,
        test_era_bias_is_visible_not_hidden,
        test_label_agreement_reports_both_recall_directions,
        test_label_agreement_with_no_overlap_is_zero_not_a_crash,
    ]
    print("Citation-label and bpref tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())