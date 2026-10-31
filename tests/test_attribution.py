"""Tests for grounded caveat attachment.

The design contract this module claims is that it cannot fabricate: the caveat rate
of its output is bounded by the caveat rate of the retrieved passages, because the
limitations are located in the passages rather than generated. These tests pin that
contract, including the half of it that is inconvenient -- the mechanism also cannot
generalise a qualification beyond what the source literally said, so a high
conditional rate is not by itself evidence of a safe answer.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from medrag.eval.attribution import (
    AttributedCaveat,
    caveat_sentences,
    compose_attributed_answer,
    extract_caveats,
    split_sentences,
)
from medrag.eval.read_test import score_answer

RESULTS = (
    "Bleeding occurred in 4.7% of the dual group versus 2.1% in single therapy. "
    "The hazard ratio was 2.31 with a 95% confidence interval of 1.44 to 3.71. "
    "No other adverse events differed significantly between groups."
)
METHODS_WITH_LIMIT = (
    "We randomised 1600 patients 1:1. "
    "The trial was not powered for rare events. "
    "Follow-up was limited to 12 months."
)
METHODS_NO_LIMIT = "We randomised 1600 patients 1:1. Follow-up was 12 months."
DECLINE_TEXT = "The passages state no limitation."


def test_split_sentences_keeps_decimals_intact() -> None:
    s = split_sentences("Rate was 2.31. Sample was 1600. No difference emerged.")
    assert len(s) == 3, s
    assert s[0] == "Rate was 2.31.", s
    print("  ok  sentence splitting does not break decimals")


def test_caveat_sentences_returns_only_the_limitation() -> None:
    """A passage with one hedge among ten result sentences yields one quote."""
    got = caveat_sentences(METHODS_WITH_LIMIT)
    assert len(got) == 1, got
    assert "not powered" in got[0], got
    print("  ok  only the limiting sentence is quoted, not the whole passage")


def test_no_caveats_found_yields_nothing() -> None:
    assert caveat_sentences(METHODS_NO_LIMIT) == []
    assert caveat_sentences("") == []
    assert caveat_sentences(RESULTS) == []
    print("  ok  clean passages produce no caveats")


def test_extract_prefers_the_higher_ranked_passage() -> None:
    """The caveat nearest the top of the ranking is the one a reader most likely needs."""
    passages = [
        (7, METHODS_NO_LIMIT),
        (9, METHODS_WITH_LIMIT),
    ]
    got = extract_caveats(passages)
    assert len(got) == 1, got
    assert got[0].rank == 1 and got[0].source_row == 9, got
    print("  ok  caveats are attributed to their source passage and rank")


def test_extract_deduplicates_a_restated_caveat() -> None:
    """The same hedge in Methods and Discussion is one piece of evidence, not two."""
    passages = [(1, METHODS_WITH_LIMIT), (2, METHODS_WITH_LIMIT)]
    assert len(extract_caveats(passages, max_caveats=5)) == 1
    print("  ok  a restated caveat is not counted twice")


def test_extract_respects_max_caveats() -> None:
    many = [(i, "Further studies are needed. The study was not powered. "
                "Generalizability is limited.") for i in range(4)]
    assert len(extract_caveats(many, max_caveats=2)) == 2
    print("  ok  max_caveats bounds the attachment")


def test_inline_citations_are_stripped_from_quotes() -> None:
    got = caveat_sentences("The trial was not powered for rare events [12,13].")
    assert got and "[12" not in got[0], got
    print("  ok  reference markers do not pollute a quoted caveat")


def test_attribution_cannot_fabricate() -> None:
    """THE contract: passages with no caveat produce no caveat in the output.

    This is the failure the two generative interventions made. Two-pass produced a
    caveat in 4.55% of answers whose passages contained none.
    """
    passages = [(1, METHODS_NO_LIMIT), (2, RESULTS)]
    out = compose_attributed_answer("Bleeding was 4.7% in the dual group.", passages)
    v = score_answer(out, qid="q", arm="attributed")
    assert not v.has_caveat, f"fabricated a caveat: {out}"
    print("  ok  no caveat in the passages means no caveat in the output")


def test_attribution_always_states_the_caveat_it_found() -> None:
    """Conditional rate should be 1.0: if the passage hedges, the reader is told."""
    passages = [(1, METHODS_WITH_LIMIT), (2, RESULTS)]
    out = compose_attributed_answer("Bleeding was 4.7%.", passages)
    v = score_answer(out, qid="q", arm="attributed")
    assert v.has_caveat, out
    assert "not powered" in out, out
    assert "[2]" in out or "[1]" in out, "the quote must stay attributable"
    print("  ok  a caveat in the passages always reaches the reader, attributed")


def test_decline_branch_explains_itself() -> None:
    """Silence must be distinguishable from omission."""
    out = compose_attributed_answer("Bleeding was 4.7%.", [(1, METHODS_NO_LIMIT)])
    assert "no limitation" in out.lower(), out
    assert "property of this evidence" in out, out
    v = score_answer(out, qid="q", arm="attributed")
    assert v.declined and not v.has_caveat, v
    print("  ok  the decline branch is explicit and is not scored as a caveat")


def test_output_rate_is_bounded_by_input_rate() -> None:
    """The bound the module exists to guarantee, checked across mixed passages."""
    combos = [
        ([(1, METHODS_NO_LIMIT), (2, RESULTS)], False),
        ([(1, METHODS_WITH_LIMIT)], True),
        ([(1, RESULTS), (2, METHODS_WITH_LIMIT)], True),
    ]
    for passages, expect in combos:
        out = compose_attributed_answer("A rate of 4.7% was reported.", passages)
        got = score_answer(out, qid="q", arm="attributed").has_caveat
        assert got == expect, (passages, out, got, expect)
    print("  ok  output caveat rate never exceeds the retrieved caveat rate")


def test_attributed_caveat_renders_with_its_source() -> None:
    c = AttributedCaveat(text="Not powered for rare events.", source_row=4, rank=2)
    assert c.render() == '"Not powered for rare events." (passage [3])', c.render()
    print("  ok  quotes render with an explicit passage reference")


def test_an_existing_answer_is_preserved_verbatim() -> None:
    """Attaching must not rewrite the answer -- only append to it."""
    ans = "Bleeding was 4.7% in the dual group versus 2.1% in single therapy."
    out = compose_attributed_answer(ans, [(1, METHODS_WITH_LIMIT)])
    assert out.startswith(ans), out
    print("  ok  the model's answer is kept intact and limitations are appended")


def main() -> int:
    tests = [
        test_split_sentences_keeps_decimals_intact,
        test_caveat_sentences_returns_only_the_limitation,
        test_no_caveats_found_yields_nothing,
        test_extract_prefers_the_higher_ranked_passage,
        test_extract_deduplicates_a_restated_caveat,
        test_extract_respects_max_caveats,
        test_inline_citations_are_stripped_from_quotes,
        test_attribution_cannot_fabricate,
        test_attribution_always_states_the_caveat_it_found,
        test_decline_branch_explains_itself,
        test_output_rate_is_bounded_by_input_rate,
        test_attributed_caveat_renders_with_its_source,
        test_an_existing_answer_is_preserved_verbatim,
    ]
    print("Grounded attribution tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())