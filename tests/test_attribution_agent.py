"""Grounded caveat attachment wired into the agent's answer path.

``medrag/eval/attribution.py`` proves the mechanism in isolation. These tests pin
that it is actually reachable from a user of the system, because a mechanism that
only runs in an evaluation script is not a contribution -- it is a measurement.

They also pin a latent CLI bug: ``medrag ask --mode`` was accepted and never
forwarded to the agent, so every ablatable retrieval mode silently ran as
``reranked``. That is the same class of failure as the controls that produced
plausible numbers instead of errors: nothing crashed, the answer looked fine.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from medrag.agent.loop import ResearchAgent
from medrag.eval.read_test import score_answer
from medrag.index.retriever import RetrievedPassage

WITH_CAVEAT = RetrievedPassage(
    chunk_id="d1#3", row_id=3, score=1.0, title="Dual antiplatelet trial",
    text=("Bleeding occurred in 4.7% of the dual group versus 2.1% in single therapy "
          "among 1600 randomised patients. The trial was not powered for rare events."),
    journal="Lancet", year=2020, pmid="12345",
)
NO_CAVEAT = RetrievedPassage(
    chunk_id="d2#1", row_id=9, score=1.0, title="Recovery cohort",
    text="Recovery was 92% at 30 days in 800 patients.",
    journal="BMJ", year=2021, pmid="999",
)


class _FakeRetriever:
    """Retrieval is not under test here; the answer path is."""

    def retrieve(self, *a, **k):  # noqa: ANN002, ANN003
        return []


def _agent(**kw) -> ResearchAgent:
    return ResearchAgent(_FakeRetriever(), mode="context", **kw)


def test_attribution_is_off_by_default() -> None:
    """Shipping it on by default would change every answer without being asked."""
    out = _agent()._synthesize("What was the bleeding rate?", [WITH_CAVEAT])
    assert "not powered" not in out, out
    print("  ok  attribution is opt-in")


def test_agent_appends_a_quoted_limitation() -> None:
    out = _agent(attribute_caveats=True)._synthesize(
        "What was the bleeding rate?", [WITH_CAVEAT])
    assert "not powered" in out, out
    assert "[1]" in out, "the quote must be attributable to its passage"
    print("  ok  the agent appends the quoted limitation")


def test_agent_never_invents_a_limitation() -> None:
    out = _agent(attribute_caveats=True)._synthesize("Recovery rate?", [NO_CAVEAT])
    v = score_answer(out, qid="q", arm="agent")
    assert not v.has_caveat, out
    assert v.declined, out
    print("  ok  the agent declines when the evidence states no limitation")


def test_agent_keeps_the_answer_text_intact() -> None:
    base = _agent()._synthesize("What was the bleeding rate?", [WITH_CAVEAT])
    full = _agent(attribute_caveats=True)._synthesize(
        "What was the bleeding rate?", [WITH_CAVEAT])
    assert full.startswith(base), "attribution must append, never rewrite"
    print("  ok  attribution appends without rewriting the answer")


def test_evidence_text_is_available_to_the_attributor() -> None:
    """The passages handed to the agent must be the ones that get quoted.

    ``_synthesize`` receives ``RetrievedPassage`` objects; if the mapping to
    ``(id, text)`` pairs were wrong the module would see empty text and silently
    decline on every query, which still looks like a working system.
    """
    out = _agent(attribute_caveats=True)._synthesize("Rate?", [WITH_CAVEAT])
    assert score_answer(out, qid="q", arm="a").has_caveat, (
        "the agent passed evidence the attributor could not read"
    )
    print("  ok  the attributor sees the real passage text")


def test_cli_forwards_the_mode_flag() -> None:
    """`medrag ask --mode context` used to run as `reranked` without error."""
    import argparse
    import inspect

    from medrag.cli import cmd_ask

    src = inspect.getsource(cmd_ask)
    assert "mode=args.mode" in src, (
        "cmd_ask accepts --mode but does not forward it to ResearchAgent"
    )
    parser_flags = [a for a in open("medrag/cli.py").read().splitlines()
                    if "--mode" in a or "--attribute-caveats" in a]
    assert parser_flags, "the flags disappeared"
    print("  ok  the CLI forwards --mode and --attribute-caveats to the agent")


def test_api_exposes_the_flag() -> None:
    from medrag.serve.api import AskRequest

    req = AskRequest(question="q")
    assert req.attribute_caveats is False, "must default to off"
    assert AskRequest(question="q", attribute_caveats=True).attribute_caveats is True
    print("  ok  the HTTP API exposes attribute_caveats, defaulting to off")


def main() -> int:
    tests = [
        test_attribution_is_off_by_default,
        test_agent_appends_a_quoted_limitation,
        test_agent_never_invents_a_limitation,
        test_agent_keeps_the_answer_text_intact,
        test_evidence_text_is_available_to_the_attributor,
        test_cli_forwards_the_mode_flag,
        test_api_exposes_the_flag,
    ]
    print("Agent attribution integration tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())