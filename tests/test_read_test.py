"""Tests for the end-to-end read-test harness.

The point of this module is to break the loop between what retrieval returns and
what a reader is actually told. Two design choices are load-bearing and pinned
here: the SAME prompt is used for both arms (varying it would confound a retrieval
comparison with a prompt comparison), and generation failures score zero rather
than being dropped (dropping them would silently improve whichever arm failed
more often).
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from medrag.eval.read_test import (
    AnswerReport,
    aggregate_answers,
    build_prompt,
    paired_delta,
    score_answer,
)

PASSAGES = [
    (0, "Major bleeding within 12 months occurred in 4.7% of the randomised patients "
        "in the dual group versus 2.1% in single therapy (hazard ratio 2.31)."),
    (1, "The trial was not powered for rare events and should be interpreted with caution."),
]


def test_prompt_is_identical_for_both_arms() -> None:
    """Only the passages differ, so any difference in the answer is retrieval's."""
    a = build_prompt("What was the bleeding rate?", PASSAGES)
    b = build_prompt("What was the bleeding rate?", PASSAGES[:1])
    # The instruction block is byte-identical.
    for line in ("Answer the question using ONLY", "Include, when the passages support it",
                 "do not infer it"):
        assert line in a and line in b
    assert a.split("Question:")[0] == b.split("Question:")[0]
    print("  ok  prompt template is identical across arms")


def test_passages_are_numbered_and_cited() -> None:
    p = build_prompt("Q?", PASSAGES)
    assert "[1]" in p and "[2]" in p, p[:200]
    assert "Major bleeding" in p
    print("  ok  passages are numbered so an answer can cite them")


def test_one_long_passage_cannot_starve_the_others() -> None:
    """Per-passage truncation, so a verbose passage cannot remove the rest.

    Without this the context-expanded arm could look worse purely because a long
    passage consumed the character budget.
    """
    long_text = "x" * 40000
    many = [(i, long_text) for i in range(10)]
    p = build_prompt("Q?", many)
    numbered = p.count("[1]")
    for i in range(1, 11):
        assert f"[{i}]" in p, f"passage {i} was truncated away entirely"
    assert numbered == 1
    print(f"  ok  all 10 long passages survive truncation ({len(p)} chars)")


def test_empty_passages_are_handled() -> None:
    p = build_prompt("Q?", [])
    assert "no passages retrieved" in p
    print("  ok  an empty retrieval still produces a valid prompt")


def test_a_declined_limitation_is_not_counted_as_a_caveat() -> None:
    """The decline sentence contains the word "limitation", so the caveat detector
    fires on it.

    Left uncorrected, any arm whose prompt invites declines gets a free win on the
    caveat rate -- the metric the experiment turns on. Measured: before the fix,
    "The passages state no limitation." scored has_caveat=True.
    """
    v = score_answer(
        "Rate was 4.7% in 1600 patients. The passages state no limitation.",
        qid="q", arm="required")
    assert v.declined, v
    assert not v.has_caveat, "a correct decline was scored as a caveat"
    real = score_answer(
        "Rate was 4.7% in 1600 patients. The trial was not powered for rare events.",
        qid="q", arm="required")
    assert real.has_caveat and not real.declined, real
    print("  ok  declines are excluded from the caveat rate")


def test_generator_variants_allow_a_correct_decline() -> None:
    """Forcing an unconditional caveat would manufacture unsupported hedging."""
    from medrag.eval.read_test import build_pass2_prompt, build_prompt

    p = [(0, "Bleeding was 4.7% in 1600 patients. The trial was not powered.")]
    for built in (build_prompt("Q?", p, required=True), build_pass2_prompt("Q?", p)):
        assert "state no limitation" in built, built[-160:]
    print("  ok  both generator variants permit declining")


def test_generate_requests_thinking_disabled() -> None:
    """A reasoning model with thinking on returns EMPTY content at a tight budget.

    Measured on the local Gemma 4 build, one prompt:

        default                     44 s, 1945 ch reasoning, 135 ch content
        enable_thinking=False        5 s,    0 ch reasoning, 266 ch content

    Thinking-on cost 9x the wall clock and, at a 700-token budget, produced empty
    content for 97.5% of queries. Scoring empty answers would have reported 0.0
    readiness for BOTH arms and looked like a catastrophic failure of the retrieval
    change rather than a harness bug. The request must therefore pin the flag.
    """
    import inspect

    from medrag.eval import read_test

    sig = inspect.signature(read_test.generate)
    assert sig.parameters["enable_thinking"].default is False, (
        "generate() must default to thinking disabled or the harness silently "
        "measures empty answers"
    )
    src = inspect.getsource(read_test.generate)
    assert '"enable_thinking"' in src, "the flag is never sent to the endpoint"
    print("  ok  generation disables reasoning by default")


def test_score_answer_detects_the_ingredients() -> None:
    good = ("Bleeding occurred in 4.7% of 1600 randomised patients in the dual group. "
            "The trial was not powered for rare events.")
    v = score_answer(good, qid="q", arm="a")
    assert v.produced and v.has_rate and v.has_population and v.has_caveat and v.ready
    print("  ok  a fully qualified answer is scored ready")


def test_score_answer_on_a_bare_answer() -> None:
    v = score_answer("Bleeding was more common in one group.", qid="q", arm="a")
    assert v.produced
    assert not v.has_rate and not v.ready
    print("  ok  a bare answer is produced but not ready")


def test_empty_generation_scores_zero_not_dropped() -> None:
    """A generation failure must count as produced-nothing.

    Dropping failures would silently favour whichever arm timed out more often,
    which is exactly the bias an end-to-end test cannot afford.
    """
    v = score_answer("", qid="q", arm="a")
    assert not v.produced and not v.has_rate and not v.ready
    rep = aggregate_answers([v, v])
    assert rep.n == 2 and rep.production == 0.0 and rep.readiness == 0.0
    print("  ok  empty generation scores zero and is retained")


def test_paired_delta_counts_flips_not_mean_differences() -> None:
    """Pairing removes query difficulty as a confound."""
    base = aggregate_answers([
        score_answer("A rate of 5% was seen.", qid="q1", arm="s"),
        score_answer("No numbers here.", qid="q2", arm="s"),
        score_answer("A rate of 3% was seen.", qid="q3", arm="s"),
    ])
    cand = aggregate_answers([
        score_answer("A rate of 5% was seen.", qid="q1", arm="c"),
        score_answer("A rate of 9% was seen.", qid="q2", arm="c"),
        score_answer("Still nothing numeric.", qid="q3", arm="c"),
    ])
    pd = paired_delta(base, cand)
    assert pd["n_paired"] == 3, pd
    # q2 gained a rate, q3 lost one. Paired delta is (1-1)/3, not the larger
    # difference of means that would result from ignoring direction.
    assert pd["has_rate"]["improved_queries"] == 1, pd["has_rate"]
    assert pd["has_rate"]["regressed_queries"] == 1, pd["has_rate"]
    assert pd["has_rate"]["paired_delta"] == 0.0, pd["has_rate"]
    assert pd["has_rate"]["better"] is False, pd["has_rate"]
    print("  ok  paired delta nets opposite flips instead of averaging them away")


def test_paired_delta_flags_a_real_improvement() -> None:
    base = aggregate_answers([score_answer("Nothing numeric.", qid=f"q{i}", arm="s") for i in range(4)])
    cand = aggregate_answers([score_answer("Rate was 5% of patients.", qid=f"q{i}", arm="c") for i in range(4)])
    pd = paired_delta(base, cand)
    assert pd["has_rate"]["improved_queries"] == 4
    assert pd["has_rate"]["regressed_queries"] == 0
    assert pd["has_rate"]["better"] is True, pd["has_rate"]
    print("  ok  a uniform improvement is reported as better")


def test_paired_delta_with_no_overlap() -> None:
    base = aggregate_answers([score_answer("x", qid="a", arm="s")])
    cand = aggregate_answers([score_answer("x", qid="b", arm="c")])
    pd = paired_delta(base, cand)
    assert pd["n_paired"] == 0, pd
    print("  ok  disjoint query sets pair to zero, not a crash")


def test_a_trickling_server_cannot_outlive_the_deadline() -> None:
    """A wall-clock deadline, not just urllib's socket timeout.

    `urlopen(timeout=...)` bounds each socket operation. A server that keeps the
    connection open and dribbles out a byte every few seconds resets that timer
    forever, so the call never returns and no timeout ever fires.

    This is not hypothetical: a serial 120-query run reached 80/360 generations and
    stopped dead for over 14 minutes, client at 0% CPU, llama-server pinned at 100%
    while reporting `is_processing: false` -- a request the server thought it had
    finished. One such request cost the whole run, because nothing was watching the
    clock.

    The fix is a daemon thread plus `join(timeout)`. Deliberately not a
    ThreadPoolExecutor: used as a context manager it calls `shutdown(wait=True)`,
    which blocks until the wedged worker returns, so the deadline fires and the
    process hangs anyway.
    """
    import socket
    import threading
    import time

    from medrag.eval.read_test import generate

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def trickle() -> None:
        try:
            conn, _ = srv.accept()
            with conn:
                conn.recv(65536)            # read the request, answer nothing
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n")
                # Never finish the body, but never go silent either -- a byte every
                # 0.2s keeps both ends' socket timers permanently fresh.
                while True:
                    conn.sendall(b" ")
                    time.sleep(0.2)
        except OSError:
            pass

    t = threading.Thread(target=trickle, daemon=True)
    t.start()
    t0 = time.time()
    try:
        out = generate("hi", endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
                       timeout=3.0)
    finally:
        srv.close()
    elapsed = time.time() - t0

    assert out == "", f"a truncated body must score as no answer, got {out!r}"
    assert elapsed < 12.0, (
        f"the deadline did not fire: generate() blocked {elapsed:.1f}s against a "
        f"3.0s deadline while the server trickled bytes"
    )
    print(f"  ok  a trickling server is cut off at the deadline ({elapsed:.1f}s / 3.0s)")


def main() -> int:
    tests = [
        test_prompt_is_identical_for_both_arms,
        test_passages_are_numbered_and_cited,
        test_one_long_passage_cannot_starve_the_others,
        test_empty_passages_are_handled,
        test_a_declined_limitation_is_not_counted_as_a_caveat,
        test_generator_variants_allow_a_correct_decline,
        test_generate_requests_thinking_disabled,
        test_score_answer_detects_the_ingredients,
        test_score_answer_on_a_bare_answer,
        test_empty_generation_scores_zero_not_dropped,
        test_paired_delta_counts_flips_not_mean_differences,
        test_paired_delta_flags_a_real_improvement,
        test_paired_delta_with_no_overlap,
        test_a_trickling_server_cannot_outlive_the_deadline,
    ]
    print("Read-test harness tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
