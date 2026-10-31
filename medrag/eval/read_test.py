"""End-to-end read test: does the generated ANSWER state rate, population, caveat?

The gap this closes
-------------------
``readiness.py`` measures whether the retrieved *passages* contain a rate, a
population and a caveat. That is an input-side property. The motivating paper
(arXiv 2502.14898) makes a claim about *output*: deployed systems produce answers
that mention complication rates 4-5% of the time and benefits 6-10% of the time.

Nothing so far connects the two. A generator can ignore every caveat it is handed.
This module closes that loop by running retrieval -> generation and measuring what
comes out.

Design, and why it avoids an LLM judge
-------------------------------------
The obvious protocol is LLM-as-judge: generate an answer, then ask a model whether
it is well qualified. Self-judging with the same model is badly biased -- a judge
that wrote the answer rates its own hedging generously, and there is no
independent model available here besides a second copy of the same weights.

Instead the *deterministic* detectors from ``readiness.py`` are applied to the
generated answer. Those detectors were written against clinical prose and know
nothing about which system produced the text, so they cannot be gamed by the
generator's style. The comparison is then:

    does the ANSWER state a rate more often when the passages contained one?

That is a genuine end-to-end test with no judge in the loop, and it is strictly
harder to pass than the input-side result: a passage-level ingredient only counts
if the model actually used it.

What this still does not establish
----------------------------------
Whether the stated rate is the *right* rate, and whether the caveat is the
right caveat. Generation can attach a correct-looking number to the wrong finding.
A human or a stronger independent judge would be needed for that, and remains the
one open question after this module.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..eval.readiness import find_benefit, find_caveat, find_population, find_rate

log = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "http://127.0.0.1:8081/v1/chat/completions"

# The instruction is deliberately the *same* for both retrieval arms. Varying it
# would confound a retrieval comparison with a prompt comparison.
PROMPT = """Answer the question using ONLY the numbered passages below.

Include, when the passages support it:
- what the finding is,
- the number (a percentage, ratio, or count),
- who the number applies to,
- anything that limits or qualifies the finding.

If a passage does not state something, do not infer it.

Question: {question}

Passages:
{passages}

Answer in at most 4 sentences."""

# Generator-side variant: require a limitation sentence explicitly.
#
# Motivated by the measured result that caveat transmission is 14% -- passages
# triple their caveat rate while the answer barely moves. This asks whether an
# instruction can recover caveats that are already present.
#
# The wording permits "the passages state no limitation", because forcing a
# caveat unconditionally would manufacture hedging that the evidence does not
# support. That is the failure mode this arm must be checked for.
PROMPT_REQUIRED = """Answer the question using ONLY the numbered passages below.

Include, when the passages support it:
- what the finding is,
- the number (a percentage, ratio, or count),
- who the number applies to.

Then add one final sentence on limitations. If the passages state a limitation,
sample size, or reason the finding may not generalise, state it. If the passages
state none, write exactly: "The passages state no limitation."

Do not infer anything the passages do not state.

Question: {question}

Passages:
{passages}

Answer in at most 5 sentences."""

# Second pass for the two-pass arm: ask about limitations in isolation.
PASS2_PROMPT = """Using ONLY the numbered passages below, answer in one sentence:

What limitations, caveats, or reasons not to over-interpret these findings do the
passages state?

If the passages state none, answer exactly: "The passages state no limitation."

Question: {question}

Passages:
{passages}"""

# Detectors for whether a "no limitation stated" answer was correctly declined.
NO_LIMIT = re.compile(
    r"no limitation|state no limitation|no caveat|not state any limitation|"
    r"none stated|no limitations", re.I)

NUM_RE = re.compile(r"^\s*\[\d+\]\s*", re.M)


def _format_passages(passages: Sequence[tuple[str, str]], max_chars: int) -> str:
    """Number the passages, truncating per-passage as well as in total.

    Per-passage truncation matters: with a single total budget one verbose passage
    can consume it and silently remove the others, which would make the
    context-expanded arm look worse purely for a formatting reason.
    """
    per = max(400, max_chars // max(1, len(passages)))
    lines: list[str] = []
    for i, (_row_id, text) in enumerate(passages, start=1):
        body = " ".join((text or "").split())[:per]
        if body:
            lines.append(f"[{i}] {body}")
    return "\n\n".join(lines) or "(no passages retrieved)"


def build_prompt(question: str, passages: Sequence[tuple[str, str]], *,
                 max_chars: int = 5500, required: bool = False) -> str:
    template = PROMPT_REQUIRED if required else PROMPT
    return template.format(question=question,
                           passages=_format_passages(passages, max_chars))


def build_pass2_prompt(question: str, passages: Sequence[tuple[str, str]], *,
                       max_chars: int = 5500) -> str:
    return PASS2_PROMPT.format(question=question,
                               passages=_format_passages(passages, max_chars))


def generate(
    prompt: str,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = "local",
    timeout: float = 300.0,
    max_tokens: int = 700,
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> str:
    """Call an OpenAI-compatible chat endpoint. Returns '' on failure.

    ``enable_thinking=False`` is essential and not an optimisation. The local
    Gemma 4 build is a reasoning model: with thinking on, a single answer consumed
    1,945 characters of ``reasoning_content`` and at a 700-token budget returned
    **empty** ``content`` for 97.5% of queries. Measured both ways on one prompt:

        default                     44 s, 1945 ch reasoning, 135 ch content
        enable_thinking=False        5 s,    0 ch reasoning, 266 ch content

    So thinking-on did not merely cost 9x the time, it destroyed the measurement:
    scoring empty answers would have reported a 0.0 readiness for both arms and
    looked like a catastrophic failure of the retrieval change rather than a
    harness bug.

    Failure returns empty rather than raising: a single timeout must not abort a
    120-query run, and an empty answer scores zero on every dimension, which is the
    correct accounting for "the system produced nothing".
    """
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }).encode()
    req = urllib.request.Request(
        endpoint, data=payload, headers={"Content-Type": "application/json"}
    )

    def _post() -> str:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
        msg = body["choices"][0]["message"]
        text = (msg.get("content") or "").strip()
        if not text:
            log.warning(
                "empty content (%d ch reasoning) -- check max_tokens/thinking",
                len((msg.get("reasoning_content") or "").strip()),
            )
        return text

    try:
        # A wall-clock deadline, not just urllib's socket timeout.
        #
        # `urlopen(timeout=300)` bounds each individual socket operation, not the
        # request. A server that keeps the connection open and trickles bytes resets
        # that timer indefinitely, so a request can hang forever and the timeout
        # never fires. Measured: a serial 120-query run reached 80/360 generations
        # and then stopped dead for 14+ minutes with zero CPU on the client and the
        # server pinned at 100% CPU reporting `is_processing: false` -- a request the
        # server believed it had finished. The whole run was lost to one such request,
        # because nothing was watching the clock.
        #
        # A plain daemon thread, deliberately not a ThreadPoolExecutor. An executor
        # used as a context manager calls shutdown(wait=True) on exit, which blocks
        # until the wedged worker returns -- so the deadline fires and the process
        # hangs anyway, which is exactly what the first version of this fix did. A
        # daemon thread is abandoned instead, and does not hold up interpreter exit.
        box: list[Any] = []

        def _run() -> None:
            try:
                box.append(_post())
            except Exception as exc:            # noqa: BLE001 - re-raised below
                box.append(exc)

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            log.warning("generation exceeded the %.0fs deadline; abandoned", timeout)
            return ""
        if not box:
            log.warning("generation produced no result")
            return ""
        if isinstance(box[0], Exception):
            raise box[0]
        return str(box[0])
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, OSError) as exc:
        log.warning("generation failed: %s", exc)
        return ""


@dataclass
class AnswerVerdict:
    """What one generated answer actually stated."""

    qid: str
    arm: str
    produced: bool = False
    has_rate: bool = False
    has_population: bool = False
    has_caveat: bool = False
    #: The benefit side of the motivating paper's taxonomy, which measured benefits in
    #: 6-10% of responses. Rates and caveats were both detected on the answer side;
    #: benefits were not, so the answer side was also two-thirds of the taxonomy.
    has_benefit: bool = False
    #: The model explicitly said the passages contain no limitation. A correct
    #: decline, and NOT a caveat -- counting it as one would inflate the metric.
    declined: bool = False
    ready: bool = False
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid, "arm": self.arm, "produced": self.produced,
            "has_rate": self.has_rate, "has_population": self.has_population,
            "has_caveat": self.has_caveat, "has_benefit": self.has_benefit,
            "declined": self.declined, "ready": self.ready,
        }


def score_answer(text: str, *, qid: str = "", arm: str = "") -> AnswerVerdict:
    """Apply the same detectors used on passages, to the generated answer."""
    body = NUM_RE.sub("", text or "")
    v = AnswerVerdict(qid=qid, arm=arm, produced=bool(body.strip()), text=body)
    v.has_rate = find_rate(body)
    v.has_population = find_population(body)
    v.has_caveat = find_caveat(body)
    v.has_benefit = find_benefit(body)
    # An explicit "the passages state no limitation" is a correct decline, not a
    # caveat -- and the decline sentence itself contains the word "limitation",
    # so the caveat detector fires on it. Counting that as a caveat would inflate
    # the very metric under test, and would hand a free win to any arm whose
    # prompt invites declines.
    v.declined = bool(NO_LIMIT.search(body))
    if v.declined:
        v.has_caveat = False
        v.has_benefit = False
    v.ready = v.has_rate and v.has_population
    return v


@dataclass
class AnswerReport:
    n: int = 0
    n_produced: int = 0
    n_rate: int = 0
    n_population: int = 0
    n_caveat: int = 0
    n_benefit: int = 0
    n_declined: int = 0
    n_ready: int = 0
    verdicts: list[AnswerVerdict] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.n_rate / self.n if self.n else 0.0

    @property
    def population(self) -> float:
        return self.n_population / self.n if self.n else 0.0

    @property
    def caveat(self) -> float:
        return self.n_caveat / self.n if self.n else 0.0

    @property
    def benefit(self) -> float:
        """Share of answers stating what the intervention is good for.

        The third ingredient of the motivating paper's taxonomy (benefits in 6-10% of
        responses there), alongside complication rates and disputed-diagnosis caveats.
        """
        return self.n_benefit / self.n if self.n else 0.0

    @property
    def declined(self) -> float:
        return self.n_declined / self.n if self.n else 0.0

    @property
    def readiness(self) -> float:
        return self.n_ready / self.n if self.n else 0.0

    @property
    def production(self) -> float:
        return self.n_produced / self.n if self.n else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "produced": round(self.production, 4),
            "rate": round(self.rate, 4),
            "population": round(self.population, 4),
            "caveat": round(self.caveat, 4),
            "benefit": round(self.benefit, 4),
            "declined": round(self.declined, 4),
            "ready": round(self.readiness, 4),
        }


def aggregate_answers(verdicts: Sequence[AnswerVerdict]) -> AnswerReport:
    rep = AnswerReport()
    for v in verdicts:
        rep.n += 1
        rep.n_produced += int(v.produced)
        rep.n_rate += int(v.has_rate)
        rep.n_population += int(v.has_population)
        rep.n_caveat += int(v.has_caveat)
        rep.n_benefit += int(v.has_benefit)
        rep.n_declined += int(v.declined)
        rep.n_ready += int(v.ready)
        rep.verdicts.append(v)
    return rep


_ATTR_TO_COUNT = {
    "has_rate": "n_rate",
    "has_population": "n_population",
    "has_caveat": "n_caveat",
    "ready": "n_ready",
}


def paired_delta(baseline: AnswerReport, candidate: AnswerReport) -> dict[str, Any]:
    """Per-query paired comparison, not a difference of means.

    The same queries run through both arms, so pairing removes query difficulty as
    a confound -- with 200 queries the unpaired difference of means can move for
    reasons that have nothing to do with the arm.
    """
    base_by_qid = {v.qid: v for v in baseline.verdicts}
    cand_by_qid = {v.qid: v for v in candidate.verdicts}
    shared = sorted(set(base_by_qid) & set(cand_by_qid))
    out: dict[str, Any] = {"n_paired": len(shared)}
    if not shared:
        return out
    for attr, count_attr in _ATTR_TO_COUNT.items():
        flips_up = sum(
            1 for q in shared
            if getattr(cand_by_qid[q], attr) and not getattr(base_by_qid[q], attr)
        )
        flips_down = sum(
            1 for q in shared
            if getattr(base_by_qid[q], attr) and not getattr(cand_by_qid[q], attr)
        )
        delta = (flips_up - flips_down) / len(shared)
        out[attr] = {
            # Rate on the PAIRED queries only, so the two sides are comparable
            # when an arm produced answers for a different subset.
            "baseline": round(
                sum(1 for q in shared if getattr(base_by_qid[q], attr)) / len(shared), 4),
            "candidate": round(
                sum(1 for q in shared if getattr(cand_by_qid[q], attr)) / len(shared), 4),
            "improved_queries": flips_up,
            "regressed_queries": flips_down,
            "paired_delta": round(delta, 4),
            "better": delta > 0,
        }
    return out