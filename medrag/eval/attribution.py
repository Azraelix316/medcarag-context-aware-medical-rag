"""Grounded caveat attachment: quote the limitation instead of generating one.

Why this module exists
----------------------
The read test found that caveat transmission from retrieval to the reader is 14%.
Two generator-side fixes were then measured (``scripts/run_generator_intervention.py``)
and both failed:

* **asking for a caveat** halved the conditional caveat rate (0.0741 -> 0.0185) and
  made the model decline to state a limitation in 82.5% of answers, including when
  one was present in the passages;
* **a second pass** asking "what limits this?" raised the conditional rate 4.2x
  (0.0741 -> 0.3148) but also produced a caveat in 4.55% of answers whose passages
  contained none -- fabricated hedging, which is worse than silence because it
  trains a reader to discount every number.

Both failures are the same failure: a *generative* step decides what the limitation
is, and a generative step can produce one when there is none.

The mechanism that cannot fail this way is to not generate. Locate the sentence in
the passage that carries the limitation, and attach it as quoted text with its
source. Then the caveat rate in the output is bounded by the caveat rate in the
retrieved passages *by construction* -- a claim that is testable, and is tested here.

That bound is the design contract, and it cuts both ways:

* it cannot fabricate (``unconditional`` caveat rate must stay at 0);
* but it cannot *generalise* either. It reports the study's own hedge rather than
  transferring the qualification to the user's specific question, which is the thing
  the motivating paper is really about. So a high ``conditional`` rate with a zero
  ``unconditional`` rate is expected and is not by itself evidence of safety.

What it does buy: the reader receives the limitation, attributed, next to the number
it qualifies. That is the artefact this project set out to produce.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..eval.readiness import _CAVEAT_PATTERNS

# Sentence splitter. Deliberately simple: clinical prose in these chunks uses
# ordinary terminal punctuation, and a dependency-free splitter keeps the module
# usable on the crawl side where extra models are not installed.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\d])")

# Strip PMC/PubMed numeric citations like "[12]" or "[3,4]" out of quoted spans.
_INLINE_CITE = re.compile(r"\s*\[\d+(?:\s*[,;-]\s*\d+)*\]")

MAX_QUOTE_CHARS = 320


@dataclass(frozen=True)
class AttributedCaveat:
    """One limitation quoted out of a retrieved passage."""

    text: str
    source_row: int
    rank: int  # 0 = highest ranked supporting passage

    def render(self) -> str:
        return f'"{self.text}" (passage [{self.rank + 1}])'


def split_sentences(text: str) -> list[str]:
    """Split into sentences, dropping empties."""
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def caveat_sentences(text: str, *, max_chars: int = MAX_QUOTE_CHARS) -> list[str]:
    """Sentences in ``text`` that state a limitation.

    Filters the *sentence*, not the passage. A passage containing one limitation
    among ten sentences of results would otherwise yield all ten sentences as
    caveats, which would bury the one that matters.
    """
    out: list[str] = []
    for sent in split_sentences(text):
        if any(p.search(sent) for p in _CAVEAT_PATTERNS):
            body = _INLINE_CITE.sub("", sent).strip()
            if len(body) > max_chars:
                body = body[:max_chars].rsplit(" ", 1)[0] + "..."
            if body and body not in out:
                out.append(body)
    return out


def extract_caveats(
    passages: Sequence[tuple[int, str]],
    *,
    max_caveats: int = 2,
) -> list[AttributedCaveat]:
    """Pull grounded caveats out of ranked passages, best passage first.

    ``passages`` is ``(row_id, text)`` in retrieval rank order. Only the first
    ``max_caveats`` are returned, and duplicates across passages are dropped --
    a Methods hedge restated in Discussion is not two pieces of evidence.
    """
    found: list[AttributedCaveat] = []
    seen: set[str] = set()
    for rank, (row_id, text) in enumerate(passages):
        for sent in caveat_sentences(text):
            key = " ".join(sent.lower().split())
            if key in seen:
                continue
            seen.add(key)
            found.append(AttributedCaveat(text=sent, source_row=row_id, rank=rank))
            if len(found) >= max_caveats:
                return found
    return found


def compose_attributed_answer(
    answer: str,
    passages: Sequence[tuple[int, str]],
    *,
    max_caveats: int = 2,
) -> str:
    """Append quoted limitations to an answer, or say plainly that there are none.

    The decline branch is essential and is not a fallback. An honest "no limitation
    was stated in the retrieved passages" tells the reader that the silence is a
    property of the evidence rather than an omission by the generator -- which is
    exactly the distinction the two failed interventions could not preserve.
    """
    body = (answer or "").strip()
    caveats = extract_caveats(passages, max_caveats=max_caveats)
    if not caveats:
        return (
            f"{body}\n\nLimitations: The retrieved passages state no limitation. "
            "That is a property of this evidence, not a guarantee that none exists."
        ).strip()
    quoted = "; ".join(c.render() for c in caveats)
    return f"{body}\n\nLimitations stated in the source: {quoted}"