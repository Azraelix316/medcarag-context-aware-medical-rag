"""The research agent loop: plan, retrieve, reflect, expand, answer.

This is the layer a research agent actually calls. It wraps the retriever with
the reasoning loop that turns single-shot RAG into iterative retrieval:

1. **Plan** - decompose the question, expand vocabulary.
2. **Retrieve** - fan out over the plan, fuse and rerank.
3. **Reflect** - score the evidence for coverage of the question's content
   terms; detect gaps without needing an LLM.
4. **Expand** - issue targeted follow-up queries for the uncovered terms.
5. **Answer** - assemble a grounded answer with per-claim citations.

Every step records a trace, so a run is fully auditable. That matters for a
paper: each query's retrieval behaviour is an inspectable artifact rather than
an opaque chain of prompts.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..index.retriever import MODE_RERANKED, MedRAGRetriever, RetrievalResult, RetrievedPassage
from .planner import QueryPlan, QueryPlanner, content_tokens

log = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """One recorded action in the agent trace."""

    step: int
    action: str                 # plan | retrieve | reflect | expand | answer
    detail: str = ""
    queries: list[str] = field(default_factory=list)
    n_results: int = 0
    elapsed: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step, "action": self.action, "detail": self.detail,
            "queries": self.queries, "n_results": self.n_results,
            "elapsed_s": round(self.elapsed, 3), "payload": self.payload,
        }


@dataclass
class CoverageReport:
    """How well the retrieved evidence covers the question."""

    covered: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"score": round(self.score, 4), "covered": self.covered, "missing": self.missing}


@dataclass
class AgentAnswer:
    """A grounded answer plus its evidence and full trace."""

    question: str
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    evidence: list[RetrievedPassage] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)
    trace: list[AgentStep] = field(default_factory=list)
    coverage: CoverageReport = field(default_factory=CoverageReport)
    iterations: int = 0
    elapsed: float = 0.0
    # Set when an LLM generator is attached; otherwise the answer is extractive.
    generator: str = "extractive"

    def to_dict(self, *, include_evidence: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "plan": self.plan,
            "trace": [s.to_dict() for s in self.trace],
            "coverage": self.coverage.to_dict(),
            "iterations": self.iterations,
            "elapsed_s": round(self.elapsed, 3),
            "generator": self.generator,
        }
        if include_evidence:
            payload["evidence"] = [p.to_dict() for p in self.evidence]
        return payload


# Structured abstracts arrive with labels glued to the body text
# ("Background: Type 2 diabetes..."), which reads badly when quoted.
_LEADING_LABEL_RE = re.compile(r"^(?:background|objective|aim|aims|purpose|"
                               r"methods?|materials and methods|patients and methods|"
                               r"design|results?|findings?|conclusions?|interpretation|"
                               r"discussion|significance|importance)\s*:\s*", re.I)


def _clean_quote(sentence: str) -> str:
    """Tidy a passage sentence so it can be quoted directly.

    Strips the leading structured-abstract label, collapses whitespace, and
    removes a dangling section heading left by an upstream split.
    """
    text = " ".join(sentence.split()).strip()
    text = _LEADING_LABEL_RE.sub("", text)
    # A lone trailing heading ("...in the control group. Conclusions") adds
    # nothing once the label has been removed. Only trim the sentence-final
    # period in that case, so an ordinary full stop survives.
    stripped = re.sub(
        r"\s*\.?\s*\b(?:Background|Objective|Aim|Aims|Methods|Results|Conclusions|"
        r"Discussion|Significance|Importance)\s*:?\s*$",
        "", text,
    )
    if stripped != text:
        text = stripped.rstrip(" .")
    return text.strip()


def assess_coverage(question: str, passages: Sequence[RetrievedPassage], *, k: int = 10) -> CoverageReport:
    """Measure whether the evidence covers the question's content terms.

    Content-term coverage is a cheap, model-free proxy for answerability. It is
    not a substitute for faithfulness grading, but it reliably detects the
    common failure where retrieval returns topically related yet off-target
    passages, which is what triggers the expansion hop.
    """
    wanted = set(content_tokens(question))
    if not wanted:
        return CoverageReport(covered=[], missing=[], score=1.0)

    seen: set[str] = set()
    for passage in passages[:k]:
        seen.update(content_tokens(passage.title))
        seen.update(content_tokens(passage.text))

    covered = sorted(wanted & seen)
    missing = sorted(wanted - seen)
    return CoverageReport(covered=covered, missing=missing, score=len(covered) / len(wanted))


def _merge_results(runs: Sequence[RetrievalResult], top_k: int) -> list[RetrievedPassage]:
    """Merge multiple retrieval runs, keeping each passage's best score.

    De-duplication is by chunk_id; when the same passage is retrieved by several
    sub-queries that is itself evidence it is central, so ranks are reinforced
    by summing the reciprocal of each rank instead of taking a plain max.
    """
    acc: dict[str, RetrievedPassage] = {}
    boost: dict[str, float] = {}
    seen_count: dict[str, int] = {}
    for run in runs:
        for rank, passage in enumerate(run.passages, start=1):
            prev = acc.get(passage.chunk_id)
            if prev is None:
                acc[passage.chunk_id] = passage
                boost[passage.chunk_id] = 0.0
                seen_count[passage.chunk_id] = 1
            else:
                if passage.score > prev.score:
                    prev.score = passage.score
                seen_count[passage.chunk_id] += 1
            boost[passage.chunk_id] += 1.0 / (60 + rank)

    merged = list(acc.values())
    for passage in merged:
        # Cross-query reinforcement plus a small bonus for multi-query agreement.
        passage.score += boost[passage.chunk_id] + 0.01 * (seen_count[passage.chunk_id] - 1)
    merged.sort(key=lambda p: (-p.score, p.chunk_id))
    return merged[:top_k]


def _dedup_by_document(passages: Sequence[RetrievedPassage], *, per_doc: int = 2) -> list[RetrievedPassage]:
    """Limit how many chunks any single document can contribute.

    Without this, one long review can occupy the whole context window and crowd
    out independent sources.
    """
    counts: dict[str, int] = {}
    out: list[RetrievedPassage] = []
    for passage in passages:
        doc_id = passage.chunk_id.split("#", 1)[0]
        n = counts.get(doc_id, 0)
        if n >= per_doc:
            continue
        counts[doc_id] = n + 1
        out.append(passage)
    return out


class ResearchAgent:
    """Iterative retrieval agent over a :class:`MedRAGRetriever`."""

    def __init__(
        self,
        retriever: MedRAGRetriever,
        *,
        planner: QueryPlanner | None = None,
        max_hops: int = 2,
        top_k: int = 10,
        mode: str = MODE_RERANKED,
        per_doc_limit: int = 2,
        coverage_threshold: float = 0.6,
        generator: Callable[[str, Sequence[RetrievedPassage]], str] | None = None,
        generator_name: str = "extractive",
        attribute_caveats: bool = False,
        max_attached_caveats: int = 2,
    ) -> None:
        self.retriever = retriever
        self.planner = planner or QueryPlanner()
        self.max_hops = max_hops
        self.top_k = top_k
        self.mode = mode
        self.per_doc_limit = per_doc_limit
        self.coverage_threshold = coverage_threshold
        self.generator = generator
        self.generator_name = generator_name
        self.attribute_caveats = attribute_caveats
        self.max_attached_caveats = max_attached_caveats

    # ------------------------------------------------------------------- main
    def research(self, question: str, *, max_hops: int | None = None) -> AgentAnswer:
        """Answer ``question`` with iterative, grounded retrieval."""
        t_start = time.time()
        max_hops = self.max_hops if max_hops is None else max_hops
        answer = AgentAnswer(question=question, generator=self.generator_name)
        trace = answer.trace
        step_no = 0

        # ---- 1. plan
        t0 = time.time()
        plan: QueryPlan = self.planner.plan(question)
        answer.plan = plan.to_dict()
        step_no += 1
        trace.append(AgentStep(step_no, "plan", plan.strategy, queries=plan.texts,
                               elapsed=time.time() - t0, payload={"is_multi_hop": plan.is_multi_hop}))

        # ---- 2. initial retrieval
        t0 = time.time()
        runs: list[RetrievalResult] = []
        for sub in plan.sub_queries[: max(1, self.planner.max_subqueries)]:
            try:
                runs.append(
                    self.retriever.retrieve(
                        sub.text if sub.kind != "expansion" else f"{question} {sub.text}",
                        top_k=self.top_k * 3,
                        mode=self.mode,
                        include_text=True,
                    )
                )
            except RuntimeError as exc:
                log.warning("sub-query %r failed: %s", sub.text[:60], exc)
        passages = _merge_results(runs, self.top_k * 3)
        step_no += 1
        trace.append(AgentStep(step_no, "retrieve", f"{len(plan.sub_queries)} queries",
                               queries=plan.texts, n_results=len(passages), elapsed=time.time() - t0))

        answer.iterations = 1

        # ---- 3. reflect + expand
        for hop in range(max_hops):
            coverage = assess_coverage(question, passages)
            answer.coverage = coverage
            t0 = time.time()
            step_no += 1
            trace.append(AgentStep(step_no, "reflect", f"coverage={coverage.score:.2f}",
                                   elapsed=time.time() - t0, payload=coverage.to_dict()))
            if coverage.score >= self.coverage_threshold or not coverage.missing:
                break

            follow_up = self._followup_queries(question, coverage.missing, passages)
            if not follow_up:
                break
            t0 = time.time()
            hop_runs = [
                self.retriever.retrieve(q, top_k=self.top_k * 2, mode=self.mode, include_text=True)
                for q in follow_up
            ]
            passages = _merge_results([*runs, *hop_runs], self.top_k * 3)
            runs.extend(hop_runs)
            answer.iterations = hop + 2
            step_no += 1
            trace.append(AgentStep(step_no, "expand", f"hop {hop + 1}",
                                   queries=follow_up, n_results=len(passages),
                                   elapsed=time.time() - t0))

        # ---- 4. select and answer
        t0 = time.time()
        evidence = _dedup_by_document(passages, per_doc=self.per_doc_limit)[: self.top_k]
        answer.evidence = evidence
        answer.citations = [p.citation for p in evidence]
        answer.answer = self._synthesize(question, evidence)
        step_no += 1
        trace.append(AgentStep(step_no, "answer", f"{len(evidence)} evidence passages",
                               n_results=len(evidence), elapsed=time.time() - t0,
                               payload={"n_documents": len({p.chunk_id.split('#')[0] for p in evidence})}))

        answer.elapsed = time.time() - t_start
        return answer

    # ------------------------------------------------------------- internals
    def _followup_queries(
        self,
        question: str,
        missing: Sequence[str],
        passages: Sequence[RetrievedPassage],
    ) -> list[str]:
        """Build targeted queries for uncovered question terms.

        Combines the missing terms with the query and, when the evidence already
        establishes part of the answer, with the top passage's title so the
        follow-up search stays inside the same clinical subfield.
        """
        if not missing:
            return []
        # Two to four uncovered terms make a focused query; more is noise.
        focus = list(missing)[:4]
        out = [f"{question} {' '.join(focus)}".strip()]
        if passages:
            anchor = passages[0].title
            if anchor and len(anchor) > 10:
                out.append(f"{anchor} {' '.join(focus)}".strip())
        return out

    def _synthesize(self, question: str, evidence: Sequence[RetrievedPassage]) -> str:
        """Produce the answer text.

        With an LLM generator attached, delegate to it. Otherwise produce a
        grounded extractive summary that quotes the highest-scoring sentences,
        so the system is fully functional with no API keys.

        When ``attribute_caveats`` is set, whatever the answer path produced is
        post-processed by :func:`medrag.eval.attribution.compose_attributed_answer`,
        which appends limitations quoted out of the retrieved passages.
        """
        text = self._synthesize_raw(question, evidence)
        if self.attribute_caveats:
            from ..eval.attribution import compose_attributed_answer

            text = compose_attributed_answer(
                text,
                [(p.chunk_id, p.text) for p in evidence],
                max_caveats=self.max_attached_caveats,
            )
        return text

    def _synthesize_raw(self, question: str, evidence: Sequence[RetrievedPassage]) -> str:
        """The answer text before any grounded caveat attachment."""
        if self.generator is not None:
            try:
                return self.generator(question, evidence)
            except Exception as exc:  # noqa: BLE001
                log.warning("generator failed, using extractive fallback: %s", exc)

        if not evidence:
            return "No relevant evidence was retrieved for this question."

        terms = set(content_tokens(question))
        scored: list[tuple[float, str, RetrievedPassage]] = []
        for rank, passage in enumerate(evidence):
            for sentence in re.split(r"(?<=[.!?])\s+", passage.text):
                sentence = _clean_quote(sentence)
                if len(sentence) < 30:
                    continue
                overlap = len(terms & set(content_tokens(sentence)))
                # Favour relevance to the question, then earlier evidence.
                scored.append((overlap / (1 + rank * 0.35), sentence, passage))

        scored.sort(key=lambda t: -t[0])
        lines: list[str] = []
        used: set[str] = set()
        for score, sentence, passage in scored:
            if score <= 0:
                continue
            key = sentence[:60].lower()
            if key in used:
                continue
            used.add(key)
            lines.append(f"- {sentence} [{passage.citation}]")
            if len(lines) >= 5:
                break

        if not lines:
            top = evidence[0]
            return f"Top match: {_clean_quote(top.text)[:500]} [{top.citation}]"

        n_docs = len({p.chunk_id.split("#")[0] for p in evidence})
        header = f'Retrieved evidence for: "{question}" ({n_docs} source{"s" if n_docs != 1 else ""})'
        return header + "\n" + "\n".join(lines)

    # ------------------------------------------------------------------ batch
    def research_batch(self, questions: Sequence[str], *, max_hops: int | None = None) -> list[AgentAnswer]:
        """Run many questions, logging throughput as it goes."""
        answers: list[AgentAnswer] = []
        t0 = time.time()
        for i, question in enumerate(questions, 1):
            answers.append(self.research(question, max_hops=max_hops))
            if i % 25 == 0 or i == len(questions):
                rate = i / max(1e-9, time.time() - t0)
                log.info("agent: %d/%d questions (%.2f q/s)", i, len(questions), rate)
        return answers
