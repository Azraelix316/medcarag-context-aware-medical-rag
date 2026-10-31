"""Query planning: rewriting, decomposition, and MeSH-aware expansion.

Agentic retrieval fails on two failure modes this module addresses:

1. **Vocabulary mismatch.** A question says "heart attack", the literature says
   "myocardial infarction". A cheap MeSH/lexical expansion bridges the gap
   without needing an LLM in the loop.
2. **Multi-hop questions.** "Does drug X treat condition Y in patients with
   condition Z?" needs three retrievals. Decomposition splits it into atomic
   sub-queries whose results are merged.

Both paths work without an LLM (heuristic mode, the default here) and accept an
LLM-backed implementation when an API key is available.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Synonym table for the highest-frequency clinical mismatches. Deliberately
# conservative: a wrong synonym costs precision, which is expensive in RAG.
_CLINICAL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "heart attack": ("myocardial infarction", "acute coronary syndrome", "MI"),
    "heart failure": ("cardiac failure", "congestive heart failure", "CHF", "HFrEF", "HFpEF"),
    "stroke": ("cerebrovascular accident", "CVA", "ischemic stroke"),
    "high blood pressure": ("hypertension", "high blood pressure", "HTN"),
    "diabetes": ("diabetes mellitus", "DM"),
    "type 2 diabetes": ("type 2 diabetes mellitus", "T2DM", "NIDDM"),
    "type 1 diabetes": ("type 1 diabetes mellitus", "T1DM", "IDDM"),
    "copd": ("chronic obstructive pulmonary disease", "COPD", "emphysema"),
    "asthma": ("asthma", "bronchial asthma"),
    "kidney disease": ("chronic kidney disease", "CKD", "renal insufficiency"),
    "cancer": ("neoplasm", "malignancy", "carcinoma"),
    "breast cancer": ("breast neoplasm", "breast carcinoma"),
    "lung cancer": ("lung neoplasm", "non-small cell lung cancer", "NSCLC"),
    "depression": ("depressive disorder", "major depressive disorder", "MDD"),
    "anxiety": ("anxiety disorder", "generalized anxiety disorder", "GAD"),
    "sepsis": ("sepsis", "septic shock", "systemic inflammatory response syndrome", "SIRS"),
    "alzheimer's": ("Alzheimer disease", "Alzheimer's disease"),
    "parkinson's": ("Parkinson disease", "Parkinson's disease"),
    "obesity": ("obesity", "adiposity", "body mass index"),
    "vitamin d": ("vitamin D", "cholecalciferol", "25-hydroxyvitamin D"),
    "blood thinner": ("anticoagulant", "anticoagulation"),
    "blood pressure": ("blood pressure", "hypertension"),
    "side effects": ("adverse effect", "adverse event", "adverse drug reaction", "toxicity"),
    "efficacy": ("efficacy", "effectiveness", "therapeutic effect"),
    "adherence": ("adherence", "compliance", "medication adherence"),
    "cost effectiveness": ("cost-effectiveness", "cost utility", "economic evaluation"),
    "screening": ("screening", "early detection", "mass screening"),
    "vaccine": ("vaccine", "vaccination", "immunization"),
}

# Clause markers that usually separate an atomic sub-question.
_SPLIT_PATTERNS = (
    r"\b(?:and|but|however|therefore|thus|while|whereas)\b",
    r"\bsuch as\b",
    r"\bin (?:patients|adults|children|women|men) with\b",
)

_STOPWORDS = frozenset(
    "what which who whom whose when where why how does do did is are was were "
    "the a an and or of to in on for with about from by as at that this these "
    "those it its be been being have has had can could would should will may "
    "might must i you we they he she".split()
)


@dataclass
class SubQuery:
    """One atomic retrieval target plus why it was produced."""

    text: str
    kind: str = "atomic"     # atomic | expansion | original
    parent: str = ""
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "kind": self.kind, "parent": self.parent, "weight": self.weight}


@dataclass
class QueryPlan:
    """The full retrieval plan for one user question."""

    original: str
    sub_queries: list[SubQuery] = field(default_factory=list)
    is_multi_hop: bool = False
    strategy: str = "single"
    notes: list[str] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for sub in self.sub_queries:
            key = sub.text.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(sub.text)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "strategy": self.strategy,
            "is_multi_hop": self.is_multi_hop,
            "notes": self.notes,
            "sub_queries": [s.to_dict() for s in self.sub_queries],
        }


# --------------------------------------------------------------- tokenisation
def content_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-']*", text.lower()) if t not in _STOPWORDS and len(t) > 2]


def detect_multi_hop(query: str, *, threshold: int = 8) -> bool:
    """Heuristic: long questions with several content terms are usually multi-hop."""
    tokens = content_tokens(query)
    if len(tokens) < threshold:
        return False
    if re.search(r"\b(?:both|and|versus|vs\.?|compared to|interaction between|relationship between)\b", query, re.I):
        return True
    # Two or more interrogative clauses.
    return len(re.findall(r"\?", query)) > 1 or bool(re.search(r",\s+(?:and|while|but)\s+", query, re.I))


def expand_query(query: str, *, max_expansions: int = 3) -> list[str]:
    """Append controlled synonyms for matched clinical phrases."""
    lowered = query.lower()
    found: list[str] = []
    for phrase, synonyms in _CLINICAL_SYNONYMS.items():
        if phrase in lowered:
            for syn in synonyms:
                if syn.lower() != phrase and syn.lower() not in lowered:
                    found.append(syn)
                    if len(found) >= max_expansions:
                        return found
    return found


def strip_question_framing(query: str) -> str:
    """Remove interrogative scaffolding so the key terms dominate the query."""
    cleaned = query.strip().rstrip("?").strip()
    cleaned = re.sub(
        r"^(?:what|which|how|why|when|where|who)\s+(?:is|are|was|were|do|does|did|can|could|would|should|will|may)\s+",
        "", cleaned, flags=re.I,
    )
    cleaned = re.sub(
        r"^(?:the\s+)?(?:current\s+|recent\s+|latest\s+)?(?:evidence|research|studies|literature)\s+"
        r"(?:on|about|regarding|for)\s+", "", cleaned, flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:")
    return cleaned or query.strip()


def decompose_heuristic(query: str, *, max_subqueries: int = 4) -> list[str]:
    """Split a compound question into atomic retrievable phrases.

    No LLM: the split uses clause boundaries and content-term extraction, which
    is enough for the templated clinical questions the benchmark generates and
    for most agent-style follow-ups.
    """
    parts: list[str] = []
    for pattern in _SPLIT_PATTERNS:
        parts = [p for chunk in re.split(pattern, query, flags=re.I) for p in ([chunk] if chunk else [])]
        if len(parts) > 1:
            break

    out: list[str] = []
    for part in parts:
        cleaned = strip_question_framing(part)
        tokens = content_tokens(cleaned)
        if len(tokens) >= 3:
            out.append(cleaned)
    if not out:
        out = [strip_question_framing(query)]

    # Keep the parts ordered and de-duplicated, preserving the original if short.
    seen: set[str] = set()
    ordered: list[str] = []
    for text in out:
        key = text.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(text)
    return ordered[:max_subqueries]


# ------------------------------------------------------------------- planning
class QueryPlanner:
    """Turns a user question into a :class:`QueryPlan`.

    ``llm_fn`` is an optional callable ``(prompt) -> str`` used when
    ``strategy='llm'``. Without it the planner stays fully local, which is the
    default so the pipeline runs with no API keys.
    """

    def __init__(
        self,
        strategy: str = "heuristic",
        *,
        mode: str = "auto",
        max_subqueries: int = 4,
        llm_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.strategy = strategy
        self.mode = mode
        self.max_subqueries = max_subqueries
        self.llm_fn = llm_fn

    def plan(self, query: str) -> QueryPlan:
        query = query.strip()
        if not query:
            return QueryPlan(original=query, strategy="empty")

        plan = QueryPlan(original=query)
        multi_hop = detect_multi_hop(query)
        plan.is_multi_hop = multi_hop

        should_decompose = multi_hop or self.mode == "decompose"

        if should_decompose:
            subs = (
                self._decompose_llm(query)
                if self.strategy == "llm" and self.llm_fn
                else decompose_heuristic(query, max_subqueries=self.max_subqueries)
            )
            plan.sub_queries = [SubQuery(text=t, kind="atomic", parent=query) for t in subs]
            plan.strategy = "decompose"
            plan.notes.append(f"decomposed into {len(plan.sub_queries)} atomic queries")
        else:
            plan.sub_queries = [SubQuery(text=query, kind="original", parent=query)]
            plan.strategy = "single"

        # Vocabulary expansion on top of the atomic queries.
        for expansion in expand_query(query):
            plan.sub_queries.append(SubQuery(text=expansion, kind="expansion", parent=query, weight=0.5))
            plan.notes.append(f"expanded with clinical synonym: {expansion}")

        if self.strategy != "none":
            stripped = strip_question_framing(query)
            if stripped.lower() != query.lower().rstrip("?") and stripped:
                plan.sub_queries.append(SubQuery(text=stripped, kind="expansion", parent=query, weight=0.8))
                plan.notes.append(f"normalized query framing: {stripped}")

        return plan

    def _decompose_llm(self, query: str) -> list[str]:
        prompt = (
            "Break this clinical research question into the minimum set of atomic "
            "search queries needed to answer it. Return one query per line, no "
            f"numbering, no commentary.\n\nQuestion: {query}"
        )
        try:
            raw = self.llm_fn(prompt)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM decomposition failed, falling back to heuristic: %s", exc)
            return decompose_heuristic(query, max_subqueries=self.max_subqueries)
        lines = [re.sub(r"^\s*[\d\-\*\.]+\s*", "", ln).strip() for ln in raw.splitlines()]
        lines = [ln for ln in lines if len(ln) > 3]
        return lines[: self.max_subqueries] or decompose_heuristic(query, max_subqueries=self.max_subqueries)


def plan_to_subquery_texts(plan: QueryPlan) -> list[str]:
    return plan.texts
