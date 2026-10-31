"""Context-provision metrics: does retrieval return a *qualified* answer?

The problem
-----------
Conventional retrieval metrics cannot see the failure documented in
arXiv 2502.14898. Measured against deployed medical RAG systems, only 4-5% of
responses mentioned complication rates and 6-10% mentioned procedure benefits,
and a disputed diagnosis's contested status appeared in 100% of direct-query
responses but only 56-69% of presupposing ones.

Every one of those responses could be scored as a success by hit@k. The
procedure's complications were listed; the sources were cited; the retrieval
"worked". What was missing was the qualifying context -- how often, in whom,
against what comparator, and what the benefit side was.

So a system can have high recall@k and still be dangerously incomplete. This
module defines metrics that can see it.

The distinction being measured
-----------------------------
For a query whose gold documents are known, a retrieved set is:

**bare**       contains a passage stating the finding, but no passage from the
               same document that frames it. This is the decontextualized
               answer: accurate, grounded, and misleading.
**qualified**  contains the finding *and* a framing passage from the same
               document -- the section that states the population the rate
               applies to, or the caveats and benefits that qualify it.

Framing is judged *within a document* on purpose. A conclusions chunk from a
different paper does not qualify this paper's result; it is a different study
with a different population. This is precisely why pooling a top-k list across
documents reads as informative while a reader of any single document would see an
unqualified claim.

Metric definitions
------------------
``qar@k``  Qualified Answer Rate. Fraction of queries whose top-k contains a gold
           document's evidential passage *and* a framing passage from that same
           document. The headline number: retrieval that supports a safe answer.

``bar@k``  Bare Answer Rate. Fraction of queries whose top-k contains a gold
           document's evidential passage but *no* framing passage from it. The
           decontextualization rate. qar + bar + (nothing found) = 1.

``sabr@k`` Same-Document Answer Rate. Of queries where a gold document was found,
           the fraction where it was found *with* its own framing. Normalised to
           exclude the "found nothing" case, so it is comparable across systems
           that retrieve different amounts.

``qual_ratio@k``  Mean number of framing passages per evidential passage in the
           top-k. More is better up to the point where framing crowds out the
           finding; this reports the balance rather than assuming it.

Honest limitations, stated because they constrain interpretation
---------------------------------------------------------------
* Framing is identified by *section role*, not by an oracle judgement of whether
  the text actually qualifies. A conclusions chunk that adds no caveat counts.
  This measures structural completeness, which is a proxy for qualification
  rather than a proof of it.
* Gold documents come from MeSH pseudo-relevance labels (Kim et al. 2018), which
  is document-level. Any retriever that scores by MeSH overlap is advantaged
  here. That is why these metrics must be reported alongside the citation-derived
  gold set, and never alone.
* A bare answer is not always wrong. For a query that genuinely asks only for an
  effect size, framing is unnecessary overhead. The metric is meaningful for the
  query families where the source literature documents that framing is what
  changes the interpretation -- outcome, complication, safety, and disputed
  condition queries.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Sections that state a finding or an effect estimate.
EVIDENTIAL_KINDS: frozenset[str] = frozenset({"results", "background", ""})
# Sections that frame a finding: population, caveats, benefits, limitations.
FRAMING_KINDS: frozenset[str] = frozenset({"conclusions", "discussion", "methods"})

# Methods is a framing section only when it carries the population a rate applies
# to. Treating every Methods chunk as framing would inflate the metric, so the
# caller may pass a stricter predicate; the default keeps it simple and the
# limitation is documented above.
_STRICT_FRAMING: frozenset[str] = frozenset({"conclusions", "discussion"})


@dataclass
class RetrievedChunk:
    """One retrieved passage, reduced to what these metrics need."""

    row_id: int
    doc_id: str
    section_kind: str


@dataclass
class ContextProvision:
    """Per-query outcome."""

    qid: str
    found_gold: bool = False
    qualified: bool = False
    bare: bool = False
    n_evidential: int = 0
    n_framing: int = 0
    qualifying_docs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "found_gold": self.found_gold,
            "qualified": self.qualified,
            "bare": self.bare,
            "n_evidential": self.n_evidential,
            "n_framing": self.n_framing,
            "n_qualifying_docs": len(self.qualifying_docs),
        }


def assess_context_provision(
    qid: str,
    retrieved: Sequence[RetrievedChunk],
    gold_doc_ids: Iterable[str],
    *,
    strict_framing: bool = True,
) -> ContextProvision:
    """Classify one query's retrieved set as bare, qualified, or nothing.

    ``strict_framing`` excludes Methods from the framing set, counting only
    Conclusions and Discussion. That is the conservative choice: a Methods chunk
    describes how the study was done, which is necessary context but not the same
    thing as a caveat or a benefit statement. With ``strict_framing=False``,
    Methods counts, which measures whether the retrieved set is internally
    self-consistent rather than whether it qualifies the finding.
    """
    gold = set(gold_doc_ids)
    out = ContextProvision(qid=qid)
    framing_kinds = _STRICT_FRAMING if strict_framing else FRAMING_KINDS

    evidential_by_doc: dict[str, int] = {}
    framing_by_doc: dict[str, int] = {}
    for chunk in retrieved:
        if chunk.doc_id not in gold:
            continue
        kind = chunk.section_kind or ""
        if kind in EVIDENTIAL_KINDS:
            evidential_by_doc[chunk.doc_id] = evidential_by_doc.get(chunk.doc_id, 0) + 1
        elif kind in framing_kinds:
            framing_by_doc[chunk.doc_id] = framing_by_doc.get(chunk.doc_id, 0) + 1

    if not evidential_by_doc:
        return out

    out.found_gold = True
    out.n_evidential = sum(evidential_by_doc.values())
    out.n_framing = sum(framing_by_doc.values())
    out.qualifying_docs = sorted(d for d in evidential_by_doc if framing_by_doc.get(d))
    out.qualified = bool(out.qualifying_docs)
    out.bare = not out.qualified
    return out


@dataclass
class ContextProvisionReport:
    """Aggregate over a set of queries."""

    n_queries: int = 0
    n_found: int = 0
    n_qualified: int = 0
    n_bare: int = 0
    qual_ratio_sum: float = 0.0
    per_query: list[ContextProvision] = field(default_factory=list)

    @property
    def qar(self) -> float:
        """Qualified Answer Rate over all queries."""
        return self.n_qualified / self.n_queries if self.n_queries else 0.0

    @property
    def bar(self) -> float:
        """Bare Answer Rate over all queries."""
        return self.n_bare / self.n_queries if self.n_queries else 0.0

    @property
    def sabr(self) -> float:
        """Same-Document Answer Rate, over queries that found a gold document."""
        return self.n_qualified / self.n_found if self.n_found else 0.0

    @property
    def qual_ratio(self) -> float:
        """Mean framing passages per evidential passage, over found queries."""
        return self.qual_ratio_sum / self.n_found if self.n_found else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            "n_found_gold": self.n_found,
            "n_qualified": self.n_qualified,
            "n_bare": self.n_bare,
            "qar": round(self.qar, 4),
            "bar": round(self.bar, 4),
            "sabr": round(self.sabr, 4),
            "qual_ratio": round(self.qual_ratio, 4),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary(), "per_query": [q.to_dict() for q in self.per_query]}


def aggregate_context_provision(
    per_query: Iterable[ContextProvision],
) -> ContextProvisionReport:
    rep = ContextProvisionReport()
    for q in per_query:
        rep.n_queries += 1
        if q.found_gold:
            rep.n_found += 1
            if q.n_evidential:
                rep.qual_ratio_sum += q.n_framing / q.n_evidential
        if q.qualified:
            rep.n_qualified += 1
        elif q.bare:
            rep.n_bare += 1
        rep.per_query.append(q)
    return rep


def context_provision_from_hits(
    hits: Sequence[tuple[int, float]],
    store: Any,
    gold_doc_ids: Iterable[str],
    qid: str = "",
    **kwargs: Any,
) -> ContextProvision:
    """Convenience wrapper: ``(row_id, score)`` pairs plus a chunk store."""
    retrieved = [
        RetrievedChunk(
            row_id=int(row),
            doc_id=(c.doc_id if (c := store.get(int(row))) else ""),
            section_kind=(c.section_kind if c else ""),
        )
        for row, _ in hits
    ]
    return assess_context_provision(qid, retrieved, gold_doc_ids, **kwargs)


def compare_provision(
    baseline: ContextProvisionReport,
    candidate: ContextProvisionReport,
) -> dict[str, Any]:
    """Baseline vs candidate, with the direction of each change made explicit."""
    out: dict[str, Any] = {}
    for name in ("qar", "bar", "sabr", "qual_ratio"):
        b, c = getattr(baseline, name), getattr(candidate, name)
        out[name] = {
            "baseline": round(b, 4),
            "candidate": round(c, 4),
            "delta": round(c - b, 4),
            # Higher is better for qar/sabr/qual_ratio; lower for bar.
            "better": (c < b) if name == "bar" else (c > b),
        }
    return out