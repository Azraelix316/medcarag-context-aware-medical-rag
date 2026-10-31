"""Citation-derived relevance labels, and the bias they inherit.

Why
---
This project's gold labels are MeSH pseudo-relevance judgments: query a MeSH
descriptor, and every document carrying it is relevant (Kim et al., BioNLP 2018).
That is convenient and model-independent, and it is exactly why it is dangerous
here. A retriever that scores documents by MeSH overlap shares its entire signal
with the labels, so a large measured gain may be the retriever recognising the
answer key. That risk was found live in this project: the MeSH graph retriever
showed hit@1 0.692 -> 0.818 over the two-way hybrid, and it was not possible to
tell how much was retrieval and how much was circularity.

Citation-derived labels break the tie because they use a disjoint signal. A
document that a high-confidence seed *cites* is relevant to the same question,
because an author who cites it in a systematic review about that topic asserts
its relevance. This is how relevance is established in the actual literature, not
an artefact of an ontology choice.

Prior art and what this borrows
-------------------------------
Citation-based relevance (CBR) is established, as is the finding that pooled
metrics are unreliable under incomplete judgments (Buckley & Voorhees 2004, hence
bpref) and that judgment-free labels should be validated against human labels
rather than trusted (Ryu & Choi 2012, who reached rank correlation up to 0.863
validating aspect-query pseudo-qrels against TREC Genomics human judgments). So
the construction below is not novel; the contribution is applying it as a
*control* on an existing evaluation, and reporting agreement between the two label
regimes rather than picking whichever flatters the method.

The bias this inherits, stated before the numbers
--------------------------------------------------
Seeding is the weak link. Documents cited by seeds are relevant *if the seeds are
right*. Seed selection here is:

* publication type restricted to systematic reviews, meta-analyses, and practice
  guidelines, which PubMed indexes carefully;
* restricted to documents whose deposited reference list is present.

Both filters are era-dependent. Measured citation coverage: 72% of 2019-2026 RCTs
carry a reference list (33 refs each) versus 20% of 2000-2009 (7.3 refs each).
So citation-derived labels are systematically denser on recent literature, and any
query whose topic peaks before ~2010 will have a thinner label set. Comparisons
against MeSH labels are therefore stratified by era rather than pooled.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Publication types trusted as seeds: the designs whose authors did the
# synthesising work, so their reference lists are topical judgements.
SEED_PUBLICATION_TYPES: frozenset[str] = frozenset({
    "Systematic Review",
    "Meta-Analysis",
    "Practice Guideline",
    "Guideline",
})

# Minimum seeds a cited document must be reached by. 1 is the bare minimum and is
# noisy; higher values tighten agreement with the MeSH labels at the cost of
# recall. Measured across the corpus for the values reported in docs/.
DEFAULT_MIN_SEEDS = 2


@dataclass
class CitationSeed:
    """A high-confidence document whose bibliography is treated as a relevance vote."""

    doc_id: str
    pmid: str
    year: int
    mesh: frozenset[str] = frozenset()
    citations: tuple[int, ...] = ()

    @property
    def n_refs(self) -> int:
        return len(self.citations)


@dataclass
class CitationLabels:
    """Query -> citation-derived relevant documents, with provenance."""

    by_descriptor: dict[str, set[str]] = field(default_factory=dict)
    n_seeds: dict[str, int] = field(default_factory=dict)
    min_seeds: int = DEFAULT_MIN_SEEDS
    era_coverage: dict[int, int] = field(default_factory=dict)

    def relevant(self, descriptor: str) -> set[str]:
        return self.by_descriptor.get(descriptor, set())

    def __len__(self) -> int:
        return len(self.by_descriptor)

    def summary(self) -> dict[str, Any]:
        sizes = [len(v) for v in self.by_descriptor.values()]
        return {
            "n_descriptors": len(self.by_descriptor),
            "min_seeds": self.min_seeds,
            "mean_gold_size": round(sum(sizes) / len(sizes), 1) if sizes else 0.0,
            "median_gold_size": sorted(sizes)[len(sizes) // 2] if sizes else 0,
            "seeds_by_decade": dict(sorted(self.era_coverage.items())),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_seeds": self.min_seeds,
            "summary": self.summary(),
            "by_descriptor": {k: sorted(v) for k, v in self.by_descriptor.items()},
        }


def is_seed_document(doc: Any) -> bool:
    """Whether a document qualifies as a relevance seed.

    Requires both a trusted publication type and a non-empty deposited reference
    list. The second condition is not optional: a seed with no bibliography casts
    no votes, and counting it would inflate seed counts without adding evidence.
    """
    types = {t for t in getattr(doc, "publication_types", ())}
    if not (types & SEED_PUBLICATION_TYPES):
        return False
    return bool(getattr(doc, "citations", ()))


def select_seeds(docs: Iterable[Any]) -> list[CitationSeed]:
    """Pull seed documents out of a corpus stream."""
    seeds: list[CitationSeed] = []
    for doc in docs:
        if not is_seed_document(doc):
            continue
        mesh = frozenset(
            t.descriptor for t in getattr(doc, "mesh_terms", ()) if t.descriptor
        )
        if not mesh:
            continue
        seeds.append(
            CitationSeed(
                doc_id=doc.doc_id,
                pmid=str(getattr(doc, "pmid", "") or ""),
                year=int(getattr(doc, "year", 0) or 0),
                mesh=mesh,
                citations=tuple(getattr(doc, "citations", ()) or ()),
            )
        )
    log.info(
        "citation seeds: %d documents with trusted publication types and bibliographies",
        len(seeds),
    )
    return seeds


def build_citation_labels(
    seeds: Sequence[CitationSeed],
    *,
    min_seeds: int = DEFAULT_MIN_SEEDS,
    restrict_descriptors: Iterable[str] | None = None,
    id_of: Callable[[Any], str] | None = None,
) -> CitationLabels:
    """Derive relevance labels by counting seed bibliographies.

    A descriptor d is the query. Every seed indexed under d votes for the
    documents it cites. A document becomes gold for d once ``min_seeds`` distinct
    seeds vote for it, which filters the long tail of one-off citations.

    ``restrict_descriptors`` limits work to the query set actually under
    evaluation; without it a 20M-document corpus would vote on every descriptor it
    contains.

    ``id_of`` maps a cited reference to the identifier space used by the rest of
    the evaluation. This is not cosmetic. Reference lists store bare PMIDs
    (``42826085``) while documents and the MeSH gold labels are keyed
    ``pmid:42826085``. Emitting bare PMIDs here produced a label set that shared
    zero identifiers with the MeSH labels, so agreement measured as Jaccard
    0.000 across all 4,466 descriptors -- indistinguishable from total
    disagreement, and entirely an artefact of the key format. ``id_of`` defaults to
    the corpus convention.
    """
    id_of = id_of or (lambda ref: f"pmid:{ref}")
    wanted = set(restrict_descriptors) if restrict_descriptors is not None else None
    descriptor_seeds: dict[str, int] = defaultdict(int)
    era: dict[int, int] = defaultdict(int)
    # (cited_doc_id, descriptor) -> number of distinct seeds voting for it.
    seed_votes: dict[tuple[str, str], int] = defaultdict(int)

    for seed in seeds:
        descriptors = seed.mesh if wanted is None else (seed.mesh & wanted)
        if not descriptors:
            continue
        for d in descriptors:
            descriptor_seeds[d] += 1
        if seed.year:
            era[(seed.year // 10) * 10] += 1
        for cited in seed.citations:
            cid = id_of(cited)
            for d in descriptors:
                seed_votes[(cid, d)] += 1

    by_descriptor: dict[str, set[str]] = defaultdict(set)
    for (doc_id, d), count in seed_votes.items():
        if count >= min_seeds:
            by_descriptor[d].add(doc_id)

    labels = CitationLabels(
        by_descriptor=dict(by_descriptor),
        n_seeds=dict(descriptor_seeds),
        min_seeds=min_seeds,
        era_coverage=dict(era),
    )
    log.info("citation labels: %s", labels.summary())
    return labels


def label_agreement(
    mesh_labels: dict[str, set[str]],
    citation_labels: dict[str, set[str]],
) -> dict[str, Any]:
    """Agreement between the two label regimes, per descriptor.

    This is the number that tells us whether the MeSH labels are trustworthy. If
    the two regimes agree, a gain measured under either is credible. If they
    diverge, a gain measured under one may be an artefact of that one's definition.

    Reported as Jaccard (overlap over union) plus recall each way, because Jaccard
    alone conflates "small sets that agree" with "large sets that overlap a lot".
    """
    shared = set(mesh_labels) & set(citation_labels)
    rows: list[dict[str, Any]] = []
    for d in sorted(shared):
        a, b = mesh_labels[d], citation_labels[d]
        if not a and not b:
            continue
        inter = len(a & b)
        union = len(a | b)
        rows.append({
            "descriptor": d,
            "mesh_size": len(a),
            "citation_size": len(b),
            "intersection": inter,
            "jaccard": round(inter / union, 4) if union else 0.0,
            "citation_recall_of_mesh": round(inter / len(a), 4) if a else 0.0,
            "mesh_recall_of_citation": round(inter / len(b), 4) if b else 0.0,
        })
    if not rows:
        return {"n_descriptors": 0, "mean_jaccard": 0.0, "rows": []}
    mean_j = sum(r["jaccard"] for r in rows) / len(rows)
    return {
        "n_descriptors": len(rows),
        "mean_jaccard": round(mean_j, 4),
        "median_jaccard": sorted(r["jaccard"] for r in rows)[len(rows) // 2],
        "rows": rows,
    }


def evaluate_under_citation_labels(
    ranked_by_query: dict[str, Sequence[str]],
    labels: CitationLabels,
    descriptors_by_query: dict[str, str],
) -> dict[str, float]:
    """Mean bpref and nDCG when relevance comes from citations, not MeSH.

    ``ranked_by_query`` is qid -> ranked document ids (doc-level, since citation
    labels are document-level). bpref is the headline because citation labels are
    incomplete by construction.
    """
    from .metrics import bpref, ndcg_at_k

    bp: list[float] = []
    nd: list[float] = []
    for qid, ranked in ranked_by_query.items():
        desc = descriptors_by_query.get(qid)
        if desc is None:
            continue
        gold = labels.relevant(desc)
        if not gold:
            continue
        b = bpref(ranked, gold)
        n = ndcg_at_k(ranked, gold, 10)
        if b == b:  # skip NaN
            bp.append(b)
        if n == n:
            nd.append(n)
    return {
        "n_queries": len(bp),
        "bpref": round(sum(bp) / len(bp), 4) if bp else 0.0,
        "ndcg@10": round(sum(nd) / len(nd), 4) if nd else 0.0,
    }