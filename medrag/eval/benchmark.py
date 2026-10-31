"""Benchmark construction from the corpus's own MeSH structure.

Rationale: a medical RAG paper needs a benchmark whose relevance labels are
independent of the retrieval system being evaluated. PubMed provides exactly
that for free. Each document carries MeSH descriptors; a query built from a
descriptor has, by construction, a large set of genuinely relevant documents
(the other articles indexed under the same descriptor). Labels come from the
indexing, not from any model.

Three query families are produced:

``mesh``       Single-hop. The descriptor itself is the query; the gold set is
               every other document carrying that descriptor. Hard negatives are
               documents from sibling descriptors, which share vocabulary but are
               wrong, so they punish pure lexical matching.

``clinical``   Natural-language clinical questions written from descriptor
               templates. Closer to real agent usage than bare MeSH terms.

``multi_hop``  Two-hop questions whose gold set is the intersection of two
               descriptors. Requires fusing evidence from two sources.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..data.mesh_taxonomy import GENERIC_ROOTS, TREE_CATEGORIES, MeshTaxonomy
from ..data.schema import Document
from .metrics import Qrels, Query

log = logging.getLogger(__name__)

# Descriptors that are too generic to yield a discriminative query.
STOP_DESCRIPTORS = frozenset("""
humans animals male female adult middle-aged aged young child infant
united states research methods, biomedical methods, research design
biomedical research research design, research methods
""".split("\n")[0].split())

# MeSH tree roots that make poor query terms. Resolved from the taxonomy, not
# from the UI string: every descriptor UI starts with "D" regardless of whether
# it is a disease (tree C) or a drug (tree D).
_GENERIC_TREE = GENERIC_ROOTS

# Clinical question templates keyed by the MeSH descriptor's tree root. These
# read as the kind of question a clinician or research agent actually asks.
_TEMPLATES: tuple[str, ...] = (
    "What is the current evidence on {topic}?",
    "How does {topic} affect patient outcomes?",
    "What treatments are effective for {topic}?",
    "What are the risk factors associated with {topic}?",
    "What diagnostic approaches are used for {topic}?",
    "What is the pathophysiology of {topic}?",
    "Which populations are most affected by {topic}?",
    "What comorbidities commonly occur with {topic}?",
)

_PHRASE_FIXES = (
    (r"\b Diseases\b", " disease"),
    (r"^\w", lambda m: m.group(0).lower()),
)


def _descriptor_phrase(descriptor: str) -> str:
    """Turn 'Diabetes Mellitus, Type 2' into the natural phrase 'type 2 diabetes mellitus'."""
    phrase = descriptor.strip()
    # Invert "Head, Qualifier" -> "qualifier head".
    if "," in phrase:
        head, _, qualifier = phrase.partition(",")
        phrase = f"{qualifier.strip()} {head.strip()}"
    phrase = phrase.lower()
    phrase = re.sub(r"\s+", " ", phrase)
    phrase = phrase.replace("'s ", " ")
    return phrase.strip()


def _clean_topic(descriptor: str) -> str | None:
    phrase = _descriptor_phrase(descriptor)
    if len(phrase) < 6 or len(phrase.split()) < 2:
        return None
    if any(bad in phrase for bad in ("unknown", "unclassified", "data", "methods")):
        return None
    return phrase


def _mesh_ui_tree_root(ui: str, label: str, taxonomy: MeshTaxonomy | None) -> str:
    """Resolve a descriptor's MeSH tree root, via taxonomy or label fallback.

    The UI's first character is NOT the tree root (all descriptor UIs begin
    with "D"), so this delegates to the taxonomy resolver and falls back to
    label heuristics when no resolver is supplied.
    """
    if taxonomy is not None:
        return taxonomy.tree_root(ui, label)
    return MeshTaxonomy.infer_root(label)


def build_mesh_qrels(
    docs: Sequence[Document],
    *,
    max_queries_per_descriptor: int = 60,
    min_gold: int = 8,
    max_gold: int = 400,
    seed: int = 13,
    taxonomy: MeshTaxonomy | None = None,
) -> Qrels:
    """Descriptor-derived single-hop queries with full-document gold sets."""
    rng = random.Random(seed)
    descriptor_docs: dict[str, list[Document]] = defaultdict(list)
    descriptor_ui: dict[str, str] = {}

    for doc in docs:
        for term in doc.mesh_terms:
            descriptor = term.descriptor.strip()
            if not descriptor or descriptor.lower() in STOP_DESCRIPTORS:
                continue
            if descriptor_ui.get(descriptor, "") == "" and term.ui:
                descriptor_ui[descriptor] = term.ui
            descriptor_docs[descriptor].append(doc)

    log.info("building MeSH qrels from %d distinct descriptors", len(descriptor_docs))

    queries: list[Query] = []
    # Prefer descriptors with enough evidence to be a meaningful retrieval task.
    eligible = [
        (d, members) for d, members in descriptor_docs.items()
        if min_gold <= len(members) <= max_gold
        and _mesh_ui_tree_root(descriptor_ui.get(d, ""), d, taxonomy) not in _GENERIC_TREE
    ]
    log.info("%d descriptors pass the gold-set size filter", len(eligible))

    for descriptor, members in eligible:
        topic = _clean_topic(descriptor)
        if not topic:
            continue
        root = _mesh_ui_tree_root(descriptor_ui.get(descriptor, ""), descriptor, taxonomy)
        doc_ids = sorted({d.doc_id for d in members})
        for i in range(min(max_queries_per_descriptor, 3)):
            # Vary phrasing so the benchmark is not one template repeated.
            template = _TEMPLATES[rng.randrange(len(_TEMPLATES))]
            queries.append(
                Query(
                    qid=f"mesh::{descriptor}::{i}",
                    text=template.format(topic=topic),
                    relevant_docs=set(doc_ids),
                    category=TREE_CATEGORIES.get(root, f"tree-{root or 'unknown'}"),
                    difficulty="single-hop",
                    answer=topic,
                )
            )

    qrels = Qrels(queries)
    log.info("built %d MeSH queries", len(queries))
    return qrels


def build_clinical_qrels(
    docs: Sequence[Document],
    *,
    n_queries: int = 2000,
    min_gold: int = 10,
    max_gold: int = 300,
    seed: int = 13,
    taxonomy: MeshTaxonomy | None = None,
) -> Qrels:
    """Bare-topic queries that mirror how an agent phrases a literature search."""
    rng = random.Random(seed)
    descriptor_docs: dict[str, list[Document]] = defaultdict(list)
    descriptor_ui: dict[str, str] = {}
    for doc in docs:
        for term in doc.mesh_terms:
            descriptor = term.descriptor.strip()
            if not descriptor or descriptor.lower() in STOP_DESCRIPTORS:
                continue
            descriptor_docs[descriptor].append(doc)
            if term.ui and not descriptor_ui.get(descriptor):
                descriptor_ui[descriptor] = term.ui

    eligible = [
        (d, members) for d, members in descriptor_docs.items()
        if min_gold <= len(members) <= max_gold
        and _mesh_ui_tree_root(descriptor_ui.get(d, ""), d, taxonomy) not in _GENERIC_TREE
    ]
    rng.shuffle(eligible)

    queries: list[Query] = []
    for descriptor, members in eligible:
        if len(queries) >= n_queries:
            break
        topic = _clean_topic(descriptor)
        if not topic:
            continue
        root = _mesh_ui_tree_root(descriptor_ui.get(descriptor, ""), descriptor, taxonomy)
        doc_ids = sorted({d.doc_id for d in members})
        # Alternate between the bare topic and a "role of X" framing.
        text = topic if len(queries) % 2 == 0 else f"{topic} recent advances"
        queries.append(
            Query(
                qid=f"clinical::{descriptor}",
                text=text,
                relevant_docs=set(doc_ids),
                category=TREE_CATEGORIES.get(root, f"tree-{root or 'unknown'}"),
                difficulty="single-hop",
                answer=topic,
            )
        )
    qrels = Qrels(queries)
    log.info("built %d clinical queries", len(queries))
    return qrels


def build_multi_hop_qrels(
    docs: Sequence[Document],
    *,
    n_queries: int = 500,
    min_gold: int = 4,
    seed: int = 13,
) -> Qrels:
    """Two-hop queries: both descriptors must be satisfied by the evidence set.

    Gold = documents carrying both descriptors. ``supporting_docs`` additionally
    lists each side separately, so success@all can distinguish "found one hop"
    from "found both".
    """
    rng = random.Random(seed)
    per_doc: dict[str, set[str]] = defaultdict(set)
    descriptor_docs: dict[str, list[str]] = defaultdict(list)

    for doc in docs:
        descriptors = {t.descriptor.strip() for t in doc.mesh_terms if t.descriptor.strip()}
        for descriptor in descriptors:
            per_doc[doc.doc_id].add(descriptor)
            descriptor_docs[descriptor].append(doc.doc_id)

    # Invert the doc->descriptors map into descriptor->docs, then find pairs
    # that share enough documents to form a non-trivial intersection.
    pairs: list[tuple[str, str, list[str]]] = []
    max_side = 500
    descriptors = [d for d, docs in descriptor_docs.items() if len(docs) <= max_side]
    for i, a in enumerate(descriptors):
        a_set = set(descriptor_docs[a])
        for b in descriptors[i + 1 :]:
            shared = a_set.intersection(descriptor_docs[b])
            if len(shared) >= min_gold:
                pairs.append((a, b, sorted(shared)))

    rng.shuffle(pairs)
    log.info("found %d co-occurring descriptor pairs", len(pairs))

    queries: list[Query] = []
    for a, b, shared in pairs[:n_queries]:
        topic_a, topic_b = _clean_topic(a), _clean_topic(b)
        if not topic_a or not topic_b:
            continue
        text = rng.choice([
            f"What is the relationship between {topic_a} and {topic_b}?",
            f"How does {topic_a} influence outcomes in {topic_b}?",
            f"Are {topic_a} and {topic_b} associated, and what evidence supports this?",
        ])
        queries.append(
            Query(
                qid=f"multihop::{a}::{b}",
                text=text,
                relevant_docs=set(shared),
                # Either side alone is acceptable supporting evidence, so a system
                # that retrieves both hops can be credited even when no single
                # document carries both descriptors.
                supporting_docs=set(shared) | set(descriptor_docs[a]) | set(descriptor_docs[b]),
                category="multi-hop",
                difficulty="multi-hop",
                answer=f"{topic_a} and {topic_b}",
            )
        )
    qrels = Qrels(queries)
    log.info("built %d multi-hop queries", len(queries))
    return qrels


def build_benchmark(
    docs: Sequence[Document],
    out_dir: Path,
    *,
    n_clinical: int = 2000,
    n_multihop: int = 500,
    seed: int = 13,
    taxonomy: MeshTaxonomy | None = None,
) -> dict[str, Any]:
    """Build and persist all three query families plus a combined file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clinical = build_clinical_qrels(docs, n_queries=n_clinical, seed=seed, taxonomy=taxonomy)
    multihop = build_multi_hop_qrels(docs, n_queries=n_multihop, seed=seed)
    mesh = build_mesh_qrels(docs, seed=seed, taxonomy=taxonomy)

    paths: dict[str, str] = {}
    for name, qrels in (("clinical", clinical), ("multihop", multihop), ("mesh", mesh)):
        paths[name] = str(qrels.save(out_dir / f"qrels_{name}.json"))

    combined = Qrels(clinical.queries + multihop.queries)
    paths["combined"] = str(combined.save(out_dir / "qrels_combined.json"))

    summary = {
        "n_documents": len(docs),
        "families": {name: len(q) for name, q in (("clinical", clinical), ("multihop", multihop), ("mesh", mesh), ("combined", combined))},
        "gold_sizes": {
            "clinical_mean": round(sum(len(q.relevant_docs) for q in clinical.queries) / max(1, len(clinical.queries)), 1),
            "multihop_mean": round(sum(len(q.relevant_docs) for q in multihop.queries) / max(1, len(multihop.queries)), 1),
        },
        "categories": {
            name: sorted({q.category for q in qrels.queries})
            for name, qrels in (("clinical", clinical), ("multihop", multihop), ("mesh", mesh))
        },
        "paths": paths,
        "seed": seed,
    }
    log.info("benchmark summary: %s", summary)
    (out_dir / "benchmark_summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_queries_from_specs(specs: Iterable[dict[str, Any]]) -> Qrels:
    """Build a Qrels from inline specs (used for ad-hoc research queries)."""
    return Qrels([Query.from_dict(spec) for spec in specs])


def build_benchmark_from_corpus(
    cfg: Any,
    out_dir: Path,
    *,
    n_clinical: int = 2000,
    n_multihop: int = 500,
    seed: int = 13,
    taxonomy: MeshTaxonomy | None = None,
    mesh_sidecar: Path | None = None,
) -> dict[str, Any]:
    """Build the benchmark by streaming the corpus, never holding all documents.

    At 624k documents a list of ``Document`` objects costs tens of GB, which
    cannot coexist with an index build. Only two compact structures are kept:
    ``descriptor -> [doc_id]`` and ``doc_id -> {descriptor}``, both of which are
    small compared to the documents themselves.

    ``mesh_sidecar`` is a ``scripts/attach_mesh_to_pmc.py`` output. PMC JATS carries
    almost no MeSH, so a full-text corpus yields no queries on its own; the sidecar
    supplies descriptors borrowed from PubMed by PMID, for documents that have none.
    """
    from ..data.acquire import iter_documents

    sidecar: dict[str, list[str]] = {}
    if mesh_sidecar is not None:
        import gzip
        with gzip.open(mesh_sidecar, "rt") as fh:
            payload = json.load(fh)
        sidecar = payload.get("by_doc_id", {})
        log.info("mesh sidecar: %d documents labelled from %s", len(sidecar), mesh_sidecar)

    descriptor_docs: dict[str, list[str]] = defaultdict(list)
    descriptor_ui: dict[str, str] = {}
    n_docs = 0

    for doc in iter_documents(cfg):
        n_docs += 1
        names = {t.descriptor.strip() for t in doc.mesh_terms
                 if t.descriptor.strip() and t.descriptor.strip().lower() not in STOP_DESCRIPTORS}
        if not names and doc.doc_id in sidecar:
            names = {n for n in sidecar[doc.doc_id] if n.lower() not in STOP_DESCRIPTORS}
        if not names:
            continue
        for name in names:
            descriptor_docs[name].append(doc.doc_id)
        for term in doc.mesh_terms:
            name = term.descriptor.strip()
            if term.ui and name and not descriptor_ui.get(name):
                descriptor_ui[name] = term.ui
        if n_docs % 100_000 == 0:
            log.info("benchmark scan: %d documents, %d descriptors", n_docs, len(descriptor_docs))

    log.info("scanned %d documents, %d distinct MeSH descriptors", n_docs, len(descriptor_docs))
    if sidecar:
        # Sidecar descriptors carry no UI on the document; without one the tree root
        # falls back to name lookup, so take the UI the abstract records gave.
        for name, ui in payload.get("ui_by_descriptor", {}).items():
            descriptor_ui.setdefault(name, ui)

    def eligible(min_gold: int, max_gold: int) -> list[tuple[str, list[str]]]:
        out = []
        for name, ids in descriptor_docs.items():
            if not (min_gold <= len(ids) <= max_gold):
                continue
            root = _mesh_ui_tree_root(descriptor_ui.get(name, ""), name, taxonomy)
            if root in _GENERIC_TREE:
                continue
            out.append((name, ids))
        return out

    rng = random.Random(seed)
    clinical_pool = eligible(10, 300)
    rng.shuffle(clinical_pool)
    mesh_pool = eligible(8, 400)

    # ---- clinical: bare-topic queries, alternating phrasing
    clinical: list[Query] = []
    for name, ids in clinical_pool[:n_clinical]:
        topic = _clean_topic(name)
        if not topic:
            continue
        root = _mesh_ui_tree_root(descriptor_ui.get(name, ""), name, taxonomy)
        text = topic if len(clinical) % 2 == 0 else f"{topic} recent advances"
        clinical.append(Query(qid=f"clinical::{name}", text=text, relevant_docs=set(ids),
                              category=TREE_CATEGORIES.get(root, f"tree-{root or 'unknown'}"),
                              difficulty="single-hop", answer=topic))

    # ---- mesh: templated clinical questions over the same descriptors
    mesh: list[Query] = []
    for name, ids in mesh_pool:
        topic = _clean_topic(name)
        if not topic:
            continue
        root = _mesh_ui_tree_root(descriptor_ui.get(name, ""), name, taxonomy)
        for i in range(3):
            template = _TEMPLATES[rng.randrange(len(_TEMPLATES))]
            mesh.append(Query(qid=f"mesh::{name}::{i}", text=template.format(topic=topic),
                              relevant_docs=set(ids),
                              category=TREE_CATEGORIES.get(root, f"tree-{root or 'unknown'}"),
                              difficulty="single-hop", answer=topic))

    # ---- multi-hop: descriptor pairs sharing >= 4 documents
    # The pair scan is quadratic in descriptors; skipped when no multi-hop queries are
    # asked for. Clinical and mesh queries are drawn from ``rng`` before this point, so
    # they are identical either way.
    pairs: list[tuple[str, str, list[str]]] = []
    names_all = ([d for d, ids in descriptor_docs.items() if len(ids) <= 500]
                 if n_multihop > 0 else [])
    for i, a in enumerate(names_all):
        a_set = set(descriptor_docs[a])
        for b in names_all[i + 1 :]:
            shared = a_set.intersection(descriptor_docs[b])
            if len(shared) >= 4:
                pairs.append((a, b, sorted(shared)))
    rng.shuffle(pairs)
    multihop: list[Query] = []
    for a, b, shared in pairs[:n_multihop]:
        topic_a, topic_b = _clean_topic(a), _clean_topic(b)
        if not topic_a or not topic_b:
            continue
        text = rng.choice([
            f"What is the relationship between {topic_a} and {topic_b}?",
            f"How does {topic_a} influence outcomes in {topic_b}?",
            f"Are {topic_a} and {topic_b} associated, and what evidence supports this?",
        ])
        multihop.append(Query(qid=f"multihop::{a}::{b}", text=text, relevant_docs=set(shared),
                              supporting_docs=set(shared) | set(descriptor_docs[a]) | set(descriptor_docs[b]),
                              category="multi-hop", difficulty="multi-hop",
                              answer=f"{topic_a} and {topic_b}"))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    families = {"clinical": clinical, "multihop": multihop, "mesh": mesh}
    for name, queries in families.items():
        paths[name] = str(Qrels(queries).save(out_dir / f"qrels_{name}.json"))
    combined = Qrels(clinical + multihop)
    paths["combined"] = str(combined.save(out_dir / "qrels_combined.json"))

    def mean_gold(queries: list[Query]) -> float:
        return round(sum(len(q.relevant_docs) for q in queries) / max(1, len(queries)), 1)

    summary = {
        "n_documents": n_docs,
        "families": {name: len(q) for name, q in families.items()},
        "families_with_combined": {**{k: len(v) for k, v in families.items()},
                                   "combined": len(combined)},
        "gold_sizes": {"clinical_mean": mean_gold(clinical), "multihop_mean": mean_gold(multihop),
                       "mesh_mean": mean_gold(mesh)},
        "categories": {name: sorted({q.category for q in queries}) for name, queries in families.items()},
        "paths": paths,
        "seed": seed,
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("benchmark summary: %s", summary)
    return summary
