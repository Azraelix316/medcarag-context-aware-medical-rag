"""Corpus record types shared by every stage of the pipeline.

A ``Document`` is the unit of acquisition (one PubMed article or one PMC
article). A ``Chunk`` is the unit of retrieval and indexing. Both are plain
dataclasses with ``to_dict``/``from_dict`` so they round-trip through JSONL
without depending on a serialization library.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

# Publication types we keep by default; everything else is filtered upstream.
DEFAULT_KEEP_TYPES = (
    "Journal Article",
    "Review",
    "Systematic Review",
    "Meta-Analysis",
    "Randomized Controlled Trial",
    "Clinical Trial",
    "Comparative Study",
    "Multicenter Study",
    "Observational Study",
    "Case Reports",
    "Practice Guideline",
    "Guideline",
    "Preprint",
)


@dataclass
class Author:
    family: str = ""
    initials: str = ""
    collective: str = ""

    @property
    def display(self) -> str:
        if self.collective:
            return self.collective
        return f"{self.family} {self.initials}".strip()


@dataclass
class MeshTerm:
    descriptor: str = ""
    ui: str = ""
    major: bool = False
    qualifiers: tuple[str, ...] = ()


@dataclass
class Section:
    """A structural block of a document, e.g. "Methods" or "Results"."""

    title: str = ""
    text: str = ""
    # Hierarchy depth, 0 for the abstract or a top-level section.
    level: int = 0
    # Section label normalised to a small closed set when recognisable.
    kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Document:
    """A single medical literature record."""

    doc_id: str
    source: str = "pubmed"
    pmid: str = ""
    pmcid: str = ""
    doi: str = ""
    title: str = ""
    abstract: str = ""
    # Structured abstract labels, e.g. {"BACKGROUND": "...", "RESULTS": "..."}.
    sections: list[Section] = field(default_factory=list)
    authors: list[Author] = field(default_factory=list)
    journal: str = ""
    year: int = 0
    publication_date: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    issn: str = ""
    language: str = "english"
    publication_types: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    mesh_terms: tuple[MeshTerm, ...] = ()
    chemicals: tuple[str, ...] = ()
    grants: tuple[str, ...] = ()
    citations: tuple[int, ...] = ()
    # MeSH-derived topical facets used for stratified evaluation.
    categories: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = asdict(self)
        d["sections"] = [s.to_dict() for s in self.sections]
        d["publication_types"] = list(self.publication_types)
        d["keywords"] = list(self.keywords)
        d["mesh_terms"] = [asdict(m) for m in self.mesh_terms]
        d["mesh_terms"] = [
            {**m, "qualifiers": list(m.get("qualifiers", []))} for m in d["mesh_terms"]
        ]
        d["chemicals"] = list(self.chemicals)
        d["grants"] = list(self.grants)
        d["citations"] = list(self.citations)
        d["categories"] = list(self.categories)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        data = dict(data)
        data["sections"] = [Section(**{k: s[k] for k in Section.__dataclass_fields__ if k in s}) for s in data.get("sections", [])]
        data["authors"] = [Author(**a) if isinstance(a, dict) else Author() for a in data.get("authors", [])]
        data["mesh_terms"] = [
            MeshTerm(**{**m, "qualifiers": tuple(m.get("qualifiers", ()))})
            for m in data.get("mesh_terms", [])
        ]
        for key in (
            "publication_types",
            "keywords",
            "chemicals",
            "grants",
            "citations",
            "categories",
        ):
            if key in data and data[key] is not None:
                data[key] = tuple(data[key])
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in allowed})

    # ------------------------------------------------------------- derived
    @property
    def text(self) -> str:
        """Full plain text: title, abstract, then section bodies."""
        parts = [self.title.strip(), self.abstract.strip()]
        parts += [f"{s.title}\n{s.text}".strip() for s in self.sections]
        return "\n\n".join(p for p in parts if p)

    def dedup_key(self) -> str:
        """Stable identity for near-duplicate detection across sources."""
        basis = f"{self.doi.lower()}" if self.doi else f"{self.pmid or self.pmcid}"
        return basis or hashlib.sha1(self.title.strip().lower().encode()).hexdigest()


@dataclass
class Chunk:
    """A retrievable unit: one passage with enough context to stand alone."""

    chunk_id: str
    doc_id: str
    text: str
    # Ordinal position of this chunk within its parent document.
    chunk_index: int = 0
    # Section the chunk came from, used for citations and provenance.
    section_title: str = ""
    section_kind: str = ""
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0
    # Denormalised document fields, so retrieval never needs a document join.
    title: str = ""
    journal: str = ""
    year: int = 0
    pmid: str = ""
    pmcid: str = ""
    doi: str = ""
    publication_types: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    # Text actually fed to the embedder (may include title/section prefixes).
    embed_text: str = ""
    is_abstract: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["publication_types"] = list(self.publication_types)
        d["categories"] = list(self.categories)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        data = dict(data)
        for key in ("publication_types", "categories"):
            if key in data and data[key] is not None:
                data[key] = tuple(data[key])
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in allowed})

    @property
    def citation(self) -> str:
        """Human-readable provenance string attached to every generated answer."""
        venue = self.journal or ""
        year = str(self.year) if self.year else "n.d."
        ident = f"PMID:{self.pmid}" if self.pmid else (f"doi:{self.doi}" if self.doi else self.doc_id)
        tail = f" [{ident}]" if ident else ""
        return f"{self.title.strip()} ({venue}, {year}){tail}".strip()


@dataclass
class CorpusManifest:
    """Accounting for one corpus build, written next to the shards."""

    corpus_id: str = ""
    source: str = "pubmed"
    query: str = ""
    n_documents: int = 0
    n_chunks: int = 0
    n_bytes: int = 0
    shard_files: list[str] = field(default_factory=list)
    # Per-stage wall-clock seconds plus arbitrary per-stage stats, for the run report.
    timings: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    built_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
