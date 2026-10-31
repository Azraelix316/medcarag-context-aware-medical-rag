"""MeSH co-occurrence graph: an expert-curated third retrieval signal.

Motivation
----------
Hybrid BM25 + dense retrieval combines two *text-derived* signals: exact
terminology matching and distributional similarity. Both are learned from raw
prose. The biomedical literature additionally carries an expert-curated semantic
structure that neither can see: the MeSH thesaurus.

Every indexed article carries 11.8 MeSH descriptors on average, and 99.2% carry
at least three. Those descriptors are assigned by professional indexers, so
co-occurrence statistics over them encode clinical relationships -- that
"pneumonia" and "sepsis" co-occur far more than "pneumonia" and "photography" --
without any of it being inferable from surface text. This module turns that into
a third retriever, giving a three-way hybrid:

    lexical (BM25)  +  semantic (dense bi-encoder)  +  curated (MeSH graph)

Query expansion
---------------
Given a query, descriptors are resolved from its text, and the descriptor set is
expanded to statistically associated descriptors using pointwise mutual
information computed over the corpus. PMI is used rather than raw co-occurrence
because it discounts descriptors that are frequent in general: it ranks
"diabetes mellitus" alongside "hypoglycemia" rather than alongside "humans".

Graph construction
------------------
Co-occurrence is sparse: with ~12 descriptors per document, a 20M-document corpus
yields ~1.3G descriptor pairs, far too many to materialise densely. Only observed
pairs are kept, in a ``scipy.sparse`` matrix, which is the same argument that
justifies the CSC postings layout in ``bm25.py``.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")

# Default floor on descriptor document-frequency for a pair to be eligible for
# PMI expansion. Common descriptors co-occur with everything and add no
# discriminative power, so they are excluded from expansion (they remain usable
# as seeds). Overridable because the right floor depends on corpus size: 50 is
# sensible at 10M documents but would exclude every pair in a small fixture.
DEFAULT_MIN_DF_FOR_EXPANSION = 50
# Documents per co-occurrence batch in MeshGraph.build.
PAIR_BATCH_DOCS = 500_000


@dataclass
class MeshQueryExpansion:
    """The descriptor set a query maps onto, plus how each was reached."""

    seeds: list[str] = field(default_factory=list)
    expanded: list[tuple[str, float]] = field(default_factory=list)
    matched_text: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": self.seeds,
            "expanded": [{"descriptor": d, "pmi": round(p, 4)} for d, p in self.expanded],
            "matched_text": self.matched_text,
        }


class MeshGraph:
    """Sparse MeSH co-occurrence graph over the corpus.

    Build with :meth:`build`, persist with :meth:`save`, reuse with
    :meth:`load`. All heavy state is typed arrays, so a 20M-document graph stays
    in the low hundreds of megabytes.
    """

    def __init__(self, min_df_for_expansion: int = DEFAULT_MIN_DF_FOR_EXPANSION) -> None:
        self.min_df_for_expansion = max(1, int(min_df_for_expansion))
        self.descriptors: list[str] = []
        self.desc_index: dict[str, int] = {}
        self.df: np.ndarray = np.zeros(0, dtype=np.int64)      # document frequency
        self.n_docs: int = 0
        self._cooc: Any = None        # scipy.sparse.csr_matrix (n_desc x n_desc)
        self._pmi: Any = None         # same shape, PMI values
        self._label_index: dict[str, int] = {}   # normalised phrase -> desc idx

    # ------------------------------------------------------------------ build
    def build(
        self,
        doc_descriptors: Iterable[tuple[str, Sequence[str]]],
        *,
        min_pair_count: int = 3,
    ) -> "MeshGraph":
        """Build the graph from ``(doc_id, descriptors)`` pairs.

        ``min_pair_count`` prunes rare pairs. Below it a co-occurrence is
        usually an indexing coincidence rather than a clinical relationship, and
        keeping them would swamp the graph with one-off noise.
        """
        from scipy import sparse

        self.descriptors = []
        self.desc_index = {}
        per_doc: list[list[int]] = []
        df_counter: Counter[int] = Counter()
        n = 0

        for _doc_id, descriptors in doc_descriptors:
            ids = sorted({self._index(d) for d in descriptors if d and d.strip()})
            if not ids:
                per_doc.append([])
                n += 1
                continue
            per_doc.append(ids)
            df_counter.update(ids)
            n += 1

        self.n_docs = n
        k = len(self.descriptors)
        self.df = np.zeros(k, dtype=np.int64)
        for i, c in df_counter.items():
            self.df[i] = c

        # Co-occurrence from unordered pairs within each document, summed over
        # batches of documents. All pairs at once is ~1.3 billion at 20M documents
        # (abstracts20m) and was OOM-killed at 31.5 GB; the batched counts are the
        # same integers (each total < 2**24, so exact in float32).
        shape = (max(1, k), max(1, k))
        counts = sparse.csr_matrix(shape, dtype=np.float32)
        for start in range(0, len(per_doc), PAIR_BATCH_DOCS):
            rows: list[np.ndarray] = []
            cols: list[np.ndarray] = []
            for ids in per_doc[start:start + PAIR_BATCH_DOCS]:
                m = len(ids)
                if m < 2:
                    continue
                arr = np.asarray(ids, dtype=np.int64)
                rr, cc = np.triu_indices(m, k=1)
                rows.append(arr[rr])
                cols.append(arr[cc])
            if rows:
                r = np.concatenate(rows)
                c = np.concatenate(cols)
                del rows, cols
                counts = counts + sparse.coo_matrix(
                    (np.ones(len(r), dtype=np.float32), (r, c)), shape=shape).tocsr()
                del r, c
        counts = counts + counts.T          # symmetrise
        counts.setdiag(0)
        counts.eliminate_zeros()

        if min_pair_count > 1:
            counts.data[counts.data < min_pair_count] = 0
            counts.eliminate_zeros()

        self._cooc = counts.astype(np.float32)
        self._pmi = self._compute_pmi()
        self._build_label_index()
        log.info(
            "mesh graph: %d docs, %d descriptors, %d co-occurrence pairs (%.1f MB)",
            self.n_docs, k, int(self._cooc.nnz / 2), self._cooc.data.nbytes / 2**20,
        )
        return self

    def _index(self, descriptor: str) -> int:
        idx = self.desc_index.get(descriptor)
        if idx is None:
            idx = len(self.descriptors)
            self.desc_index[descriptor] = idx
            self.descriptors.append(descriptor)
        return idx

    def _compute_pmi(self) -> Any:
        """Pointwise mutual information over the symmetric co-occurrence matrix.

        PMI(a,b) = log( P(a,b) / (P(a) P(b)) ), with the standard +1 smoothing
        implied by using corpus counts. Pairs above ``_MIN_DF_FOR_EXPANSION`` are
        masked out so that ubiquitous descriptors do not dominate.
        """
        from scipy import sparse

        k = len(self.descriptors)
        if k == 0 or self.n_docs == 0:
            return sparse.csr_matrix((max(1, k), max(1, k)), dtype=np.float32)
        n = float(self.n_docs)
        cooc = self._cooc.tocoo()
        pa = self.df[cooc.row].astype(np.float64) / n
        pb = self.df[cooc.col].astype(np.float64) / n
        pab = cooc.data.astype(np.float64) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(pab / np.maximum(pa * pb, 1e-12))
        # Mask pairs involving a very common descriptor: they are associated but
        # not discriminative, and would otherwise flood the candidate set.
        mask = (self.df[cooc.row] >= self.min_df_for_expansion) & (
            self.df[cooc.col] >= self.min_df_for_expansion
        )
        pmi = np.where(np.isfinite(pmi), pmi, 0.0).astype(np.float32)
        pmi[~mask] = 0.0
        pmi[pmi <= 0] = 0.0
        return sparse.coo_matrix((pmi, (cooc.row, cooc.col)), shape=(k, k)).tocsr()

    def _build_label_index(self) -> None:
        """Map normalised descriptor phrases to indices, for query resolution.

        Each descriptor contributes several keys so that ordinary user phrasing
        resolves against curated MeSH wording:

        * the normalised form ("diabetic nephropathies")
        * the comma-inverted natural form ("type 2 diabetes mellitus")
        * the light-stemmed form of each ("diabetic nephropathi"), because MeSH
          headings are plural and verb forms ("Diabetes Complications",
          "Diabetic Neuropathies") while users type the singular
          ("diabetic nephropathy"). Without stemming, the most common clinical
          phrasing fails to resolve at all.

        Hyphens are stripped by :meth:`_normalise`, which is what lets
        "Pneumonia, Ventilator-Associated" match "ventilator associated
        pneumonia".
        """
        self._label_index = {}
        for idx, descriptor in enumerate(self.descriptors):
            keys = {self._normalise(descriptor), self._normalise(self._invert(descriptor))}
            for key in list(keys):
                stemmed = self._stem_phrase(key)
                if stemmed:
                    keys.add(stemmed)
            for key in keys:
                if len(key) >= 4:
                    self._label_index.setdefault(key, idx)

    @staticmethod
    def _stem_phrase(phrase: str) -> str:
        from .bm25 import _light_stem

        return " ".join(_light_stem(t) for t in phrase.split())

    @staticmethod
    def _normalise(text: str) -> str:
        return " ".join(_WORD_RE.findall(text.lower()))

    @staticmethod
    def _invert(descriptor: str) -> str:
        """'Diabetes Mellitus, Type 2' -> 'type 2 diabetes mellitus'."""
        if "," in descriptor:
            head, _, qualifier = descriptor.partition(",")
            return f"{qualifier.strip()} {head.strip()}".lower()
        return descriptor.lower()

    # ------------------------------------------------------------------ query
    def resolve_descriptors(self, text: str) -> list[str]:
        """Map free text onto corpus descriptors.

        Tries the whole normalised phrase first, then falls back to individual
        content tokens so that "metformin in renal impairment" still reaches
        "Metformin" and "Renal Impairment".
        """
        found: list[str] = []
        seen: set[str] = set()
        normalised = self._normalise(text)

        def add(desc: str) -> None:
            if desc not in seen:
                seen.add(desc)
                found.append(desc)

        # Longest phrases first so a specific descriptor wins over its tokens.
        tokens = normalised.split()
        for size in (6, 5, 4, 3, 2):
            for i in range(len(tokens) - size + 1):
                phrase = " ".join(tokens[i : i + size])
                idx = self._label_index.get(phrase)
                if idx is not None:
                    add(self.descriptors[idx])
                # Retry stemmed, so "diabetic nephropathy" meets
                # "Diabetic Nephropathies".
                stemmed = self._stem_phrase(phrase)
                if stemmed != phrase:
                    idx = self._label_index.get(stemmed)
                    if idx is not None:
                        add(self.descriptors[idx])
        for token in tokens:
            if len(token) >= 5:
                idx = self._label_index.get(token) or self._label_index.get(self._stem_phrase(token))
                if idx is not None:
                    add(self.descriptors[idx])
        return found

    def expand(
        self,
        seeds: Sequence[str],
        *,
        top_k: int = 12,
        min_pmi: float = 1.0,
    ) -> list[tuple[str, float]]:
        """Descriptors most strongly associated with ``seeds``, by PMI.

        Aggregates over multiple seeds by summing PMI, so a descriptor
        associated with two seeds outranks one associated with a single seed.
        """
        if self._pmi is None or not seeds:
            return []
        from scipy import sparse

        idxs = [self.desc_index[s] for s in seeds if s in self.desc_index]
        if not idxs:
            return []
        agg = np.zeros(len(self.descriptors), dtype=np.float32)
        for i in idxs:
            agg += np.asarray(self._pmi[i].todense()).ravel()
        for i in idxs:
            agg[i] = 0.0
        candidates = np.flatnonzero(agg > 0)
        if candidates.size == 0:
            return []
        scored = candidates[np.argsort(-agg[candidates])][:top_k]
        return [(self.descriptors[i], float(agg[i])) for i in scored if agg[i] >= min_pmi]

    def plan(self, query: str, *, top_k: int = 12, min_pmi: float = 1.0) -> MeshQueryExpansion:
        """Resolve and expand a query in one step."""
        seeds = self.resolve_descriptors(query)
        matched: dict[str, list[str]] = {d: [query] for d in seeds}
        expanded = self.expand(seeds, top_k=top_k, min_pmi=min_pmi) if seeds else []
        for descriptor, _pmi in expanded:
            matched.setdefault(descriptor, []).append("pmi-expansion")
        return MeshQueryExpansion(seeds=seeds, expanded=expanded, matched_text=matched)

    # ------------------------------------------------------------------ search
    def score_documents(
        self,
        descriptors: dict[str, list[int]],
        weights: dict[str, float],
        *,
        idf_power: float = 1.0,
    ) -> dict[str, float]:
        """Score documents by IDF-weighted coverage of a descriptor set.

        ``descriptors`` maps doc_id -> descriptor indices (the ``doc_descriptors``
        side of the graph). Weights are per-descriptor importance, so an expanded
        descriptor contributes less than a seed that the user actually typed.
        """
        n = len(self.descriptors)
        if n == 0 or self.n_docs == 0:
            return {}
        idf = np.log(1.0 + self.n_docs / np.maximum(self.df, 1)).astype(np.float32) ** idf_power
        scores: dict[str, float] = {}
        for doc_id, descs in descriptors.items():
            total = 0.0
            for d in descs:
                w = weights.get(d)
                if w:
                    total += w * float(idf[d])
            if total > 0:
                scores[doc_id] = total
        return scores

    def query_weights(self, query: str, *, expand_top_k: int = 12, min_pmi: float = 1.0,
                      seed_weight: float = 1.0, expansion_weight: float = 0.35,
                      ) -> dict[int, float] | None:
        """Per-descriptor weights search() scores with; None when no seed resolves."""
        expansion = self.plan(query, top_k=expand_top_k, min_pmi=min_pmi)
        if not expansion.seeds:
            return None
        weights: dict[int, float] = {}
        for d in expansion.seeds:
            weights[self.desc_index[d]] = seed_weight
        for d, pmi in expansion.expanded:
            # Diminishing return with depth, so the top association dominates.
            weights[self.desc_index[d]] = expansion_weight * min(1.0, pmi / 6.0)
        return weights

    def search(
        self,
        query: str,
        doc_descriptors: dict[str, list[int]],
        *,
        top_k: int = 50,
        expand_top_k: int = 12,
        min_pmi: float = 1.0,
        seed_weight: float = 1.0,
        expansion_weight: float = 0.35,
    ) -> list[tuple[str, float]]:
        """Third-way retrieval: rank documents by curated MeSH overlap."""
        weights = self.query_weights(query, expand_top_k=expand_top_k, min_pmi=min_pmi,
                                     seed_weight=seed_weight,
                                     expansion_weight=expansion_weight)
        if weights is None:
            return []
        scored = self.score_documents(doc_descriptors, weights)
        ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        from scipy import sparse

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "descriptors": self.descriptors,
            "n_docs": self.n_docs,
            "min_df_for_expansion": self.min_df_for_expansion,
            "df": self.df,
            "cooc_data": self._cooc.data if self._cooc is not None else np.zeros(0, np.float32),
            "cooc_indices": self._cooc.indices if self._cooc is not None else np.zeros(0, np.int64),
            "cooc_indptr": self._cooc.indptr if self._cooc is not None else np.zeros(1, np.int64),
            "cooc_shape": self._cooc.shape if self._cooc is not None else (0, 0),
            "pmi_data": self._pmi.data if self._pmi is not None else np.zeros(0, np.float32),
            "pmi_indices": self._pmi.indices if self._pmi is not None else np.zeros(0, np.int64),
            "pmi_indptr": self._pmi.indptr if self._pmi is not None else np.zeros(1, np.int64),
            "pmi_shape": self._pmi.shape if self._pmi is not None else (0, 0),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            import pickle

            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        log.info("saved mesh graph -> %s (%.1f MB)", path, path.stat().st_size / 2**20)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MeshGraph":
        import pickle

        from scipy import sparse

        path = Path(path)
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        obj = cls(min_df_for_expansion=payload.get("min_df_for_expansion", DEFAULT_MIN_DF_FOR_EXPANSION))
        obj.descriptors = payload["descriptors"]
        obj.desc_index = {d: i for i, d in enumerate(obj.descriptors)}
        obj.n_docs = payload["n_docs"]
        obj.df = payload["df"]
        obj._cooc = sparse.csr_matrix(
            (payload["cooc_data"], payload["cooc_indices"], payload["cooc_indptr"]),
            shape=payload["cooc_shape"],
        )
        obj._pmi = sparse.csr_matrix(
            (payload["pmi_data"], payload["pmi_indices"], payload["pmi_indptr"]),
            shape=payload["pmi_shape"],
        )
        obj._build_label_index()
        log.info("loaded mesh graph <- %s (%d descriptors, %d docs)", path, len(obj.descriptors), obj.n_docs)
        return obj

    def stats(self) -> dict[str, Any]:
        return {
            "n_docs": self.n_docs,
            "n_descriptors": len(self.descriptors),
            "n_cooc_pairs": int(self._cooc.nnz / 2) if self._cooc is not None else 0,
            "n_pmi_pairs": int(self._pmi.nnz / 2) if self._pmi is not None else 0,
            "mean_df": round(float(self.df.mean()), 1) if len(self.df) else 0,
            "bytes": int(
                (self._cooc.data.nbytes + self._cooc.indices.nbytes if self._cooc is not None else 0)
                + (self._pmi.data.nbytes + self._pmi.indices.nbytes if self._pmi is not None else 0)
            ),
        }


def iter_doc_descriptors(
    docs: Iterable[Any],
) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(doc_id, descriptors)`` from Document-like objects."""
    for doc in docs:
        descriptors = [t.descriptor for t in getattr(doc, "mesh_terms", ()) if t.descriptor]
        if descriptors:
            yield doc.doc_id, descriptors


def top_associated(graph: MeshGraph, descriptor: str, top_k: int = 15) -> list[tuple[str, float]]:
    """Descriptors most associated with ``descriptor``; used for graph reports."""
    return graph.expand([descriptor], top_k=top_k, min_pmi=0.5)