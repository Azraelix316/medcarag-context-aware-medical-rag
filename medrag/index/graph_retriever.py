"""Graph-based retrieval integrated with the lexical and dense retrievers.

Why a separate module
--------------------
``MeshGraph`` is a document-level index; BM25 and the dense bi-encoder are
chunk-level. Fusing them requires an explicit bridge, which is what this module
provides:

1. :func:`build_doc_descriptor_map` scans the corpus once and records
   ``doc_id -> MeSH descriptor indices``. At 20M documents this is a few hundred
   megabytes of typed arrays rather than Python objects.
2. :class:`GraphRetriever` scores documents with the graph and then projects
   each document's score onto its chunks, so graph evidence participates in the
   same chunk-level fusion as the other two signals.
3. :func:`fuse_three` merges three independently ranked lists.

The projection is deliberately conservative: a document's graph score is shared
by its chunks, so the graph contributes a *document prior* rather than a passage
judgement. That is the correct semantics -- MeSH indexing is assigned at the
article level -- and it is why the graph helps as a complementary signal instead
of competing with the passage-level reranker.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..index.hybrid import Candidate
from ..index.mesh_graph import MeshGraph, iter_doc_descriptors

log = logging.getLogger(__name__)


class DocDescriptorMap:
    """``doc_id -> MeSH descriptor indices``, stored as typed arrays.

    Kept separate from the graph itself so a graph can be reused across corpus
    growth and so this map can be rebuilt without touching the PMI matrix.
    """

    def __init__(self) -> None:
        self.doc_ids: list[str] = []
        self._offsets: np.ndarray = np.zeros(1, dtype=np.int64)
        self._descriptors: np.ndarray = np.zeros(0, dtype=np.int32)
        self._index: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.doc_ids)

    def build(self, graph: MeshGraph, docs: Iterable[Any]) -> "DocDescriptorMap":
        rows: list[np.ndarray] = []
        offsets = [0]
        self.doc_ids = []
        self._index = {}
        n = 0
        for doc in docs:
            descs = sorted({graph.desc_index[t.descriptor] for t in getattr(doc, "mesh_terms", ())
                            if t.descriptor and t.descriptor in graph.desc_index})
            self.doc_ids.append(doc.doc_id)
            self._index[doc.doc_id] = n
            n += 1
            if descs:
                rows.append(np.asarray(descs, dtype=np.int32))
            offsets.append(offsets[-1] + len(descs))
            if n % 500_000 == 0:
                log.info("doc descriptor map: %d documents", n)
        self._descriptors = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int32)
        self._offsets = np.asarray(offsets, dtype=np.int64)
        log.info("doc descriptor map: %d documents, %d descriptor slots (%.1f MB)",
                 n, len(self._descriptors), self.nbytes / 2**20)
        return self

    def descriptors_for(self, doc_id: str) -> np.ndarray:
        i = self._index.get(doc_id)
        if i is None:
            return np.zeros(0, dtype=np.int32)
        return self._descriptors[self._offsets[i] : self._offsets[i + 1]]

    def as_dict(self, graph: MeshGraph) -> dict[str, list[int]]:
        """Materialise as ``doc_id -> descriptor indices`` for graph scoring."""
        out: dict[str, list[int]] = {}
        for i, doc_id in enumerate(self.doc_ids):
            descs = self._descriptors[self._offsets[i] : self._offsets[i + 1]]
            if len(descs):
                out[doc_id] = [int(d) for d in descs]
        return out

    @property
    def nbytes(self) -> int:
        return int(self._descriptors.nbytes + self._offsets.nbytes)

    def save(self, path: str | Path) -> Path:
        import json
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "doc_ids": self.doc_ids,
                    "offsets": self._offsets,
                    "descriptors": self._descriptors,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        log.info("saved doc descriptor map -> %s (%.1f MB)", path, path.stat().st_size / 2**20)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DocDescriptorMap":
        import pickle

        with open(Path(path), "rb") as fh:
            payload = pickle.load(fh)
        obj = cls()
        obj.doc_ids = payload["doc_ids"]
        obj._offsets = payload["offsets"]
        obj._descriptors = payload["descriptors"]
        obj._index = {d: i for i, d in enumerate(obj.doc_ids)}
        log.info("loaded doc descriptor map <- %s (%d documents)", path, len(obj.doc_ids))
        return obj


class GraphRetriever:
    """Third retrieval signal: expert-curated MeSH proximity."""

    def __init__(self, graph: MeshGraph, doc_map: DocDescriptorMap) -> None:
        self.graph = graph
        self.doc_map = doc_map
        # Cache of doc_id -> descriptor indices; built lazily and shared.
        self._descriptors: dict[str, list[int]] | None = None

    @property
    def descriptors(self) -> dict[str, list[int]]:
        if self._descriptors is None:
            self._descriptors = self.doc_map.as_dict(self.graph)
        return self._descriptors

    def search(self, query: str, top_k: int = 100, **kwargs: Any) -> list[tuple[str, float]]:
        return self.graph.search(query, self.descriptors, top_k=top_k, **kwargs)

    def plan(self, query: str, **kwargs: Any) -> Any:
        return self.graph.plan(query, **kwargs)


class InvertedGraphSearch:
    """GraphRetriever.search without materialising ``doc_id -> descriptors``.

    GraphRetriever scores every document in a Python loop over a dict of all of
    them: at 20M documents (abstracts20m) that is a 10+ GB dict and minutes per
    query. This keeps the descriptor -> documents inverse of the DocDescriptorMap
    arrays and scores only documents carrying a weighted descriptor. Each score is
    the same float64 sum in the same order (ascending descriptor index, as the map
    stores them, starting from 0.0), and the ranking the same (-score, doc_id) sort,
    so the hits are identical (tests/test_mesh_graph.py).
    """

    def __init__(self, graph: MeshGraph, doc_map: DocDescriptorMap) -> None:
        self.graph = graph
        self.doc_ids = doc_map.doc_ids
        offsets = doc_map._offsets
        n = len(self.doc_ids)
        # as_dict keeps, for a repeated doc_id, its last occurrence that has any
        # descriptors; documents without descriptors never score.
        valid = np.diff(offsets) > 0
        if len(doc_map._index) != n:
            last: dict[str, int] = {}
            for i in np.flatnonzero(valid).tolist():
                last[self.doc_ids[i]] = i
            valid[:] = False
            valid[np.fromiter(last.values(), dtype=np.int64, count=len(last))] = True
            del last
        doc_of = np.repeat(np.arange(n, dtype=np.int32), np.diff(offsets))
        keep = valid[doc_of]
        desc = doc_map._descriptors[keep]
        doc_of = doc_of[keep]
        order = np.argsort(desc, kind="stable")
        self._docs = doc_of[order]
        n_desc = max(len(graph.descriptors), int(desc.max()) + 1 if len(desc) else 0)
        self._ptr = np.zeros(n_desc + 1, dtype=np.int64)
        np.cumsum(np.bincount(desc, minlength=n_desc), out=self._ptr[1:])
        self._idf = np.log(1.0 + graph.n_docs / np.maximum(graph.df, 1)).astype(np.float32)

    def search(self, query: str, top_k: int = 100, **kwargs: Any) -> list[tuple[str, float]]:
        if len(self.graph.descriptors) == 0 or self.graph.n_docs == 0:
            return []
        weights = self.graph.query_weights(query, **kwargs)
        if weights is None:
            return []
        descs = sorted(d for d, w in weights.items() if w)
        if not descs:
            return []
        contrib = {d: weights[d] * float(self._idf[d]) for d in descs}
        docs = np.concatenate([self._docs[self._ptr[d]:self._ptr[d + 1]] for d in descs])
        vals = np.concatenate([np.full(self._ptr[d + 1] - self._ptr[d], contrib[d])
                               for d in descs])
        dkey = np.concatenate([np.full(self._ptr[d + 1] - self._ptr[d], d, dtype=np.int64)
                               for d in descs])
        if not len(docs):
            return []
        order = np.lexsort((dkey, docs))
        docs, vals = docs[order], vals[order]
        starts = np.flatnonzero(np.r_[True, docs[1:] != docs[:-1]])
        pos = np.arange(len(docs)) - np.repeat(starts, np.diff(np.r_[starts, len(docs)]))
        total = np.zeros(len(starts), dtype=np.float64)
        group = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(docs)]))
        for j in range(int(pos.max()) + 1):
            at = pos == j
            total[group[at]] += vals[at]
        uniq = docs[starts]
        hit = total > 0
        uniq, total = uniq[hit], total[hit]
        if len(total) > top_k:
            kth = np.partition(-total, top_k - 1)[top_k - 1]
            sel = -total <= kth
            uniq, total = uniq[sel], total[sel]
        ranked = sorted(((self.doc_ids[i], float(t)) for i, t in zip(uniq.tolist(), total.tolist())),
                        key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]


_DOC_ROWS_CACHE: dict[str, Any] = {}


def doc_to_chunks(store_rows: Sequence[str], ranked_docs: Sequence[tuple[str, float]],
                  *, per_doc: int = 3) -> list[tuple[str, float]]:
    """Project document-level graph scores onto chunk ids.

    ``store_rows`` is ``row_id -> chunk_id`` where ``chunk_id`` is
    ``"<doc_id>#<n>"``. Each ranked document contributes up to ``per_doc`` of its
    chunks, preserving the document's score so fusion sees a flat chunk ranking.

    The ``doc_id -> first per_doc rows`` map is built once per ``store_rows`` and
    reused: rebuilding it per query walked every row, about a minute per query at
    48M rows (abstracts20m).
    """
    key = (id(store_rows), len(store_rows), per_doc)
    if _DOC_ROWS_CACHE.get("key") != key:
        first: dict[str, list[int]] = {}
        for row_id, chunk_id in enumerate(store_rows):
            rows_ = first.setdefault(chunk_id.split("#", 1)[0], [])
            if len(rows_) < per_doc:
                rows_.append(row_id)
        _DOC_ROWS_CACHE.update(key=key, by_doc=first)
    by_doc: dict[str, list[int]] = _DOC_ROWS_CACHE["by_doc"]

    out: list[tuple[str, float]] = []
    for doc_id, score in ranked_docs:
        rows = by_doc.get(doc_id)
        if not rows:
            continue
        # Take this document's first few chunks (row order follows chunk order).
        for row_id in rows[:per_doc]:
            out.append((str(row_id), score))
    return out


def fuse_three(
    sparse: Sequence[tuple[str, float]],
    dense: Sequence[tuple[str, float]],
    graph: Sequence[tuple[str, float]],
    *,
    k: int = 60,
    top_k: int = 100,
    weights: dict[str, float] | None = None,
) -> list[Candidate]:
    """Reciprocal Rank Fusion over three independently ranked candidate lists.

    RRF is used rather than score blending because the three signals live on
    incomparable scales: BM25 is unbounded, cosine lives in [-1, 1], and graph
    scores are IDF-weighted descriptor overlaps. Rank fusion needs no
    calibration between them.
    """
    weights = weights or {"sparse": 1.0, "dense": 1.0, "graph": 1.0}
    lists = {"sparse": list(sparse), "dense": list(dense), "graph": list(graph)}
    acc: dict[str, Candidate] = {}

    for name, ranked in lists.items():
        w = weights.get(name, 1.0)
        for rank, (row_id, raw) in enumerate(ranked, start=1):
            cand = acc.get(row_id)
            if cand is None:
                cand = Candidate(chunk_id=row_id, sources=[])
                acc[row_id] = cand
            cand.score += w / (k + rank)
            setattr(cand, f"{name}_score", raw)
            setattr(cand, f"{name}_rank", rank)
            if name not in cand.sources:
                cand.sources.append(name)

    ordered = sorted(acc.values(), key=lambda c: (-c.score, c.chunk_id))
    return ordered[:top_k]


def build_graph_index(
    graph: MeshGraph,
    doc_map: DocDescriptorMap,
    index_dir: str | Path,
) -> dict[str, str]:
    """Persist both halves of the graph index."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph.save(index_dir / "mesh_graph.pkl")
    docmap_path = doc_map.save(index_dir / "mesh_docmap.pkl")
    return {"graph": str(graph_path), "doc_map": str(docmap_path)}


def load_graph_index(index_dir: str | Path) -> tuple[MeshGraph, DocDescriptorMap]:
    index_dir = Path(index_dir)
    graph = MeshGraph.load(index_dir / "mesh_graph.pkl")
    doc_map = DocDescriptorMap.load(index_dir / "mesh_docmap.pkl")
    return graph, doc_map


def iter_corpus_documents(cfg: Any) -> Iterator[Any]:
    from ..data.acquire import iter_documents

    return iter_documents(cfg)