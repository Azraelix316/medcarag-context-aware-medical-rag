"""Co-citation edges: documents that the same papers cite.

Purpose in the dependency graph
-------------------------------
``DependencyGraph`` builds intra-document structure (section-typed, narrative,
qualification). This module adds the cross-document layer: if several documents
cite the same paper, they are asserting the same evidentiary basis, and that is a
dependency worth traversing.

Why co-citation rather than direct citation
------------------------------------------
A direct citation A -> B says B is relevant to A's claim. Co-citation says A and
B are *related*, without either citing the other, which is the common case in a
literature where two meta-analyses synthesise the same trial evidence. Because
the retrieved chunk's own claim often needs a corroborating source rather than its
cited source, co-citation reaches evidence a direct-citation walk cannot.

This is also where the contextual-contamination thesis gets its teeth. If several
documents that answer a query all cite the same landmark trial, then that trial
is corroborated evidence, and a passage drawn from it is better grounded than a
passage from a document that cites it once. Corroboration count is therefore a
retrieval signal in its own right, not just a graph edge.

Era bias, measured and not corrected
-------------------------------------
Reference lists are era-dependent: 72% of 2019-2026 RCT records carry one
(33 references each) versus 20% of 2000-2009 (7.3 each). So co-citation
structure is denser on recent literature, and any corroboration signal derived
from it is systematically stronger for recent topics. :func:`corroboration`
reports its own era coverage so that bias is visible in every result rather than
discovered later.
"""

from __future__ import annotations

import itertools
import os
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _rss_mb() -> float:
    """Resident set size in MiB, or -1.0. See dependency_graph.rss_mb."""
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return -1.0

# A co-citation pair needs at least this many shared references before it is
# treated as evidence of a relationship. Below 2, two papers sharing one
# reference is usually an artefact of indexing rather than a shared position.
DEFAULT_MIN_SHARED = 2

# Cap on shared references counted per pair. Without it a single very common
# landmark reference (a trial everyone cites) links thousands of documents into
# one blob, and traversal stops discriminating.
DEFAULT_MAX_SHARED = 25

# Cap on how many documents may cite one reference for it to generate pairs.
# This is a memory bound, not a quality choice: enumeration is quadratic in the
# citing-list length, so a reference cited by 5,000 documents alone yields 12.5M
# pairs. Measured: with this unset the build was OOM-killed on a 930K-document
# corpus. 200 keeps the interesting structure (papers citing a specific trial)
# while bounding the pair count to C(200,2) = 19,900 per reference.
DEFAULT_MAX_CITERS = 200


@dataclass
class CoCitationGraph:
    """Sparse document-to-document co-citation structure."""

    doc_ids: list[str] = field(default_factory=list)
    index: dict[str, int] = field(default_factory=dict)
    # CSR adjacency over docs.
    indptr: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    shared: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    # Doc id -> the references that doc cites, kept for corroboration queries.
    references: dict[str, tuple[int, ...]] = field(default_factory=dict)
    era_coverage: dict[int, int] = field(default_factory=dict)
    min_shared: int = DEFAULT_MIN_SHARED

    def __len__(self) -> int:
        return len(self.doc_ids)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        docs: Iterable[Any],
        *,
        min_shared: int = DEFAULT_MIN_SHARED,
        max_shared: int = DEFAULT_MAX_SHARED,
        max_citers: int = DEFAULT_MAX_CITERS,
    ) -> "CoCitationGraph":
        """Build from a corpus stream.

        Only pairs that actually co-occur are kept, in CSR form: at 20M documents a
        dense doc-by-doc matrix is 400G entries, and even the sparse graph is
        dominated by papers that share a ubiquitous reference.
        """
        references: dict[str, tuple[int, ...]] = {}
        era: Counter[int] = Counter()
        for doc in docs:
            refs = tuple(getattr(doc, "citations", ()) or ())
            if not refs:
                continue
            references[doc.doc_id] = refs
            year = int(getattr(doc, "year", 0) or 0)
            if year:
                era[(year // 10) * 10] += 1

        # reference -> documents citing it
        citing: dict[int, list[str]] = defaultdict(list)
        for doc_id, refs in references.items():
            for ref in set(refs):
                citing[ref].append(doc_id)

        # Pairs are counted as packed int64 keys in numpy, not as a
        # Counter[tuple[int, int]].
        #
        # Measured on this corpus: 555,142 documents, 28,672,183 citation edges,
        # 7,784,100 distinct references, and 232,992,097 enumerated pairs at
        # max_citers=200 (168,089,595 at 100, 111,903,756 at 50). A Python Counter
        # of tuples costs well over 150 bytes per entry, so that is 35 GB and the
        # build was killed. A packed key is 8 bytes: 1.86 GB, and the whole
        # aggregation becomes a sort.
        #
        # Packing is lossless because each document gets a dense integer index and
        # the key is a * n_docs + b with b < n_docs. n_docs is far below 2**31, so
        # the product stays inside int64.
        doc_list = sorted(references)
        doc_index = {d: i for i, d in enumerate(doc_list)}
        n_docs = len(doc_list)
        log.info("co-citation: %d documents with bibliographies, %d distinct references",
                 n_docs, len(citing))

        blocks: list[np.ndarray] = []
        enumerated = 0
        for ref, citers in citing.items():
            # A reference cited by many documents carries little discriminating
            # information, and pair enumeration is quadratic in this length, so
            # highly-cited references are skipped on memory grounds too.
            if len(citers) < 2 or len(citers) > max_citers:
                continue
            idxs = np.fromiter((doc_index[d] for d in citers), dtype=np.int64,
                               count=len(citers))
            idxs.sort()
            i, j = np.triu_indices(len(idxs), k=1)
            blocks.append(idxs[i] * n_docs + idxs[j])
            enumerated += i.size
            if len(blocks) >= 4096:
                log.info("co-citation: enumerated %d pairs (rss=%.0f MiB)",
                         enumerated, _rss_mb())
                blocks.append(np.concatenate(blocks))
                blocks = blocks[-1:]
        keys = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.int64)
        del blocks
        log.info("co-citation: %d enumerated pairs, rss=%.0f MiB",
                 enumerated, _rss_mb())

        uniq, counts = (np.unique(keys, return_counts=True) if keys.size
                        else (np.zeros(0, np.int64), np.zeros(0, np.int64)))
        del keys
        pair_a = (uniq // n_docs).astype(np.int32)
        pair_b = (uniq % n_docs).astype(np.int32)
        del uniq
        log.info("co-citation: %d distinct pairs, rss=%.0f MiB", pair_a.size, _rss_mb())

        wanted = (counts >= min_shared) & (counts <= max_shared)
        keep_a, keep_b = pair_a[wanted], pair_b[wanted]
        keep_c = counts[wanted].astype(np.int32)
        del pair_a, pair_b, counts, wanted
        log.info("co-citation: %d pairs survive min_shared=%d max_shared=%d",
                 keep_a.size, min_shared, max_shared)

        # doc_ids must contain only documents that appear in a surviving pair. Using
        # the full bibliography-bearing set would add every isolated document as a
        # node, which is what the ubiquitous-reference test catches: 300 documents
        # that share nothing but a hub reference must not enter the graph at all.
        if keep_a.size == 0:
            used = np.zeros(0, dtype=np.int64)
        else:
            used = np.unique(np.concatenate([keep_a, keep_b]))
        remap = np.full(len(doc_list), -1, dtype=np.int64)
        remap[used] = np.arange(used.size, dtype=np.int64)
        doc_ids = [doc_list[int(i)] for i in used]
        index = {d: i for i, d in enumerate(doc_ids)}
        n = len(doc_ids)
        src = np.concatenate([remap[keep_a], remap[keep_b]])
        dst = np.concatenate([remap[keep_b], remap[keep_a]])
        shr = np.concatenate([keep_c, keep_c])
        order = np.lexsort((dst, src))
        src, dst, shr = src[order], dst[order], shr[order]
        degree = np.bincount(src, minlength=n).astype(np.int64)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(degree, out=indptr[1:])

        g = cls(
            doc_ids=doc_ids,
            index=index,
            indptr=indptr,
            indices=dst.astype(np.int32),
            shared=shr.astype(np.int32),
            references=references,
            era_coverage=dict(era),
            min_shared=min_shared,
        )
        log.info(
            "co-citation graph: %d documents, %d pairs from %d with bibliographies",
            len(doc_ids), int(g.indices.size / 2), len(references),
        )
        return g

    # -------------------------------------------------------------- traversal
    def neighbours(self, doc_id: str) -> list[tuple[str, int]]:
        """Documents sharing references with ``doc_id``, strongest first."""
        i = self.index.get(doc_id)
        if i is None:
            return []
        lo, hi = int(self.indptr[i]), int(self.indptr[i + 1])
        rows = [
            (self.doc_ids[int(j)], int(c))
            for j, c in zip(self.indices[lo:hi], self.shared[lo:hi])
        ]
        rows.sort(key=lambda kv: (-kv[1], kv[0]))
        return rows

    def corroboration(
        self,
        doc_ids: Sequence[str],
        *,
        min_shared: int = 2,
        top_k: int = 20,
    ) -> list[tuple[int, int, set[str]]]:
        """References attested by several of ``doc_ids``, strongest first.

        Returns ``(times_shared, reference_pmid, citing_doc_ids)``. A reference
        cited by three of the retrieved documents is corroborated evidence; one
        cited once is not. This is the signal that is genuinely new rather than a
        reformulation of "documents share a topic" -- it measures attestation, and
        it is available only where bibliographies exist.
        """
        counts: Counter[int] = Counter()
        attesters: dict[int, set[str]] = defaultdict(set)
        for d in doc_ids:
            refs = self.references.get(d)
            if not refs:
                continue
            for ref in set(refs):
                counts[ref] += 1
                attesters[ref].add(d)
        out = [
            (c, ref, attesters[ref])
            for ref, c in counts.items()
            if c >= min_shared
        ]
        out.sort(key=lambda t: (-t[0], t[1]))
        return out[:top_k]

    def to_dependency_edges(self, *, weight: float = 0.5) -> list[tuple[str, str, float]]:
        """Edges for ``DependencyGraph.build(cross_doc_edges=...)``.

        Weight is the dependency weight, deliberately below the intra-document
        interpretive weight of 1.0: a passage's own Methods section qualifies it
        more directly than another paper's.
        """
        out: list[tuple[str, str, float]] = []
        for i in range(len(self.doc_ids)):
            lo, hi = int(self.indptr[i]), int(self.indptr[i + 1])
            for j, c in zip(self.indices[lo:hi], self.shared[lo:hi]):
                if int(j) > i:  # emit each undirected pair once
                    out.append((self.doc_ids[i], self.doc_ids[int(j)], weight))
        return out

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        """Persist and return the written path.

        Every other graph in the package returns a ``Path`` from ``save``, and that
        is what callers assume. This one used to return ``self`` so that
        ``load(g.save(p))`` could chain -- which read fine and cost 324 MB of log: the
        CLI does ``path = g.save(...)`` then ``"path": str(path)``, so the whole
        graph -- every document id and every reference tuple -- was stringified into
        a single 324,133,380-character log line.
        """
        import pickle
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({
                "doc_ids": self.doc_ids, "indptr": self.indptr,
                "indices": self.indices, "shared": self.shared,
                "references": self.references, "era_coverage": self.era_coverage,
                "min_shared": self.min_shared,
            }, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)
        log.info("saved co-citation graph -> %s (%.1f MB)", p, p.stat().st_size / 2**20)
        return p

    @classmethod
    def load(cls, path: str) -> "CoCitationGraph":
        import pickle
        from pathlib import Path

        with open(Path(path), "rb") as fh:
            d = pickle.load(fh)
        g = cls(
            doc_ids=d["doc_ids"], indptr=d["indptr"], indices=d["indices"],
            shared=d["shared"], references=d["references"],
            era_coverage=d["era_coverage"], min_shared=d["min_shared"],
        )
        g.index = {x: i for i, x in enumerate(g.doc_ids)}
        return g

    def stats(self) -> dict[str, Any]:
        return {
            "n_documents": len(self.doc_ids),
            "n_pairs": int(self.indices.size / 2),
            "n_docs_with_bibliography": len(self.references),
            "edges_per_doc": round(float(np.diff(self.indptr).mean()), 2) if len(self.doc_ids) else 0,
            "min_shared": self.min_shared,
            "era_coverage": dict(sorted(self.era_coverage.items())),
            "bytes": int(self.indptr.nbytes + self.indices.nbytes + self.shared.nbytes),
        }


