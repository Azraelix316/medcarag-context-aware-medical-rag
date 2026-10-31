"""Section-dependency graph over chunks.

Why this exists
---------------
Chunk-level retrieval in biomedical RAG fails by returning *unqualified* facts.
arXiv 2502.14898 documents this empirically against deployed systems: only 4-5%
of responses mentioned complication rates and 6-10% mentioned procedure benefits,
and a disputed diagnosis's contested status was surfaced 100% of the time by a
direct query but only 56-69% of the time by a presupposing one. The facts were
present in the cited sources; retrieval simply never surfaced the passage that
qualifies them.

Our own data shows the same shape. On a 30,000-chunk sample of the production
store, sections are structured and lopsided:

    background 34.5%   results 18.7%   methods 18.4%   conclusions 18.0%

So a lexical query about an outcome ("what happened in the trial") matches
Results, while the qualifiers that make Results interpretable -- the population
and comparator in Methods, the rate and caveats in Conclusions -- live in
different chunks that never match. The evidence for "this complication is rare"
and the evidence for "this complication happened" are indexed as unrelated
passages.

This module makes those dependencies explicit and traversable, giving a third
retrieval signal that is neither lexical (BM25), distributional (dense), nor
ontology-level (MeSH): **structural**.

What the edges mean
-------------------
Medical documents have a conventional argument order, and readers lean on it.
Three edge types, in decreasing strength:

``interpretive`` (Methods <-> Results <-> Conclusions)
    A result is not interpretable without the population and comparator that
    produced it; a stated rate is not trustworthy without its denominator. These
    are the edges that matter for the decontextualization failure.

``narrative`` (Background -> Methods -> Results -> Conclusions)
    Document order. Background states the question, Methods the approach,
    Results the findings, Conclusions the interpretation.

``qualification`` (anything -> Conclusions/Discussion)
    The passage that hedges, qualifies, or limits is usually in the conclusion.
    This is the edge that reaches the paper's "benefits and risks" failure mode.

Distinct from prior work
-----------------------
Personalised PageRank over a graph is long-established (Haveliwala 2002), and
citation-graph retrieval is well trodden. What is specific here is that the edge
*weights are derived from medical document structure* -- section role and
argument position -- rather than from citation links or random-walk preferences.
The graph is a claim about how a clinical paper is built, not about how often
papers cite each other.

Cost
----
Building is one streaming pass over the chunk store and costs O(chunks). Traversal
is bounded by ``neighbour_depth`` and a per-node fan-out cap, so query time does
not grow with graph size.
"""

from __future__ import annotations

import logging
import math
import os
import time
from array import array
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def rss_mb() -> float:
    """Resident set size in MiB, or -1.0 when it cannot be read.

    Present so that a build killed by the OOM killer leaves evidence. The kernel
    writes nothing to a dying process's own log, so without this the only symptom
    of an out-of-memory build is a progress line that stops and a process that is
    gone -- which is how a 57-minute dependency-graph build over 16.1M chunks
    vanished with nothing in its log.
    """
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return -1.0

# Canonical argument order. Unknown kinds sort first so a document with only
# unlabelled chunks still produces a monotone chain.
SECTION_ORDER: tuple[str, ...] = (
    "background",
    "objectives",
    "methods",
    "results",
    "discussion",
    "conclusions",
)
_SECTION_RANK = {kind: i for i, kind in enumerate(SECTION_ORDER)}

# Edge weights. Interpretive edges are worth more than narrative ones because
# they are the ones whose absence causes a fact to be read as unconditional.
W_INTERPRETIVE = 1.0
W_NARRATIVE = 0.6
W_QUALIFICATION = 0.9

# Pairings that define an interpretive dependency.
INTERPRETIVE_PAIRS: tuple[tuple[str, str], ...] = (
    ("methods", "results"),      # no result without the population that produced it
    ("results", "conclusions"),  # no finding without its stated interpretation
    ("methods", "conclusions"),  # no conclusion without knowing what was done
    ("discussion", "conclusions"),
)


def section_rank(kind: str) -> int:
    """Position of a section in the argument; unknown kinds sort first."""
    return _SECTION_RANK.get(kind, -1)


@dataclass
class DependencyGraph:
    """Chunk-level dependency structure.

    Stored as typed arrays rather than Python dicts: at 20M documents and ~76M
    chunks, a dict-of-lists per node would cost hundreds of bytes per edge in
    interpreter overhead alone.

    ``indptr``/``indices`` are a CSR-style adjacency over chunks; ``weights`` are
    parallel. ``doc_start``/``doc_end`` give each document's contiguous row range
    so document-level operations need no lookup table.
    """

    n_nodes: int = 0
    indptr: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    section_kind: list[str] = field(default_factory=list)
    doc_start: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    doc_end: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    doc_ids: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        rows: Sequence[str],
        *,
        kind_of: Any,
        doc_of: Any,
        intra_doc_edge: bool = True,
        cross_doc_max_docs: int = 0,
        cross_doc_edges: Sequence[tuple[int, int, float]] = (),
        progress_every: int = 0,
    ) -> "DependencyGraph":
        """Build the graph from chunk metadata.

        ``rows`` is the store's row-id list, so row index == node id. ``kind_of``
        and ``doc_of`` are callables row -> section kind / doc id (a chunk object
        accessor, so the store is streamed once and not held in memory).

        ``cross_doc_edges`` are doc-to-doc dependencies supplied by the caller,
        e.g. from citations. They are attached to each document's first chunk so
        traversal can leave a document, which is how a Methods chunk reaches a
        citing paper that qualifies it.
        """
        n = len(rows)
        kinds: list[str] = []
        docs: list[str] = []
        # Per-document chunk lists, kept as index ranges once contiguous.
        by_doc: dict[str, list[int]] = defaultdict(list)
        for row in range(n):
            k = kind_of(row) or ""
            d = doc_of(row) or ""
            kinds.append(k)
            docs.append(d)
            by_doc[d].append(row)
            if row % 2_000_000 == 0 and row:
                log.info("dependency-graph: indexed %d/%d rows, rss=%.0f MiB",
                         row, n, rss_mb())
        log.info("dependency-graph: %d documents, rss=%.0f MiB", len(by_doc), rss_mb())

        # Edges accumulate into typed arrays, not a dict of lists of tuples.
        #
        # The dict form was OOM-killed twice on the 16.1M-chunk PMC corpus: RSS went
        # from 6.1 GiB to 21.9 GiB inside the edge-construction phase and the kernel
        # killed it (dmesg: anon-rss 21870008 kB). A dict[int, list[tuple[int, float]]]
        # over ~16.1M nodes and tens of millions of directed edges costs well over
        # 100 bytes per edge once the tuple, the two list slots and the boxed row id
        # are counted -- and every byte of it is thrown away, because _finalize
        # immediately converts the whole thing to CSR numpy arrays.
        #
        # array('i') and array('f') store raw 4-byte values, so the same edge set is
        # 12 bytes per directed edge. The strongest-edge-per-neighbour dedup that
        # _finalize performed in Python now happens vectorised in numpy after the
        # arrays are built.
        edge_src = array("i")
        edge_dst = array("i")
        edge_w = array("f")

        def add(a: int, b: int, w: float) -> None:
            if a == b:
                return
            # Both directions. The graph is undirected and traversal uses
            # `neighbours`, so storing only a->b leaves every edge one-way: the
            # array rewrite above did exactly that, and the asymmetry test caught
            # 250 of 250 pairs unreciprocated.
            edge_src.append(a)
            edge_dst.append(b)
            edge_w.append(w)
            edge_src.append(b)
            edge_dst.append(a)
            edge_w.append(w)

        # --- intra-document structure
        for doc, chunk_rows in by_doc.items():
            chunk_rows.sort()
            ks = [kinds[r] for r in chunk_rows]
            ranks = [section_rank(k) for k in ks]
            n_chunks = len(chunk_rows)

            for i in range(n_chunks):
                ri = chunk_rows[i]
                ki = ks[i]
                if progress_every and ri and ri % progress_every == 0:
                    log.info("dependency-graph: edges at row %d/%d, %d directed, rss=%.0f MiB",
                             ri, n, len(edge_src), rss_mb())

                # Interpretive: same document, paired section roles.
                for a_kind, b_kind in INTERPRETIVE_PAIRS:
                    if ki not in (a_kind, b_kind):
                        continue
                    other = b_kind if ki == a_kind else a_kind
                    for j in range(n_chunks):
                        if j == i or ks[j] != other:
                            continue
                        # Only the nearest chunk of the paired role, else a long
                        # methods section creates a near-complete graph.
                        add(ri, chunk_rows[j], W_INTERPRETIVE)
                        break

                # Qualification: everything reaches the interpretive summary.
                if ki in ("conclusions", "discussion"):
                    for j in range(n_chunks):
                        if j != i and ks[j] in ("results", "methods"):
                            add(ri, chunk_rows[j], W_QUALIFICATION)
                else:
                    for j in range(n_chunks):
                        if ks[j] in ("conclusions", "discussion"):
                            add(ri, chunk_rows[j], W_QUALIFICATION)
                            break

                # Narrative: document order, adjacent chunks only, and only
                # forward along a non-decreasing argument order. Connecting
                # Background->Results directly would be wrong: Methods is what
                # sits between them.
                if intra_doc_edge:
                    if i + 1 < n_chunks and ranks[i + 1] >= ranks[i]:
                        add(ri, chunk_rows[i + 1], W_NARRATIVE)
                    if i > 0 and ranks[i - 1] <= ranks[i]:
                        add(ri, chunk_rows[i - 1], W_NARRATIVE)

        log.info("dependency-graph: %d directed edges built, rss=%.0f MiB",
                 len(edge_src), rss_mb())

        # --- optional cross-document edges
        if cross_doc_edges:
            for src_doc, dst_doc, w in list(cross_doc_edges)[:cross_doc_max_docs or None]:
                s_rows = by_doc.get(src_doc)
                d_rows = by_doc.get(dst_doc)
                if not s_rows or not d_rows:
                    continue
                # Anchor on each document's conclusion chunk when it has one, so
                # traversal lands on the passage that carries the framing.
                for rows_ in (s_rows, d_rows):
                    concl = [r for r in rows_ if kinds[r] in ("conclusions", "discussion")]
                    add(concl[0] if concl else rows_[0], d_rows[0], w)

        return cls._finalize(n, kinds, docs, by_doc, edge_src, edge_dst, edge_w)

    @classmethod
    def _finalize(
        cls,
        n: int,
        kinds: list[str],
        docs: list[str],
        by_doc: dict[str, list[int]],
        edge_src: array,
        edge_dst: array,
        edge_w: array,
    ) -> "DependencyGraph":
        log.info("dependency-graph: finalising %d nodes, %d directed edges, rss=%.0f MiB",
                 n, len(edge_src), rss_mb())

        # Document extents first, so the per-document row lists -- about 4 GB at 20M
        # documents -- can be released before the edge arrays are sorted.
        doc_list = sorted(by_doc)
        doc_start = np.zeros(len(doc_list), dtype=np.int64)
        doc_end = np.zeros(len(doc_list), dtype=np.int64)
        for i, d in enumerate(doc_list):
            rs = by_doc[d]
            doc_start[i], doc_end[i] = min(rs), max(rs) + 1
        by_doc.clear()

        # Strongest edge per (src, dst) pair, vectorised. The previous per-node
        # Python dict did this one node at a time, which is what made the whole
        # structure worth 22 GiB before it was thrown away.
        #
        # One int64 key per edge, src * n + dst, sorts in exactly (src, dst) order, so
        # it replaces the lexsort over separate int64 src and dst arrays. At 220M
        # directed edges (abstracts20m) the lexsort form held five 1.8 GB arrays at
        # once on top of the graph and was OOM-killed in this step; the input arrays
        # are also emptied as soon as they are copied out.
        if len(edge_src):
            key = np.frombuffer(edge_src, dtype=np.int32).astype(np.int64) * n
            key += np.frombuffer(edge_dst, dtype=np.int32)
            wts = np.frombuffer(edge_w, dtype=np.float32).copy()
            del edge_src[:], edge_dst[:], edge_w[:]
            order = np.argsort(key, kind="stable")
            key = key[order]
            wts = wts[order]
            del order
            first = np.empty(len(key), dtype=bool)
            first[0] = True
            np.not_equal(key[1:], key[:-1], out=first[1:])
            group = np.cumsum(first) - 1
            g_key = key[first]
            del key, first
            g_w = np.zeros(len(g_key), dtype=np.float32)
            np.maximum.at(g_w, group, wts)
            del wts, group
            g_src = g_key // n
            indices = (g_key - g_src * n).astype(np.int32)
            del g_key
            degree = np.bincount(g_src, minlength=n).astype(np.int64)
            del g_src
            indptr = np.zeros(n + 1, dtype=np.int64)
            np.cumsum(degree, out=indptr[1:])
            weights = g_w
        else:
            indptr = np.zeros(n + 1, dtype=np.int64)
            indices = np.zeros(0, dtype=np.int32)
            weights = np.zeros(0, dtype=np.float32)

        g = cls(
            n_nodes=n,
            indptr=indptr,
            indices=indices,
            weights=weights,
            section_kind=kinds,
            doc_start=doc_start,
            doc_end=doc_end,
            doc_ids=doc_list,
        )
        log.info(
            "dependency graph: %d nodes, %d edges (%.2f/node), %d documents",
            n, len(indices) // 2, len(indices) / max(1, n), len(doc_list),
        )
        return g

    # ------------------------------------------------------------- properties
    def degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    def neighbours(self, node: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = int(self.indptr[node]), int(self.indptr[node + 1])
        return self.indices[lo:hi], self.weights[lo:hi]

    def degree_by_kind(self) -> dict[str, float]:
        kinds = np.asarray(self.section_kind)
        deg = self.degree()
        return {k: float(deg[kinds == k].mean()) for k in set(self.section_kind)}

    # -------------------------------------------------------------- traversal
    def expand(
        self,
        seeds: Sequence[int],
        weights: dict[int, float] | None = None,
        *,
        depth: int = 1,
        top_k: int = 200,
        max_per_seed: int = 8,
        damping: float = 0.85,
    ) -> list[tuple[int, float]]:
        """Walk the dependency graph outward from seed chunks.

        Score propagation is damped by depth so a direct neighbour of a seed
        outranks a two-hop dependency, which is what keeps "the passage that
        qualifies this one" reachable without letting the graph flood the results
        with loosely related chunks.

        ``max_per_seed`` bounds fan-out per seed; without it one hub chunk (a
        40-chunk Results section) would dominate the expansion.
        """
        weights = weights or {}
        base: dict[int, float] = {}
        for seed in seeds:
            base[seed] = max(base.get(seed, 0.0), weights.get(seed, 1.0))
        frontier = {s: base[s] for s in base}
        seen: dict[int, float] = dict(base)

        for _ in range(depth):
            nxt: dict[int, float] = {}
            for node, score in frontier.items():
                idx, w = self.neighbours(node)
                if len(idx) == 0:
                    continue
                order = np.argsort(-w)[:max_per_seed]
                for j in order:
                    nb = int(idx[j])
                    gain = score * float(w[j]) * damping
                    if gain <= 0:
                        continue
                    if nb not in seen or seen[nb] < gain:
                        seen[nb] = max(seen.get(nb, 0.0), gain)
                        nxt[nb] = max(nxt.get(nb, 0.0), gain)
            if not nxt:
                break
            frontier = nxt

        ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(n, s) for n, s in ranked if n not in base][:top_k]

    def context_pairs(
        self,
        seeds: Sequence[int],
        *,
        top_k: int = 10,
        max_per_seed: int = 6,
    ) -> list[tuple[int, int, float]]:
        """Interpreting (seed, context) pairs -- the passage plus what qualifies it.

        This is the retrieval shape the decontextualization failure calls for:
        return the matching passage *and* its interpretive anchor, so a generator
        cannot read an unqualified fact. Pairs are returned rather than a flat
        list because the pairing is the point -- a context chunk with no seed to
        qualify is just noise.
        """
        seed_set = set(seeds)
        out: list[tuple[int, int, float]] = []
        for seed in seeds:
            idx, w = self.neighbours(seed)
            if len(idx) == 0:
                continue
            order = np.argsort(-w)[:max_per_seed]
            for j in order:
                nb = int(idx[j])
                if nb in seed_set:
                    continue
                # A conclusion qualifying a result is the highest-value edge we
                # can return; both directions carry the same weight, so accept
                # either orientation.
                out.append((seed, nb, float(w[j])))
        out.sort(key=lambda t: -t[2])
        return out[:top_k]

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        import json
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(
                {
                    "n_nodes": self.n_nodes,
                    "indptr": self.indptr,
                    "indices": self.indices,
                    "weights": self.weights,
                    "section_kind": self.section_kind,
                    "doc_start": self.doc_start,
                    "doc_end": self.doc_end,
                    "doc_ids": self.doc_ids,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp.replace(path)
        log.info("saved dependency graph -> %s (%.1f MB)", path, path.stat().st_size / 2**20)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DependencyGraph":
        import pickle

        with open(Path(path), "rb") as fh:
            p = pickle.load(fh)
        return cls(
            n_nodes=p["n_nodes"],
            indptr=p["indptr"],
            indices=p["indices"],
            weights=p["weights"],
            section_kind=p["section_kind"],
            doc_start=p["doc_start"],
            doc_end=p["doc_end"],
            doc_ids=p["doc_ids"],
        )

    def stats(self) -> dict[str, Any]:
        deg = self.degree()
        return {
            "n_nodes": self.n_nodes,
            "n_edges": int(len(self.indices) / 2),
            "edges_per_node": round(float(deg.mean()), 2),
            "max_degree": int(deg.max()) if len(deg) else 0,
            "n_documents": len(self.doc_ids),
            "n_isolated": int((deg == 0).sum()),
            "bytes": int(
                self.indptr.nbytes + self.indices.nbytes
                + self.weights.nbytes + self.doc_start.nbytes + self.doc_end.nbytes
            ),
        }




# ------------------------------------------------------------- streamed inputs
#
# cmd_dependency_graph feeds DependencyGraph.build with store.get() per row. At
# 48,255,655 rows that is one seek per row into a 95 GB blob on NFS plus a resident
# offsets list of 48M tuples, which measured at the 16M scale as hours. The graph
# only needs two facts per row -- its section kind and which document it belongs to
# -- so they are scanned once, in parallel, straight from the raw record bytes, and
# build() is called unchanged with accessors over the scanned arrays.

_KIND_KEY = b'"section_kind":"'
_CHUNK_KEY = b'"chunk_id":"'
_DOC_KEY = b'"doc_id":"'


def _field(raw: bytes, key: bytes) -> bytes:
    i = raw.find(key)
    if i < 0:
        return b""
    i += len(key)
    return raw[i: raw.index(b'"', i)]


def _doc_of_chunk_id(cid: bytes) -> bytes:
    j = cid.rfind(b"#")
    return cid[:j] if j >= 0 else cid


def _scan_range(task: tuple[str, int, int]) -> tuple[int, list[bytes], bytes, int, list[bytes]]:
    """Section kinds, run-start flags, id mismatches and run doc ids for rows lo..hi-1.

    A row's flag is 1 when its ``doc_id`` -- what cmd_dependency_graph groups by --
    differs from the previous row's (row 0's is always 1), so the flags mark where
    each run of one document's rows starts; the doc id of every run is returned in
    order. A document can have more than one run. A mismatch is a row whose
    ``doc_id`` is not its chunk id up to '#', the key ChunkStore.document_rows uses;
    build refuses any.
    """
    store_dir, lo, hi = task
    store_dir_p = Path(store_dir)
    kinds: list[bytes] = []
    flags = bytearray()
    first = max(0, lo - 1)
    offs = np.fromfile(store_dir_p / "offsets.bin", dtype="<u8",
                       count=2 * (hi - first), offset=16 * first).reshape(-1, 2)
    prev: bytes | None = None
    mismatches = 0
    run_docs: list[bytes] = []
    with open(store_dir_p / "blob.jsonl", "rb") as fh:
        step = 20_000
        for a in range(0, len(offs), step):
            part = offs[a:a + step]
            start, end = int(part[0, 0]), int(part[-1, 0] + part[-1, 1])
            fh.seek(start)
            raw = fh.read(end - start)
            for k, (off, length) in enumerate(part):
                rec = raw[off - start: off - start + length]
                doc = _field(rec, _DOC_KEY)
                if first + a + k >= lo:
                    kinds.append(_field(rec, _KIND_KEY))
                    flags.append(1 if doc != prev else 0)
                    if doc != prev:
                        run_docs.append(doc)
                    mismatches += doc != _doc_of_chunk_id(_field(rec, _CHUNK_KEY))
                prev = doc
    return lo, kinds, bytes(flags), mismatches, run_docs


def scan_graph_inputs(index_dir: str | Path, *, workers: int = 6,
                      rows_per_task: int = 1_000_000) -> dict[str, Any]:
    """Write graph_inputs/{kind_codes.npy, doc_start.npy, run_docs.txt, kind_names.json}.

    ``run_docs.txt`` holds the doc id of each run, in row order. A document whose rows
    form more than one run is listed in ``split_docs.json`` with its run start rows.
    """
    import json
    from multiprocessing import Pool

    index_dir = Path(index_dir)
    store_dir = index_dir / "store"
    n = (store_dir / "offsets.bin").stat().st_size // 16
    out = index_dir / "graph_inputs"
    out.mkdir(parents=True, exist_ok=True)
    codes = np.zeros(n, dtype=np.uint8)
    starts = np.zeros(n, dtype=np.uint8)
    names: dict[bytes, int] = {}
    mismatches = 0
    runs_by_lo: dict[int, list[bytes]] = {}
    tasks = [(str(store_dir), lo, min(n, lo + rows_per_task)) for lo in range(0, n, rows_per_task)]
    t0 = time.time()
    with Pool(workers) as pool:
        for done, (lo, kinds, flags, bad, runs) in enumerate(
                pool.imap_unordered(_scan_range, tasks), 1):
            mismatches += bad
            runs_by_lo[lo] = runs
            for k in set(kinds):
                if k not in names:
                    names[k] = len(names)
            if len(names) > 255:
                raise ValueError("more than 255 distinct section kinds")
            codes[lo:lo + len(kinds)] = [names[k] for k in kinds]
            starts[lo:lo + len(flags)] = np.frombuffer(flags, dtype=np.uint8)
            log.info("graph inputs: %d/%d row ranges (%.0fs)", done, len(tasks), time.time() - t0)
    np.save(out / "kind_codes.npy", codes)
    np.save(out / "doc_start.npy", starts)
    run_starts = np.flatnonzero(starts)
    first_run: dict[bytes, int] = {}
    split: dict[str, list[int]] = {}
    with open(out / "run_docs.txt", "wb") as fh:
        i = 0
        for lo in sorted(runs_by_lo):
            for doc in runs_by_lo.pop(lo):
                fh.write(doc + b"\n")
                if doc in first_run:
                    split.setdefault(doc.decode(), [int(run_starts[first_run[doc]])]).append(
                        int(run_starts[i]))
                else:
                    first_run[doc] = i
                i += 1
    del first_run
    if i != len(run_starts):
        raise ValueError(f"{i} run doc ids for {len(run_starts)} run starts")
    (out / "split_docs.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
    kind_names = [k.decode() for k, _ in sorted(names.items(), key=lambda kv: kv[1])]
    n_runs = int(len(run_starts))
    manifest = {"n_rows": n, "n_doc_starts": n_runs, "kind_names": kind_names,
                "doc_id_chunk_id_mismatches": mismatches, "n_runs": n_runs,
                "n_docs": n_runs - sum(len(v) - 1 for v in split.values()),
                "docs_with_several_runs": len(split)}
    (out / "kind_names.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("graph inputs: %d rows, %d documents, %d kinds in %.0fs",
             n, manifest["n_doc_starts"], len(kind_names), time.time() - t0)
    return manifest


def build_from_scanned_inputs(index_dir: str | Path, doc_ids: Sequence[str] | None = None,
                              first_rows: Sequence[int] | None = None,
                              **kwargs: Any) -> DependencyGraph:
    """DependencyGraph.build over the scanned arrays.

    Each row's document is the doc id of the run it falls in, so a document split
    into several runs is grouped exactly as build() groups it from store.get().
    ``doc_ids``/``first_rows`` (ChunkStore.document_rows' output, or the page texts)
    are optional cross-checks: the runs' first-appearance order must reproduce them.
    """
    import json

    index_dir = Path(index_dir)
    src = index_dir / "graph_inputs"
    manifest = json.loads((src / "kind_names.json").read_text(encoding="utf-8"))
    if manifest["doc_id_chunk_id_mismatches"]:
        raise ValueError(f"{manifest['doc_id_chunk_id_mismatches']} rows whose doc_id is not "
                         f"their chunk id's document; the document names would differ")
    with open(src / "run_docs.txt", "rb") as fh:
        run_docs = [line.rstrip(b"\n").decode() for line in fh]
    flags = np.load(src / "doc_start.npy")
    starts = np.flatnonzero(flags)
    if len(run_docs) != len(starts):
        raise ValueError(f"{len(run_docs)} run doc ids for {len(starts)} run starts")
    if doc_ids is not None:
        seen: set[str] = set()
        order, rows = [], []
        for d, r in zip(run_docs, starts.tolist()):
            if d not in seen:
                seen.add(d)
                order.append(d)
                rows.append(r)
        del seen
        if order != list(doc_ids) or (first_rows is not None and rows != list(first_rows)):
            raise ValueError("scanned runs do not reproduce the documents' first appearances")
        del order, rows
    run_index = (np.cumsum(flags, dtype=np.int64) - 1).tolist()
    del flags, starts
    codes = np.load(src / "kind_codes.npy")
    kind_names = manifest["kind_names"]
    kinds = [kind_names[c] for c in codes.tolist()]
    del codes
    return DependencyGraph.build(range(len(kinds)), kind_of=kinds.__getitem__,
                                 doc_of=lambda r: run_docs[run_index[r]], **kwargs)
