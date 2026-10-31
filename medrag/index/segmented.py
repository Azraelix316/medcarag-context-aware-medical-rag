"""Exact BM25 and dense search over a corpus too large to hold in RAM.

At 48M chunks the in-RAM indexes do not fit on a 31 GB host: a flat FAISS index is
the full 74 GB matrix, and BM25 extrapolates to about 30 GB. Both searches here read
the corpus one segment at a time, score a whole batch of queries against it, keep
each query's running top-k, and move on. Nothing larger than one segment is resident.

They are exact, not approximations of the in-RAM versions:

* **Dense** is the inner product of every query with every row, which is what
  ``IndexFlatIP`` computes (equal up to float32 summation order), streamed from
  ``embeddings.f32``.
* **BM25** is built as independent per-segment indexes, but scored with the
  *global* N, average document length and document frequency. Those three are the
  only corpus-wide quantities in the BM25 formula, so a segment scored with them
  gives each row the score the monolithic index would. ``test_segmented.py`` checks
  this against ``BM25Index.search`` row for row.

Ties break on the lower row id, as ``BM25Index.search`` does, so the merged ranking
is identical too, not just the score set.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .bm25 import BM25Index, tokenize

log = logging.getLogger(__name__)

TEXT_READ_ROWS = 20_000   # rows per sequential read of the blob while building


# ------------------------------------------------------------------ shared
def _merge_topk(best_s: np.ndarray, best_i: np.ndarray, s: np.ndarray, i: np.ndarray,
                k: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge two (nq, *) score/row arrays into the top-k per query.

    Sorted by score descending, then row ascending, with a lexsort rather than an
    argsort on score alone, so tied rows land in the same order however the corpus
    was segmented.
    """
    s = np.concatenate([best_s, s], axis=1)
    i = np.concatenate([best_i, i], axis=1)
    out_s = np.empty((s.shape[0], min(k, s.shape[1])), np.float32)
    out_i = np.empty(out_s.shape, np.int64)
    for q in range(s.shape[0]):
        order = np.lexsort((i[q], -s[q]))[:k]
        out_s[q], out_i[q] = s[q, order], i[q, order]
    return out_s, out_i


def _hits(scores: np.ndarray, rows: np.ndarray) -> list[list[tuple[int, float]]]:
    return [[(int(r), float(v)) for r, v in zip(rr, ss) if r >= 0]
            for ss, rr in zip(scores, rows)]


# ------------------------------------------------------------------ dense
def load_embedding_matrix(index_dir: str | Path) -> np.memmap:
    """Open ``embeddings.f32`` as described by its ``embeddings.json`` manifest."""
    index_dir = Path(index_dir)
    manifest = json.loads((index_dir / "embeddings.json").read_text(encoding="utf-8"))
    n, dim = manifest["shape"]
    return np.memmap(index_dir / manifest["file"], dtype="<f4", mode="r", shape=(n, dim))


def stream_dense_search(matrix: np.ndarray, queries: np.ndarray, top_k: int = 100, *,
                        score_budget_bytes: int = 512 * 2**20,
                        max_slice_rows: int = 1 << 18,
                        progress_every_s: float = 60.0) -> list[list[tuple[int, float]]]:
    """Exact inner-product top-k of every query against every row of ``matrix``.

    ``matrix`` is normally the 74 GB memmap. Rows are read in slices sized so the
    (queries x slice) score matrix stays under ``score_budget_bytes``; the read
    itself is sequential, so the cost is one pass over the file per call, however
    many queries are in it. Batch all queries into one call.
    """
    q = np.ascontiguousarray(np.asarray(queries, dtype=np.float32))
    n, nq = int(matrix.shape[0]), int(q.shape[0])
    k = min(top_k, n)
    # Capped too, so a single query does not read a 74 GB slice into RAM at once.
    step = min(max_slice_rows, max(4096, score_budget_bytes // (4 * max(1, nq))))
    best_s = np.full((nq, 0), -np.inf, np.float32)
    best_i = np.full((nq, 0), -1, np.int64)
    t0 = last = time.time()
    for lo in range(0, n, step):
        hi = min(n, lo + step)
        scores = q @ np.asarray(matrix[lo:hi], dtype=np.float32).T
        kk = min(k, hi - lo)
        part = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
        best_s, best_i = _merge_topk(best_s, best_i,
                                     np.take_along_axis(scores, part, axis=1), part + lo, k)
        if time.time() - last > progress_every_s:
            last = time.time()
            log.info("dense stream: %d/%d rows (%.0f rows/s)", hi, n, hi / (last - t0))
    log.info("dense stream: %d queries x %d rows in %.0fs", nq, n, time.time() - t0)
    return _hits(best_s, best_i)


# ------------------------------------------------------------------ BM25 build
def segment_bounds(n_rows: int, rows_per_segment: int) -> list[tuple[int, int]]:
    return [(lo, min(n_rows, lo + rows_per_segment)) for lo in range(0, n_rows, rows_per_segment)]


def store_n_rows(store_dir: str | Path) -> int:
    return (Path(store_dir) / "offsets.bin").stat().st_size // 16


def iter_range_texts(store_dir: str | Path, lo: int, hi: int) -> Iterator[str]:
    """Yield ``embed_text or text`` for rows ``lo..hi-1``, as build_bm25_from_store indexes.

    Reads the blob in large sequential ranges rather than one seek per row, which
    is what makes a pass over 95 GB on NFS take hours rather than days.
    """
    store_dir = Path(store_dir)
    with open(store_dir / "blob.jsonl", "rb") as fh:
        for a in range(lo, hi, TEXT_READ_ROWS):
            b = min(hi, a + TEXT_READ_ROWS)
            offs = np.fromfile(store_dir / "offsets.bin", dtype="<u8",
                               count=2 * (b - a), offset=16 * a).reshape(-1, 2)
            start, end = int(offs[0, 0]), int(offs[-1, 0] + offs[-1, 1])
            fh.seek(start)
            raw = fh.read(end - start)
            for off, length in offs:
                rec = json.loads(raw[off - start: off - start + length])
                yield rec.get("embed_text") or rec.get("text") or ""


def segment_path(out_dir: str | Path, seg: int) -> Path:
    return Path(out_dir) / f"bm25-seg{seg:03d}.pkl"


def segment_stats_path(out_dir: str | Path, seg: int) -> Path:
    return Path(out_dir) / f"bm25-seg{seg:03d}.stats.pkl"


def build_bm25_segment(texts: Iterator[str], out_dir: str | Path, seg: int, lo: int, hi: int, *,
                       k1: float, b: float, min_token_length: int,
                       spill_dir: str | Path | None = None) -> Path:
    """Build one segment's BM25 over rows ``lo..hi-1`` (local ids 0..hi-lo-1).

    Writes the full index and, beside it, a small stats file holding only what the
    global scoring needs: the vocabulary, per-term df, row count and total length.
    The stats file is published last, so its presence marks a finished segment.
    """
    index = BM25Index(k1=k1, b=b, min_token_length=min_token_length)
    t0 = time.time()
    for local, text in enumerate(texts):
        index.add(local, text)
        if (local + 1) % 500_000 == 0:
            log.info("bm25 seg %d: %d/%d rows (%.0fs)", seg, local + 1, hi - lo, time.time() - t0)
    if index._cur_doc != hi - lo:
        raise ValueError(f"segment {seg}: read {index._cur_doc} rows, expected {hi - lo}")
    index.finalize(spill_dir=spill_dir)
    path = index.save(segment_path(out_dir, seg))
    stats = {"seg": seg, "lo": lo, "hi": hi, "n_docs": index.n_docs,
             "total_len": int(index._doc_len.astype(np.int64).sum()),
             "k1": index.k1, "b": index.b, "min_token_length": index.min_token_length,
             "terms": index._terms, "df": index._df}
    final = segment_stats_path(out_dir, seg)
    tmp = final.with_name(final.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        pickle.dump(stats, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(final)
    log.info("bm25 seg %d: rows %d..%d built in %.0fs -> %s", seg, lo, hi, time.time() - t0, path)
    return path


def claim_segment(out_dir: str | Path, n_segments: int) -> int | None:
    """Claim the lowest unbuilt segment with an atomic mkdir, for several workers."""
    claims = Path(out_dir) / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    for seg in range(n_segments):
        if segment_stats_path(out_dir, seg).exists():
            continue
        try:
            os.mkdir(claims / f"seg{seg:03d}")
            return seg
        except FileExistsError:
            continue
    return None


# ------------------------------------------------------------------ BM25 search
def _query_counts(query: str, min_token_length: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokenize(query, min_length=min_token_length):
        counts[token] = counts.get(token, 0) + 1
    return counts


def global_bm25_stats(stats_paths: Sequence[Path], query_terms: set[str]) -> dict[str, Any]:
    """Corpus-wide N, avgdl and df (for the query terms only) from the segment stats."""
    n = total = 0
    df: dict[str, int] = {t: 0 for t in query_terms}
    params = None
    for p in stats_paths:
        with open(p, "rb") as fh:
            st = pickle.load(fh)
        p_params = (st["k1"], st["b"], st["min_token_length"])
        if params is not None and p_params != params:
            raise ValueError(f"{p}: BM25 parameters {p_params} differ from {params}")
        params = p_params
        n += st["n_docs"]
        total += st["total_len"]
        index = {t: i for i, t in enumerate(st["terms"])} if query_terms else {}
        for t in query_terms:
            tid = index.get(t)
            if tid is not None:
                df[t] += int(st["df"][tid])
    if params is None:
        raise ValueError("no BM25 segment stats found")
    return {"n_docs": n, "avgdl": float(total / n) if n else 0.0, "df": df,
            "k1": params[0], "b": params[1], "min_token_length": params[2]}


def _score_segment(seg: BM25Index, lo: int, query_counts: Sequence[dict[str, int]],
                   g: dict[str, Any], top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """BM25Index.search's scoring for each query, with the global statistics."""
    total = float(g["n_docs"]) or 1.0
    avgdl = g["avgdl"] or 1.0
    k1, b = g["k1"], g["b"]
    denom_base = k1 * (1.0 - b + b * seg._doc_len.astype(np.float32) / avgdl)
    matrix = seg._matrix
    k = min(top_k, seg.n_docs)
    out_s = np.full((len(query_counts), k), -np.inf, np.float32)
    out_i = np.full((len(query_counts), k), -1, np.int64)
    for qi, counts in enumerate(query_counts):
        scores = np.zeros(seg.n_docs, dtype=np.float32)
        for term, qtf in counts.items():
            tid = seg._vocab.get(term)
            # shape[1]: the CSC is (docs x terms). A segment can have more terms
            # than rows, which is how BM25Index.search's old shape[0] bound was found.
            if tid is None or tid >= matrix.shape[1]:
                continue
            # Same expression and dtypes as BM25Index's _idf array, on a 1-element
            # array: float32 scalars would promote to float64 under numpy < 2.
            df = np.array([g["df"][term]], dtype=np.float32)
            idf = float(np.log(1.0 + (total - df + 0.5) / (df + 0.5)).astype(np.float32)[0])
            if idf <= 0.0:
                continue
            start, end = matrix.indptr[tid], matrix.indptr[tid + 1]
            if end <= start:
                continue
            cols = matrix.indices[start:end].astype(np.int64, copy=False)
            tf = matrix.data[start:end]
            contrib = idf * (tf * (k1 + 1.0) / (tf + denom_base[cols]))
            contrib *= 1.0 + 0.1 * math.log(1 + qtf)
            np.add.at(scores, cols, contrib.astype(np.float32))
        pos = np.flatnonzero(scores > 0.0)
        if pos.size == 0:
            continue
        order = np.lexsort((pos, -scores[pos]))[:k]
        out_s[qi, :order.size] = scores[pos[order]]
        out_i[qi, :order.size] = pos[order] + lo
    return out_s, out_i


def segmented_bm25_search(seg_dir: str | Path, queries: Sequence[str], top_k: int = 100,
                          only: Sequence[int] | None = None, partial: bool = False):
    """Exact BM25 top-k for every query over all segments under ``seg_dir``.

    Two passes: the small stats files give the global statistics, then each
    segment is loaded once, scores every query, and is released before the next.

    ``only`` scores just those segments (statistics stay corpus-wide) and, with
    ``partial``, returns the raw (scores, rows) top-k arrays for merge_partial_topk,
    so the segments can be split across hosts and merged to the identical result.
    """
    seg_dir = Path(seg_dir)
    stats_paths = sorted(seg_dir.glob("bm25-seg[0-9][0-9][0-9].stats.pkl"))
    seg_ids = [int(p.name[8:11]) for p in stats_paths]
    if seg_ids != list(range(len(seg_ids))):
        raise ValueError(f"BM25 segments under {seg_dir} are not contiguous: {seg_ids}")
    with open(stats_paths[0], "rb") as fh:
        mtl = pickle.load(fh)["min_token_length"]
    counts = [_query_counts(q, mtl) for q in queries]
    g = global_bm25_stats(stats_paths, {t for c in counts for t in c})
    log.info("bm25 global: %d rows over %d segments, avgdl %.2f",
             g["n_docs"], len(seg_ids), g["avgdl"])
    best_s = np.full((len(queries), 0), -np.inf, np.float32)
    best_i = np.full((len(queries), 0), -1, np.int64)
    expected_lo = 0
    for seg in seg_ids:
        with open(segment_stats_path(seg_dir, seg), "rb") as fh:
            st = pickle.load(fh)
        if st["lo"] != expected_lo:
            raise ValueError(f"segment {seg} starts at row {st['lo']}, expected {expected_lo}")
        expected_lo = st["hi"]
        if only is not None and seg not in only:
            continue
        t0 = time.time()
        index = BM25Index.load(segment_path(seg_dir, seg))
        s, i = _score_segment(index, st["lo"], counts, g, top_k)
        del index
        best_s, best_i = _merge_topk(best_s, best_i, s, i, top_k)
        log.info("bm25 seg %d: %d queries in %.0fs", seg, len(queries), time.time() - t0)
    if expected_lo != g["n_docs"]:
        raise ValueError(f"segments end at row {expected_lo} but cover {g['n_docs']} rows")
    if partial:
        return best_s, best_i
    return [[(r, v) for r, v in hits if v > 0.0] for hits in _hits(best_s, best_i)]


def merge_partial_topk(parts: Sequence[tuple[np.ndarray, np.ndarray]], top_k: int,
                       ) -> list[list[tuple[int, float]]]:
    """Merge partial=True results that together cover every segment once."""
    best_s = np.full((parts[0][0].shape[0], 0), -np.inf, np.float32)
    best_i = np.full((parts[0][0].shape[0], 0), -1, np.int64)
    for s, i in parts:
        best_s, best_i = _merge_topk(best_s, best_i, s, i, top_k)
    return [[(r, v) for r, v in hits if v > 0.0] for hits in _hits(best_s, best_i)]


# ------------------------------------------------------------------ retriever adapter
class PrecomputedHits:
    """Any precomputed per-query hit list, looked up by query text like PrecomputedBM25.

    Serves search-segmented dense hits and precomputed MeSH graph hits (doc ids,
    ``as_int=False``) to the retriever's hybrid3 and graph modes. A query that was
    not precomputed, or a deeper top_k than was kept, raises.
    """

    def __init__(self, hits: dict[str, list[tuple[Any, float]]], top_k: int) -> None:
        self._hits = hits
        self.top_k = top_k

    @classmethod
    def from_run(cls, run_path: str | Path, queries_path: str | Path, field: str, *,
                 as_int: bool = True, id_field: str = "qid",
                 query_field: str = "query") -> "PrecomputedHits":
        texts = {}
        for line in Path(queries_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                texts[str(rec[id_field])] = str(rec[query_field])
        hits: dict[str, list[tuple[Any, float]]] = {}
        top_k = 0
        for line in Path(run_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                h = [(int(r) if as_int else str(r), float(s)) for r, s in rec[field]]
                hits[texts[str(rec["qid"])]] = h
                top_k = max(top_k, len(h))
        return cls(hits, top_k)

    def search(self, query: str, top_k: int = 100, **_: Any) -> list[tuple[Any, float]]:
        if query not in self._hits:
            raise KeyError(f"query not in the precomputed run: {query[:80]!r}")
        if top_k > self.top_k and len(self._hits[query]) >= self.top_k:
            raise ValueError(f"top_k={top_k} but the run kept only {self.top_k} hits per query")
        return self._hits[query][:top_k]

    def stats(self) -> dict[str, Any]:
        return {"precomputed_queries": len(self._hits), "top_k": self.top_k}


class PrecomputedBM25:
    """Serves ``search-segmented`` sparse hits through ``BM25Index.search``'s interface.

    The retriever's sparse-based modes (sparse, bleed, context, page, pagectx) call
    only ``bm25.search(query, top_k)``. At 48M rows that search runs as one batched
    pass over the segments, so its output is loaded here and looked up per query.
    A query that was not precomputed raises rather than returning nothing, so a
    missing sub-query cannot silently shrink a ranking.
    """

    def __init__(self, hits: dict[str, list[tuple[int, float]]], n_docs: int, top_k: int) -> None:
        self._hits = hits
        self.n_docs = n_docs
        self.top_k = top_k

    @classmethod
    def from_run(cls, run_path: str | Path, queries_path: str | Path, *, n_docs: int,
                 id_field: str = "qid", query_field: str = "query") -> "PrecomputedBM25":
        """Join a search-segmented run file back to its query texts by qid."""
        texts = {}
        for line in Path(queries_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                texts[str(rec[id_field])] = str(rec[query_field])
        hits: dict[str, list[tuple[int, float]]] = {}
        top_k = 0
        for line in Path(run_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                h = [(int(r), float(s)) for r, s in rec["sparse"]]
                hits[texts[str(rec["qid"])]] = h
                top_k = max(top_k, len(h))
        return cls(hits, n_docs, top_k)

    def search(self, query: str, top_k: int = 100, *, candidates: int | None = None,
               ) -> list[tuple[int, float]]:
        if candidates is not None:
            raise ValueError("precomputed BM25 hits were computed without candidates")
        if query not in self._hits:
            raise KeyError(f"query not in the precomputed sparse run: {query[:80]!r}")
        if top_k > self.top_k and len(self._hits[query]) >= self.top_k:
            raise ValueError(f"top_k={top_k} but the run kept only {self.top_k} hits per query")
        return self._hits[query][:top_k]

    def stats(self) -> dict[str, Any]:
        return {"n_docs": self.n_docs, "precomputed_queries": len(self._hits), "top_k": self.top_k}
