"""Dense vector index over chunk embeddings.

FAISS is used with an inner-product index on L2-normalised vectors, which makes
cosine similarity a dot product. Index type adapts to corpus size: flat is
exhaustive and exact but O(N) per query, while IVF+HNSW approximates. The
pipeline builds flat first (exactness matters for benchmark numbers) and
switches to IVF-PQ only when the chunk count exceeds a threshold where exact
search stops being interactive.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _rss_gib() -> float:
    """Resident set size in GiB, or -1.0."""
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 2**30
    except Exception:  # noqa: BLE001
        return -1.0

# Index-type policy.
#
# `auto` uses an EXACT flat (IndexFlatIP) search up to this many vectors and
# only then falls back to approximate IVF-PQ. The threshold is far higher than
# it first appears it should be, because the quality cost of approximate search
# is severe and the latency cost of exact search is not.
#
# Measured on this host at 2,368,622 vectors x 384 dims:
#   IndexFlatIP (exact)  99.4 ms/query, nDCG@10 0.542
#   IndexIVFPQ nprobe=16  0.1 ms/query, nDCG@10 0.274   <- halves dense quality
#
# 16-byte PQ codes over 384 dims are simply too coarse for clinical text. A 99 ms
# query is entirely usable for research retrieval, batch evaluation, and an agent
# fanning out a handful of sub-queries, so approximate search must be an explicit
# choice rather than a silent automatic one.
#
# Lower this (or pass index_type=) only when sub-10ms latency is a hard
# requirement and a quality regression is acceptable.
IVF_THRESHOLD = 20_000_000


class DenseIndex:
    """FAISS-backed dense retrieval over normalised embeddings."""

    def __init__(self, dim: int, index_type: str = "auto") -> None:
        import faiss

        self.faiss = faiss
        self.dim = dim
        self.index_type = index_type
        self.index: Any = None
        # Chunk ids for provenance/debugging, and the integer row ids that the
        # retrieval layer actually uses to resolve payloads. Keeping both
        # explicit avoids the position/row-id ambiguity that silently breaks
        # fusion when the two retrievers disagree about what an id is.
        self.ids: list[str] = []
        self.row_ids: list[int] = []
        self._nlist = 0
        #: Contiguous flat shards, when the index was built sharded.
        self.shards: list[Any] = []
        self.shard_offsets: list[int] = []

    # ------------------------------------------------------------------ build
    def _index_type_for(self, n_total: int) -> str:
        kind = self.index_type
        if kind == "auto":
            kind = "flat" if n_total < IVF_THRESHOLD else "ivfpq"
        return kind

    def _build_flat_sharded(self, matrix: np.ndarray, batch_size: int,
                            flat_shards: int) -> "DenseIndex":
        """Build ``flat_shards`` contiguous IndexFlat shards over the matrix."""
        n = int(matrix.shape[0])
        faiss = self.faiss
        bounds = np.linspace(0, n, flat_shards + 1).astype(int)
        self.shards: list[Any] = []
        self.shard_offsets: list[int] = []
        t0 = time.time()
        for s in range(flat_shards):
            lo, hi = int(bounds[s]), int(bounds[s + 1])
            if hi <= lo:
                continue
            idx = faiss.IndexFlatIP(self.dim)
            for start in range(lo, hi, batch_size):
                stop = min(hi, start + batch_size)
                idx.add(np.ascontiguousarray(matrix[start:stop]))
            self.shards.append(idx)
            self.shard_offsets.append(lo)
            log.info("dense shard %d/%d: rows %d..%d (%d vectors), rss=%.1f GiB",
                     s + 1, flat_shards, lo, hi, hi - lo, _rss_gib())
        log.info("built %d flat shards over %d rows in %.0f s",
                 len(self.shards), n, time.time() - t0)
        # A stand-in so save()/load() and any `self.index.ntotal` reader work.
        self.index = self.shards[0] if self.shards else self._make(n, "flat")
        return self

    def _make(self, n_total: int, index_type: str) -> Any:
        faiss = self.faiss
        kind = index_type
        if kind == "auto":
            kind = "flat" if n_total < IVF_THRESHOLD else "ivfpq"

        if kind == "flat":
            return faiss.IndexFlatIP(self.dim)
        if kind == "ivfflat":
            nlist = max(1, min(4096, int(round(4 * np.sqrt(max(1, n_total))))) // 2)
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self._nlist = nlist
            return index
        if kind == "ivfpq":
            # Approximate. 16-byte PQ codes are too coarse for clinical text:
            # measured nDCG@10 0.542 (exact) vs 0.274 (IVFPQ) at 2.37M vectors.
            # Only reachable via an explicit index_type; `auto` never selects it.
            nlist = max(1, min(4096, int(round(4 * np.sqrt(max(1, n_total))))) // 2)
            m = 16 if self.dim % 16 == 0 else 8
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
            self._nlist = nlist
            return index
        raise ValueError(f"unknown index_type: {index_type}")

    def build(
        self,
        matrix: np.ndarray,
        ids: list[str],
        *,
        row_ids: list[int] | None = None,
        batch_size: int = 50_000,
        train_size: int = 200_000,
    ) -> "DenseIndex":
        """Build the index. ``matrix`` must be (n, dim) float32, L2-normalised."""
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) matrix, got {matrix.shape}")
        n = matrix.shape[0]
        if row_ids is not None and len(row_ids) != n:
            raise ValueError(f"row_ids length {len(row_ids)} != matrix rows {n}")
        # Default: positions are row ids, which is how the store is built.
        self.row_ids = list(range(n)) if row_ids is None else list(row_ids)
        if n == 0:
            self.index = self._make(1, self.index_type)
            return self

        self.index = self._make(n, self.index_type)
        faiss = self.faiss

        if not self.index.is_trained:
            # IVF training needs a sample; more than ~256k points adds cost, not recall.
            rng = np.random.default_rng(17)
            sample_n = min(train_size, n)
            sample_idx = rng.choice(n, size=sample_n, replace=False) if sample_n < n else np.arange(n)
            t0 = time.time()
            self.index.train(np.ascontiguousarray(matrix[sample_idx]))
            log.info("trained %s on %d samples in %.1fs", type(self.index).__name__, sample_n, time.time() - t0)
        if hasattr(self.index, "nprobe"):
            # nprobe=16 balances recall/latency for a typical query mix.
            self.index.nprobe = min(16, max(1, self._nlist))

        for start in range(0, n, batch_size):
            block = np.ascontiguousarray(matrix[start : start + batch_size])
            self.index.add(block)
            log.info("dense index: added %d/%d vectors", min(start + batch_size, n), n)

        self.ids = list(ids)
        log.info("built %s with %d vectors", type(self.index).__name__, n)
        return self

    def build_from_memmap(
        self,
        matrix: np.ndarray,
        ids: Sequence[str],
        *,
        batch_size: int = 100_000,
        row_ids: Sequence[int] | None = None,
        flat_shards: int = 1,
    ) -> "DenseIndex":
        """Build the index by streaming a (possibly memory-mapped) matrix.

        Equivalent to :meth:`build` but never materialises the whole matrix as
        a second array. For a memmap input this keeps peak memory at one FAISS
        batch regardless of corpus size, which is what makes multi-million
        chunk corpora buildable without a RAM-sized staging buffer.

        ``flat_shards`` splits an exhaustive flat index into that many contiguous
        row ranges, searched and merged. It exists for a specific failure: FAISS's
        ``IndexFlat`` stores vectors in a ``std::vector`` that doubles, so adding
        the last vectors transiently needs the old and new buffers at once. At
        16,133,551 x 384 the index is 24.8 GiB and the doubling peak is ~37 GiB,
        which died with ``std::bad_alloc`` on a 31 GiB host after 12.2M of 16.1M
        vectors.

        Sharding is not an approximation. Every shard is exhaustive and the search
        merges the per-shard top-k, so the result set is identical to a single flat
        index over the same vectors -- only the merge costs an extra sort. The
        default is 1, which keeps small corpora on the simpler path.
        """
        n = int(matrix.shape[0])
        if matrix.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {matrix.shape[1]}")
        if len(ids) != n:
            raise ValueError(f"got {len(ids)} ids for {n} rows")
        self.row_ids = list(range(n)) if row_ids is None else list(row_ids)
        self.ids = list(ids)
        if flat_shards > 1 and self._index_type_for(n) == "flat":
            return self._build_flat_sharded(matrix, batch_size, flat_shards)
        if n == 0:
            self.index = self._make(1, self.index_type)
            return self

        self.index = self._make(n, self.index_type)
        if not self.index.is_trained:
            rng = np.random.default_rng(17)
            sample_n = min(200_000, n)
            if n > sample_n:
                idx = np.sort(rng.choice(n, size=sample_n, replace=False))
                sample = np.ascontiguousarray(matrix[idx])
            else:
                sample = np.ascontiguousarray(matrix)
            t0 = time.time()
            self.index.train(sample)
            log.info("trained %s on %d samples in %.1fs", type(self.index).__name__, len(sample), time.time() - t0)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = min(16, max(1, self._nlist))

        for start in range(0, n, batch_size):
            block = np.ascontiguousarray(matrix[start : start + batch_size])
            self.index.add(block)
            if start % (batch_size * 10) == 0 or start + batch_size >= n:
                log.info("dense index: added %d/%d vectors", min(start + batch_size, n), n)
        self.ids = list(ids)
        self.row_ids = list(range(n)) if row_ids is None else list(row_ids)
        log.info("built %s with %d vectors", type(self.index).__name__, n)
        return self

    def add(self, matrix: np.ndarray, ids: list[str], row_ids: list[int] | None = None) -> "DenseIndex":
        """Incrementally add vectors to an existing trained index."""
        if self.index is None:
            raise RuntimeError("call build() first")
        self.index.add(np.ascontiguousarray(matrix))
        self.ids.extend(ids)
        if row_ids is not None:
            self.row_ids.extend(row_ids)
        else:
            start = len(self.row_ids)
            self.row_ids.extend(range(start, start + len(ids)))
        return self

    # ------------------------------------------------------------------ query
    def search(self, queries: np.ndarray, top_k: int = 100, *, nprobe: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, positions)`` where positions index into the vector list."""
        shards = getattr(self, "shards", None)
        if shards:
            q = np.ascontiguousarray(queries.astype(np.float32))
            k = min(top_k, int(sum(sh.ntotal for sh in shards)))
            all_s, all_i = [], []
            for off, sh in zip(self.shard_offsets, shards):
                s_, i_ = sh.search(q, min(k, int(sh.ntotal)))
                all_s.append(s_)
                all_i.append(i_ + off)
            scores = np.concatenate(all_s, axis=1)
            positions = np.concatenate(all_i, axis=1)
            # Each shard returns its own top-k, so the global top-k is the best of
            # the union. -1 marks a shard that returned fewer than k.
            order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
            rows = np.arange(len(q))[:, None]
            return scores[rows, order], positions[rows, order]
        if self.index is None or not self.index.ntotal:
            empty_s = np.zeros((len(queries), 0), dtype=np.float32)
            empty_i = np.zeros((len(queries), 0), dtype=np.int64)
            return empty_s, empty_i
        if nprobe is not None and hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe
        k = min(top_k, int(self.index.ntotal))
        scores, positions = self.index.search(np.ascontiguousarray(queries.astype(np.float32)), k)
        return scores, positions

    def search_rows(self, queries: np.ndarray, top_k: int = 100) -> list[tuple[int, float]]:
        """Search a batch and return ``(row_id, score)`` pairs merged by score."""
        scores, positions = self.search(queries, top_k)
        merged: list[tuple[int, float]] = []
        for q_scores, q_pos in zip(scores, positions):
            for pos, score in zip(q_pos, q_scores):
                if 0 <= pos < len(self.row_ids):
                    merged.append((self.row_ids[pos], float(score)))
        merged.sort(key=lambda kv: -kv[1])
        return merged

    def search_one_rows(self, query: np.ndarray, top_k: int = 100) -> list[tuple[int, float]]:
        scores, positions = self.search(np.asarray([query], dtype=np.float32), top_k)
        out: list[tuple[int, float]] = []
        for pos, score in zip(positions[0], scores[0]):
            if 0 <= pos < len(self.row_ids):
                out.append((self.row_ids[pos], float(score)))
        return out

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path, ids_path: str | Path | None = None) -> Path:
        """Persist the index.

        A sharded flat index is written as ``<stem>.shard0``, ``.shard1``, ... plus a
        small manifest, because ``write_index`` only ever sees one FAISS handle.

        This bit us: ``save`` wrote ``self.index``, which for a sharded build is the
        *first shard* kept as a stand-in, so a 4-shard index over 16,133,551 vectors
        saved as 6.20 GB -- one quarter of the data — while the log cheerfully reported
        all 16,133,551 vectors. Nothing raised; the file was simply wrong, and it
        would have surfaced much later as missing three quarters of the corpus.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # with_name, not with_suffix: Path("dense.faiss").with_suffix(".ids.json")
        # yields "dense.ids.json", because it replaces ".faiss" wholesale. Sidecar
        # names are built by appending so they stay anchored to the index filename.
        ids_path = Path(ids_path) if ids_path else path.with_name(path.name + ".ids.json")

        if getattr(self, "shards", None):
            written: list[str] = []
            for i, sh in enumerate(self.shards):
                sp = path.with_name(f"{path.name}.shard{i}")
                self.faiss.write_index(sh, str(sp))
                written.append(sp.name)
            manifest = {"sharded": True, "shards": written,
                        "offsets": list(self.shard_offsets),
                        "dim": self.dim, "n_vectors": len(self.ids)}
            mp = path.with_name(path.name + ".manifest.json")
            mp.write_text(json.dumps(manifest), encoding="utf-8")
            path.unlink(missing_ok=True)
            out = mp
        else:
            self.faiss.write_index(self.index, str(path))
            out = path

        # row_ids are persisted alongside chunk ids: fusion needs the integers,
        # and re-deriving them as 0..n-1 would break any partial build.
        ids_path.write_text(
            json.dumps({"ids": self.ids, "row_ids": self.row_ids}), encoding="utf-8"
        )
        log.info("saved dense index -> %s (%d vectors)", out, len(self.ids))
        return out

    @classmethod
    def load(cls, path: str | Path, ids_path: str | Path | None = None, *, nprobe: int = 16) -> "DenseIndex":
        import faiss

        path = Path(path)
        manifest_path = path.with_name(path.name + ".manifest.json")
        if manifest_path.exists():
            # Sharded flat index: several FAISS files plus a manifest.
            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            obj = cls(dim=int(man["dim"]), index_type="loaded")
            obj.shards = [faiss.read_index(str(path.with_name(n)))
                          for n in man["shards"]]
            obj.shard_offsets = [int(o) for o in man["offsets"]]
            obj.index = obj.shards[0] if obj.shards else None
            index = obj.index
        else:
            index = faiss.read_index(str(path))
            obj = cls(dim=index.d, index_type="loaded")
            obj.index = index
        ids_path = Path(ids_path) if ids_path else path.with_name(path.name + ".ids.json")
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            obj.ids = list(payload.get("ids", []))
            obj.row_ids = list(payload.get("row_ids", []))
        else:  # legacy format: bare id list, row ids implied by position
            obj.ids = list(payload)
            obj.row_ids = list(range(len(obj.ids)))
        if not obj.row_ids:
            obj.row_ids = list(range(len(obj.ids)))
        if hasattr(index, "nprobe"):
            index.nprobe = min(nprobe, max(1, obj._nlist or nprobe))
        log.info("loaded dense index <- %s (%d vectors, ntotal=%d)", path, len(obj.ids), index.ntotal)
        return obj

    def stats(self) -> dict[str, Any]:
        return {
            "type": type(self.index).__name__ if self.index is not None else "none",
            "n_vectors": int(self.index.ntotal) if self.index is not None else 0,
            "dim": self.dim,
            "nprobe": getattr(self.index, "nprobe", None),
            "n_ids": len(self.ids),
        }
