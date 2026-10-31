"""Index construction: chunk store, BM25 postings, and dense FAISS vectors.

This is the heaviest stage of the pipeline, so it is built around
recoverability. BM25 and the dense index are written to temporary files and
atomically renamed, and each stage skips itself if its artifact already exists,
so a killed run resumes from the last completed stage instead of starting over.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..config import PipelineConfig
from ..data.schema import Chunk
from .bm25 import BM25Index
from .dense import DenseIndex
from .embeddings import load_embeddings, save_embeddings
from .multi_gpu import build_embedder

# Rows per persisted embedding block. One block is the unit of parallel work and
# the unit of restartability: a shard writes whole blocks, so an interrupted run
# leaves a readable set of complete blocks rather than a partially written matrix.
EMBED_BLOCK_ROWS = 20_000
from .store import ChunkStore, build_chunk_store

log = logging.getLogger(__name__)


def build_store_from_chunks(
    chunks: Iterable[Chunk],
    *,
    store_dir: Path,
    cache_size: int = 200_000,
    progress_every: int = 50_000,
) -> ChunkStore:
    """Materialise the chunk store, streaming to ``store_dir`` as it grows.

    Chunks are written to the backing blob immediately, so peak memory is
    O(cache_size) regardless of how many chunks the corpus produces. Keeping the
    cache small is therefore safe and desirable at scale.
    """
    t0 = time.time()
    store = ChunkStore(cache_size=cache_size, dir_path=store_dir)
    n = 0
    for chunk in chunks:
        store.add(chunk)
        n += 1
        if n % progress_every == 0:
            log.info("store: %d chunks (%.1fs, %.1f MiB blob)",
                     n, time.time() - t0, store._cursor / 2**20)
    store.finalize_build()
    log.info("store built: %d chunks in %.1fs (%.1f MiB blob)",
             n, time.time() - t0, store._cursor / 2**20)
    return store


def build_bm25_from_store(store: ChunkStore, cfg: PipelineConfig, *, progress_every: int = 50_000,
                          spill_dir: str | None = None) -> BM25Index:
    """Build BM25 over chunk text.

    Indexes ``embed_text`` (title + section + body) rather than raw ``text``:
    BM25 then benefits from the same document context that dense retrieval gets,
    which measurably improves recall for passages with no self-describing nouns.
    """
    index = BM25Index(k1=cfg.sparse.k1, b=cfg.sparse.b, min_token_length=cfg.sparse.min_token_length)
    t0 = time.time()
    for n, (row_id, text) in enumerate(store.iter_texts(use_embed_text=True), 1):
        index.add(row_id, text)
        if n % progress_every == 0:
            log.info("bm25: indexed %d/%d chunks (%.1fs)", n, len(store), time.time() - t0)
    # Spill by default: at 16.1M chunks the accumulated postings do not fit alongside
    # the CSC they are converted into. See BM25Index.finalize.
    index.finalize(spill_dir=spill_dir)
    log.info("bm25 built over %d chunks in %.1fs", len(store), time.time() - t0)
    return index

def _embed_store_to_memmap(
    store: ChunkStore,
    cfg: PipelineConfig,
    memmap_path: Path,
    *,
    batch_rows: int = EMBED_BLOCK_ROWS,
    block_shard: int = 0,
    block_nshards: int = 1,
    reverse_blocks: bool = False,
) -> tuple[np.memmap | None, int]:
    """Embed every chunk into a memory-mapped float32 matrix on disk.

    Design note. The obvious implementation accumulates all blocks in a list and
    calls ``np.vstack`` at the end, which holds three full copies of the matrix
    at once (per-block list, the vstack result, and FAISS's own storage). At
    2.37M chunks x 384 dims that is ~10 GiB of avoidable peak RSS on top of the
    chunk cache, and it OOMs a 31 GiB host.

    Instead each block is written straight into a pre-allocated ``np.memmap``:
    peak extra memory is one block, and the FAISS index is then filled by
    streaming the memmap in slices. Blocks stay on disk individually so a
    restarted run skips completed work.

    Sharding. ``block_shard``/``block_nshards`` let several nodes embed disjoint
    subsets of blocks concurrently. Each block is written to its own
    ``embeddings-blockNNNNN.npz`` via a temp file plus atomic rename, so blocks
    are the unit of sharing and a partial or interrupted run leaves a readable
    set of complete blocks.
    """
    memmap_path.parent.mkdir(parents=True, exist_ok=True)
    # build_embedder picks MultiGpuEmbedder when >=2 CUDA devices are visible,
    # which measured 1.6x the single-device path on this host.
    embedder = build_embedder(cfg.embed).load()
    log.info("embedder backend: %s", type(embedder).__name__)

    n = len(store)
    dim = int(embedder.dim)
    n_blocks = (n + batch_rows - 1) // batch_rows

    # A sharded worker MUST NOT touch the shared memmap at all -- not even to
    # create or size it. Measured failure: two hosts each writing disjoint byte
    # ranges of one mmap-backed file over NFS lost a writer's entire contribution
    # (4000/4000 rows zeroed) because each client holds its own page cache and
    # only pushes on flush/close. Opening it here would reintroduce exactly that.
    # Block files are immune: each is written by exactly one writer and published
    # by atomic rename, which NFS does guarantee. So a sharded run writes only
    # block files, and the single-process assembly pass builds the memmap.
    shard_writes_memmap = block_nshards == 1

    if shard_writes_memmap:
        # Pre-allocate the full matrix. Each block is written in place, so a
        # crash loses at most the block in flight, and resume skips the rest.
        if memmap_path.exists() and memmap_path.stat().st_size == n * dim * 4:
            matrix = np.memmap(memmap_path, dtype=np.float32, mode="r+", shape=(n, dim))
        else:
            tmp = memmap_path.with_suffix(".tmp")
            matrix = np.memmap(tmp, dtype=np.float32, mode="w+", shape=(n, dim))
            matrix.flush()
            tmp.replace(memmap_path)
            matrix = np.memmap(memmap_path, dtype=np.float32, mode="r+", shape=(n, dim))
    else:
        matrix = None

    t0 = time.time()
    done_rows = 0
    embedded_rows = 0     # excludes reused blocks, so the rate/ETA stay honest
    embedded_time = 0.0
    # Each node claims blocks block_shard, block_shard+block_nshards, ...
    # The start MUST be block_shard: range(0, n, k) would hand every shard the
    # same first block, so all nodes would redundantly embed block 0 and
    # overwrite each other, and no other block would ever be written.
    blocks = (
        range(block_shard, n_blocks, block_nshards)
        if 0 <= block_shard < block_nshards
        else range(n_blocks)
    )
    if reverse_blocks:
        # A second worker on the same shard (e.g. a CPU worker beside the GPU one)
        # walks from the far end, so the two meet once instead of racing block by
        # block; the per-block exists() check below skips whatever the other did.
        blocks = reversed(blocks)
    if block_nshards > 1:
        log.info("embedding blocks %d,%d,... of %d (shard %d/%d)",
                 block_shard, block_shard + block_nshards, n_blocks, block_shard, block_nshards)

    for block in blocks:
        block_path = memmap_path.parent / f"{memmap_path.stem}-block{block:05d}.npz"
        lo = block * batch_rows
        hi = min(n, lo + batch_rows)

        if block_path.exists():
            if shard_writes_memmap:
                # Already embedded by an earlier run: copy it into the matrix.
                data = np.load(block_path, allow_pickle=True)
                matrix[lo:hi] = data["embeddings"]
            done_rows += (hi - lo)
            log.info("embed block %d/%d: reusing cached shard (%d rows)", block + 1, n_blocks, hi - lo)
            continue

        texts: list[str] = []
        for row_id in range(lo, hi):
            chunk = store.get(row_id)
            texts.append((chunk.embed_text or chunk.text) if chunk else "")
        t_block = time.time()
        vecs = embedder.embed(texts, is_query=False, batch_size=cfg.embed.batch_size)
        block_vecs = np.asarray(vecs, dtype=np.float32)

        # Every block records which shard wrote it.
        #
        # Without this, two shards computing the same block is invisible: each writes
        # its own temp file and renames it into place, and the second rename simply
        # replaces the first. The result is not corrupt -- both computed the same rows
        # -- but the run cannot tell anyone it happened, which means the one property
        # the sharded path exists to guarantee, "no row written twice", is unverifiable
        # after the fact. `verify_shard_blocks.py` had to describe this as a gap it
        # could not check, because the filesystem genuinely does not record it.
        #
        # The pid and block id go in alongside so a collision is diagnosable rather
        # than merely detectable. Backward compatible: blocks written before this
        # change carry no `shard` key and are read as unowned, not as corrupt.
        owner = np.array([block_shard], dtype=np.int32)
        meta = np.array([block, n_blocks, os.getpid()], dtype=np.int64)

        if shard_writes_memmap:
            matrix[lo:hi] = block_vecs
            np.savez(block_path.with_suffix(".tmp.npz"),
                     embeddings=block_vecs, shard=owner, meta=meta)
            block_path.with_suffix(".tmp.npz").replace(block_path)
            matrix.flush()
        else:
            # Publish the block atomically. Never open the memmap for write.
            # np.savez appends ".npz" unless the name already ends in it, so the
            # temp name must be built as a .npz path or the rename target is
            # missing and the write fails with FileNotFoundError.
            tmp = block_path.with_name(f"{block_path.stem}.tmp{os.getpid()}.npz")
            np.savez(tmp, embeddings=block_vecs, shard=owner, meta=meta)
            os.replace(tmp, block_path)

        done_rows += (hi - lo)
        embedded_rows += (hi - lo)
        embedded_time += time.time() - t_block
        # Rate is measured over embedded rows only: reused blocks complete in
        # milliseconds and would otherwise report an absurd throughput.
        rate = embedded_rows / max(1e-9, embedded_time)
        remaining = max(0, n - done_rows)
        eta_min = remaining / max(1e-9, rate) / 60.0
        if remaining:
            log.info("embed block %d/%d: %d rows in %.1fs | %d/%d total | %.0f rows/s | ETA %.1f min",
                     block + 1, n_blocks, hi - lo, time.time() - t_block,
                     done_rows, n, rate, eta_min)
        else:
            log.info("embed block %d/%d: %d rows in %.1fs | %d/%d total | complete",
                     block + 1, n_blocks, hi - lo, time.time() - t_block, done_rows, n)

    log.info("embeddings complete: %d x %d in %.1fs (%.0f rows/s)",
             n, dim, time.time() - t0, n / max(1e-9, time.time() - t0))

    if not shard_writes_memmap:
        # Report the true block coverage rather than the rows this shard touched,
        # so a caller cannot mistake a partial run for a complete matrix.
        missing = missing_embedding_blocks(memmap_path, n, n_blocks, batch_rows)
        log.info("shard %d/%d finished; %d/%d blocks present on disk, %d missing",
                 block_shard, block_nshards, n_blocks - len(missing), n_blocks, len(missing))
        return None, dim  # type: ignore[return-value]

    # Single-process path: ensure the memmap reflects every block on disk, which
    # also repairs a run that was previously sharded and is now being finished.
    _sync_memmap_from_blocks(matrix, memmap_path, n_blocks, batch_rows)
    return matrix, dim


def missing_embedding_blocks(
    memmap_path: Path,
    n: int,
    n_blocks: int,
    batch_rows: int,
) -> list[int]:
    """Blocks with no published ``.npz`` yet."""
    return [
        b for b in range(n_blocks)
        if not (memmap_path.parent / f"{memmap_path.stem}-block{b:05d}.npz").exists()
    ]


def _sync_memmap_from_blocks(matrix, memmap_path: Path, n_blocks: int, batch_rows: int) -> int:
    """Copy every block file into the memmap. Single-writer only.

    Run by whichever node finishes the index build. Block files are the source of
    truth; the memmap is a derived artifact assembled by one process, so it
    never sees concurrent writers.
    """
    n, dim = matrix.shape
    copied = 0
    for block in range(n_blocks):
        block_path = memmap_path.parent / f"{memmap_path.stem}-block{block:05d}.npz"
        lo = block * batch_rows
        hi = min(n, lo + batch_rows)
        data = np.load(block_path, allow_pickle=True)["embeddings"]
        if data.shape[0] != hi - lo or data.shape[1] != dim:
            raise ValueError(
                f"block {block} has shape {data.shape}, expected {(hi - lo, dim)}; "
                f"the chunk set changed under it. Delete {memmap_path.parent} and rebuild."
            )
        matrix[lo:hi] = data
        copied += 1
    matrix.flush()
    log.info("assembled memmap %s from %d block file(s)", memmap_path.name, copied)
    return copied


def write_embedding_manifest(emb_path: Path, *, n: int, dim: int, embed_cfg) -> Path:
    """Describe ``embeddings.f32`` so it can be loaded without this codebase.

    The file is a headerless row-major float32 matrix; row i embeds store row i.
    """
    import json as _json

    manifest = {
        "file": emb_path.name,
        "dtype": "float32",
        "byte_order": "little",
        "layout": "row-major, no header",
        "shape": [n, dim],
        "bytes": n * dim * 4,
        "row_order": "row i is chunk row id i of store/ (store/rowmap.json)",
        "model": embed_cfg.model_name,
        "pooling": embed_cfg.pooling,
        "normalized": embed_cfg.normalize,
        "query_prefix": embed_cfg.query_prefix,
        "max_seq_length": embed_cfg.max_seq_length,
        "fp16_inference": embed_cfg.fp16,
        "load": f"np.memmap('{emb_path.name}', dtype='<f4', mode='r', shape=({n}, {dim}))",
    }
    path = emb_path.with_name("embeddings.json")
    path.write_text(_json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def build_dense_from_store(
    store: ChunkStore,
    cfg: PipelineConfig,
    index_dir: Path,
    *,
    index_type: str = "auto",
    force: bool = False,
    faiss_add_batch: int = 100_000,
    block_shard: int = 0,
    block_nshards: int = 1,
    skip_faiss: bool = False,
    reverse_blocks: bool = False,
) -> DenseIndex:
    """Embed the corpus and build the FAISS index, streaming from a memmap.

    Two stages, both bounded in memory:
      1. ``_embed_store_to_memmap`` writes vectors into a memory-mapped matrix.
      2. FAISS is filled by iterating that memmap in slices, so the full matrix
         is never resident alongside the index.

    ``skip_faiss`` stops after stage 1. A flat FAISS index holds every vector in
    RAM (48.3M x 384 float32 = 74 GB for abstracts20m, on 31 GB hosts), so a
    corpus released as vectors ships the on-disk matrix plus
    ``embeddings.json`` and leaves the index choice to whoever loads it.
    """
    emb_path = index_dir / "embeddings.f32"
    dense_path = index_dir / "dense.faiss"

    if force:
        # A forced rebuild changes the chunk set, so every cached embedding is
        # stale. Reusing them would silently misalign dense rows against store
        # row ids and corrupt every dense-retrieval result.
        stale = list(index_dir.glob("embeddings-block*.npz"))
        if emb_path.exists():
            emb_path.unlink()
        for path in stale:
            path.unlink()
        log.info("force rebuild: discarded %d stale embedding block(s)", len(stale))

    n_chunks = len(store)
    batch_rows = EMBED_BLOCK_ROWS
    matrix, dim = _embed_store_to_memmap(
        store, cfg, emb_path, batch_rows=batch_rows,
        block_shard=block_shard, block_nshards=block_nshards,
        reverse_blocks=reverse_blocks,
    )
    if matrix is None:
        # Sharded worker: it wrote only its own block files and deliberately did
        # not open the shared memmap. Report coverage and stop.
        n_blocks = (n_chunks + batch_rows - 1) // batch_rows
        missing = missing_embedding_blocks(emb_path, n_chunks, n_blocks, batch_rows)
        import json as _json

        try:
            (index_dir / "embed_progress.json").write_text(_json.dumps({
                "n_blocks": n_blocks,
                "block_nshards": block_nshards,
                "n_present": n_blocks - len(missing),
                "missing_blocks": missing[:50],
                "n_missing": len(missing),
                "note": "blocks only; run block_nshards=1 to assemble the memmap and build FAISS",
            }, indent=2), encoding="utf-8")
        except OSError:
            pass
        log.warning(
            "shard %d/%d done: %d/%d blocks present, %d missing. "
            "FAISS deferred -- finish with --block-nshards 1 to assemble and build.",
            block_shard, block_nshards, n_blocks - len(missing), n_blocks, len(missing),
        )
        return None  # type: ignore[return-value]

    n = matrix.shape[0]
    if n != len(store):
        raise ValueError(
            f"dense/row misalignment: {n} vectors for {len(store)} chunks. "
            f"Delete {index_dir} and rebuild."
        )

    if skip_faiss:
        write_embedding_manifest(emb_path, n=n, dim=dim, embed_cfg=cfg.embed)
        log.info("FAISS skipped by request; matrix assembled at %s", emb_path)
        return None  # type: ignore[return-value]

    # Row ids are 0..n-1 by construction: the memmap is filled in row order and
    # every block's range is exactly its row range.
    index = DenseIndex(dim=dim, index_type=index_type)
    index.build_from_memmap(matrix, store.rows, batch_size=faiss_add_batch, row_ids=list(range(n)))
    return index


def build_index(
    cfg: PipelineConfig,
    chunks: Iterable[Chunk],
    *,
    index_dir: Path | None = None,
    store_cache_size: int = 200_000,
    force: bool = False,
    bm25_spill_dir: str | None = None,
    skip_dense: bool = False,
    skip_bm25: bool = False,
    max_index_cache: int = 300_000,
    block_shard: int = 0,
    block_nshards: int = 1,
    skip_faiss: bool = False,
    reverse_blocks: bool = False,
) -> dict[str, Any]:
    """Build every index artifact under ``index_dir``. Idempotent per stage.

    ``max_index_cache`` bounds how many chunks are held in RAM while building
    the BM25 and dense indexes. Anything beyond that is re-read from the store
    blob, which is slower but keeps peak memory bounded. Measured: caching all
    2.37M chunks costs ~14 GiB and leaves no room for the vector index.

    ``block_shard``/``block_nshards`` distribute dense embedding across nodes.
    Each node embeds a disjoint subset of blocks against the shared memmap; the
    FAISS index is built only by whichever node finds all blocks present.
    """
    index_dir = Path(index_dir) if index_dir else cfg.paths.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    report: dict[str, Any] = {"index_dir": str(index_dir)}

    # ---------------------------------------------------------------- store
    store_dir = index_dir / "store"
    if force and store_dir.exists():
        # Offsets are absolute blob positions, so a forced rebuild must start
        # from an empty store; otherwise new chunks append after stale bytes.
        shutil.rmtree(store_dir)
        log.info("force rebuild: removed existing store at %s", store_dir)
    if not (store_dir / "rowmap.json").exists():
        t0 = time.time()
        store = build_store_from_chunks(chunks, store_dir=store_dir, cache_size=store_cache_size)
        timings["store"] = time.time() - t0
        report["n_chunks"] = len(store)
    else:
        log.info("store exists at %s; reusing", store_dir)
        store = ChunkStore.load(store_dir)
        report["n_chunks"] = len(store)

    # The chunk cache is a speed knob, not a correctness requirement: anything
    # evicted is re-read from the store blob. It must be bounded, because
    # caching 2.37M chunk objects costs ~14 GiB of RSS and leaves no headroom
    # for the FAISS build.
    store.cache_size = max(10_000, min(store_cache_size, max_index_cache))
    log.info("store cache during index build: %d chunks (of %d total)",
             store.cache_size, len(store))
    if store._rfh is not None:
        store._rfh.close()
        store._rfh = None

    # ----------------------------------------------------------------- bm25
    #
    # Shard 0 owns BM25. Every other shard skips it.
    #
    # BM25 is a whole-corpus artifact with no notion of sharding, so the
    # `if not bm25_path.exists()` guard is not enough when several nodes start
    # together: all four saw no bm25.pkl, all four spent 2.8 hours building the
    # same index over the same store, and all four wrote the same file. The write
    # itself is atomic (tmp + rename) so nothing was corrupted -- it was purely
    # four-fold duplicated work, discovered only by watching all four processes
    # report identical "bm25: indexed 1550000/16133551 chunks" progress lines.
    #
    # A node assigned to dense blocks has no use for BM25; the retriever reads it
    # at query time, long after embedding is done.
    bm25_path = index_dir / "bm25.pkl"
    if skip_bm25:
        log.info("skipping BM25 by request")
        report["bm25"] = {"skipped": "by request"}
    elif block_shard != 0 and block_nshards > 1:
        log.info("shard %d/%d: skipping BM25 (owned by shard 0)",
                 block_shard, block_nshards)
        report["bm25"] = {"skipped": f"owned by shard 0 of {block_nshards}"}
    elif force or not bm25_path.exists():
        t0 = time.time()
        bm25 = build_bm25_from_store(store, cfg, spill_dir=bm25_spill_dir)
        bm25.save(bm25_path)
        timings["bm25"] = time.time() - t0
        report["bm25"] = bm25.stats()
    else:
        log.info("bm25 index exists; reusing %s", bm25_path)
        report["bm25"] = {"reused": True}

    # ---------------------------------------------------------------- dense
    if skip_dense:
        report["dense"] = {"skipped": True}
    else:
        dense_path = index_dir / "dense.faiss"
        if force or skip_faiss or not dense_path.exists():
            t0 = time.time()
            dense = build_dense_from_store(
                store, cfg, index_dir, force=force,
                block_shard=block_shard, block_nshards=block_nshards,
                skip_faiss=skip_faiss, reverse_blocks=reverse_blocks,
            )
            if dense is None and skip_faiss and block_nshards == 1:
                timings["dense"] = time.time() - t0
                report["dense"] = {"faiss": "skipped by request",
                                   "matrix": str(index_dir / "embeddings.f32"),
                                   "manifest": str(index_dir / "embeddings.json")}
            elif dense is None:
                # Sharded embedding incomplete; FAISS deferred to a later pass.
                report["dense"] = {"deferred": "awaiting other embedding shards"}
            else:
                dense.save(dense_path)
                timings["dense"] = time.time() - t0
                report["dense"] = dense.stats()
        else:
            log.info("dense index exists; reusing %s", dense_path)
            report["dense"] = {"reused": True}

    report["timings"] = {k: round(v, 1) for k, v in timings.items()}
    store.close()
    log.info("build_index complete: %s", report)
    return report
