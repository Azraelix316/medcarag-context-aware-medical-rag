"""Tests for sharded embedding: block publication and memmap assembly.

Regression guard for a measured, silent data-loss bug.

The original design had every node write its own byte range of one shared
``embeddings.f32`` memmap, on the reasoning that "blocks are disjoint byte ranges,
so nodes never write the same region". That reasoning is false on NFS.

Measured, two hosts, disjoint ranges, concurrent:

    writer 0: 4000/4000 rows correct
    writer 1:    0/4000 rows correct   ** entire contribution lost **

Each NFS client keeps its own page cache, so a writer's ``matrix.flush()`` can
push pages that overlap, in the server's view, the ranges another client already
committed. Disjointness of *offsets* does not imply disjointness of *flushes*.

The fix is that a block file is the unit of sharing: exactly one writer per
block, published by atomic rename, which POSIX and NFS both guarantee. A sharded
worker must therefore never open the shared memmap for writing. These tests pin
that invariant and the assembly pass that turns blocks into a matrix.
"""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from medrag.index.builder import (
    EMBED_BLOCK_ROWS,
    _sync_memmap_from_blocks,
    missing_embedding_blocks,
)


def _write_block(memmap_path: Path, block: int, rows: int, dim: int, marker: float) -> None:
    """Publish one block the way a shard does: temp file, then atomic rename."""
    block_path = memmap_path.parent / f"{memmap_path.stem}-block{block:05d}.npz"
    # np.savez forces a .npz suffix, so the temp name must already end in one.
    tmp = block_path.with_name(f"{block_path.stem}.tmp.npz")
    np.savez(tmp, embeddings=np.full((rows, dim), marker, dtype=np.float32))
    tmp.replace(block_path)


def test_shard_never_opens_shared_memmap() -> None:
    """A sharded worker must not create or open the memmap at all.

    This is the invariant that prevents the measured loss: if the file is never
    opened r+ by a shard, no page cache can overlap.
    """
    import inspect

    from medrag.index import builder

    src = inspect.getsource(builder._embed_store_to_memmap)
    # The guard must be keyed on sharding, and must gate BOTH the reuse copy and
    # the fresh write, since either would put shard pages in a shared cache.
    assert "shard_writes_memmap = block_nshards == 1" in src, (
        "sharded workers are no longer gated out of the memmap"
    )
    # The guard must appear three times: allocation, the reuse-copy path, and the
    # fresh-write path. Missing the allocation guard was a real bug: the memmap
    # was created before the sharding branch, so a shard opened it anyway and
    # hit a permission error on a file it should never have touched.
    assert src.count("if shard_writes_memmap:") >= 3, (
        f"memmap access not gated on allocation+reuse+write "
        f"(found {src.count('if shard_writes_memmap:')} guards, need >=3)"
    )
    # Allocation must be inside the guard, not merely guarded elsewhere. Match
    # calls only (np.memmap() not the type annotation or docstring prose).
    alloc = src.index("if shard_writes_memmap:")
    assert "np.memmap(" not in src[:alloc], (
        "the memmap is opened before the sharding guard, so shards still touch it"
    )
    print("  ok  sharded workers never open, create, or size the shared memmap")


def test_blocks_are_published_atomically() -> None:
    """Blocks appear whole or not at all; no partial file is ever visible."""
    with tempfile.TemporaryDirectory() as tmp:
        mm = Path(tmp) / "embeddings.f32"
        _write_block(mm, 0, rows=8, dim=4, marker=1.0)
        path = mm.parent / "embeddings-block00000.npz"
        assert path.exists()
        data = np.load(path)["embeddings"]
        assert data.shape == (8, 4), data.shape
        # No temp files left behind.
        assert not list(mm.parent.glob("*.tmp*")), list(mm.parent.glob("*.tmp*"))
    print("  ok  blocks are published whole via atomic rename")


def test_assembly_reproduces_every_block_exactly() -> None:
    """Assembly must place each block at its own row offset, unchanged."""
    dim, block_rows = 4, EMBED_BLOCK_ROWS
    n_blocks = 3
    n = n_blocks * block_rows
    with tempfile.TemporaryDirectory() as tmp:
        mm = Path(tmp) / "embeddings.f32"
        matrix = np.memmap(mm, dtype=np.float32, mode="w+", shape=(n, dim))
        truth = {}
        for b in range(n_blocks):
            arr = np.arange(block_rows * dim, dtype=np.float32).reshape(block_rows, dim)
            arr += b * 1_000_000.0
            truth[b] = arr
            _write_block(mm, b, block_rows, dim, 0.0)
            np.savez(mm.parent / f"embeddings-block{b:05d}.npz", embeddings=arr)

        copied = _sync_memmap_from_blocks(matrix, mm, n_blocks, block_rows)
        assert copied == n_blocks, copied
        for b in range(n_blocks):
            got = np.asarray(matrix[b * block_rows : (b + 1) * block_rows, :])
            assert np.array_equal(got, truth[b]), f"block {b} corrupted during assembly"
    print(f"  ok  {n_blocks} blocks assembled at correct offsets, bit-exact")


def test_assembly_rejects_a_mismatched_block() -> None:
    """A block from a different chunk set must fail loudly, not silently skew.

    Reusing embeddings after the chunker changes misaligns every dense row
    against store row ids, which corrupts retrieval with no other symptom.
    """
    with tempfile.TemporaryDirectory() as tmp:
        mm = Path(tmp) / "embeddings.f32"
        n, dim, rows = 100, 4, 10
        matrix = np.memmap(mm, dtype=np.float32, mode="w+", shape=(n, dim))
        # Wrong row count for block 1.
        np.savez(mm.parent / "embeddings-block00000.npz",
                 embeddings=np.zeros((rows, dim), np.float32))
        np.savez(mm.parent / "embeddings-block00001.npz",
                 embeddings=np.zeros((rows + 3, dim), np.float32))
        try:
            _sync_memmap_from_blocks(matrix, mm, 2, rows)
        except ValueError as exc:
            assert "shape" in str(exc), str(exc)
            print(f"  ok  wrong-sized block rejected: {str(exc)[:58]}...")
        else:
            raise AssertionError("a mis-shaped block was accepted silently")


def test_missing_blocks_are_detected() -> None:
    """Coverage reporting must name the gaps, so a partial run is visible."""
    with tempfile.TemporaryDirectory() as tmp:
        mm = Path(tmp) / "embeddings.f32"
        for b in (0, 2, 3):
            _write_block(mm, b, rows=4, dim=2, marker=1.0)
        missing = missing_embedding_blocks(mm, n=16, n_blocks=4, batch_rows=4)
        assert missing == [1], missing
    print("  ok  missing blocks identified exactly")


def test_shard_block_assignment_is_disjoint_and_complete() -> None:
    """The union of all shards' blocks must equal every block, with no overlap.

    Regression guard. The stride was written as ``range(0, n_blocks,
    block_nshards)``, which ignores ``block_shard`` entirely: every shard computed
    ``range(0, 3, 4) == [0]``, so all four workers redundantly embedded block 0
    and overwrote one another, and blocks 1 and 2 were never written. Observed as
    "1/3 blocks present" after four successful-looking workers, which is exactly
    the kind of silent under-completion this must catch.
    """
    import re

    from medrag.index import builder

    src = inspect.getsource(builder._embed_store_to_memmap)
    match = re.search(
        r"blocks = \(\s*range\(block_shard,\s*n_blocks,\s*block_nshards\)", src
    )
    assert match, (
        "block stride does not start at block_shard, so all shards embed the "
        "same first block"
    )

    # Simulate the assignment rule for a range of configurations.
    for n_blocks, nshards in [(3, 4), (119, 4), (1, 1), (10, 3), (7, 7), (8, 5)]:
        claimed = [
            list(range(s, n_blocks, nshards)) if 0 <= s < nshards else list(range(n_blocks))
            for s in range(nshards)
        ]
        flat = [b for c in claimed for b in c]
        assert sorted(flat) == list(range(n_blocks)), (
            f"n_blocks={n_blocks} nshards={nshards}: union {sorted(flat)} != all blocks"
        )
        assert len(flat) == len(set(flat)), (
            f"n_blocks={n_blocks} nshards={nshards}: blocks claimed twice: "
            f"{[b for b in set(flat) if flat.count(b) > 1]}"
        )
    print("  ok  shard block sets are disjoint and cover every block")


def test_block_rows_constant_matches_usage() -> None:
    """Guard the constant against drifting away from the call sites."""
    import inspect

    from medrag.index import builder

    assert builder.EMBED_BLOCK_ROWS == 20_000
    src = inspect.getsource(builder.build_dense_from_store)
    assert "EMBED_BLOCK_ROWS" in src, "call site bypasses the block-size constant"
    print(f"  ok  EMBED_BLOCK_ROWS={EMBED_BLOCK_ROWS} used consistently")


def test_skip_faiss_assembles_the_matrix_and_builds_no_index() -> None:
    """--no-faiss writes embeddings.f32 + embeddings.json and never builds FAISS.

    A flat FAISS index holds every vector in RAM; 48.3M x 384 float32 is 74 GB,
    which a 31 GB host cannot hold. The on-disk matrix is the release artifact.
    """
    import json

    from medrag.config import PipelineConfig
    from medrag.data.schema import Chunk
    from medrag.index import builder

    class FakeEmbedder:
        dim = 3

        def load(self):
            return self

        def embed(self, texts, is_query=False, batch_size=64):
            return [[float(len(t)), 1.0, 0.0] for t in texts]

    def no_faiss(*a, **k):
        raise AssertionError("DenseIndex built despite skip_faiss")

    chunks = [Chunk(chunk_id=f"d{i}:0", doc_id=f"d{i}", text="x" * (i + 1)) for i in range(5)]
    real_embedder, real_dense = builder.build_embedder, builder.DenseIndex
    builder.build_embedder, builder.DenseIndex = (lambda cfg: FakeEmbedder()), no_faiss
    try:
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            store = builder.build_store_from_chunks(chunks, store_dir=index_dir / "store")
            cfg = PipelineConfig()
            out = builder.build_dense_from_store(store, cfg, index_dir, skip_faiss=True)
            store.close()
            assert out is None
            assert not (index_dir / "dense.faiss").exists()
            manifest = json.loads((index_dir / "embeddings.json").read_text())
            assert manifest["shape"] == [5, 3], manifest
            assert manifest["model"] == cfg.embed.model_name
            mat = np.memmap(index_dir / "embeddings.f32", dtype="<f4", mode="r", shape=(5, 3))
            assert [float(v) for v in mat[:, 0]] == [1.0, 2.0, 3.0, 4.0, 5.0], mat[:, 0]
            assert (index_dir / "embeddings.f32").stat().st_size == manifest["bytes"]
            del mat
    finally:
        builder.build_embedder, builder.DenseIndex = real_embedder, real_dense
    print("  ok  skip_faiss assembles the on-disk matrix + manifest, no FAISS")


def test_reverse_blocks_walks_the_shard_backwards_and_skips_done_blocks() -> None:
    """--reverse-blocks visits the same blocks last-to-first and still skips done ones."""
    from medrag.config import PipelineConfig
    from medrag.data.schema import Chunk
    from medrag.index import builder

    seen: list[int] = []

    class FakeEmbedder:
        dim = 2

        def load(self):
            return self

        def embed(self, texts, is_query=False, batch_size=64):
            seen.append(len(texts[0]))  # first row of the block identifies it
            return [[1.0, 0.0] for _ in texts]

    # 10 rows, 2 rows per block -> blocks 0..4; shard 0 of 2 owns 0, 2, 4.
    chunks = [Chunk(chunk_id=f"d{i}:0", doc_id=f"d{i}", text="x" * (i + 1)) for i in range(10)]
    real = builder.build_embedder
    builder.build_embedder = lambda cfg: FakeEmbedder()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            store = builder.build_store_from_chunks(chunks, store_dir=index_dir / "store")
            mm = index_dir / "embeddings.f32"
            _write_block(mm, 2, rows=2, dim=2, marker=1.0)   # block 2 already done
            builder._embed_store_to_memmap(store, PipelineConfig(), mm, batch_rows=2,
                                           block_shard=0, block_nshards=2, reverse_blocks=True)
            store.close()
    finally:
        builder.build_embedder = real
    # Block 4 starts at row 8 (text length 9), block 0 at row 0 (length 1).
    assert seen == [9, 1], seen
    print("  ok  reverse_blocks visits 4 then 0 and skips the finished block 2")


def main() -> int:
    tests = [
        test_shard_never_opens_shared_memmap,
        test_blocks_are_published_atomically,
        test_assembly_reproduces_every_block_exactly,
        test_assembly_rejects_a_mismatched_block,
        test_missing_blocks_are_detected,
        test_shard_block_assignment_is_disjoint_and_complete,
        test_block_rows_constant_matches_usage,
        test_skip_faiss_assembles_the_matrix_and_builds_no_index,
        test_reverse_blocks_walks_the_shard_backwards_and_skips_done_blocks,
    ]
    print("Embedding sharding tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())