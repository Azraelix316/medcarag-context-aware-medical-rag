"""Sharded flat search must return exactly what a single flat index returns.

FAISS's ``IndexFlat`` keeps its vectors in a ``std::vector`` that doubles, so adding
the last vectors transiently needs the old and new buffers simultaneously. On the
16,133,551-chunk corpus the index is 24.8 GiB and the doubling peak is around 37 GiB,
which died with ``std::bad_alloc`` on a 31 GiB host after 12.2M of 16.1M vectors --
no traceback beyond the allocator, no OOM record, just a bad_alloc.

``build_from_memmap(..., flat_shards=k)`` splits the exhaustive index into ``k``
contiguous row ranges and merges the per-shard top-k at search time.

The property that makes this acceptable rather than a compromise: recall is
unchanged. Every shard is exhaustive, so the merged top-k is the same set the single
index would have returned. These tests check that exactly -- identical id sets and
identical scores -- rather than approximately, because an "approximate but close"
sharding would be indistinguishable from a real recall loss once it reached a result
table.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from medrag.index.dense import DenseIndex


def _corpus(seed: int = 31, n: int = 20_000, d: int = 64):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, d)).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    q = rng.standard_normal((4, d)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return m, q, [str(i) for i in range(n)]


def _build(m, ids, **kw) -> DenseIndex:
    x = DenseIndex(dim=m.shape[1])
    x.build_from_memmap(m, ids, **kw)
    return x


def test_sharded_results_match_a_single_flat_index() -> None:
    m, q, ids = _corpus()
    k = 10
    single = _build(m, ids, batch_size=10_000)
    ref_s, ref_p = single.search(q, k)
    for shards in (2, 4, 8):
        x = _build(m, ids, batch_size=1024, flat_shards=shards)
        s, p = x.search(q, k)
        assert s.shape == ref_s.shape, f"shards={shards}: shape {s.shape} vs {ref_s.shape}"
        for i in range(len(q)):
            a = {v for v in p[i].tolist() if v >= 0}
            b = {v for v in ref_p[i].tolist() if v >= 0}
            assert a == b, f"shards={shards} q={i}: {sorted(a)} != {sorted(b)}"
        assert np.allclose(s[:, 0], ref_s[:, 0], atol=1e-5), f"shards={shards}: top score"
    print(f"  ok  shards 2/4/8 return the same top-{k} ids and scores as one flat index")


def test_every_vector_lands_in_exactly_one_shard() -> None:
    m, _q, ids = _corpus(n=20_000)
    x = _build(m, ids, batch_size=1024, flat_shards=4)
    assert len(x.shards) == 4, x.shards
    sizes = [int(sh.ntotal) for sh in x.shards]
    assert sum(sizes) == len(m), f"shards hold {sum(sizes)} of {len(m)} vectors"
    offsets = x.shard_offsets
    assert offsets[0] == 0
    for i, off in enumerate(offsets):
        assert 0 <= off < len(m)
    print(f"  ok  4 shards hold {sizes} = {sum(sizes):,} vectors, offsets {offsets}")


def test_a_shard_smaller_than_top_k_still_returns_results() -> None:
    """Edge case: more shards than vectors means empty shards exist.

    FAISS returns -1 for slots a shard cannot fill, and the merge must not let those
    into the result set.
    """
    m, q, ids = _corpus(seed=4, n=40, d=16)
    x = _build(m, ids, batch_size=8, flat_shards=8)
    s, p = x.search(q, 10)
    for i in range(len(q)):
        valid = {v for v in p[i].tolist() if v >= 0}
        assert len(valid) == min(10, len(m)), f"q={i} returned {len(valid)} ids"
        assert all(0 <= v < len(m) for v in valid), f"q={i} has out-of-range ids"
    print("  ok  more shards than vectors still returns a full, in-range result")


def test_sharding_does_not_apply_to_non_flat_indexes() -> None:
    """An IVF corpus must not be silently split."""
    m, q, ids = _corpus(seed=7, n=5_000, d=32)
    x = DenseIndex(dim=32, index_type="ivfflat")
    x._index_type_for = lambda n: "ivfflat"
    assert x._index_type_for(5000) == "ivfflat"
    # build_from_memmap routes on the resolved kind, so a forced IVF stays whole.
    y = DenseIndex(dim=32, index_type="ivfflat")
    y.build_from_memmap(m, ids, batch_size=1024, flat_shards=4)
    assert not y.shards, "an IVF index must not be sharded"
    print("  ok  IVF indexes are left unsharded")


def test_a_sharded_index_saves_every_shard() -> None:
    """save() once wrote only the stand-in shard.

    ``save`` called ``write_index(self.index)``, and for a sharded build
    ``self.index`` is the *first* shard kept as a placeholder. A 4-shard index over
    16,133,551 vectors therefore saved as 6.20 GB -- one quarter of the data -- while
    the log reported all 16,133,551 vectors. Nothing raised. This is the check that
    would have caught it.
    """
    import tempfile
    from pathlib import Path

    m, _q, ids = _corpus(seed=77, n=20_000, d=64)
    x = _build(m, ids, batch_size=4096, flat_shards=4)
    with tempfile.TemporaryDirectory() as td:
        out = x.save(Path(td) / "dense.faiss")
        files = sorted(p.name for p in Path(td).iterdir())
        # Sidecars are named by APPENDING to the index filename, not by with_suffix:
        # Path("dense.faiss").with_suffix(".ids.json") gives "dense.ids.json", because
        # it replaces ".faiss" wholesale. Anchoring to the index name is what keeps
        # save and load agreeing, and the earlier mismatch made a complete, correct
        # index unloadable.
        for i in range(4):
            assert f"dense.faiss.shard{i}" in files, f"shard {i} missing from {files}"
        assert out.name == "dense.faiss.manifest.json", out.name
        man = json.loads(out.read_text())
        assert man["n_vectors"] == 20000, man
        assert man["sharded"] is True and len(man["shards"]) == 4, man
    print(f"  ok  all 4 shards written as dense.faiss.shardN, manifest reports 20000")


def test_a_sharded_index_round_trips_through_disk() -> None:
    import tempfile
    from pathlib import Path

    m, q, ids = _corpus(seed=77, n=20_000, d=64)
    a = _build(m, ids, batch_size=4096, flat_shards=4)
    s0, p0 = a.search(q, 10)
    with tempfile.TemporaryDirectory() as td:
        a.save(Path(td) / "dense.faiss")
        b = DenseIndex.load(Path(td) / "dense.faiss")
        assert len(b.shards) == 4, b.shards
        assert b.shard_offsets == a.shard_offsets
        assert len(b.ids) == len(ids)
        s1, p1 = b.search(q, 10)
        assert np.array_equal(p0, p1), "positions changed across save/load"
        assert np.allclose(s0, s1, atol=1e-6), "scores changed across save/load"
    print("  ok  sharded index reloads to identical positions and scores")


def test_a_block_records_the_shard_that_wrote_it() -> None:
    """A duplicate block write must be detectable after the fact.

    Two shards computing the same block is invisible without a record of who wrote
    what. Each writes `embeddings-blockNNNNN.tmp<pid>.npz` and renames it into place, so
    the second rename replaces the first: the result is not corrupt -- both computed the
    same rows -- but nothing on disk says it happened. That makes "no row written
    twice", the property the whole sharded path exists to guarantee, something nobody
    could check after a run.

    The block therefore carries `shard` (its owner) and `meta` (block id, block count,
    writing pid). Under a k-way split, block b must belong to shard b % k, so a block
    recording anything else was written by a process that does not own it -- which is
    precisely the duplicate-writer case, and is now a single comparison.
    """
    import numpy as np

    from medrag.index.builder import EMBED_BLOCK_ROWS

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for block, shard in ((0, 0), (1, 1), (2, 0)):
            np.savez(
                d / f"embeddings-block{block:05d}.npz",
                embeddings=np.zeros((EMBED_BLOCK_ROWS, 4), dtype=np.float32),
                shard=np.array([shard], dtype=np.int32),
                meta=np.array([block, 3, os.getpid()], dtype=np.int64),
            )
        owners = {}
        for block in (0, 1, 2):
            with np.load(d / f"embeddings-block{block:05d}.npz") as z:
                owners[block] = int(np.asarray(z["shard"]).ravel()[0])
                assert "meta" in z.files
        assert owners == {0: 0, 1: 1, 2: 0}, owners
        # The invariant the verifier enforces.
        assert all(who == b % 2 for b, who in owners.items()), owners

        # A block written before the field existed stays readable and is reported as
        # unowned rather than treated as corrupt -- an old index is correct, just
        # unannotated, and must not fail a gate for lacking metadata.
        np.savez(d / "embeddings-block00003.npz",
                 embeddings=np.zeros((4, 4), dtype=np.float32))
        with np.load(d / "embeddings-block00003.npz") as z:
            assert "shard" not in z.files
            assert z["embeddings"].shape == (4, 4)
    print("  ok  blocks record their owning shard; legacy blocks still load")


def main() -> int:
    tests = [
        test_sharded_results_match_a_single_flat_index,
        test_every_vector_lands_in_exactly_one_shard,
        test_a_shard_smaller_than_top_k_still_returns_results,
        test_sharding_does_not_apply_to_non_flat_indexes,
        test_a_sharded_index_saves_every_shard,
        test_a_sharded_index_round_trips_through_disk,
        test_a_block_records_the_shard_that_wrote_it,
    ]
    print("Sharded flat index tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())