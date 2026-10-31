"""Build dense.faiss from a LOCAL copy of the embedding memmap.

The memmap lives on NFS. FAISS streams it in 100,000-row batches while holding the
vectors it has accumulated, and by the time it holds ~23 GiB there is no page cache
left for a 24.78 GiB file -- so every batch goes back to the network. Measured on
this host: 79.9 GB read from disk to get through a 24.78 GB file, at roughly 5 MB/s,
with the process in D (disk sleep) at 0% CPU for over an hour.

This is the same lesson the chunk store already taught here: NFS random access is
~10x slower than local, and a multi-gigabyte file that is streamed repeatedly needs
to be local. The copy costs about eight minutes once and turns a multi-hour stall
into a few minutes.

The index itself is still written to the shared location, since that is a single
sequential write of 24.8 GiB and NFS handles that fine.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.config import PipelineConfig
from medrag.index.dense import DenseIndex
from medrag.index.store import ChunkStore

N_CHUNKS = 16_133_551
DIM = 384
LOCAL = Path("emb_local.f32")
INDEX_DIR = Path("data/pmcmerged/index")

log = logging.getLogger("build_dense_local")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    expected = N_CHUNKS * DIM * 4
    actual = LOCAL.stat().st_size
    if actual != expected:
        log.error("local memmap is %d bytes, expected %d -- copy incomplete",
                  actual, expected)
        return 1
    log.info("local memmap verified: %d bytes = %d x %d", actual, N_CHUNKS, DIM)

    matrix = np.memmap(LOCAL, dtype=np.float32, mode="r", shape=(N_CHUNKS, DIM))
    # A memmap can be a short file if a copy was interrupted; check the ends read.
    if float(np.linalg.norm(matrix[0])) <= 0 or float(np.linalg.norm(matrix[-1])) <= 0:
        log.error("first or last row of the local memmap is zero")
        return 1

    store = ChunkStore.load(INDEX_DIR / "store", cache_size=20_000)
    log.info("store has %d rows", len(store))
    assert len(store) == N_CHUNKS, f"store has {len(store)} rows, expected {N_CHUNKS}"

    t0 = time.time()
    # 4 shards: FAISS's IndexFlat doubles its internal vector, so a single index of
    # 16,133,551 x 384 needs a ~37 GiB transient and died with std::bad_alloc on this
    # 31 GiB host. Sharding is exact -- every shard is exhaustive and the search
    # merges the per-shard top-k -- so recall is unchanged, only the merge is extra.
    index = DenseIndex(dim=DIM)
    index.build_from_memmap(matrix, store.rows, batch_size=200_000,
                            row_ids=list(range(N_CHUNKS)), flat_shards=4)
    log.info("FAISS built in %.0f s", time.time() - t0)

    # save() returns the path it actually wrote: for a sharded index that is the
    # manifest, with the shard files beside it. Renaming a `.tmp` here was wrong --
    # save() does its own atomic replace per shard and never creates that file, so the
    # script died on a missing path *after* a complete, correct index was written.
    out = index.save(INDEX_DIR / "dense.faiss")
    man = json.loads((INDEX_DIR / "dense.faiss.manifest.json").read_text())
    total = sum((INDEX_DIR / n).stat().st_size for n in man["shards"])
    log.info("wrote %s plus %d shards (%.2f GB) in %.0f s total",
             out, len(index.shards), total / 1e9, time.time() - t0)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())