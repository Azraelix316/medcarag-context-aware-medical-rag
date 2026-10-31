"""Build the shared chunk store from already-chunked shard files.

``build-index`` takes a chunk stream and builds the store itself, guarded by
``if not (store_dir / "rowmap.json").exists()`` so that concurrent nodes reuse it
rather than each rebuilding it. That guard is correct but it assumes one node wins
the race and does the work.

For the 16.1M-chunk PMC corpus that assumption is wasteful in a specific way: the
stream is ``chunk_corpus(iter_documents(cfg))``, so every node that loses the race
still spends ~30 minutes re-chunking 555,142 documents only to find the store
already exists. Four nodes, four chunkings, three of them pointless -- and the ones
that do build it race on the same output directory.

This builds the store once, from the chunk shards the two nodes already produced,
in a single explicit step before any embedder starts. It also means the store is
built from the exact chunks that were verified by independent recount (16,133,551
across 3,228 shards) rather than from a second, unverified chunking pass.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.config import PipelineConfig
from medrag.data.schema import Chunk
from medrag.index.builder import build_store_from_chunks

log = logging.getLogger("build_pmc_store")


def chunk_stream(paths: list[str]):
    """Yield Chunks from gzip JSONL shards, in a deterministic order."""
    for path in paths:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                yield Chunk.from_dict(json.loads(line))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--chunk-dir", required=True)
    ap.add_argument("--store-cache", type=int, default=400_000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = PipelineConfig.load(args.config)._finalize()

    paths = sorted(glob.glob(f"{args.chunk_dir}/*.jsonl.gz"))
    if not paths:
        print(f"no chunk shards under {args.chunk_dir}")
        return 1
    log.info("building store from %d chunk shards", len(paths))

    # Count first: the store must not be built from a partial read, and an
    # unexplained shortfall is exactly what has to be caught here.
    t0 = time.time()
    expected = 0
    for path in paths:
        with gzip.open(path, "rt") as fh:
            expected += sum(1 for line in fh if line.strip())
    log.info("counted %d chunks in %d shards (%.0f s)", expected, len(paths),
             time.time() - t0)

    store_dir = Path(cfg.paths.index_dir) / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    store = build_store_from_chunks(chunk_stream(paths), store_dir=store_dir,
                                    cache_size=args.store_cache)
    built = len(store)
    log.info("store holds %d rows in %.0f s", built, time.time() - t0)

    if built != expected:
        log.error("STORE INCOMPLETE: built %d of %d chunks read from shards",
                  built, expected)
        return 1
    print(json.dumps({"chunks_read": expected, "store_rows": built,
                      "shards": len(paths),
                      "store_dir": str(store_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())