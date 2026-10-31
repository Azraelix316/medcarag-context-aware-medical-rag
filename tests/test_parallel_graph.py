"""scripts/parallel_graph.py: the merged parts equal one DependencyGraph.build over the store."""

from __future__ import annotations

import gzip
import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from medrag.data.schema import Chunk  # noqa: E402
from medrag.index.builder import build_store_from_chunks  # noqa: E402
from medrag.index.dependency_graph import DependencyGraph  # noqa: E402
from medrag.index.store import ChunkStore  # noqa: E402

spec = importlib.util.spec_from_file_location("parallel_graph", "scripts/parallel_graph.py")
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)

KINDS = ["background", "methods", "results", "discussion", "conclusions", "references", ""]


def test_merged_parts_equal_one_build() -> None:
    rng = random.Random(5)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        chunk_dir = d / "chunks"
        chunk_dir.mkdir()
        all_chunks = []
        for s in range(9):
            shard = []
            for i in range(rng.randint(0, 12)):         # some shards hold no documents
                doc = f"pmcid:PMC{s:02d}{i:03d}"
                for c in range(rng.randint(1, 9)):
                    shard.append(Chunk(chunk_id=f"{doc}#{c:04d}", doc_id=doc, text="x",
                                       section_kind=rng.choice(KINDS)))
            all_chunks += shard
            stem = f"pmc-{s:05d}"
            with gzip.open(chunk_dir / f"{stem}.chunks.jsonl.gz", "wb") as fh:
                for c in shard:
                    fh.write(json.dumps(c.to_dict()).encode() + b"\n")
            (chunk_dir / f"{stem}.done").write_text(json.dumps({"n_chunks": len(shard)}))
        build_store_from_chunks(all_chunks, store_dir=d / "store")
        store = ChunkStore.load(d / "store")
        ref = DependencyGraph.build(
            store.rows, kind_of=lambda r: store.get(r).section_kind or "",
            doc_of=lambda r: store.get(r).doc_id or "")

        stems = pg.shard_stems(chunk_dir)
        counts = {s: json.loads((chunk_dir / f"{s}.done").read_text())["n_chunks"] for s in stems}
        for n_parts in (1, 2, 4, 9):
            parts, row0 = [], 0
            for rng_ in pg.part_ranges(stems, n_parts):
                parts.append(pg.build_part(chunk_dir, rng_, row0))
                row0 += sum(counts[s] for s in rng_)
            got = pg.merge_parts(parts)
            assert got.n_nodes == ref.n_nodes
            for k in ("indptr", "indices", "weights", "doc_start", "doc_end"):
                assert np.array_equal(getattr(got, k), getattr(ref, k)), (n_parts, k)
                assert getattr(got, k).dtype == getattr(ref, k).dtype, (n_parts, k)
            assert got.doc_ids == ref.doc_ids and list(got.section_kind) == list(ref.section_kind)
    print(f"  merged parts == DependencyGraph.build over the store ({ref.n_nodes} rows, "
          f"{len(ref.indices)} directed edges, 1/2/4/9 parts)")


def main() -> int:
    tests = [test_merged_parts_equal_one_build]
    print("parallel dependency graph")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
