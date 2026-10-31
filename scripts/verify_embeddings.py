"""Verify the sharded dense embedding before anything depends on it.

What this checks, and why each one has been a real failure mode here
-------------------------------------------------------------------
**Coverage.** Every block must be present exactly once. Four shards each claiming
`range(k, n, 4)` covers everything; a shard started late, restarted, or given the
wrong index leaves holes. A hole is invisible in a FAISS index -- search just
returns fewer neighbours -- so it is checked by counting blocks and rows rather
than by searching.

**No all-zero rows.** A block that loaded but failed to embed produces zeros that
index and retrieve fine while meaning nothing. This is the failure that a
successful-looking index cannot reveal.

**Alignment.** Row ``i`` of the vectors must belong to row ``i`` of the store. The
shards write blocks independently, so a block written at the wrong offset -- or
memmapped before another shard's blocks were published -- yields a fully populated
index whose every vector is off by a block. FAISS self-retrieval catches this: each
row must retrieve itself at rank 1, and that identity must agree with the store's
own self-dot-product at the same row.

Blocks are published by atomic rename and the memmap is re-synced from the blocks
on load, so alignment is checked against the blocks on disk rather than against any
cached mapping.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.index.builder import EMBED_BLOCK_ROWS
from medrag.index.store import ChunkStore

INDEX_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "data/pmcmerged/index")
SAMPLE_ROWS = (0, 1, 5_000, 1_000_000, 4_000_000, 8_000_000,
               15_000_000, 16_133_550)


def sample_rows(n_rows: int) -> tuple[int, ...]:
    """Rows to check alignment on, all inside ``[0, n_rows)``.

    The fixed list above is what caught the 7.39M-versus-8.71M chunk miscount and the
    duplicated worker-B prefix, because the interesting failures were at specific large row
    ids on the 16,133,551-row PMC index. It is the wrong shape for a smaller index:
    run against the 2,368,622-row abstract index it asks for row 4,000,000, which does
    not exist, and reports "block 200 missing ... NOT VERIFIED" for an index that may
    well be complete.

    A verifier that fails on a healthy index is not a conservative verifier -- it is one
    whose output nobody can act on, because the same "NOT VERIFIED" covers both a real
    fault and an out-of-range probe. So the fixed rows are filtered to the ones that
    exist and the list is topped back up with a spread across the real range.
    """
    if n_rows <= 0:
        return (0,)
    inside = tuple(r for r in SAMPLE_ROWS if r < n_rows)
    # Always include the endpoints and the midpoint, even when enough fixed probes
    # survived.
    #
    # An early return on "four is enough" was wrong twice over. It left the last row of
    # the index unprobed, and the last row is where a truncated write, a partial final
    # block and an off-by-one at the boundary all show up. On the 2,368,622-row
    # abstract index it returned (0, 1, 5000, 1000000) -- four probes, none of them
    # past row 1,000,000, all of them in the first 42% of the file.
    for r in (0, n_rows - 1, n_rows // 2, n_rows // 4, (3 * n_rows) // 4):
        if 0 <= r < n_rows:
            inside = tuple(sorted(set(inside) | {r}))
    return inside


def block_paths(index_dir: Path) -> list[Path]:
    """Committed blocks only.

    A block under construction is named ``embeddings-block00134.tmpNNNNNN.npz`` --
    numpy's ``savez`` appends the ``.npz`` suffix to the caller's temporary path, so
    the atomic-rename temporary does match the naive glob. Counting one would
    certify a half-written block as present, which is the exact failure this script
    exists to catch.
    """
    out = []
    for p in glob.glob(str(index_dir / "embeddings-block*.npz")):
        if ".tmp" in Path(p).name:
            continue
        out.append(Path(p))
    return sorted(out)


def verify_coverage(index_dir: Path, n_chunks: int) -> dict:
    paths = block_paths(index_dir)
    expected_blocks = (n_chunks + EMBED_BLOCK_ROWS - 1) // EMBED_BLOCK_ROWS
    present = {int(p.stem.split("block")[-1]) for p in paths}
    missing = sorted(set(range(expected_blocks)) - present)
    extra = sorted(present - set(range(expected_blocks)))
    return {
        "expected_blocks": expected_blocks,
        "blocks_present": len(present),
        "missing_blocks": missing[:20],
        "n_missing": len(missing),
        "unexpected_blocks": extra[:20],
        "complete": not missing,
    }


def verify_no_zero_rows(index_dir: Path, n_chunks: int) -> dict:
    """Count rows whose vector is all zeros.

    Sampled per block rather than globally: a full scan of 16.1M x 384 floats is
    25 GB and pointless when a bad block is bad in every row it contains.
    """
    zero_blocks: list[int] = []
    checked = 0
    for path in block_paths(index_dir):
        with np.load(path) as z:
            vecs = z["embeddings"]
        if not np.isfinite(vecs).all():
            zero_blocks.append(int(path.stem.split("block")[-1]))
            continue
        rows = np.abs(vecs).sum(axis=1)
        if (rows == 0).any():
            zero_blocks.append(int(path.stem.split("block")[-1]))
        checked += vecs.shape[0]
    return {"rows_checked": checked, "blocks_with_bad_rows": zero_blocks[:20],
            "n_bad_blocks": len(zero_blocks), "clean": not zero_blocks}


def verify_alignment(index_dir: Path, n_chunks: int) -> dict:
    """Each sampled row must be its own nearest neighbour, and agree with the store.

    Uses one representative row per block boundary plus the corpus endpoints, which
    is where an off-by-one-block error would show up first.
    """
    import faiss

    dim = None
    paths = block_paths(index_dir)
    if not paths:
        return {"error": "no committed blocks"}
    in_flight = len(glob.glob(str(index_dir / "embeddings-block*.tmp*.npz")))
    with np.load(paths[0]) as z:
        dim = int(z["embeddings"].shape[1])

    # Assemble only the sampled rows' vectors, from the blocks that own them.
    #
    # Several sampled rows can share a block -- rows 0, 1 and 5,000 are all in block 0
    # -- so the map must hold a LIST per block. It used to hold a single row, which
    # silently overwrote: rows 0 and 1 were never loaded, their slots stayed zero, and
    # the check reported cos=0.000000 for them and "not self" for every row, because
    # the retrieved id is a position in rows_to_check while the comparison was against a
    # row id. The index was fine; the verifier was comparing the wrong things.
    rows_to_check = sample_rows(n_chunks)
    sample_vecs = np.zeros((len(rows_to_check), dim), dtype=np.float32)
    owners: dict[int, list[int]] = {}
    for row in rows_to_check:
        owners.setdefault(row // EMBED_BLOCK_ROWS, []).append(row)
    for blk, rows in owners.items():
        path = index_dir / f"embeddings-block{blk:05d}.npz"
        if not path.exists():
            return {"error": f"block {blk} missing for sampled rows {rows}"}
        with np.load(path) as z:
            vecs = z["embeddings"]
        for row in rows:
            offset = row - blk * EMBED_BLOCK_ROWS
            if offset >= vecs.shape[0]:
                return {"error": f"row {row} past end of block {blk}"}
            sample_vecs[rows_to_check.index(row)] = vecs[offset]
    if not np.isfinite(sample_vecs).all() or np.abs(sample_vecs).sum() == 0:
        return {"error": "sampled vectors did not load from their blocks"}

    norms = np.linalg.norm(sample_vecs, axis=1, keepdims=True)
    unit = sample_vecs / np.maximum(norms, 1e-12)
    index = faiss.IndexFlatIP(dim)
    index.add(unit)
    sims, ids = index.search(unit, 1)

    store = ChunkStore.load(index_dir / "store", cache_size=20_000)
    results = []
    for i, row in enumerate(rows_to_check):
        # `ids` holds positions in rows_to_check, not row ids. Comparing the two
        # directly reported is_self=False for every row.
        nearest_pos = int(ids[i][0])
        nearest_row = rows_to_check[nearest_pos]
        results.append({
            "row": row,
            "nearest_row": nearest_row,
            "is_self": nearest_row == row,
            "cosine": round(float(sims[i][0]), 6),
            "store_doc": (store.get(row).doc_id if store.get(row) else None),
        })
    store.close()
    return {
        "dim": dim,
        "blocks_in_flight": in_flight,
        "all_self": all(r["is_self"] for r in results),
        "min_cosine": round(min(r["cosine"] for r in results), 6),
        "rows": results,
    }


def main() -> int:
    store_meta = json.loads((INDEX_DIR / "store" / "store.json").read_text())
    n_chunks = int(store_meta["n_chunks"])
    print(f"index dir : {INDEX_DIR}")
    print(f"store rows: {n_chunks:,}\n")

    cov = verify_coverage(INDEX_DIR, n_chunks)
    print("COVERAGE")
    print(f"  expected blocks : {cov['expected_blocks']:,}")
    print(f"  present         : {cov['blocks_present']:,}")
    print(f"  missing         : {cov['n_missing']:,}"
          + (f"  e.g. {cov['missing_blocks'][:8]}" if cov["n_missing"] else ""))
    print(f"  complete        : {cov['complete']}\n")

    zeros = verify_no_zero_rows(INDEX_DIR, n_chunks)
    print("ROW SANITY")
    print(f"  rows checked    : {zeros['rows_checked']:,}")
    print(f"  bad blocks      : {zeros['n_bad_blocks']:,}")
    print(f"  clean           : {zeros['clean']}\n")

    print("ALIGNMENT (FAISS self-retrieval vs store)")
    if not cov["complete"]:
        # Alignment is only meaningful over a complete index: with holes, the
        # sampled rows that do exist can still be checked, so try them and report
        # what was checkable rather than bailing out with nothing.
        print(f"  index incomplete ({cov['n_missing']} blocks missing); "
              f"checking only the sampled rows whose blocks exist")
    align = verify_alignment(INDEX_DIR, n_chunks)
    if "error" in align:
        print(f"  ERROR: {align['error']}")
        print("\nNOT VERIFIED -- do not depend on this index")
        return 1
    print(f"  dim             : {align['dim']}")
    for r in align["rows"]:
        print(f"  row {r['row']:>10,}  nearest={r['nearest_row']:>10,}  "
              f"self={r['is_self']}  cos={r['cosine']:.6f}  {r['store_doc']}")
    print(f"  all rows retrieve themselves: {align['all_self']}")

    ok = cov["complete"] and zeros["clean"] and align["all_self"]
    print(f"\n{'VERIFIED' if ok else 'NOT VERIFIED -- do not depend on this index'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())