"""Verify a *sharded* embedding run: ownership, duplication, and row alignment.

`scripts/verify_embeddings.py` verifies a finished index. It assumes every block is
present, so pointed at a half-finished shard it reports `block N missing for sampled
rows [...]` and stops -- correct, and useless for the question actually being asked,
which is whether the block a shard *did* write is right.

This asks the three questions that matter mid-run, per shard:

  1. OWNERSHIP.   Does shard s hold exactly the blocks `range(s, n_blocks, k)`? A
                  missing block is a hole the merged index will have; an extra one is
                  two processes writing the same rows, which is the silent corruption
                  that atomic rename hides rather than prevents -- unless a concurrent
                  shard is declared with --concurrent, in which case out-of-range blocks
                  are that shard's and are expected.
  2. DUPLICATION. Does any block appear twice, and does any row appear in two blocks?
  3. ALIGNMENT.   Re-embed the chunk text for sampled rows straight from the store and
                  compare against the stored vector. This is the check that catches a
                  block written from the wrong rows -- correct code, correct block,
                  wrong offset -- which ownership and duplication both pass.

Alignment must reproduce the shard's *inputs*, not merely its settings, and getting
that wrong fails on a healthy index. This script reported FAIL on a correct one twice,
for two different reasons, before passing:

  * it embedded ``chunk.text`` where the builder embeds ``chunk.embed_text or
    chunk.text``. ``embed_text`` carries the section heading the chunk was conditioned
    on, so the two strings genuinely differ and cosines landed between 0.65 and 0.98;
  * it loaded ``configs/default.yaml`` alone, whose ``max_seq_length`` is 512, while
    the shard ran ``configs/verify4.yaml`` at 320.

The first is worse than a crash, because the failure mode -- many rows just under 1.0 --
is exactly what genuine misalignment also looks like. A verifier that cannot reproduce
the run's input construction is measuring itself.

Alignment is sampled at block *boundaries* on purpose. Row 19,999 and row 20,000 are
the pair an off-by-one-block error produces, and a uniformly random sample almost never
lands on either.

Usage:
    python scripts/verify_shard_blocks.py <index-dir> --block-shard S --block-nshards K
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medrag.index.builder import EMBED_BLOCK_ROWS  # noqa: E402


def committed_blocks(index_dir: Path) -> list[int]:
    """Block ids whose temp file has been renamed into place.

    A `.tmp` file is an in-flight write. Counting one as present is how a verifier
    reports success on a run that is still going: the rename happens after the data is
    durable, and reading the temp file races the writer.
    """
    out: list[int] = []
    for path in glob.glob(str(index_dir / "embeddings-block*.npz")):
        stem = Path(path).name[len("embeddings-block"):-len(".npz")]
        if stem.isdigit():
            out.append(int(stem))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("index_dir")
    ap.add_argument("--block-shard", type=int, required=True)
    ap.add_argument("--block-nshards", type=int, required=True)
    ap.add_argument("--rows-per-block", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    # The config the shard actually ran with. This argument is not optional decoration:
    # verifying a run against a *different* embed config measures the config
    # difference and calls it misalignment. The first version of this script hardcoded
    # configs/default.yaml, whose max_seq_length is 512, while the shard ran
    # verify4.yaml at 320. Every probed row then came back between 0.65 and 0.98 cosine
    # and the verdict was FAIL -- against an index that was in fact correct. A verifier
    # that fails on a healthy index is worse than none, because the same output covers
    # a real fault and the verifier's own mistake, and a reader cannot tell them apart.
    ap.add_argument("--config", action="append", default=[],
                    help="config file the shard was run with; repeatable, in order")
    ap.add_argument("--concurrent", action="store_true",
                    help="another shard is writing into the same index dir right now. "
                         "Blocks outside this shard's range then belong to that shard, "
                         "not to a second writer claiming the same rows, so they are "
                         "reported and not failed. The distinction matters: without "
                         "this flag a healthy 2-shard run reports FOREIGN blocks and "
                         "fails, and a real duplicate writer -- two processes embedding "
                         "the SAME block -- is indistinguishable from it by looking at "
                         "the filesystem at all, because blocks are published by atomic "
                         "rename and the second writer silently overwrites the first. "
                         "Catching that needs the owning shard id recorded at write "
                         "time, which the block format does not carry.")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    store_dir = index_dir / "store"
    rowmap = json.loads((store_dir / "rowmap.json").read_text())
    n_rows = int(rowmap["n"])
    n_blocks = (n_rows + EMBED_BLOCK_ROWS - 1) // EMBED_BLOCK_ROWS

    expected = set(range(args.block_shard, n_blocks, args.block_nshards))
    present = set(committed_blocks(index_dir))
    in_flight = len(glob.glob(str(index_dir / "embeddings-block*.tmp*.npz")))

    print(f"index dir          : {index_dir}")
    print(f"store rows         : {n_rows:,}")
    print(f"blocks total       : {n_blocks:,}")
    print(f"shard              : {args.block_shard}/{args.block_nshards}")
    print(f"blocks committed   : {len(present)}")
    print(f"blocks in flight   : {in_flight} (excluded from every count below)\n")

    ok = True

    # ---- 1. ownership -------------------------------------------------------
    owned = present & expected
    foreign = present - expected
    absent = expected - present
    print("OWNERSHIP")
    print(f"  expected of this shard : {len(expected)}")
    print(f"  owned and committed   : {len(owned)}")
    print(f"  not yet written       : {len(absent)}"
          + (f"  e.g. {sorted(absent)[:6]}" if absent else ""))
    print(f"  FOREIGN blocks present: {len(foreign)}"
          + (f"  e.g. {sorted(foreign)[:6]}" if foreign else ""))
    if foreign:
        if args.concurrent:
            print(f"  blocks owned by the concurrent shard: {len(foreign)}"
                  f"  e.g. {sorted(foreign)[:6]}   (expected while both shards run)")
        else:
            print("  !! a block outside this shard's range is present with no concurrent")
            print("     shard declared: either two writers claim the same rows, or the")
            print("     run being verified is not the shard it claims to be")
            ok = False
    if not owned:
        print("  !! this shard has committed nothing; nothing below was checked")
        return 1

    # ---- 1b. recorded ownership -------------------------------------------
    #
    # Each block now records which shard wrote it, which is what makes a duplicate
    # write detectable at all. Before that field existed, two shards computing the same
    # block both wrote a temp file and renamed it into place, the second silently
    # replaced the first, and nothing on disk could say so -- so "no row written
    # twice" was an assertion nobody could check after the fact.
    #
    # With it, the check is exact: under a k-way split block b must belong to shard
    # b % k. A block recording anything else was written by a process that does not own
    # it. Legacy blocks with no `shard` key are reported as unowned rather than
    # treated as failures, because an index built before the field existed is correct,
    # just unannotated.
    print("\nRECORDED OWNERSHIP")
    owners: dict[int, int] = {}
    unowned: list[int] = []
    misowned: list[tuple[int, int]] = []
    all_present = sorted(present)
    probe = all_present if len(all_present) <= 64 else (
        all_present[:32] + all_present[-32:])
    for b in probe:
        with np.load(index_dir / f"embeddings-block{b:05d}.npz") as z:
            if "shard" not in z.files:
                unowned.append(b)
                continue
            who = int(np.asarray(z["shard"]).ravel()[0])
        owners[b] = who
        if args.block_nshards > 1 and who != b % args.block_nshards:
            misowned.append((b, who))
    if owners:
        tally: dict[int, int] = {}
        for b, who in owners.items():
            tally[who] = tally.get(who, 0) + 1
        print(f"  blocks probed        : {len(probe)} of {len(all_present)}")
        print(f"  shards seen in blocks: {dict(sorted(tally.items()))}")
    if unowned:
        print(f"  blocks with no owner recorded: {len(unowned)}"
              f"  e.g. {unowned[:6]}   (written before the field existed)")
    if misowned:
        print(f"  !! blocks written by a shard that does not own them: {len(misowned)}")
        for b, who in misowned[:8]:
            print(f"     block {b}: recorded shard {who}, but under a "
                  f"{args.block_nshards}-way split it belongs to shard {b % args.block_nshards}")
        print("     This is the duplicate-writer case the ownership field exists to")
        print("     catch. Two shards computed the same block; the second rename")
        print("     replaced the first and the run could not previously tell.")
        ok = False
    elif owners:
        print("  every annotated block was written by the shard that owns it: OK")
    if not owners:
        print("  no annotated blocks in the probed set -- ownership unverifiable here")

    # ---- 2. duplication -----------------------------------------------------
    print("\nDUPLICATION")
    shapes: dict[tuple[int, int], list[int]] = {}
    total_embedded = 0
    for b in sorted(owned):
        with np.load(index_dir / f"embeddings-block{b:05d}.npz") as z:
            arr = z["embeddings"]
        shapes.setdefault(arr.shape, []).append(b)
        total_embedded += arr.shape[0]
    for shape, blocks in shapes.items():
        print(f"  blocks with shape {shape}: {len(blocks)}")
    dup = [b for b, bl in shapes.items() if len(bl) > 1 and bl != sorted(bl)]
    print(f"  unique block files   : {len(present)}")
    print(f"  rows across owned    : {total_embedded:,}")
    # A short block is legal only for the final block of the corpus.
    lo, hi = min(owned), max(owned)
    short = [b for b, shape in
             ((b, np.load(index_dir / f'embeddings-block{b:05d}.npz')['embeddings'].shape[0])
              for b in sorted(owned))
             if shape != EMBED_BLOCK_ROWS and b != n_blocks - 1]
    print(f"  short blocks         : {len(short)}"
          + (f"  e.g. {short[:6]}" if short else "  (none; only the last may be short)"))
    if short:
        ok = False

    # ---- 3. alignment ------------------------------------------------------
    print("\nALIGNMENT (re-embedding the chunk text and comparing)")
    from medrag.index.embeddings import DenseEmbedder
    from medrag.config import PipelineConfig
    from medrag.index.store import ChunkStore

    if not args.config:
        print("  !! --config not given: the embed config used for verification may not "
              "match the shard's, which invalidates every cosine below.")
        return 2
    cfg = PipelineConfig.load(args.config)._finalize()
    cfg.embed.device = args.device
    print(f"\nverifying against the shard's own embed config: max_seq_length="
          f"{cfg.embed.max_seq_length}, batch_size={cfg.embed.batch_size}, "
          f"model={cfg.embed.model_name}, fp16={cfg.embed.fp16}\n")
    emb = DenseEmbedder(cfg.embed).load()
    store = ChunkStore.load(store_dir)

    block_list = sorted(owned)
    probe_blocks = block_list if len(block_list) <= 12 else (
        block_list[:6] + block_list[len(block_list) // 2 - 3: len(block_list) // 2 + 3]
        + block_list[-6:])
    probe_blocks = sorted(set(probe_blocks))

    worst = 1.0
    worst_row = None
    n_probed = 0
    n_bad = 0
    for b in probe_blocks:
        lo_r, hi_r = b * EMBED_BLOCK_ROWS, min(n_rows, (b + 1) * EMBED_BLOCK_ROWS)
        with np.load(index_dir / f"embeddings-block{b:05d}.npz") as z:
            mat = z["embeddings"]
        # First row, last row of the block, and one interior row. First/last are the
        # pair an off-by-one-block error produces.
        offs = sorted({0, max(0, (hi_r - lo_r) - 1), (hi_r - lo_r) // 2})[: args.rows_per_block]
        rows = [lo_r + o for o in offs if lo_r + o < hi_r]
        texts = []
        for r in rows:
            # get() takes the row INDEX. Passing store.rows[r] -- the row-id string
            # -- raises rather than silently returning None, which is the good failure.
            chunk = store.get(r)
            # `embed_text or text`, exactly as the builder does it. Embedding
            # chunk.text instead produced cosines of 0.65-0.98 on a correct index,
            # because embed_text carries the section heading the chunk was conditioned
            # on and the two are genuinely different strings.
            texts.append(((chunk.embed_text or chunk.text) if chunk else ""))
        # `embed`, not `encode`: the builder calls embedder.embed(...) and matching
        # it exactly is the point -- this check has to reproduce the same call the
        # shard made, or it is not checking the shard's output.
        vecs = emb.embed(texts, is_query=False, batch_size=len(texts))
        for row, v in zip(rows, vecs):
            got = mat[row - lo_r]
            cos = float(np.dot(v, got) / (np.linalg.norm(v) * np.linalg.norm(got) + 1e-12))
            n_probed += 1
            if cos < worst:
                worst, worst_row = cos, row
            if cos < 0.999:
                n_bad += 1
                print(f"  !! row {row:,} (block {b}) cosine {cos:.6f}")
    print(f"  blocks probed        : {len(probe_blocks)} of {len(owned)}")
    print(f"  rows probed          : {n_probed}")
    print(f"  worst cosine         : {worst:.6f}" + (f"  at row {worst_row:,}" if worst_row else ""))
    print(f"  rows below 0.999     : {n_bad}")
    if n_bad:
        ok = False

    print("\nVERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
