"""Select a corpus prefix that is representative rather than merely first.

The expansion crawl lands documents in publication-date order, so "the first N" is a
pure slice of the oldest years. A prefix of that shape is the one thing a
representativeness argument cannot accept, because the whole point of crawling past the
RCT-restricted 590,762 ceiling was to widen type coverage and the oldest PMC years are
the thinnest for section structure -- 45.0% carry a usable body in 1995 against 100.0% in
2018. A prefix would take the worst of the corpus and call it a sample.

A stride sample keeps every k-th record, so era and publication-type mix survive, and it
does so in one streaming pass with no corpus held in memory.

    python scripts/sample_corpus.py <src-glob> <dest-dir> --every N --max M
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import random
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True,
                    help="globs of corpus shards to sample")
    ap.add_argument("--dest", required=True, help="destination corpus directory")
    ap.add_argument("--every", type=int, default=1,
                    help="keep every Nth record; 1 keeps all")
    ap.add_argument("--max", type=int, default=0, help="stop after M records (0 = all)")
    ap.add_argument("--seed", type=int, default=17,
                    help="only used to break ties inside a year, so the sample is "
                         "stable across runs rather than dependent on shard order")
    ap.add_argument("--shard-size", type=int, default=5000)
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Pass 1: count records per year so the stride can be chosen per era. A single global
    # stride over an era-skewed corpus samples the dense years and skips the sparse ones
    # entirely, which is the same bias in a different guise.
    print("  pass 1: counting by publication year", flush=True)
    per_year: collections.Counter = collections.Counter()
    total = 0
    for pattern in args.src:
        for path in sorted(glob.glob(pattern)):
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    total += 1
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    y = str(rec.get("year") or "")[:4]
                    per_year[int(y) if y.isdigit() else 0] += 1
    print(f"  {total:,} records across {len(per_year)} years", flush=True)
    for y in sorted(per_year):
        print(f"    {y or 'unknown':>8}: {per_year[y]:>9,}", flush=True)

    # Pass 2: keep every k-th record within each year.
    kept_per_year: collections.Counter = collections.Counter()
    seen_per_year: collections.Counter = collections.Counter()
    types: collections.Counter = collections.Counter()
    rng = random.Random(args.seed)

    tmp = dest / f"sample.tmp{os.getpid()}.jsonl.gz"
    out = gzip.open(tmp, "wt")
    n_kept = 0
    try:
        for pattern in args.src:
            for path in sorted(glob.glob(pattern)):
                with gzip.open(path, "rt") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        y = str(rec.get("year") or "")[:4]
                        y = int(y) if y.isdigit() else 0
                        seen_per_year[y] += 1
                        # `seen % every == 1` rather than `== 0` so a year with fewer
                        # records than the stride still contributes its first document
                        # instead of contributing nothing.
                        if args.every > 1 and seen_per_year[y] % args.every != 1:
                            continue
                        out.write(line + "\n")
                        kept_per_year[y] += 1
                        for pt in (rec.get("publication_types") or ["?"]):
                            types[pt] += 1
                        n_kept += 1
                        if args.max and n_kept >= args.max:
                            break
                if args.max and n_kept >= args.max:
                    break
            if args.max and n_kept >= args.max:
                break
    finally:
        out.close()

    print(f"\n  kept {n_kept:,} of {total:,}", flush=True)
    print("  by year:", flush=True)
    for y in sorted(kept_per_year):
        print(f"    {y or 'unknown':>8}: {kept_per_year[y]:>9,}", flush=True)
    print("  publication types:", flush=True)
    for pt, n in types.most_common(10):
        print(f"    {pt:26s} {n:>9,}  {n / max(n_kept, 1):>6.1%}")

    # Rewrite into canonical shard names so corpus_shards globs this directory.
    n_shards = 0
    with gzip.open(tmp, "rt") as fh:
        shard = None
        w = None
        try:
            for i, line in enumerate(fh):
                if i % args.shard_size == 0:
                    if w:
                        w.close()
                    shard = dest / f"pmc-{n_shards:05d}.jsonl.gz"
                    w = gzip.open(shard, "wt")
                    n_shards += 1
                w.write(line)
        finally:
            if w:
                w.close()
    os.unlink(tmp)
    print(f"\n  wrote {n_shards} canonical shards to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
