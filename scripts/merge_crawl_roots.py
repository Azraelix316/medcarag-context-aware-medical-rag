"""Merge several crawl roots into one corpus directory.

Two nodes crawl into separate roots -- they must, or both will claim the same shard
index -- but the downstream pipeline wants one corpus. Merging by concatenating
shard files is not possible: both roots name their shards ``pmc-00000.jsonl.gz``,
so a naive copy silently overwrites one node's work with the other's. Measured on
the first two-node crawl: c7 held 31 shards and worker-B 29, and both started at 00000.

So every shard is renamed on the way in, keeping the source root in the name. That
makes the provenance of each file recoverable from its name alone, and makes a
half-finished merge obvious rather than looking like a short corpus.

Merging is verified rather than assumed: the document count, the unique-identifier
count, and the cross-root duplicate count are all reported, because a merge that
loses or duplicates documents produces a corpus that still runs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def merge_roots(roots: list[Path], dest: Path, *, link: bool = False,
                source: str = "pmc") -> dict[str, int]:
    """Copy (or hard-link) every shard from ``roots`` into ``dest``.

    Shard names are prefixed with the source directory name. Overwriting is not
    possible by construction, which is the whole point.
    """
    dest.mkdir(parents=True, exist_ok=True)
    moved = 0
    for root in roots:
        src_dir = root / "corpus" / source
        if not src_dir.is_dir():
            log.warning("no %s directory under %s", source, root)
            continue
        for src in sorted(src_dir.glob("*.jsonl.gz")):
            target = dest / f"{root.name}__{src.name}"
            if target.exists():
                continue
            if link:
                try:
                    target.hardlink_to(src)
                except OSError:
                    shutil.copy2(src, target)
            else:
                shutil.copy2(src, target)
            moved += 1
    return {"shards": moved}


def verify_corpus(corpus_dir: Path) -> dict[str, int]:
    """Count documents and unique identifiers across every shard.

    Also reports documents that appear in more than one shard, which is the
    signature of a merge that duplicated work.
    """
    docs = 0
    ids: set[str] = set()
    per_shard: dict[str, int] = {}
    broken: list[str] = []
    for path in sorted(corpus_dir.glob("*.jsonl.gz")):
        n = 0
        try:
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    n += 1
                    ident = rec.get("pmcid") or rec.get("pmid") or rec.get("doc_id")
                    if ident:
                        ids.add(str(ident))
        except (EOFError, OSError) as exc:
            # A shard still being appended to by a live crawl is not an error here,
            # but it must be reported rather than counted as short.
            log.warning("shard %s is incomplete (%s); counted as partial", path.name, exc)
            broken.append(path.name)
        docs += n
        per_shard[path.name] = n
    dupes = docs - len(ids)
    return {"documents": docs, "unique_ids": len(ids), "duplicate_documents": dupes,
            "shards": len(per_shard), "partial_shards": len(broken)}


def dedupe_corpus(corpus_dir: Path, *, rewrite: bool = True,
                  salvage_partial: bool = False) -> dict[str, int]:
    """Drop documents already seen, keeping the first occurrence.

    Necessary, not cosmetic. Two nodes crawling overlapping windows produce a corpus
    where every duplicated document would be chunked twice, embedded twice, and
    retrieved twice -- inflating every count downstream while producing no error.
    Measured on the first two-node crawl: 289,479 records, 142,383 unique,
    147,096 duplicates (50.8%).

    First occurrence wins, and shards are walked in sorted name order, so the
    surviving copy comes from the lexicographically first root. Records are compared
    by pmcid, then pmid, then doc_id -- the same precedence the crawler uses, so a
    document reachable by two identifier schemes is still recognised as one.

    A truncated shard is left untouched by default, because during a live crawl it
    is still being written. Once the crawl is over it is simply damaged, and leaving
    it untouched keeps every duplicate inside it. ``salvage_partial`` keeps the
    complete records read before the break and rewrites the shard without the tail.
    """
    seen: set[str] = set()
    dropped = 0
    kept = 0
    salvaged = 0
    for path in sorted(corpus_dir.glob("*.jsonl.gz")):
        rows: list[dict] = []
        try:
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    ident = str(rec.get("pmcid") or rec.get("pmid") or rec.get("doc_id") or "")
                    if not ident:
                        kept += 1
                        rows.append(rec)
                        continue
                    if ident in seen:
                        dropped += 1
                        continue
                    seen.add(ident)
                    kept += 1
                    rows.append(rec)
        except (EOFError, OSError, json.JSONDecodeError) as exc:
            if not salvage_partial:
                log.warning("shard %s incomplete (%s); leaving it untouched", path.name, exc)
                continue
            log.warning("shard %s incomplete (%s); salvaged %d complete records",
                        path.name, exc, len(rows))
            salvaged += 1
        if not rewrite:
            continue
        tmp = path.with_name(path.name + ".dedup")
        with gzip.open(tmp, "wt") as fout:
            for rec in rows:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(path)
    return {"kept": kept, "dropped": dropped, "unique": len(seen), "salvaged_shards": salvaged}


def canonicalise_names(corpus_dir: Path, *, prefix: str = "pmc") -> int:
    """Renumber shards to the canonical ``<prefix>-NNNNN.jsonl.gz`` the pipeline reads.

    The prefixed names that make a merge collision-free also stop the corpus from
    being readable by every downstream stage: ``corpus_shards`` globs
    ``pmc-*.jsonl.gz``, so a directory of ``pmcoa2_c7__pmc-*.jsonl.gz`` files looks
    empty and the stage raises. That is the correct behaviour now -- a read path
    must not silently crawl -- but it means the merged corpus has to end up wearing
    the names the rest of the pipeline expects.

    Safe because it runs after deduplication: renumbering cannot merge or split a
    record, and ordering by current name keeps the merge deterministic.
    """
    paths = sorted(corpus_dir.glob("*.jsonl.gz"))
    for i, src in enumerate(paths):
        target = corpus_dir / f"{prefix}-{i:05d}.jsonl.gz"
        if target == src:
            continue
        os.rename(src, target)
    return len(paths)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", required=True, help="crawl roots to merge")
    ap.add_argument("--dest", required=True, help="destination corpus directory")
    ap.add_argument("--link", action="store_true",
                    help="hard-link instead of copying (same filesystem only)")
    ap.add_argument("--prefix", default="pmc",
                    help="canonical shard-name prefix (corpus_shards globs this)")
    ap.add_argument("--source", default="pmc",
                    help="corpus subdirectory under each root: pmc or pubmed")
    ap.add_argument("--salvage-partial", action="store_true",
                    help="keep the complete records of truncated shards (finished crawls only)")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="report duplicates without removing them")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    stats = merge_roots([Path(p) for p in args.roots], Path(args.dest), link=args.link,
                        source=args.source)
    log.info("merged %d shards into %s", stats["shards"], args.dest)

    if args.no_dedupe:
        report = verify_corpus(Path(args.dest))
        log.info(json.dumps(report, indent=2))
        return 0 if report["duplicate_documents"] == 0 else 1

    dedup = dedupe_corpus(Path(args.dest), salvage_partial=args.salvage_partial)
    log.info("dedupe: kept=%d dropped=%d unique=%d salvaged_shards=%d",
             dedup["kept"], dedup["dropped"], dedup["unique"], dedup["salvaged_shards"])
    # --prefix was parsed and then dropped here, so every merge was renamed pmc-*,
    # which a pubmed corpus reader does not glob.
    n = canonicalise_names(Path(args.dest), prefix=args.prefix)
    log.info("renamed %d shards to canonical %s-NNNNN.jsonl.gz names", n, args.prefix)
    report = verify_corpus(Path(args.dest))
    log.info(json.dumps(report, indent=2))
    ok = report["duplicate_documents"] == 0 and report["documents"] == report["unique_ids"]
    # Non-zero exit when the merge lost or duplicated documents: a corpus that is
    # merely smaller than expected is far harder to notice than one that failed.
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())