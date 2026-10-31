"""Backfill fields on an already-crawled corpus without re-crawling.

Why this exists
---------------
A crawl is expensive and a parser improvement arrives afterwards. Here the PMC
citation parser was fixed hours into a six-hour crawl, so ~590,000 already-fetched
documents carry ``citations: []``. Discarding them to re-crawl would have thrown
away the long pole for the sake of one field.

The cost asymmetry makes backfilling obviously right:

    full crawl      ~2,938 efetch requests PLUS the esearch window walk (~5 h)
    backfill        ~2,950 efetch requests, no esearch at all  (~5 min)

Both request counts are similar; the crawl's cost is dominated by identifier
partitioning, not by fetching. So the correct move is to fetch the records again
and keep everything else.

What it does
------------
Streams the shard files, collects the identifiers, re-fetches in batches, and
rewrites the shards with the newly parsed field merged in. Only documents that
actually gain something are touched, and the rewrite is atomic per shard.

Deliberately narrow: it patches one named field. A general "re-parse everything"
tool would tempt a rewrite of the whole corpus from a different parser version,
which silently mixes vintages -- the trap already hit once in this project when
the section-kind fix landed after 15,000 PMC records had been fetched.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterator
from pathlib import Path

from .eutils import EutilsClient
from .pubmed_parser import parse_efetch
from .shards import read_shards

log = logging.getLogger(__name__)

# Fields that may be backfilled. Each is a Document field name that the parser
# populates from a single efetch response.
BACKFILLABLE = ("citations",)


def iter_corpus_ids(shard_paths: list[Path], *, field: str) -> dict[str, str]:
    """Map ``identifier -> shard path`` for documents that lack ``field``.

    PMC records are addressed by PMCID and PubMed records by PMID, because
    efetch resolves either and they are both present in the stored records.
    """
    out: dict[str, str] = {}
    for path in shard_paths:
        for doc in read_shards([path]):
            ident = doc.get("pmcid") or doc.get("pmid") or doc.get("doc_id")
            if not ident:
                continue
            if doc.get(field):
                continue  # already populated; nothing to gain
            out.setdefault(str(ident), str(path))
    return out


def backfill_field(
    shard_paths: list[Path],
    field: str,
    *,
    client: EutilsClient,
    source: str,
    batch_size: int = 200,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Re-fetch documents and merge a parsed ``field`` back into the shards.

    Returns counts. With ``dry_run`` the fetch is skipped, which is how the cost of
    a backfill should be checked before committing the API budget to it.
    """
    if field not in BACKFILLABLE:
        raise ValueError(f"field {field!r} is not backfillable; expected one of {BACKFILLABLE}")

    todo = iter_corpus_ids(shard_paths, field=field)
    if limit:
        todo = dict(list(todo.items())[:limit])
    log.info("backfill %s: %d documents lack it", field, len(todo))
    stats = {"candidates": len(todo), "fetched": 0, "updated": 0, "still_empty": 0}
    if dry_run or not todo:
        return stats

    # Keyed by identifier, never by position. Patches arrive in fetch order while
    # shards are in crawl order, so a positional list would attach each document's
    # citations to whichever document happened to sit at the same offset -- a
    # silent corruption that leaves the field populated and the graph meaningless.
    patches: dict[str, list[int]] = {}
    ids = list(todo)
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        try:
            xml = client.efetch(batch, db=source, retmode="xml")
        except Exception as exc:  # noqa: BLE001
            log.warning("efetch batch %d failed: %s", i, str(exc)[:120])
            continue
        stats["fetched"] += len(batch)
        for doc in parse_efetch(xml, source=source):
            values = getattr(doc, field)
            if not values:
                stats["still_empty"] += 1
                continue
            for key in filter(None, (doc.pmcid, doc.pmid)):
                patches.setdefault(str(key), list(values))
        if stats["fetched"] % 20_000 < batch_size:
            log.info("backfill %s: fetched=%d updated=%d", field, stats["fetched"],
                     len(patches))

    stats["updated"] = apply_patches(shard_paths, field, patches)
    return stats


def apply_patches(shard_paths: list[Path], field: str,
                  patches: dict[str, list[int]]) -> int:
    """Rewrite shards with ``field`` set, atomically, only where it was empty.

    Lookups are by the document's own ``pmcid``/``pmid``, so the mapping cannot
    drift from the shard order. Returns the number of documents updated.
    """
    todo = {
        str(doc.get("pmcid") or doc.get("pmid") or doc.get("doc_id") or ""): path
        for path in shard_paths
        for doc in read_shards([path])
        if not doc.get(field)
    }
    updated = 0
    files = 0
    for path in sorted(set(todo.values())):
        src = Path(path)
        if not src.exists():
            continue
        tmp = src.with_name(src.name + ".tmp")
        with gzip.open(src, "rt") as fin, gzip.open(tmp, "wt") as fout:
            for line in fin:
                if not line.strip():
                    continue
                doc = json.loads(line)
                if not doc.get(field):
                    ident = str(doc.get("pmcid") or doc.get("pmid") or doc.get("doc_id") or "")
                    got = patches.get(ident)
                    if got:
                        doc[field] = got
                        updated += 1
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
        tmp.replace(src)
        files += 1
    log.info("backfill %s: rewrote %d shards, set %d documents", field, files, updated)
    return updated