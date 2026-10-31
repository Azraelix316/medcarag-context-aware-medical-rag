"""Corpus acquisition driver: crawl PubMed/PMC into sharded JSONL.

The crawler is designed to run unattended for days. It streams PMIDs, fetches
batches, parses XML, filters for clinical utility, and appends documents to
gzip shards. Every stage logs throughput so an operator can see whether it is
progressing or stalled on NCBI rate limits.
"""

from __future__ import annotations

import calendar
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AcquireConfig, PipelineConfig
from .eutils import EutilsClient
from .mesh_taxonomy import MeshTaxonomy, taxonomy_cache_path
from .pubmed_parser import parse_efetch
from .schema import CorpusManifest, Document, MeshTerm
from .shards import ShardWriter, _Progress, atomic_write_json

log = logging.getLogger(__name__)

# Queries used when config.acquire.query is empty. Each is a coherent clinical
# evidence slice, which yields a better retrieval benchmark than one giant sweep.
DEFAULT_QUERIES: dict[str, tuple[str, ...]] = {
    "pubmed": (
        'randomized controlled trial[pt]',
        'systematic review[pt]',
        'meta-analysis[pt]',
        'clinical practice guideline[pt]',
        'randomized controlled trial[pt] AND ("2020"[dp] : "2026"[dp])',
        'multicenter study[pt] AND ("2018"[dp] : "2026"[dp])',
    ),
    # `has_text_ml[filter]` is a PubMed-side filter and does not exist in the PMC
    # database. Measured against the live API:
    #
    #   '"open access"[filter] AND has_text_ml[filter]'   count = 0
    #   '"has_text_ml"[filter]'                          count = 0
    #   '"open access"[filter]'                           count = 8,298,989
    #
    # and NCBI echoes the term back untranslated, so nothing warns: the query parses,
    # the esearch succeeds, and the crawl acquires zero records and exits clean. Every
    # PMC config in this repo overrode this default, which is the only reason it was
    # never noticed -- `acquire --source pmc` with no override produced an empty corpus
    # and no error.
    #
    # `"open access"[filter]` is the right default on its own terms: in PMC that filter
    # *is* the subset with retrievable full text, which is what `has_text_ml` was
    # reaching for. Note it does not restrict by publication type -- adding
    # `AND "randomized controlled trial"[pt]` here would silently cap the corpus at
    # 590,762 records, which is what pmcmerged did (see configs/pmcsweep_c7.yaml).
    "pmc": ('"open access"[filter]',),
}

# Publication types worth keeping for reasoning-grounded generation.
#
# PubMed and PMC do NOT share a vocabulary, and a single list silently loses most
# of the PMC corpus. Measured on 360 live PMC OA RCT records (2005/2018/2022):
#
#     269  research article      <- 75% of the corpus
#      51  review article        <- 14%
#      15  other
#       6  editorial             <- should be dropped
#       5  systematic review
#
# Only "journal article" appears in the PubMed vocabulary, so a PubMed-shaped
# whitelist keeps 22% of live PMC records and discards the 75% that are ordinary
# research articles. The PMC list is kept separate rather than merged so the two
# vocabularies cannot be confused again.
_KEEP_PT_PREFIXES_PUBMED = (
    "journal article", "review", "systematic review", "meta-analysis",
    "randomized controlled trial", "clinical trial", "comparative study",
    "multicenter study", "observational study", "case reports",
    "practice guideline", "guideline", "preprint", "research support",
    "clinical study", "validation study",
)

_KEEP_PT_PREFIXES_PMC = (
    "research article", "review article", "systematic review", "meta-analysis",
    "randomized controlled trial", "clinical trial", "comparative study",
    "multicenter study", "observational study", "case report",
    "practice guideline", "guideline", "methods article",
)

_KEEP_PT_PREFIXES_BY_SOURCE = {
    "pubmed": _KEEP_PT_PREFIXES_PUBMED,
    "pmc": _KEEP_PT_PREFIXES_PMC,
}

# Kept for callers that only care about the PubMed vocabulary.
_KEEP_PT_PREFIXES = _KEEP_PT_PREFIXES_PUBMED


@dataclass
class CrawlStats:
    fetched: int = 0
    kept: int = 0
    skipped_no_abstract: int = 0
    skipped_pubtype: int = 0
    duplicates: int = 0
    requests: int = 0
    query_failures: int = 0
    batch_failures: int = 0
    elapsed: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "fetched": self.fetched, "kept": self.kept,
            "skipped_no_abstract": self.skipped_no_abstract,
            "skipped_pubtype": self.skipped_pubtype, "duplicates": self.duplicates,
            "requests": self.requests, "query_failures": self.query_failures,
            "batch_failures": self.batch_failures, "elapsed_s": round(self.elapsed, 1),
        }


def _section_chars(doc: Document) -> int:
    """Total characters across a document's parsed sections.

    Accepts both ``Section`` objects and the plain dicts that :meth:`Document.to_dict`
    produces. Reading only the object form would silently return 0 for anything
    loaded from a shard file, which is exactly where a re-crawl reads from.
    """
    total = 0
    for s in doc.sections or ():
        if isinstance(s, dict):
            total += len(s.get("text") or "")
        else:
            total += len(getattr(s, "text", "") or "")
    return total


def _keep_document(doc: Document, cfg: AcquireConfig, require_abstract: bool = True) -> bool:
    if len(doc.title) < 8:
        return False

    if cfg.source == "pmc":
        # Content is mandatory, and it is checked *before* publication type rather
        # than instead of it.
        #
        # PMC's "open access" filter marks the PDF as free, not the JATS body as
        # present. Probed live across nine eras at 40 records each:
        #
        #     era     OA records   has <body>   has <sec>
        #     1995        3,762        45.0%       12.5%
        #     2000        5,536        85.0%       35.0%
        #     2005       16,724        67.5%       50.0%
        #     2010       82,931        97.5%       92.5%
        #     2018      354,927       100.0%       95.0%
        #     2024      792,631       100.0%      100.0%
        #     all sampled           86.4%       72.5%
        #
        # So an all-OA sweep draws roughly one record in seven with no body at all:
        # empty abstract, empty sections, no text anywhere. Those records then sail
        # through the publication-type test below, because "Research Article" is a
        # perfectly good type for an article that contains nothing. Measured on a live
        # 50,000-record sweep corpus, that admitted **65.4% empty records**.
        #
        # The 5,000-character floor is the one already used on the no-pubtype path and
        # is not a new threshold: it is the same signal, applied where it was missing.
        if _section_chars(doc) < 5000:
            return False
    elif require_abstract and not doc.abstract.strip():
        return False

    prefixes = _KEEP_PT_PREFIXES_BY_SOURCE.get(cfg.source, _KEEP_PT_PREFIXES_PUBMED)
    if doc.publication_types:
        lowered = tuple(pt.lower() for pt in doc.publication_types)
        # Keep when at least one type is a substantive research/review type. This is a
        # filter *on top of* the content requirement, never a substitute for it.
        return any(pt.startswith(prefixes) for pt in lowered)
    # No publication types to judge by. An abstract is a reasonable proxy, but for
    # PMC the body text is the better signal: these records typically carry NO
    # abstract (measured: 100% of a live 60-record sample) while carrying genuine
    # full text (93% exceed 5,000 characters). Requiring a 200-character abstract
    # here discards almost the whole PMC corpus.
    if cfg.source == "pmc":
        return _section_chars(doc) >= 5000
    return len(doc.abstract) >= 200


def build_query(cfg: AcquireConfig, base: str) -> str:
    """Compose the E-utilities query from base + non-date filters.

    The date range is deliberately omitted: :meth:`iter_ids_partitioned` adds
    one window of it per request, which is what keeps every esearch under the
    10,000-record service cap.
    """
    parts = [base.strip()] if base.strip() else ["*"]
    if cfg.filter_query.strip():
        parts.append(cfg.filter_query.strip())
    if cfg.lang.strip():
        parts.append(f'"{cfg.lang}"[language]')
    return " AND ".join(parts)


def _dedup_state(existing_shards: list[Path]) -> tuple[set[str], int]:
    """Dedup key set from the shards already on disk, and how many records they hold.

    Records that were already written are excluded from the next run's output,
    so an interrupted crawl can simply be restarted with the same query list.

    The record count comes back because this function already reads every record to
    build the key set, and discarding the number is what let the corpus manifest lie:
    see the note at the manifest construction below. Every other record in this
    project reads a manifest and believes it.
    """
    from .shards import read_shards

    seen: set[str] = set()
    n_records = 0
    for path in existing_shards:
        for record in read_shards([path]):
            n_records += 1
            ident = record.get("doi") or record.get("pmid") or record.get("pmcid") or record.get("doc_id", "")
            if ident:
                seen.add(str(ident).lower())
    return seen, n_records


def sweep_queries(
    min_date: str,
    max_date: str,
    *,
    granularity: str = "yearly",
    base: str = "",
    tag: str = "dp",
) -> tuple[str, ...]:
    """One query per date window, covering ``[min_date, max_date]`` exactly once.

    This is what makes the 20M target reachable. The default evidence slices are six
    publication-type queries capped at ``max_records`` each, so the crawl is bounded by
    ``6 * 200,000`` -- 1.2M records -- however long it runs. Measured: it reached
    1,022,707 and stopped. PubMed holds ~37M.

    Partitioning by date is also the only way past NCBI's own ceiling: esearch will not
    page beyond roughly 10,000,000 records for a single query, so a single sweep query
    is not an option no matter how ``max_records`` is set. Date windows stay under that
    individually while together covering the whole range.

    Windows are half-open and contiguous: ``["1950/01/01", "1960/12/31"]`` followed by
    ``["1961/01/01", "1970/12/31"]`` and so on, with no gap and no overlap, so a document
    falls in exactly one slice and two nodes cannot both fetch it.
    """
    def _parse(text: str) -> tuple[int, int]:
        parts = text.replace("-", "/").split("/")
        if len(parts) < 3 or not all(p.isdigit() for p in parts[:3]):
            raise ValueError(
                f"date {text!r} is not YYYY/MM/DD; the sweep cannot partition it"
            )
        return int(parts[0]), int(parts[1])

    y0, m0 = _parse(min_date)
    y1, _ = _parse(max_date)
    if y1 < y0:
        raise ValueError(f"max_date {max_date!r} precedes min_date {min_date!r}")
    if granularity not in ("yearly", "monthly"):
        raise ValueError(
            f"unknown sweep granularity {granularity!r}; expected 'yearly' or 'monthly'"
        )

    out: list[str] = []
    for year in range(y0, y1 + 1):
        if granularity == "yearly":
            spans = ((f"{year}/01/01", f"{year}/12/31"),)
        else:
            # Real last day per month, not a hard-coded 31: "2015/04/31" is not a date,
            # and a filter that names one is either rejected or silently narrowed by
            # NCBI, which would quietly drop records from the sweep.
            spans = tuple(
                (f"{year}/{m:02d}/01", f"{year}/{m:02d}/{calendar.monthrange(year, m)[1]}")
                for m in range(1, 13)
            )
        for lo, hi in spans:
            # Clamp the first and last windows to the configured range. Without this a
            # monthly sweep of 2016/02/01..2016/02/28 still emits all twelve months of
            # 2016, because partitioning is by month and the day-level bounds are
            # dropped -- so the caller gets records outside the range it asked for.
            if lo < min_date:
                lo = min_date
            if hi > max_date:
                hi = max_date
            if lo > hi:
                continue
            window = f'("{lo}"[{tag}] : "{hi}"[{tag}])'
            out.append(f"{base} AND {window}".strip() if base else window)
    return tuple(out)


def crawl_corpus(
    cfg: PipelineConfig,
    queries: tuple[str, ...] | None = None,
    *,
    shard: int = 0,
    nshards: int = 1,
) -> CorpusManifest:
    """Crawl the corpus, optionally restricted to this node's query slices.

    Sharding is round-robin over query slices, so two nodes on different IPs
    crawl disjoint slices and neither re-fetches the other's work. The global
    dedup set still applies across shards, because slices overlap (RCTs also
    appear in the "recent RCT" slice).
    """
    """Crawl the corpus described by ``cfg.acquire`` into sharded JSONL."""
    acq = cfg.acquire
    client = EutilsClient(
        api_key=acq.api_key, email=acq.email, tool=acq.tool,
        rate=acq.max_requests_per_second, timeout=acq.request_timeout,
        max_retries=acq.max_retries,
    )
    if not client.health_check():
        raise RuntimeError("NCBI E-utilities unreachable; refusing to start crawl")

    out_dir = cfg.paths.corpus_dir / acq.source
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{acq.source}"

    if acq.sweep.strip():
        # A date-partitioned sweep replaces the default evidence slices entirely.
        query_list = sweep_queries(
            acq.min_date, acq.max_date,
            granularity=acq.sweep_granularity or "yearly",
            base=acq.query.strip() or acq.filter_query.strip(),
        )
        log.info("date-partitioned sweep: %d slices over %s..%s (%s), cap %s per slice, "
                 "ceiling ~%s raw records",
                 len(query_list), acq.min_date, acq.max_date,
                 acq.sweep_granularity or "yearly", acq.max_records or "unbounded",
                 f"{(acq.max_records or 0) * len(query_list):,}")
    else:
        query_list = queries or DEFAULT_QUERIES[acq.source]
        if acq.query.strip():
            query_list = (acq.query.strip(),)

    if nshards > 1:
        if not (0 <= shard < nshards):
            raise ValueError(f"shard {shard} out of range for nshards={nshards}")
        mine = tuple(q for i, q in enumerate(query_list) if i % nshards == shard)
        log.info("shard %d/%d owns %d of %d query slices", shard, nshards, len(mine), len(query_list))
        query_list = mine
        if not query_list:
            log.warning("shard %d has no query slices; nothing to crawl", shard)
            query_list = ()

    stats = CrawlStats()
    started = time.time()
    # Resume-safe: keys already on disk are skipped by the dedup set.
    seen, n_existing = _dedup_state(sorted(out_dir.glob(f"{prefix}-*.jsonl.gz")))
    log.info("resuming: %d documents already on disk (%d distinct keys)",
             n_existing, len(seen))
    if seen:
        log.info("resuming: %d documents already on disk", len(seen))
    writer = ShardWriter(out_dir, prefix, shard_size=acq.shard_size)
    # Categories come from the taxonomy cache only -- NO API calls here.
    # A 200-record batch carries ~3k mostly-unique MeSH terms; resolving those
    # over HTTP at 8 req/s would add ~6 minutes of blocking per batch and stall
    # the crawl. Authoritative resolution is a separate offline pass
    # (`medrag taxonomy`) followed by `medrag recategorize`.
    taxonomy = MeshTaxonomy(taxonomy_cache_path(cfg.paths.data_root), use_api=False)
    log.info("mesh taxonomy cache: %d descriptors already resolved (API disabled during crawl)",
             len(taxonomy._cache))

    try:
        for qi, base in enumerate(query_list, 1):
            # A single bad query must not abort a multi-hour crawl: log it,
            # record the failure, and continue with the remaining slices.
            try:
                _crawl_one_query(base, query_list, qi, client, acq, writer, seen, stats, taxonomy)
            except Exception as exc:  # noqa: BLE001
                log.exception("query %d/%d failed (%s); continuing", qi, len(query_list), str(exc)[:120])
                stats.query_failures += 1
                continue
    finally:
        shards = writer.close()
        taxonomy.flush()
        log.info("mesh taxonomy stats: %s", taxonomy.stats())

    stats.elapsed = time.time() - started
    # n_documents is the SIZE OF THE CORPUS, not the size of this run.
    #
    # It used to be stats.kept, which on a resumed crawl describes the increment. The
    # restarted run over this corpus wrote 30,488 documents into a directory holding
    # 960,880, and recorded n_documents: 30488. Anyone reading that to size the corpus
    # would be wrong by 31x, and nothing in the file hints at the difference -- the
    # field is named n_documents and it is the only count in there.
    #
    # An independent recount of the shards confirms 960,880 rows over 51 files, all
    # unique, so the true total is recoverable and was available for free: this loop
    # had already read every record to build the dedup set.
    all_shards = sorted(out_dir.glob(f"{prefix}-*.jsonl.gz"))
    manifest = CorpusManifest(
        corpus_id=f"{acq.source}-{len(query_list)}q",
        source=acq.source,
        query=" | ".join(query_list),
        n_documents=n_existing + stats.kept,
        n_bytes=n_existing + stats.kept,
        shard_files=[str(p) for p in all_shards],
        timings={"acquire": stats.elapsed,
                 "n_documents_before_run": n_existing,
                 "n_documents_added_by_this_run": stats.kept},
        config=cfg.to_dict().get("acquire", {}),
        built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    manifest.timings["stats"] = stats.as_dict()
    atomic_write_json(out_dir / "manifest.json", manifest.to_dict())
    log.info("crawl complete: %s", stats.as_dict())
    return manifest


def _crawl_one_query(
    base: str,
    query_list: tuple[str, ...],
    qi: int,
    client: EutilsClient,
    acq: AcquireConfig,
    writer: ShardWriter,
    seen: set[str],
    stats: CrawlStats,
    taxonomy: MeshTaxonomy | None,
) -> None:
    """Crawl one query by walking date-partitioned windows.

    A single PubMed query cannot return more than 10,000 records, so the query
    is split by publication date (year, then month, then day as needed) and each
    window is fetched separately. Windows are walked newest-first, which keeps
    the corpus weighted toward recent evidence while staying deterministic.
    """
    query = build_query(acq, base)
    log.info("[query %d/%d] %s", qi, len(query_list), query)

    progress = _Progress(acq.max_records or 0, f"acquire[{base[:36]}]")
    batch: list[str] = []
    consumed = 0
    cap = acq.max_records or None
    current_window = ""

    for label, pmid in client.iter_ids_partitioned(
        query,
        db=acq.source,
        limit=cap,
        min_date=acq.min_date,
        max_date=acq.max_date,
        page=min(9_000, client.MAX_RECORDS_PER_QUERY - 500),
    ):
        if label != current_window:
            log.info("[query %d/%d] window %s (consumed %d)", qi, len(query_list), label, consumed)
            current_window = label
        batch.append(pmid)
        if len(batch) < acq.fetch_batch:
            continue
        consumed += len(batch)
        _process_batch(batch, client, acq, writer, seen, stats, progress, taxonomy)
        batch = []
        if cap and consumed >= cap:
            break

    if batch:
        consumed += len(batch)
        _process_batch(batch, client, acq, writer, seen, stats, progress, taxonomy)
    progress.close()
    log.info("[query %d/%d] consumed %d records, kept %d total", qi, len(query_list), consumed, stats.kept)


def _process_batch(
    batch: list[str],
    client: EutilsClient,
    acq: AcquireConfig,
    writer: ShardWriter,
    seen: set[str],
    stats: CrawlStats,
    progress: _Progress,
    taxonomy: MeshTaxonomy | None = None,
) -> None:
    try:
        xml = client.efetch(batch, db=acq.source, retmode="xml")
    except Exception as exc:  # noqa: BLE001
        # Skip the batch rather than losing the whole crawl to one bad response.
        log.warning("efetch batch of %d failed (%s); skipping", len(batch), str(exc)[:120])
        stats.batch_failures += 1
        return
    stats.requests += 1
    docs = parse_efetch(xml, source=acq.source, taxonomy=taxonomy)
    stats.fetched += len(docs)
    for doc in docs:
        ident = (doc.doi or doc.pmid or doc.pmcid or doc.doc_id).lower()
        if ident in seen:
            stats.duplicates += 1
            continue
        require_abstract = acq.source == "pubmed"
        if not _keep_document(doc, acq, require_abstract=require_abstract):
            # Attribute the rejection to the check that actually ran. The previous
            # version counted anything rejected while the abstract happened to be
            # empty as skipped_no_abstract, which mislabelled all 445,142 dropped
            # PMC records: they had no abstract because PMC rarely supplies one,
            # not because they lacked content. A stat that names the wrong cause is
            # worse than no stat, because it points the fix in the wrong direction.
            if doc.publication_types:
                stats.skipped_pubtype += 1
            elif require_abstract and not doc.abstract.strip():
                stats.skipped_no_abstract += 1
            else:
                stats.skipped_pubtype += 1
            continue
        seen.add(ident)
        writer.write(doc.to_dict())
        stats.kept += 1
    progress.update(len(docs))
    if stats.kept and stats.kept % 20_000 < len(docs):
        log.info("acquire: kept=%d fetched=%d dup=%d shards=%d",
                 stats.kept, stats.fetched, stats.duplicates, len(writer.shard_files))


def resolve_corpus_taxonomy(cfg: PipelineConfig, *, limit: int | None = None) -> dict[str, Any]:
    """Pre-resolve MeSH tree positions for every descriptor in the corpus.

    Doing this in one explicit pass keeps the hot crawl loop free of API calls
    and guarantees every article gets authoritative specialty facets. Results
    are cached to disk, so this is a one-time cost per corpus.
    """
    taxonomy = MeshTaxonomy(
        taxonomy_cache_path(cfg.paths.data_root),
        use_api=True,
        rate_limit=30.0,
        workers=12,
    )
    distinct: dict[str, str] = {}
    for doc in iter_documents(cfg):
        for term in doc.mesh_terms:
            if term.ui and term.ui not in distinct:
                distinct[term.ui] = term.descriptor
        if limit and len(distinct) >= limit:
            break
    log.info("resolving %d distinct MeSH descriptors", len(distinct))
    pairs = sorted(distinct.items())
    taxonomy.resolve_many(pairs)
    return taxonomy.stats()


def recategorize_corpus(cfg: PipelineConfig) -> dict[str, Any]:
    """Rewrite each document's ``categories`` using the resolved taxonomy.

    The crawl writes categories from label heuristics alone, because resolving
    MeSH tree numbers over HTTP inside the crawl loop would stall it. This pass
    applies the authoritative tree roots afterwards, rewriting the shards in
    place. It is a streaming rewrite, so memory stays bounded at corpus scale.
    """
    from .pubmed_parser import derive_categories
    from .shards import read_shards

    taxonomy = MeshTaxonomy(taxonomy_cache_path(cfg.paths.data_root), use_api=False)
    log.info("recategorize using %d cached MeSH tree numbers", len(taxonomy._cache))
    if not taxonomy._cache:
        log.warning("taxonomy cache is empty; run `medrag taxonomy` first")

    out_dir = cfg.paths.corpus_dir / cfg.acquire.source
    prefix = cfg.acquire.source
    shards = sorted(out_dir.glob(f"{prefix}-*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"no corpus shards under {out_dir}")

    changed = 0
    total = 0
    # Rewrite into a temp directory, then swap, so an interrupted pass never
    # leaves a half-written corpus behind.
    tmp_dir = out_dir.parent / f"{prefix}__recat_tmp"
    if tmp_dir.exists():
        for stale in tmp_dir.glob("*.jsonl.gz"):
            stale.unlink()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    shard_size = cfg.acquire.shard_size
    writer = ShardWriter(tmp_dir, prefix, shard_size=shard_size, resume=False)
    for shard in shards:
        for record in read_shards([shard]):
            total += 1
            mesh = [MeshTerm(**m) if isinstance(m, dict) else m for m in record.get("mesh_terms", [])]
            pub_types = tuple(record.get("publication_types", ()))
            new_cats = derive_categories(mesh, pub_types, taxonomy)
            if list(new_cats) != list(record.get("categories", ())):
                changed += 1
                record["categories"] = list(new_cats)
            writer.write(record)
    new_shards = writer.close()

    # Swap atomically: remove old shards, move the new ones into place.
    for shard in shards:
        shard.unlink()
    for shard in new_shards:
        shard.replace(out_dir / shard.name)
    tmp_dir.rmdir()

    stats = {
        "documents": total,
        "categories_changed": changed,
        "fraction_changed": round(changed / max(1, total), 4),
        "taxonomy_cache": len(taxonomy._cache),
        "shards": len(new_shards),
    }
    log.info("recategorize complete: %s", stats)
    return stats


def corpus_shards(cfg: PipelineConfig, *, crawl_if_missing: bool = False) -> list[Path]:
    """Locate corpus shards for the configured source.

    Fails loudly when they are absent. It used to call :func:`crawl_corpus` instead,
    which is wrong for every caller that merely wants to read: ``build-index``,
    ``chunk`` and the graph stages all come through here, and a naming mismatch
    between the shards on disk and the glob pattern turned ``medrag build-index``
    into a several-hour crawl that wrote 30,000 documents into the corpus directory
    nobody was looking at, with no error and no obvious symptom beyond a build that
    would not finish.

    Callers that genuinely mean "crawl if needed" -- the end-to-end pipeline -- pass
    ``crawl_if_missing=True`` explicitly.
    """
    out_dir = cfg.paths.corpus_dir / cfg.acquire.source
    shards = sorted(out_dir.glob(f"{cfg.acquire.source}-*.jsonl.gz"))
    if not shards:
        if not crawl_if_missing:
            present = sorted(out_dir.glob("*.jsonl.gz")) if out_dir.is_dir() else []
            hint = ""
            if present:
                hint = (
                    f" ({len(present)} file(s) are present but none match the expected "
                    f"'{cfg.acquire.source}-*.jsonl.gz' prefix; first is "
                    f"'{present[0].name}')"
                )
            raise FileNotFoundError(
                f"no corpus shards under {out_dir}{hint}. Crawl with "
                f"`medrag acquire`, or pass crawl_if_missing=True if that is what "
                f"you want -- silently crawling from a read path has already cost "
                f"one build-index run a full crawl."
            )
        crawl_corpus(cfg)
        shards = sorted(out_dir.glob(f"{cfg.acquire.source}-*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"no corpus shards under {out_dir} after crawling")
    return shards


def iter_documents(cfg: PipelineConfig) -> Iterator[Document]:
    """Stream Documents from shards without materialising the whole corpus.

    This is the path the large-scale stages use: a million-document corpus is
    tens of GB as Python objects, which does not fit alongside the indexes.
    """
    from .shards import read_shards

    shards = corpus_shards(cfg)
    n = 0
    for record in read_shards(shards):
        try:
            yield Document.from_dict(record)
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping malformed document record: %s", exc)
            continue
        n += 1
        if n % 100_000 == 0:
            log.info("streamed %d documents", n)
    log.info("streamed %d documents from %d shards", n, len(shards))


def crawl_or_load(cfg: PipelineConfig) -> list[Document]:
    """Return corpus documents as a list. Prefer :func:`iter_documents` at scale."""
    docs = list(iter_documents(cfg))
    log.info("loaded %d documents", len(docs))
    return docs
