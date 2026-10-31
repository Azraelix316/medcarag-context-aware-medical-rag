"""``medrag`` command line interface.

Subcommands form the pipeline stages, in execution order::

    medrag doctor          # environment + network self-check
    medrag acquire         # crawl PubMed/PMC into sharded JSONL
    medrag chunk           # structure-aware chunking into JSONL
    medrag build-index     # BM25 + dense indexes over the chunks
    medrag serve           # FastAPI retrieval service
    medrag query "..."     # one-off retrieval from the CLI
    medrag ask "..."       # full agentic research loop
    medrag benchmark       # construct the evaluation qrels
    medrag evaluate        # retrieval ablation across modes
    medrag pipeline        # acquire -> chunk -> index in one background run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Module-level logger. It was previously defined only as a local inside
# cmd_backfill_citations, so cmd_chunk's progress logging raised NameError on the
# first 50,000-document milestone -- i.e. it failed *after* doing the work, on a
# multi-hour job, with the corpus already partly written.
log = logging.getLogger(__name__)
from typing import Any

from .config import PipelineConfig

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(verbose: bool = False, log_file: str | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)
    # These libraries are extremely chatty at INFO.
    for noisy in ("urllib3", "faiss", "sentence_transformers", "transformers", "filelock", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_config(args: argparse.Namespace) -> PipelineConfig:
    paths = [Path(p) for p in (args.config or [])]
    cfg = PipelineConfig.load(paths if paths else None)
    overrides: dict[str, Any] = {}
    if getattr(args, "data_root", None):
        overrides["paths"] = {"data_root": args.data_root}
    if getattr(args, "max_records", None):
        overrides.setdefault("acquire", {})["max_records"] = args.max_records
    if getattr(args, "source", None):
        overrides.setdefault("acquire", {})["source"] = args.source
    if getattr(args, "index_dir", None):
        overrides.setdefault("paths", {})["index_dir"] = args.index_dir
    if getattr(args, "no_rerank", False):
        overrides["rerank"] = {"enabled": False}
    if overrides:
        base = cfg.to_dict()
        cfg = PipelineConfig.from_dict(_merge(base, overrides))._finalize()
    cfg.paths.ensure()
    return cfg


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


# ------------------------------------------------------------------- commands
def cmd_doctor(args: argparse.Namespace) -> int:
    """Environment and connectivity self-check."""
    import importlib
    import platform

    print("=" * 68)
    print("MedRAG doctor")
    print("=" * 68)
    print(f"python     : {platform.python_version()} ({platform.machine()})")
    print(f"data root  : {Path('data').resolve()}")

    print("\n-- python packages --")
    missing: list[str] = []
    for mod in ("numpy", "scipy", "torch", "transformers", "sentence_transformers",
                "faiss", "lxml", "fastapi", "uvicorn", "yaml", "requests", "tqdm"):
        try:
            m = importlib.import_module(mod)
            print(f"  ok   {mod:22s} {getattr(m, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  MISS {mod:22s} {exc}")
            missing.append(mod)

    print("\n-- gpu --")
    try:
        import torch

        print(f"  cuda available : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  gpu{i}          : {p.name} sm_{p.major}{p.minor} {p.total_memory / 2**30:.1f}GiB")
    except Exception as exc:  # noqa: BLE001
        print(f"  torch probe failed: {exc}")

    print("\n-- network --")
    try:
        from .data.eutils import EutilsClient

        client = EutilsClient(rate=1.0)
        print(f"  ncbi e-utilities : {'ok' if client.health_check() else 'UNREACHABLE'}")
        print(f"  ncbi api key     : {'set' if client.api_key else 'not set (3 req/s limit)'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ncbi probe failed: {exc}")

    for url, label in (
        ("https://huggingface.co/api/models?limit=1", "huggingface"),
        ("https://pypi.org/pypi/numpy/json", "pypi"),
    ):
        try:
            import requests

            code = requests.get(url, timeout=15).status_code
            print(f"  {label:16s}: {code}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:16s}: {exc}")

    print("\n-- disk --")
    import shutil

    usage = shutil.disk_usage(".")
    print(f"  free : {usage.free / 2**30:.1f} GiB of {usage.total / 2**30:.1f} GiB")

    if missing:
        print(f"\nRESULT: {len(missing)} package(s) missing: {', '.join(missing)}")
        print("Run: bash scripts/setup_env.sh")
        return 1
    print("\nRESULT: environment looks good")
    return 0


def cmd_acquire(args: argparse.Namespace) -> int:
    from .data.acquire import crawl_corpus

    cfg = load_config(args)
    manifest = crawl_corpus(cfg, shard=args.shard, nshards=args.nshards)
    print(json.dumps(manifest.to_dict(), indent=2)[:2000])
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    from .data.acquire import iter_documents
    from .data.chunk import chunk_corpus, chunk_document
    from .data.shards import ShardWriter, atomic_write_json

    cfg = load_config(args)
    out_dir = cfg.paths.corpus_dir / cfg.acquire.source / "chunks"
    writer = ShardWriter(out_dir, "chunks", shard_size=cfg.acquire.shard_size)

    t0 = time.time()
    n_chunks = 0
    n_docs = 0
    n_tokens = 0
    for doc in iter_documents(cfg):
        n_docs += 1
        for chunk in chunk_document(doc, cfg.chunk):
            writer.write(chunk.to_dict())
            n_chunks += 1
            n_tokens += chunk.token_count
        if n_docs % 50_000 == 0:
            log.info("chunked %d docs -> %d chunks (%.0f docs/s)",
                     n_docs, n_chunks, n_docs / max(1e-9, time.time() - t0))
    shards = writer.close()
    written = writer.report()["written"]
    elapsed = time.time() - t0

    manifest = {
        "n_documents": n_docs,
        "n_chunks": n_chunks,
        "mean_tokens": round(n_tokens / max(1, n_chunks), 1),
        # Only shards this run wrote. `shards` also lists shards it merely resumed
        # over, which made a duplicated 1,325,000-chunk prefix look correct: the
        # manifest reported 7,387,592 while the directory held 8,712,592.
        "shards": written,
        "n_shards_written": len(written),
        "n_shards_resumed_over": len(shards) - len(written),
        "elapsed_s": round(elapsed, 1),
        "chunk_config": cfg.to_dict()["chunk"],
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(out_dir / "manifest.json", manifest)
    if manifest["n_shards_resumed_over"]:
        log.warning(
            "chunk: resumed over %d pre-existing shards; those were NOT rewritten "
            "and their records are not in n_chunks. A resumed directory is only "
            "safe if the earlier run stopped before writing any record.",
            manifest["n_shards_resumed_over"])
    print(json.dumps(manifest, indent=2)[:2000])
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    from .data.acquire import iter_documents
    from .data.chunk import chunk_corpus
    from .index.builder import build_index

    cfg = load_config(args)
    # Stream documents: a million-doc corpus cannot be held as Python objects
    # alongside the indexes.
    report = build_index(
        cfg,
        chunk_corpus(iter_documents(cfg), cfg.chunk),
        store_cache_size=args.store_cache,
        force=args.force,
        skip_dense=args.skip_dense,
        bm25_spill_dir=(None if args.no_bm25_spill else str(cfg.paths.index_dir / "bm25_spill")),
        skip_bm25=args.skip_bm25,
        block_shard=args.block_shard,
        block_nshards=args.block_nshards,
        skip_faiss=args.no_faiss,
        reverse_blocks=args.reverse_blocks,
    )
    print(json.dumps(report, indent=2)[:3000])
    return 0


def cmd_bm25_segments(args: argparse.Namespace) -> int:
    """Build BM25 one segment at a time; several workers may share --out-dir."""
    import shutil

    from .index import segmented as sg

    cfg = load_config(args)
    store_dir = cfg.paths.index_dir / "store"
    out_dir = Path(args.out_dir) if args.out_dir else cfg.paths.index_dir / "bm25_segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = sg.segment_bounds(sg.store_n_rows(store_dir), args.rows_per_segment)
    built = 0
    while True:
        seg = sg.claim_segment(out_dir, len(bounds)) if args.segment is None else args.segment
        if seg is None or (args.segment is not None and built):
            break
        lo, hi = bounds[seg]
        spill = (Path(args.spill_root) if args.spill_root else out_dir) / f"spill-seg{seg:03d}"
        sg.build_bm25_segment(sg.iter_range_texts(store_dir, lo, hi), out_dir, seg, lo, hi,
                              k1=cfg.sparse.k1, b=cfg.sparse.b,
                              min_token_length=cfg.sparse.min_token_length,
                              spill_dir=spill)
        shutil.rmtree(spill, ignore_errors=True)
        built += 1
    done = len(list(out_dir.glob("bm25-seg[0-9][0-9][0-9].stats.pkl")))
    print(json.dumps({"built_here": built, "segments_done": done, "segments": len(bounds)}))
    return 0


def cmd_search_segmented(args: argparse.Namespace) -> int:
    """Exact sparse and/or dense top-k for a query file, streamed from disk."""
    import numpy as np

    from .index import segmented as sg

    cfg = load_config(args)
    index_dir = cfg.paths.index_dir
    rows = [json.loads(line) for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    qids = [str(r[args.id_field]) for r in rows]
    texts = [str(r[args.query_field]) for r in rows]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results: dict[str, list[list[tuple[int, float]]]] = {}
    if "sparse" in modes:
        seg_dir = Path(args.bm25_dir) if args.bm25_dir else index_dir / "bm25_segments"
        k = args.top_k or cfg.fusion.sparse_top_k
        if args.partial_out:
            # One host's share of the segments; --merge-partials joins them.
            only = [int(x) for x in args.segments.split(",")]
            s, i = sg.segmented_bm25_search(seg_dir, texts, k, only=only, partial=True)
            np.savez(args.partial_out, scores=s, rows=i, segments=np.array(only))
            print(json.dumps({"queries": len(qids), "segments": only, "out": args.partial_out}))
            return 0
        if args.merge_partials:
            parts, covered = [], []
            for p in args.merge_partials.split(","):
                z = np.load(p)
                parts.append((z["scores"], z["rows"]))
                covered += z["segments"].tolist()
            n_seg = len(list(seg_dir.glob("bm25-seg[0-9][0-9][0-9].stats.pkl")))
            if sorted(covered) != list(range(n_seg)):
                raise SystemExit(f"partials cover segments {sorted(covered)}, need 0..{n_seg - 1} once")
            results["sparse"] = sg.merge_partial_topk(parts, k)
        else:
            results["sparse"] = sg.segmented_bm25_search(seg_dir, texts, k)
    if "dense" in modes:
        from .index.embeddings import DenseEmbedder

        vecs = DenseEmbedder(cfg.embed).load().embed(texts, is_query=True)
        results["dense"] = sg.stream_dense_search(
            sg.load_embedding_matrix(index_dir), vecs, args.top_k or cfg.fusion.dense_top_k)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for i, qid in enumerate(qids):
            fh.write(json.dumps({"qid": qid, **{m: results[m][i] for m in results}}) + "\n")
    tmp.replace(out)
    print(json.dumps({"queries": len(qids), "modes": list(results), "out": str(out)}))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .serve.api import run_server

    cfg = load_config(args)
    run_server(cfg, host=args.host or cfg.serve.host, port=args.port or cfg.serve.port,
               warmup=not args.no_warmup)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from .index.retriever import load_index

    cfg = load_config(args)
    retriever = load_index(cfg, Path(args.index_dir) if args.index_dir else None,
                           with_reranker=not args.no_rerank)
    result = retriever.retrieve(args.query, top_k=args.top_k, mode=args.mode)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    print(f"\nquery: {result.query}   [mode={result.mode}]")
    timings = ", ".join(f"{k}={v * 1000:.0f}ms" for k, v in result.timings.items())
    print(f"timings: {timings}")
    for i, passage in enumerate(result.passages, 1):
        print(f"\n[{i}] score={passage.score:.4f}  {passage.citation}")
        print(f"    section: {passage.section_title or '-'} | sources={','.join(passage.sources)}")
        text = passage.text[: args.char_limit].replace("\n", " ")
        print(f"    {text}...")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .agent.loop import ResearchAgent
    from .agent.planner import QueryPlanner
    from .index.retriever import load_index

    cfg = load_config(args)
    retriever = load_index(cfg, Path(args.index_dir) if args.index_dir else None,
                           with_reranker=not args.no_rerank)
    planner = QueryPlanner(strategy=args.planner, mode=args.mode if args.mode != "reranked" else "auto")
    # `mode` was accepted on the command line but never forwarded, so
    # `medrag ask --mode context` silently ran the default retrieval. The agent's
    # default is MODE_RERANKED, so this only bit on the ablatable modes.
    agent = ResearchAgent(retriever, planner=planner, max_hops=args.max_hops,
                          top_k=args.top_k, mode=args.mode,
                          attribute_caveats=args.attribute_caveats)
    answer = agent.research(args.question)

    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
        return 0
    print(f"\nQ: {answer.question}\n")
    print(f"plan: {answer.plan.get('strategy')} ({len(answer.plan.get('sub_queries', []))} sub-queries, "
          f"multi_hop={answer.plan.get('is_multi_hop')})")
    print(f"iterations: {answer.iterations}  coverage: {answer.coverage.score:.2f}  "
          f"elapsed: {answer.elapsed:.2f}s")
    print(f"\n{answer.answer}")
    if args.show_trace:
        print("\n-- trace --")
        for step in answer.trace:
            print(f"  {step.step}. {step.action:8s} {step.detail} ({step.elapsed:.2f}s)")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .data.mesh_taxonomy import MeshTaxonomy, taxonomy_cache_path
    from .eval.benchmark import build_benchmark_from_corpus

    cfg = load_config(args)
    # MeSH tree roots decide which descriptors are clinical enough to form a
    # benchmark query; resolve them authoritatively rather than guessing.
    taxonomy = MeshTaxonomy(taxonomy_cache_path(cfg.paths.data_root), use_api=not args.offline)
    log_note(f"mesh taxonomy cache: {len(taxonomy._cache)} descriptors")
    out_dir = cfg.paths.runs_dir / "benchmark"
    # Streamed from shards: 624k Documents as Python objects would not fit
    # alongside the index build.
    summary = build_benchmark_from_corpus(
        cfg, out_dir, n_clinical=args.n_clinical,
        n_multihop=args.n_multihop, seed=args.seed, taxonomy=taxonomy,
        mesh_sidecar=Path(args.mesh_sidecar) if args.mesh_sidecar else None,
    )
    summary["taxonomy"] = taxonomy.stats()
    print(json.dumps(summary, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .eval.metrics import Qrels, compare_modes, significance_bootstrap
    from .index.retriever import load_index

    cfg = load_config(args)
    qrels_path = Path(args.qrels) if args.qrels else cfg.paths.runs_dir / "benchmark" / "qrels_combined.json"
    if not qrels_path.exists():
        print(f"qrels not found: {qrels_path}\nRun `medrag benchmark` first.", file=sys.stderr)
        return 1
    qrels = Qrels.load(qrels_path)
    if args.sample and args.sample < len(qrels):
        qrels = Qrels(qrels.sample(args.sample, seed=cfg.evaluation.seed))
    print(f"evaluating {len(qrels)} queries over {args.modes}")

    retriever = load_index(cfg, Path(args.index_dir) if args.index_dir else None,
                           with_reranker=not args.no_rerank)
    results = compare_modes(retriever, qrels, modes=tuple(args.modes),
                            k_values=tuple(cfg.evaluation.metrics_at_k))

    # Paired bootstrap of reranked vs the strongest non-reranked baseline.
    significance: dict[str, Any] = {}
    if len(results) >= 2 and results[0].per_query and results[-1].per_query:
        best_base = results[-2]
        key = next((k for k in results[-1].metrics if k.startswith("ndcg@")), None)
        if key:
            significance[f"reranked_vs_{best_base.name}:{key}"] = significance_bootstrap(
                [r.get(key, float("nan")) for r in results[-1].per_query],
                [r.get(key, float("nan")) for r in best_base.per_query],
            )

    out = {
        "qrels": str(qrels_path),
        "n_queries": len(qrels),
        "results": [r.to_dict() for r in results],
        "significance": significance,
        "index": retriever.stats(),
    }
    print("\n" + "\n\n".join(r.format_table() for r in results))
    if significance:
        print("\n-- significance (paired bootstrap) --")
        print(json.dumps(significance, indent=2))
    if args.out:
        from .data.shards import atomic_write_json

        atomic_write_json(Path(args.out), out)
        print(f"\nwrote {args.out}")
    return 0


def cmd_mesh_graph(args: argparse.Namespace) -> int:
    """Build the MeSH co-occurrence graph and the doc->descriptor map."""
    from .data.acquire import iter_documents
    from .index.graph_retriever import DocDescriptorMap, build_graph_index
    from .index.mesh_graph import MeshGraph, iter_doc_descriptors

    cfg = load_config(args)
    t0 = time.time()
    graph = MeshGraph(min_df_for_expansion=args.min_df).build(
        iter_doc_descriptors(iter_documents(cfg)),
        min_pair_count=args.min_pair_count,
    )
    log_note(f"graph: {graph.stats()} in {time.time() - t0:.0f}s")

    # Second pass for the doc -> descriptor map, so both halves come from the
    # same corpus revision.
    doc_map = DocDescriptorMap().build(graph, iter_documents(cfg))
    paths = build_graph_index(graph, doc_map, cfg.paths.index_dir)
    print(json.dumps({
        "graph": graph.stats(),
        "doc_map": {"n_docs": len(doc_map), "bytes": doc_map.nbytes},
        "paths": paths,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2))
    return 0


def cmd_backfill_citations(args: argparse.Namespace) -> int:
    """Re-fetch an already-crawled corpus to fill its citation fields.

    A parser improvement can land hours into a multi-hour crawl, leaving the fetched
    records with a field the schema implies is populated but is not. Re-crawling to
    fix one field discards the long pole; re-fetching is far cheaper, because the
    crawl's cost is dominated by esearch window iteration rather than by efetch.
    """
    from .data.backfill import backfill_field
    from .data.eutils import EutilsClient
    from .data.shards import read_shards

    cfg = load_config(args)
    shard_dir = cfg.paths.corpus_dir / cfg.acquire.source
    shards = sorted(shard_dir.glob(args.shard_prefix + "*.jsonl.gz"))
    if not shards:
        print(f"no shards under {shard_dir} matching {args.shard_prefix!r}*")
        return 1

    client = EutilsClient(api_key=cfg.acquire.api_key, email=cfg.acquire.email,
                          tool=cfg.acquire.tool)
    # Only documents that actually lack the field are re-fetched, so a corpus that
    # is already complete costs nothing.
    docs = sum(1 for p in shards for _ in read_shards([p]))
    log.info("backfilling citations over %d shards (%d documents)", len(shards), docs)
    stats = backfill_field(shards, "citations", client=client,
                           source=cfg.acquire.source, limit=args.limit,
                           dry_run=args.dry_run)
    print(json.dumps(stats, indent=2))
    if args.dry_run:
        print("dry run: nothing fetched and nothing written")
    return 0


def cmd_dependency_graph(args: argparse.Namespace) -> int:
    """Build the section-dependency graph over chunks."""
    from .data.acquire import iter_documents
    from .index.dependency_graph import DependencyGraph
    from .index.store import ChunkStore

    cfg = load_config(args)
    t0 = time.time()
    if args.scan_inputs:
        from .index.dependency_graph import scan_graph_inputs

        print(json.dumps(scan_graph_inputs(cfg.paths.index_dir, workers=args.workers), indent=2))
        return 0
    if args.from_scan:
        # Documents and their first rows from the page texts, which record exactly
        # ChunkStore.document_rows' output without loading the row map.
        from .index.dependency_graph import build_from_scanned_inputs
        from .index.page_index import read_page_text_block

        pages = json.loads((cfg.paths.index_dir / "page_texts" / "pages.json").read_text())
        doc_ids: list[str] = []
        first_rows: list[int] = []
        for b in range(pages["n_blocks"]):
            for rec in read_page_text_block(cfg.paths.index_dir, b):
                doc_ids.append(rec["doc"])
                first_rows.append(rec["row"])
        graph = build_from_scanned_inputs(cfg.paths.index_dir, doc_ids, first_rows,
                                          progress_every=10_000_000)
        del doc_ids, first_rows
        return _save_dependency_graph(cfg, graph, t0)
    store = ChunkStore.load(cfg.paths.index_dir / "store", cache_size=args.store_cache)
    log_note(f"dependency graph over {len(store):,} chunks")

    # Separate accessors, not two walrus lambdas: a walrus assigned in one lambda
    # is invisible in the other, which fails at the first empty chunk.
    def kind_of(row: int) -> str:
        chunk = store.get(row)
        return (chunk.section_kind if chunk else "") or ""

    def doc_of(row: int) -> str:
        chunk = store.get(row)
        return (chunk.doc_id if chunk else "") or ""

    graph = DependencyGraph.build(store.rows, kind_of=kind_of, doc_of=doc_of,
                                    progress_every=args.store_cache * 50)
    return _save_dependency_graph(cfg, graph, t0)


def _save_dependency_graph(cfg: PipelineConfig, graph: Any, t0: float) -> int:
    path = graph.save(cfg.paths.index_dir / "dependency_graph.pkl")
    print(json.dumps({
        "stats": graph.stats(),
        "degree_by_kind": {k: round(v, 3) for k, v in graph.degree_by_kind().items()},
        "path": str(path),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2))
    return 0


def cmd_page_index(args: argparse.Namespace) -> int:
    """Build document-level vectors stored alongside the chunk index."""
    from .index.page_index import PageIndex
    from .index.store import ChunkStore

    cfg = load_config(args)
    if args.write_texts or args.assemble_blocks:
        # The blocked build, for corpora whose page matrix does not fit in RAM.
        from .index.page_index import assemble_page_blocks, write_page_text_blocks

        if args.write_texts:
            print(json.dumps(write_page_text_blocks(cfg.paths.index_dir), indent=2))
        if args.assemble_blocks:
            idx = PageIndex.load(assemble_page_blocks(cfg.paths.index_dir))
            print(json.dumps({"page_index": idx.stats()}, indent=2))
        return 0
    store = ChunkStore.load(cfg.paths.index_dir / "store", cache_size=args.store_cache)
    log_note(f"page index over {len(store):,} chunks")
    idx = PageIndex.build(cfg, store.rows, store=store, max_docs=args.max_docs)
    path = idx.save(cfg.paths.index_dir / "page_index.pkl")
    print(json.dumps({"page_index": idx.stats(), "path": str(path)}, indent=2))
    return 0


def cmd_cocitation(args: argparse.Namespace) -> int:
    """Build the co-citation graph and report corroboration coverage."""
    from .data.acquire import iter_documents
    from .index.cocitation import CoCitationGraph

    cfg = load_config(args)
    t0 = time.time()
    g = CoCitationGraph.build(
        iter_documents(cfg),
        min_shared=args.min_shared,
        max_shared=args.max_shared,
        max_citers=args.max_citers,
    )
    path = g.save(str(cfg.paths.index_dir / "cocitation.pkl"))
    print(json.dumps({
        "cocitation": g.stats(),
        "path": str(path),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2))
    return 0


def cmd_taxonomy(args: argparse.Namespace) -> int:
    """Resolve MeSH tree positions for every descriptor in the corpus.

    MeSH tree numbers are not present in PubMed XML and cannot be derived from
    descriptor UIs (every UI starts with ``D`` for both diseases and drugs), so
    they are looked up against the NLM MeSH API and cached to disk. The crawl
    falls back to label heuristics until this has been run at least once.
    """
    from .data.acquire import resolve_corpus_taxonomy
    from .data.mesh_taxonomy import MeshTaxonomy, taxonomy_cache_path

    cfg = load_config(args)
    taxonomy = MeshTaxonomy(taxonomy_cache_path(cfg.paths.data_root))
    print(f"cache before: {taxonomy.stats()}  ({taxonomy.cache_path})")
    stats = resolve_corpus_taxonomy(cfg, limit=args.limit)
    print(json.dumps({"cache_after": stats, "path": str(taxonomy_cache_path(cfg.paths.data_root))}, indent=2))
    return 0


def cmd_recategorize(args: argparse.Namespace) -> int:
    """Rewrite document categories from the resolved MeSH taxonomy."""
    from .data.acquire import recategorize_corpus

    cfg = load_config(args)
    print(json.dumps(recategorize_corpus(cfg), indent=2))
    return 0


def cmd_paths(args: argparse.Namespace) -> int:
    """Print the resolved artifact paths, one per line.

    Shell scripts use this instead of parsing the config dump, so that stage
    markers always point at the real locations even under config layering.
    """
    cfg = load_config(args)
    for label, path in (
        ("data_root", cfg.paths.data_root),
        ("corpus_dir", cfg.paths.corpus_dir),
        ("index_dir", cfg.paths.index_dir),
        ("runs_dir", cfg.paths.runs_dir),
        ("cache_dir", cfg.paths.cache_dir),
    ):
        print(f"{label}={path}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    index_dir = Path(args.index_dir) if args.index_dir else cfg.paths.index_dir
    print(json.dumps({"index_dir": str(index_dir), "config": cfg.to_dict()}, indent=2))
    for name in ("manifest.json", "chunks/manifest.json"):
        for candidate in cfg.paths.corpus_dir.rglob(name):
            try:
                print(f"\n-- {candidate.relative_to(cfg.paths.data_root)} --")
                print(json.dumps(json.loads(candidate.read_text()), indent=2)[:1500])
            except Exception as exc:  # noqa: BLE001
                print(f"  unreadable: {exc}")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Run acquire -> chunk -> build-index end to end."""
    from .data.acquire import crawl_corpus, iter_documents
    from .data.chunk import chunk_corpus
    from .index.builder import build_index

    cfg = load_config(args)
    stages: list[str] = ["acquire", "chunk", "index"]
    if args.skip_acquire:
        stages.remove("acquire")
    if args.skip_dense:
        log_note("dense indexing disabled (--skip-dense)")

    timings: dict[str, float] = {}
    t_all = time.time()

    if "acquire" in stages:
        t0 = time.time()
        crawl_corpus(cfg)
        timings["acquire"] = time.time() - t0

    t0 = time.time()
    # Stream from shards so a large corpus never lands in RAM as Python objects.
    report = build_index(cfg, chunk_corpus(iter_documents(cfg), cfg.chunk),
                         store_cache_size=args.store_cache, skip_dense=args.skip_dense)
    timings["chunk+index"] = time.time() - t0

    timings["total"] = time.time() - t_all
    print(json.dumps({"timings": {k: round(v, 1) for k, v in timings.items()}, "report": report},
                     indent=2)[:4000])
    return 0


def log_note(msg: str) -> None:
    print(f"[note] {msg}", file=sys.stderr)


# ------------------------------------------------------------------- argument
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medrag",
        description="Large-scale medical RAG corpus, retrieval index, and research agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--config", action="append", help="YAML config file (repeatable; later files win)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--log-file", help="also write logs here")
    parser.add_argument("--data-root", help="override data_root")
    parser.add_argument("--index-dir", help="override index_dir")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="environment and connectivity self-check")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("acquire", help="crawl PubMed/PMC into sharded JSONL")
    p.add_argument("--source", choices=["pubmed", "pmc"], help="override acquire.source")
    p.add_argument("--max-records", type=int, help="override acquire.max_records")
    p.add_argument("--shard", type=int, default=0,
                   help="this node's shard index (0-based)")
    p.add_argument("--nshards", type=int, default=1,
                   help="total shards; query slices are assigned round-robin")
    p.set_defaults(func=cmd_acquire)

    p = sub.add_parser("chunk", help="chunk the corpus into retrievable passages")
    p.set_defaults(func=cmd_chunk)

    p = sub.add_parser("build-index", help="build BM25 + dense indexes")
    p.add_argument("--store-cache", type=int, default=500_000, help="chunk payloads held in RAM")
    p.add_argument("--force", action="store_true", help="rebuild even if artifacts exist")
    p.add_argument("--skip-dense", action="store_true", help="build BM25 only")
    p.add_argument("--no-bm25-spill", action="store_true",
                   help="keep BM25 postings in RAM instead of spilling them to disk; "
                        "needed above ~22 GiB of postings, where holding inputs and "
                        "the CSC together no longer fits")
    p.add_argument("--skip-bm25", action="store_true",
                   help="build dense only. BM25 over 16.1M chunks needs roughly "
                        "15 GiB for the sparse matrix alone, so running it "
                        "alongside an embedding shard on a 31 GiB box gets the "
                        "process OOM-killed with no log entry")
    p.add_argument("--block-shard", type=int, default=0,
                   help="embed every Nth block (0-based); for multi-node embedding")
    p.add_argument("--block-nshards", type=int, default=1,
                   help="total embedding shards; 1 = do everything locally")
    p.add_argument("--no-faiss", action="store_true",
                   help="assemble embeddings.f32 on disk plus embeddings.json and stop; "
                        "no FAISS index. A flat index holds every vector in RAM, which "
                        "a 48M-chunk corpus (74 GB) cannot do on a 31 GB host")
    p.add_argument("--reverse-blocks", action="store_true",
                   help="walk this shard's blocks last-to-first, so a second worker on "
                        "the same shard (e.g. CPU beside GPU) meets the first only once")
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("bm25-segments",
                       help="build BM25 as row segments, for corpora whose index exceeds RAM")
    p.add_argument("--rows-per-segment", type=int, default=4_000_000,
                   help="rows per segment; peak build RAM is roughly 1.5 GiB per million rows")
    p.add_argument("--segment", type=int, help="build only this segment, unclaimed")
    p.add_argument("--out-dir", help="default: <index_dir>/bm25_segments")
    p.add_argument("--spill-root",
                   help="where each segment spills its postings while finalizing (a few GB, "
                        "deleted after); a local disk is faster than NFS. default: --out-dir")
    p.set_defaults(func=cmd_bm25_segments)

    p = sub.add_parser("search-segmented",
                       help="exact sparse/dense top-k for a JSONL query file, streamed from disk")
    p.add_argument("--queries", required=True, help="JSONL, one query object per line")
    p.add_argument("--id-field", default="qid")
    p.add_argument("--query-field", default="query")
    p.add_argument("--modes", default="sparse,dense")
    p.add_argument("--top-k", type=int, help="default: fusion.sparse_top_k / dense_top_k")
    p.add_argument("--bm25-dir", help="default: <index_dir>/bm25_segments")
    p.add_argument("--out", required=True, help="JSONL: qid plus [row, score] lists per mode")
    p.add_argument("--segments", help="with --partial-out: comma-separated segment ids to score")
    p.add_argument("--partial-out", help="write this host's sparse top-k over --segments (.npz)")
    p.add_argument("--merge-partials",
                   help="comma-separated --partial-out files covering every segment once")
    p.set_defaults(func=cmd_search_segmented)

    p = sub.add_parser("serve", help="run the FastAPI retrieval service")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--no-warmup", action="store_true", help="skip preloading model weights")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("query", help="one-off hybrid retrieval")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--mode", default="reranked",
                   choices=["sparse", "dense", "hybrid", "reranked", "graph",
                            "hybrid3", "hybrid3r", "bleed", "context", "bleedctx",
                            "page", "pagectx"])
    p.add_argument("--char-limit", type=int, default=400)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("ask", help="full agentic research loop")
    p.add_argument("question")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-hops", type=int, default=2)
    p.add_argument("--mode", default="reranked",
                   choices=["sparse", "dense", "hybrid", "reranked", "graph",
                            "hybrid3", "hybrid3r", "bleed", "context", "bleedctx",
                            "page", "pagectx"])
    p.add_argument("--planner", default="heuristic", choices=["none", "heuristic", "llm"])
    p.add_argument("--show-trace", action="store_true")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--attribute-caveats", action="store_true",
                   help="append limitations quoted from the retrieved passages "
                        "(conditional caveat rate 0.0556 -> 1.0000, unconditional 0.0000)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("backfill-citations",
                       help="re-fetch an already-crawled corpus to fill citations "
                            "(PMC JATS bibliographies were unread until recently)")
    p.add_argument("--limit", type=int, help="cap documents, for a cost probe")
    p.add_argument("--dry-run", action="store_true",
                   help="report how many documents lack citations without fetching")
    p.add_argument("--shard-prefix", default="",
                   help="only shards whose name starts with this; lets two nodes "
                        "backfill disjoint halves of one merged corpus")
    p.set_defaults(func=cmd_backfill_citations)

    p = sub.add_parser("mesh-graph", help="build the MeSH co-occurrence graph index")
    p.add_argument("--min-df", type=int, default=50,
                   help="descriptor df floor for PMI expansion eligibility")
    p.add_argument("--min-pair-count", type=int, default=3,
                   help="prune co-occurrence pairs seen fewer times than this")
    p.set_defaults(func=cmd_mesh_graph)

    p = sub.add_parser("dependency-graph", help="build the section-dependency graph over chunks")
    p.add_argument("--store-cache", type=int, default=400_000,
                   help="chunks held in RAM while streaming the store")
    p.add_argument("--scan-inputs", action="store_true",
                   help="scan section kinds and document starts from the blob in parallel "
                        "into <index_dir>/graph_inputs, for stores too large for store.get()")
    p.add_argument("--from-scan", action="store_true",
                   help="build from graph_inputs plus page_texts (same graph as the default)")
    p.add_argument("--workers", type=int, default=6, help="processes for --scan-inputs")
    p.set_defaults(func=cmd_dependency_graph)

    p = sub.add_parser("page-index", help="build document-level (page) vectors")
    p.add_argument("--store-cache", type=int, default=300_000)
    p.add_argument("--max-docs", type=int, help="cap documents, for a quick probe")
    p.add_argument("--write-texts", action="store_true",
                   help="blocked build, step 1: write page texts to <index_dir>/page_texts")
    p.add_argument("--assemble-blocks", action="store_true",
                   help="blocked build, step 3: page_blocks/*.npz -> page_vectors.f32 + "
                        "page_index.pkl (step 2 is scripts/remote_embed.py --kind page)")
    p.set_defaults(func=cmd_page_index)

    p = sub.add_parser("cocitation", help="build the co-citation graph over documents")
    p.add_argument("--min-shared", type=int, default=2,
                   help="minimum shared references for a co-citation edge")
    p.add_argument("--max-shared", type=int, default=25,
                   help="drop pairs sharing more than this many references")
    p.add_argument("--max-citers", type=int, default=200,
                   help="memory bound: skip references cited by more documents")
    p.set_defaults(func=cmd_cocitation)

    p = sub.add_parser("taxonomy", help="resolve MeSH tree numbers for the corpus")
    p.add_argument("--limit", type=int, help="stop after resolving this many descriptors")
    p.set_defaults(func=cmd_taxonomy)

    p = sub.add_parser("recategorize", help="rewrite document categories from the MeSH taxonomy")
    p.set_defaults(func=cmd_recategorize)

    p = sub.add_parser("benchmark", help="build evaluation qrels from the corpus")
    p.add_argument("--n-clinical", type=int, default=2000)
    p.add_argument("--n-multihop", type=int, default=500)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--offline", action="store_true", help="do not call the MeSH API; use label heuristics")
    p.add_argument("--mesh-sidecar",
                   help="pmc_mesh.json.gz from scripts/attach_mesh_to_pmc.py, for full-text corpora")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("evaluate", help="retrieval ablation across modes")
    p.add_argument("--qrels", help="path to qrels JSON")
    p.add_argument("--sample", type=int, default=1000, help="queries to evaluate (0 = all)")
    p.add_argument("--modes", nargs="+",
                   default=["sparse", "dense", "hybrid", "graph", "hybrid3", "reranked"])
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--out", help="write the JSON report here")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("stats", help="print corpus and index statistics")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("paths", help="print resolved artifact paths (key=value per line)")
    p.set_defaults(func=cmd_paths)

    p = sub.add_parser("pipeline", help="acquire -> chunk -> index end to end")
    p.add_argument("--source", choices=["pubmed", "pmc"])
    p.add_argument("--max-records", type=int)
    p.add_argument("--store-cache", type=int, default=500_000)
    p.add_argument("--skip-acquire", action="store_true")
    p.add_argument("--skip-dense", action="store_true")
    p.set_defaults(func=cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose, log_file=args.log_file)
    if args.verbose:
        import transformers

        transformers.logging.set_verbosity_info()
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("medrag.cli").exception("command failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
