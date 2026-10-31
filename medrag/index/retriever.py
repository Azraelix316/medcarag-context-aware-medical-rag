"""The retriever: hybrid BM25 + dense, fused, then cross-encoder reranked.

This is the object an agent calls. It owns the three indexes (lexical, vector,
chunk store) plus optional reranking, and exposes one ``retrieve`` entry point
that returns ranked, provenance-carrying results.

Retrieval modes let the evaluation isolate each component's contribution:
``sparse`` (BM25 only), ``dense`` (vectors only), ``hybrid`` (fusion, no
rerank), and ``reranked`` (the full pipeline).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import PipelineConfig
from .bm25 import BM25Index
from .dense import DenseIndex
from .embeddings import DenseEmbedder
from .hybrid import Candidate, candidate_stats, fuse
from .rerank import CrossEncoderReranker
from .store import ChunkStore

log = logging.getLogger(__name__)

MODE_SPARSE = "sparse"
MODE_DENSE = "dense"
MODE_HYBRID = "hybrid"
MODE_RERANKED = "reranked"
MODE_GRAPH = "graph"
MODE_HYBRID3 = "hybrid3"          # sparse + dense + MeSH graph
MODE_HYBRID3_RERANKED = "hybrid3r"  # the above, then cross-encoder
MODE_BLEED = "bleed"              # sparse, then sibling-bleed document conditioning
MODE_CONTEXT = "context"          # sparse, plus dependency-graph framing passages
MODE_BLEED_CONTEXT = "bleedctx"   # sibling bleed, plus framing passages
MODE_PAGE = "page"                # sparse, then page-level document conditioning
MODE_PAGE_CONTEXT = "pagectx"     # page conditioning, plus framing passages
ALL_MODES = (
    MODE_PAGE, MODE_PAGE_CONTEXT,
    MODE_SPARSE, MODE_DENSE, MODE_HYBRID, MODE_RERANKED,
    MODE_GRAPH, MODE_HYBRID3, MODE_HYBRID3_RERANKED,
    MODE_BLEED, MODE_CONTEXT, MODE_BLEED_CONTEXT,
)
# Modes that require the MeSH graph index.
GRAPH_MODES = frozenset({MODE_GRAPH, MODE_HYBRID3, MODE_HYBRID3_RERANKED})
# Modes that require the section-dependency graph.
DEPENDENCY_MODES = frozenset({MODE_CONTEXT, MODE_BLEED_CONTEXT, MODE_PAGE_CONTEXT})
PAGE_MODES = frozenset({MODE_PAGE, MODE_PAGE_CONTEXT})
# Modes that apply sibling-bleed document conditioning.
BLEED_MODES = frozenset({MODE_BLEED, MODE_BLEED_CONTEXT})


@dataclass
class RetrievedPassage:
    """One result, with enough context to cite it without a second lookup."""

    chunk_id: str
    row_id: int
    text: str
    score: float
    title: str = ""
    section_title: str = ""
    section_kind: str = ""
    journal: str = ""
    year: int = 0
    pmid: str = ""
    pmcid: str = ""
    doi: str = ""
    is_abstract: bool = False
    publication_types: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    sources: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        venue = self.journal or ""
        year = str(self.year) if self.year else "n.d."
        ident = f"PMID:{self.pmid}" if self.pmid else (f"doi:{self.doi}" if self.doi else self.chunk_id)
        return f"{self.title.strip()} ({venue}, {year}) [{ident}]".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "row_id": self.row_id,
            "text": self.text,
            "score": round(self.score, 6),
            "title": self.title,
            "section_title": self.section_title,
            "section_kind": self.section_kind,
            "journal": self.journal,
            "year": self.year,
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "doi": self.doi,
            "is_abstract": self.is_abstract,
            "publication_types": list(self.publication_types),
            "categories": list(self.categories),
            "dense_score": None if self.dense_score is None else round(self.dense_score, 6),
            "sparse_score": None if self.sparse_score is None else round(self.sparse_score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
            "sources": self.sources,
            "citation": self.citation,
        }


@dataclass
class RetrievalResult:
    query: str
    passages: list[RetrievedPassage]
    mode: str
    timings: dict[str, float] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "timings_ms": {k: round(v * 1000, 2) for k, v in self.timings.items()},
            "stats": self.stats,
            "passages": [
                {k: v for k, v in p.to_dict().items() if include_text or k != "text"}
                for p in self.passages
            ],
        }


class MedRAGRetriever:
    """Hybrid retriever over a chunk store."""

    def __init__(
        self,
        *,
        bm25: BM25Index | None = None,
        dense: DenseIndex | None = None,
        store: ChunkStore | None = None,
        embedder: DenseEmbedder | None = None,
        reranker: CrossEncoderReranker | None = None,
        graph: Any = None,
        dependency: Any = None,
        page: Any = None,
        cfg: PipelineConfig | None = None,
    ) -> None:
        self.cfg = cfg or PipelineConfig.load()
        self.bm25 = bm25
        self.dense = dense
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        # Optional document-level vectors. Used by the `page`/`pagectx` arms.
        self.page = page
        self.dense_embedder = embedder
        # Optional third signal: a GraphRetriever over the MeSH co-occurrence graph.
        self.graph = graph
        # Optional structural signal: the section-dependency graph over chunks.
        self.dependency = dependency
        if self.reranker is None and self.cfg.rerank.enabled:
            self.reranker = CrossEncoderReranker(
                self.cfg.rerank.model_name,
                device=self.cfg.rerank.device,
                fp16=self.cfg.rerank.fp16,
                batch_size=self.cfg.rerank.batch_size,
                max_length=self.cfg.rerank.max_seq_length,
            )

    # ------------------------------------------------------------------ access
    @property
    def is_ready(self) -> bool:
        return self.store is not None and (self.bm25 is not None or self.dense is not None)

    def n_chunks(self) -> int:
        return len(self.store) if self.store else 0

    def _require(self, mode: str) -> None:
        if self.store is None:
            raise RuntimeError("retriever has no chunk store; call build_index() or load_index() first")
        if mode in (MODE_SPARSE, MODE_HYBRID, MODE_RERANKED, MODE_HYBRID3, MODE_HYBRID3_RERANKED) and self.bm25 is None:
            raise RuntimeError(f"mode={mode} requires a BM25 index")
        if mode in (MODE_DENSE, MODE_HYBRID, MODE_RERANKED, MODE_HYBRID3, MODE_HYBRID3_RERANKED) and self.dense is None:
            raise RuntimeError(f"mode={mode} requires a dense index")
        if mode in GRAPH_MODES and self.graph is None:
            raise RuntimeError(
                f"mode={mode} requires the MeSH graph index; run `medrag mesh-graph` first"
            )
        if mode in DEPENDENCY_MODES and self.dependency is None:
            raise RuntimeError(
                f"mode={mode} requires the section-dependency graph; "
                f"run `medrag dependency-graph` first"
            )

    # --------------------------------------------------------------- retrieval
    def _sparse_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        assert self.bm25 is not None
        return self.bm25.search(query, top_k=top_k)

    def _apply_bleed(self, hits: Sequence[tuple[int, float]]) -> list[tuple[int, float]]:
        """Re-score by the strongest sibling in the same document.

        Distinct from contextual retrieval (Anthropic 2024), which rewrites chunk
        text with document context and so requires re-embedding the corpus. This
        runs against the existing index at query time, which is the entire
        practical argument for it.
        """
        from .sibling_bleed import apply_sibling_bleed

        assert self.store is not None
        return apply_sibling_bleed(
            hits, self.store, alpha=self.cfg.fusion.bleed_alpha, mode=self.cfg.fusion.bleed_mode
        )

    def _apply_page(
        self, hits: Sequence[tuple[int, float]], queries: Sequence[str]
    ) -> list[tuple[int, float]]:
        """Re-score chunk hits by how well their *document* matches the query.

        Needs the page index and the dense encoder. Both are optional: without them
        the hits come back unchanged, so the arm degrades to ``sparse`` rather than
        raising. That matters because a missing page index is a deployment state, not
        a programming error, and silently failing would make ``page`` and
        ``sparse`` indistinguishable in a results table.
        """
        from .page_scoring import PageConfig, apply_page_scores, normalise

        if self.page is None or self.page.matrix is None or self.dense_embedder is None:
            log.warning("page scoring unavailable (page index or dense encoder missing); "
                        "returning the base ranking unchanged")
            return list(hits)
        import numpy as np

        doc_of_hit: dict[int, str | None] = {}
        doc_ids: list[str | None] = []
        for row, _ in hits:
            chunk = self.store.get(row) if self.store is not None else None  # type: ignore[union-attr]
            did = chunk.doc_id if chunk is not None else None
            doc_of_hit[row] = did
            doc_ids.append(did)
        # Position within the page MATRIX, which is not the chunk-store row id.
        #
        # These are different numbers and using one for the other is invisible until
        # it isn't: PageIndex.row_ids holds chunk-store row ids, so row 0 of the page
        # matrix and row 0 of the chunk store are different documents entirely. The
        # first version of this built the lookup from row_ids and indexed the matrix
        # with it, which happened to work while every candidate came from an early
        # chunk and then died with IndexError on row 7,843,341 of a 554,069-row
        # matrix. Had the page index been smaller rather than larger, it would have
        # kept running and scored most documents against the wrong page vector --
        # plausible cosines attached to the wrong papers.
        pos_of = self.page.position_of()
        known = [d for d in dict.fromkeys(d for d in doc_ids if d) if d in pos_of]
        if not known:
            return list(hits)
        qv = self.dense_embedder.embed(list(queries), is_query=True)
        qv = np.asarray(qv, dtype=np.float32).reshape(len(queries), -1)
        qn = qv / np.maximum(np.linalg.norm(qv, axis=1, keepdims=True), 1e-12)
        mat = np.asarray(self.page.matrix)
        sims = (qn @ mat[[pos_of[d] for d in known]].T).max(axis=0)
        scores = normalise([float(s) for s in sims])
        page_scores = dict(zip(known, scores))
        return apply_page_scores(hits, page_scores, doc_of_hit.get,
                                 PageConfig(alpha=self.cfg.fusion.page_alpha))

    def _apply_context(
        self, hits: Sequence[tuple[int, float]], top_k: int
    ) -> list[tuple[int, float]]:
        """Append the interpreting passages of the top-ranked seeds.

        ``context_budget`` is passed explicitly. Without it ``expand_with_context``
        defaults the budget to ``top_k``, which silently caps expansion at ten
        passages no matter what ``context_seeds`` and ``context_per_seed`` say.

        That cap is invisible at the shipped default, where
        ``5 * 2 == top_k == 10``, so the numbers look right and the knob does nothing.
        It appears as soon as the budget changes: at 10 x 12, ``retrieve`` computes
        ``final_k = top_k + 120 = 130`` for its output window while the expansion
        could only ever supply 20. Falsifier 15b was run through this path and
        therefore compared two ten-passage arms while believing it compared two
        budgets.
        """
        from .sibling_bleed import expand_with_context

        assert self.store is not None
        return expand_with_context(
            hits, self.store, self.dependency,
            top_k=top_k,
            seed_count=self.cfg.fusion.context_seeds,
            context_per_seed=self.cfg.fusion.context_per_seed,
            context_budget=self.cfg.fusion.context_seeds
            * self.cfg.fusion.context_per_seed,
        )

    def _graph_search(self, queries: Sequence[str], top_k: int) -> list[tuple[int, float]]:
        """MeSH-graph retrieval, projected from documents onto chunk row ids.

        The graph scores documents (MeSH indexing is assigned per article), so
        each ranked document donates a few of its chunks carrying the same score.
        That makes the graph a document prior in the fusion, which is exactly
        the right granularity to complement a passage-level signal.
        """
        assert self.graph is not None and self.store is not None
        from .graph_retriever import doc_to_chunks

        merged: list[tuple[int, float]] = []
        seen: dict[int, float] = {}
        for q in queries:
            ranked_docs = self.graph.search(q, top_k=max(top_k // 3, 20))
            for row_id, score in doc_to_chunks(self.store.rows, ranked_docs, per_doc=3):
                rid = int(row_id)
                if rid not in seen or seen[rid] < score:
                    seen[rid] = score
        merged = sorted(((r, s) for r, s in seen.items()), key=lambda kv: -kv[1])
        return merged[:top_k]

    def _dense_search(self, queries: Sequence[str], top_k: int) -> list[tuple[int, float]]:
        """Dense retrieval returning ``(row_id, score)``, matching the BM25 path."""
        assert self.dense is not None
        if self.embedder is None:
            self.embedder = DenseEmbedder(self.cfg.embed).load()
        vectors = self.embedder.embed(list(queries), is_query=True)
        if len(queries) == 1:
            return self.dense.search_one_rows(vectors[0], top_k)
        return self.dense.search_rows(vectors, top_k)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        mode: str = MODE_RERANKED,
        sub_queries: Sequence[str] | None = None,
        include_text: bool = True,
    ) -> RetrievalResult:
        """Retrieve the most relevant passages for ``query``.

        ``sub_queries`` fans out retrieval for multi-hop questions; results are
        merged by max score so a passage found by any hop survives fusion.
        """
        if mode not in ALL_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {ALL_MODES}")
        self._require(mode)

        timings: dict[str, float] = {}
        queries = [query] + [q for q in (sub_queries or []) if q and q != query]

        # ---- sparse
        sparse: list[tuple[int, float]] = []
        if mode in (
            MODE_SPARSE, MODE_HYBRID, MODE_RERANKED, MODE_HYBRID3, MODE_HYBRID3_RERANKED,
            MODE_BLEED, MODE_CONTEXT, MODE_BLEED_CONTEXT, MODE_PAGE, MODE_PAGE_CONTEXT,
        ):
            t0 = time.perf_counter()
            for q in queries:
                sparse.extend(self._sparse_search(q, self.cfg.fusion.sparse_top_k))
            timings["sparse"] = time.perf_counter() - t0

        # ---- dense
        dense_hits: list[tuple[int, float]] = []
        if mode in (MODE_DENSE, MODE_HYBRID, MODE_RERANKED, MODE_HYBRID3, MODE_HYBRID3_RERANKED):
            t0 = time.perf_counter()
            dense_hits = self._dense_search(queries, self.cfg.fusion.dense_top_k)
            timings["dense"] = time.perf_counter() - t0

        # ---- graph (third, expert-curated signal)
        graph_hits: list[tuple[int, float]] = []
        if mode in GRAPH_MODES:
            t0 = time.perf_counter()
            graph_hits = self._graph_search(queries, self.cfg.fusion.dense_top_k)
            timings["graph"] = time.perf_counter() - t0

        # ---- fusion
        t0 = time.perf_counter()
        if mode == MODE_SPARSE:
            ranked_ids = [row for row, _s in sparse]
            by_row = {row: Candidate(chunk_id=str(row), sparse_score=s, sparse_rank=i + 1,
                                     sources=[MODE_SPARSE], score=s)
                      for i, (row, s) in enumerate(sparse)}
            candidates = [by_row[r] for r in ranked_ids]
        elif mode == MODE_DENSE:
            by_row = {row: Candidate(chunk_id=str(row), dense_score=s, dense_rank=i + 1,
                                     sources=[MODE_DENSE], score=s)
                      for i, (row, s) in enumerate(dense_hits)}
            candidates = [by_row[row] for row, _s in dense_hits]
        elif mode == MODE_GRAPH:
            by_row = {row: Candidate(chunk_id=str(row), score=s,
                                     sources=[MODE_GRAPH]) for row, s in graph_hits}
            candidates = [by_row[row] for row, _s in graph_hits]
        elif mode in PAGE_MODES:
            # Page-level arms. Same starting point as the bleed/context arms -- the
            # sparse ranking -- so the ablation isolates the document-conditioned
            # signal rather than confounding it with a different candidate set.
            base = sparse[: max(top_k * 4, 40)]
            t_page = time.perf_counter()
            ranked = self._apply_page(base, queries)
            timings["page"] = time.perf_counter() - t_page
            if mode in DEPENDENCY_MODES:
                t_ctx = time.perf_counter()
                ranked = self._apply_context(ranked, top_k=top_k)
                timings["context"] = time.perf_counter() - t_ctx
            candidates = [
                Candidate(chunk_id=str(row), sparse_score=s, score=s, sources=[mode])
                for row, s in ranked
            ]
        elif mode in (MODE_BLEED, MODE_CONTEXT, MODE_BLEED_CONTEXT):
            # Contextual-contamination arms. Each starts from the same sparse
            # ranking and then applies one or both post-hoc mechanisms, so the
            # ablation against `sparse` isolates the mechanism rather than
            # confounding it with a different candidate generator.
            base = sparse[: max(top_k * 4, 40)]
            t_bleed = time.perf_counter()
            ranked = self._apply_bleed(base) if mode in BLEED_MODES else list(base)
            timings["bleed"] = time.perf_counter() - t_bleed
            if mode in DEPENDENCY_MODES:
                t_ctx = time.perf_counter()
                ranked = self._apply_context(ranked, top_k=top_k)
                timings["context"] = time.perf_counter() - t_ctx
            sources = [mode]
            candidates = [
                Candidate(chunk_id=str(row), sparse_score=s, score=s, sources=list(sources))
                for row, s in ranked
            ]
        elif mode in (MODE_HYBRID3, MODE_HYBRID3_RERANKED):
            from .graph_retriever import fuse_three

            candidates = fuse_three(
                [(str(r), s) for r, s in sparse],
                [(str(r), s) for r, s in dense_hits],
                [(str(r), s) for r, s in graph_hits],
                k=self.cfg.fusion.rrf_k,
                top_k=self.cfg.rerank.depth if mode == MODE_HYBRID3_RERANKED else top_k * 5,
            )
        else:
            candidates = fuse(
                [(str(r), s) for r, s in dense_hits],
                [(str(r), s) for r, s in sparse],
                method=self.cfg.fusion.method,
                top_k=self.cfg.rerank.depth if mode == MODE_RERANKED else top_k * 5,
                rrf_k=self.cfg.fusion.rrf_k,
                dense_weight=self.cfg.fusion.dense_weight,
                sparse_weight=self.cfg.fusion.sparse_weight,
            )
        timings["fusion"] = time.perf_counter() - t0

        # ---- rerank
        if mode in (MODE_RERANKED, MODE_HYBRID3_RERANKED) and self.reranker is not None:
            t0 = time.perf_counter()
            head = candidates[: self.cfg.rerank.depth]
            row_ids = [int(c.chunk_id) for c in head]
            texts = self.store.texts_bulk(row_ids) if self.store else {}
            # Rerank a deeper window than top_k so that dropping duplicate
            # passages during materialisation still leaves top_k results.
            candidates = self.reranker.rerank(query, head, texts, top_k=max(top_k * 3, self.cfg.rerank.top_k))
            timings["rerank"] = time.perf_counter() - t0

        # Context modes deliberately return MORE than top_k: the framing passages
        # come after the results and are the whole point.
        final_k = top_k + self.cfg.fusion.context_seeds * self.cfg.fusion.context_per_seed \
            if mode in DEPENDENCY_MODES else top_k
        # The window must be at least final_k. It used to be sliced to top_k when
        # include_text was False, which is the cheap path every offline driver and
        # metric script takes -- so the context passages were cut before anyone saw
        # them and `context` returned byte-identical rankings to `sparse` on 400 of
        # 400 queries. The same slicing was only avoided on the include_text=True
        # path, so the bug hid exactly where the arms were being measured.
        #
        # _materialise also drops duplicate and missing chunks, so the window is
        # wider still when text is wanted, to avoid returning fewer than asked for.
        window = candidates[: max(top_k * 3, final_k)] if include_text \
            else candidates[:final_k]
        passages = self._materialise(window, include_text=include_text)[:final_k]
        return RetrievalResult(
            query=query,
            passages=passages,
            mode=mode,
            timings=timings,
            stats={
                "n_queries": len(queries),
                "n_candidates": len(candidates),
                "n_results": len([p for p in passages[:top_k]]),
                "n_context": len(passages) - min(top_k, len(passages)),
                **candidate_stats(candidates),
            },
        )

    def _materialise(self, candidates: Sequence[Candidate], *, include_text: bool = True) -> list[RetrievedPassage]:
        """Resolve row ids into full passages with provenance.

        Overlapping chunks mean the same sentence can appear as two distinct
        chunk ids, so results are de-duplicated on normalised text. Without
        this, a single well-matched passage occupies several of the top-k slots
        and crowds out independent evidence.
        """
        assert self.store is not None
        row_ids = [int(c.chunk_id) for c in candidates]
        texts = self.store.texts_bulk(row_ids) if include_text else [""] * len(row_ids)
        out: list[RetrievedPassage] = []
        seen_text: set[str] = set()
        seen_docs: set[str] = set()
        for cand, row_id, text in zip(candidates, row_ids, texts):
            chunk = self.store.get(row_id)
            if chunk is None:
                continue
            if include_text:
                fingerprint = " ".join(text.split())[:240].lower()
                if fingerprint in seen_text:
                    log.debug("dropping duplicate passage %s", chunk.chunk_id)
                    continue
                seen_text.add(fingerprint)
            seen_docs.add(chunk.doc_id)
            out.append(
                RetrievedPassage(
                    chunk_id=chunk.chunk_id,
                    row_id=row_id,
                    text=text,
                    score=cand.score,
                    title=chunk.title,
                    section_title=chunk.section_title,
                    section_kind=chunk.section_kind,
                    journal=chunk.journal,
                    year=chunk.year,
                    pmid=chunk.pmid,
                    pmcid=chunk.pmcid,
                    doi=chunk.doi,
                    is_abstract=chunk.is_abstract,
                    publication_types=chunk.publication_types,
                    categories=chunk.categories,
                    dense_score=cand.dense_score,
                    sparse_score=cand.sparse_score,
                    rerank_score=cand.rerank_score,
                    dense_rank=cand.dense_rank,
                    sparse_rank=cand.sparse_rank,
                    sources=list(cand.sources),
                )
            )
        return out

    # --------------------------------------------------------------- lifecycle
    def warmup(self) -> "MedRAGRetriever":
        """Load model weights and run a throwaway query to trigger lazy inits.

        Model load is ~4-8s and lazy CUDA/FAISS init adds more. Doing it up front
        keeps first-request latency in line with steady state, which matters for
        latency measurements in the paper.
        """
        if self.embedder is not None:
            self.embedder.load()
        if self.reranker is not None:
            self.reranker.load()
        if self.embedder is None and self.dense is not None:
            # Query embedding needs a bi-encoder even in sparse-only mode.
            self.embedder = DenseEmbedder(self.cfg.embed).load()
        if self.is_ready:
            try:
                self.retrieve("warmup query", top_k=1, mode=MODE_HYBRID, include_text=False)
            except Exception as exc:  # noqa: BLE001
                log.warning("retriever warmup query failed: %s", exc)
        log.info("retriever warmed up (%d chunks)", self.n_chunks())
        return self

    def stats(self) -> dict[str, Any]:
        return {
            "n_chunks": self.n_chunks(),
            "bm25": self.bm25.stats() if self.bm25 else None,
            "dense": self.dense.stats() if self.dense else None,
            "graph": self.graph.graph.stats() if self.graph else None,
            "dependency": self.dependency.stats() if self.dependency else None,
            "store": self.store.stats() if self.store else None,
            "reranker": self.reranker.stats() if self.reranker else None,
            "cfg": {
                "fusion": {"method": self.cfg.fusion.method, "rrf_k": self.cfg.fusion.rrf_k},
                "rerank": {"enabled": self.cfg.rerank.enabled, "model": self.cfg.rerank.model_name},
                "embed": {"model": self.cfg.embed.model_name},
            },
        }


# --------------------------------------------------------------------- loader
def load_index(cfg: PipelineConfig, index_dir: Path | None = None, *,
               with_reranker: bool | None = None, with_graph: bool | None = None,
               with_dependency: bool | None = None,
               with_dense: bool | None = None) -> MedRAGRetriever:
    """Load a previously built index from disk.

    ``with_dense=False`` skips the dense index entirely. Worth having at this corpus
    size: loading 16,133,551 x 384 from NFS takes 2,168 s, and the sparse arm that
    every context-provision ablation is measured against does not need it.
    """
    index_dir = Path(index_dir) if index_dir else cfg.paths.index_dir
    if not index_dir.exists():
        raise FileNotFoundError(f"index directory not found: {index_dir} (run `medrag build-index` first)")

    store = ChunkStore.load(index_dir / "store")
    bm25 = None
    bm25_path = index_dir / "bm25.pkl"
    if bm25_path.exists():
        bm25 = BM25Index.load(bm25_path)

    page = None
    page_path = index_dir / "page_index.pkl"
    if page_path.exists():
        try:
            from .page_index import PageIndex

            page = PageIndex.load(page_path)
        except Exception as exc:  # noqa: BLE001
            # A page index that will not load must not take the whole retriever with
            # it: the `page` arms degrade to `sparse` and say so, and every other
            # mode keeps working.
            log.warning("page index present but failed to load (%s); page arms will "
                        "degrade to the base ranking", exc)

    dense = None
    dense_path = index_dir / "dense.faiss"
    sharded_manifest = index_dir / (dense_path.name + ".manifest.json")
    has_dense = dense_path.exists() or sharded_manifest.exists()
    if has_dense and with_dense is not False:
        dense = DenseIndex.load(dense_path)

    # Guard against a partially rebuilt index: all three components address the
    # same row ids, so any count mismatch means scores would be attributed to
    # the wrong passages. Fail loudly rather than returning plausible nonsense.
    n = len(store)
    if bm25 is not None and bm25.n_docs != n:
        raise ValueError(
            f"BM25 index covers {bm25.n_docs} rows but the chunk store has {n}. "
            f"Rebuild with `medrag build-index --force`."
        )
    if dense is not None and len(dense.row_ids) != n:
        raise ValueError(
            f"dense index covers {len(dense.row_ids)} rows but the chunk store has {n}. "
            f"Rebuild with `medrag build-index --force`."
        )

    use_reranker = cfg.rerank.enabled if with_reranker is None else with_reranker
    retriever = MedRAGRetriever(
        bm25=bm25, dense=dense, store=store, page=page, cfg=cfg,
        reranker=CrossEncoderReranker(cfg.rerank.model_name, device=cfg.rerank.device,
                                     fp16=cfg.rerank.fp16, batch_size=cfg.rerank.batch_size,
                                     max_length=cfg.rerank.max_seq_length) if use_reranker else None,
    )
    # Preload the query embedder so the first request is not seconds slower. The
    # page arms need it too -- they embed the query to score documents -- so it is
    # loaded whenever either the dense index or the page index is present.
    if dense is not None or page is not None:
        retriever.embedder = DenseEmbedder(cfg.embed).load()
        retriever.dense_embedder = retriever.embedder

    # Optional third signal. Absent is normal: graph modes then raise a clear
    # error rather than silently degrading to two-way fusion.
    if with_graph is None:
        with_graph = (index_dir / "mesh_graph.pkl").exists()
    if with_graph:
        try:
            from .graph_retriever import GraphRetriever, load_graph_index

            graph, doc_map = load_graph_index(index_dir)
            retriever.graph = GraphRetriever(graph, doc_map)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load the MeSH graph index: %s", exc)
            retriever.graph = None

    # Optional structural signal. Absent is normal; the context modes then raise a
    # clear error rather than silently degrading to a bare ranking.
    if with_dependency is None:
        with_dependency = (index_dir / "dependency_graph.pkl").exists()
    if with_dependency:
        try:
            from .dependency_graph import DependencyGraph

            retriever.dependency = DependencyGraph.load(index_dir / "dependency_graph.pkl")
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load the dependency graph: %s", exc)
            retriever.dependency = None

    log.info("loaded index from %s: %d chunks (bm25=%s dense=%s graph=%s dep=%s rerank=%s)",
             index_dir, retriever.n_chunks(), bm25 is not None, dense is not None,
             retriever.graph is not None, retriever.dependency is not None,
             retriever.reranker is not None)
    return retriever
