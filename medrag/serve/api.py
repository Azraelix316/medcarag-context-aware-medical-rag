"""FastAPI retrieval service.

Exposes the retriever and the research agent over HTTP so external agents can
use the corpus without linking against the package. Endpoints:

``GET  /health``      liveness plus index statistics
``POST /retrieve``    hybrid retrieval, returns passages with citations
``POST /ask``         full agentic research loop
``POST /plan``        query planning only (cheap; no index needed)
``GET  /docs``        OpenAPI UI
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import PipelineConfig
from ..index.retriever import MedRAGRetriever, load_index

log = logging.getLogger(__name__)

Mode = Literal["sparse", "dense", "hybrid", "reranked"]


# ------------------------------------------------------------------- schemas
class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="the question or search string")
    top_k: int = Field(10, ge=1, le=100, description="number of passages to return")
    mode: Mode = "reranked"
    sub_queries: list[str] = Field(default_factory=list, description="explicit fan-out queries")
    include_text: bool = True


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(8, ge=1, le=50)
    max_hops: int = Field(2, ge=0, le=6)
    mode: Mode = "reranked"
    include_evidence: bool = True
    show_trace: bool = False
    #: Append limitations quoted out of the retrieved passages. Measured effect:
    #: conditional caveat rate 0.0556 -> 1.0000 with an unconditional rate of
    #: exactly 0.0000, so it cannot invent a limitation the evidence lacks.
    attribute_caveats: bool = False


class PlanRequest(BaseModel):
    question: str = Field(..., min_length=1)
    strategy: Literal["none", "heuristic", "llm"] = "heuristic"


# ------------------------------------------------------------------- service
class MedRAGService:
    """Holds the loaded index and agent; shared by all requests.

    The index is loaded once at startup and reused. Model weights are warmed up
    eagerly so the first request is not several seconds slower than the rest.
    """

    def __init__(self, cfg: PipelineConfig, index_dir: Any = None) -> None:
        self.cfg = cfg
        self.index_dir = index_dir
        self.retriever: MedRAGRetriever | None = None
        self.agent: Any = None
        self.started_at = time.time()

    def load(self, *, warmup: bool = True) -> None:
        self.retriever = load_index(self.cfg, self.index_dir)
        if warmup:
            try:
                self.retriever.warmup()
                # A throwaway query triggers lazy FAISS/torch init paths.
                self.retriever.retrieve("warmup query", top_k=1, mode="hybrid", include_text=False)
            except Exception as exc:  # noqa: BLE001
                log.warning("warmup incomplete: %s", exc)
        from ..agent.loop import ResearchAgent
        from ..agent.planner import QueryPlanner

        self.agent = ResearchAgent(self.retriever, planner=QueryPlanner(strategy=self.cfg.query.rewrite))
        log.info("service ready with %d chunks", self.retriever.n_chunks())

    def require_retriever(self) -> MedRAGRetriever:
        if self.retriever is None:
            raise HTTPException(status_code=503, detail="index not loaded")
        return self.retriever


def create_app(cfg: PipelineConfig, index_dir: Any = None, *, warmup: bool = True) -> FastAPI:
    """Build the FastAPI application."""
    service = MedRAGService(cfg, index_dir)

    app = FastAPI(
        title="MedRAG",
        description="Large-scale medical RAG retrieval and research agent.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.serve.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        service.load(warmup=warmup)

    @app.get("/health")
    def health() -> dict[str, Any]:
        retriever = service.retriever
        return {
            "status": "ok" if retriever and retriever.is_ready else "loading",
            "uptime_s": round(time.time() - service.started_at, 1),
            "n_chunks": retriever.n_chunks() if retriever else 0,
            "index": retriever.stats() if retriever else None,
        }

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest) -> dict[str, Any]:
        retriever = service.require_retriever()
        t0 = time.time()
        try:
            result = retriever.retrieve(
                req.query, top_k=req.top_k, mode=req.mode,
                sub_queries=req.sub_queries or None, include_text=req.include_text,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = result.to_dict(include_text=req.include_text)
        payload["wall_s"] = round(time.time() - t0, 3)
        return payload

    @app.post("/ask")
    def ask(req: AskRequest) -> dict[str, Any]:
        if service.agent is None:
            raise HTTPException(status_code=503, detail="agent not loaded")
        from ..agent.loop import ResearchAgent

        agent = ResearchAgent(
            service.require_retriever(),
            planner=service.agent.planner,
            max_hops=req.max_hops,
            top_k=req.top_k,
            mode=req.mode,
            attribute_caveats=req.attribute_caveats,
        )
        answer = agent.research(req.question)
        payload = answer.to_dict(include_evidence=req.include_evidence)
        if not req.show_trace:
            payload.pop("trace", None)
        return payload

    @app.post("/plan")
    def plan(req: PlanRequest) -> dict[str, Any]:
        from ..agent.planner import QueryPlanner

        return QueryPlanner(strategy=req.strategy).plan(req.question).to_dict()

    return app


def run_server(cfg: PipelineConfig, index_dir: Any = None, *, host: str | None = None,
               port: int | None = None, warmup: bool = True) -> None:
    """Blocking server entry point used by ``medrag serve``."""
    import uvicorn

    host = host or cfg.serve.host
    port = port or cfg.serve.port
    app = create_app(cfg, index_dir, warmup=warmup)
    log.info("starting MedRAG service on http://%s:%d/docs", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
