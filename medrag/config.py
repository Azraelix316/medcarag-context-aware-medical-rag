"""Configuration objects for the MedRAG pipeline.

Every stage (acquire, chunk, embed, index, serve, eval) reads a single frozen
``PipelineConfig`` so that a run is fully described by one YAML file plus the
corpus manifest it points at.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data"


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class PathsConfig:
    """Filesystem layout. All paths are resolved relative to ``data_root``."""

    data_root: Path = DEFAULT_DATA_ROOT
    corpus_dir: Path = Path("corpus")
    index_dir: Path = Path("index")
    cache_dir: Path = Path("cache")
    runs_dir: Path = Path("runs")
    log_dir: Path = Path("../logs")

    def resolve(self) -> "PathsConfig":
        root = Path(self.data_root).expanduser().resolve()
        for f in fields(self):
            if f.name == "data_root":
                continue
            val = Path(getattr(self, f.name))
            setattr(self, f.name, val if val.is_absolute() else (root / val))
        self.data_root = root
        return self

    def ensure(self) -> "PathsConfig":
        for f in fields(self):
            Path(getattr(self, f.name)).mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class AcquireConfig:
    """NCBI E-utilities acquisition settings for the medical corpus."""

    # "pubmed" -> bibliographic records (title/abstract/MeSH); "pmc" -> full text.
    source: str = "pubmed"
    # Free-text query. Empty means the default evidence slices; see `sweep`.
    query: str = ""
    # Date-partitioned sweep. "yearly" turns [min_date, max_date] into one query per
    # year, so the sweep has slices that can be sharded across nodes.
    #
    # This is what makes the 20M target reachable. Without it the crawl is bounded by
    # len(DEFAULT_QUERIES) * max_records: six publication-type slices at 200,000 each,
    # so 1.2M raw records no matter how long it runs. Measured on this corpus -- 6
    # slices, cap 200,000 per slice, ceiling ~1.2M, and it reached 1,022,707 and
    # stopped. PubMed holds ~37M records; the six slices cover the publication types
    # with good evidence, which is the right benchmark and a tiny sample of the corpus.
    #
    # Partitioning by date is also the only way around NCBI's own limit: esearch
    # refuses to page past ~10,000,000 records for a single query, so one big sweep
    # query is not an option.
    sweep: str = ""
    # Granularity of the partition: "yearly" or "monthly".
    sweep_granularity: str = "yearly"
    # MeSH/field-restricted filter applied server side, e.g.
    # 'AND ("randomized controlled trial"[pt] OR systematic review[pt])'.
    filter_query: str = ""
    # Language filter; empty string means all languages.
    lang: str = "english"
    # datetype=pdat&mindate=.../maxdate=... bounds the sweep by publication date.
    min_date: str = "1950/01/01"
    max_date: str = "2026/12/31"
    # Records to fetch. The full PubMed corpus is ~37M abstracts; 0 means "unbounded".
    max_records: int = 200_000
    # efetch batch size (NCBI caps at 10k POST / 200 GET per call).
    fetch_batch: int = 200
    # Politeness: NCBI asks for <=3 req/s without an API key, <=10 with one.
    max_requests_per_second: float = 3.0
    api_key: str | None = None
    email: str | None = None
    tool: str = "medrag"
    # Retries with exponential backoff on 429/5xx.
    max_retries: int = 5
    request_timeout: int = 120
    # gzip-compressed JSONL shards written to corpus_dir.
    shard_size: int = 10_000
    # Skip these; they dominate the corpus without adding medical reasoning value.
    exclude_publication_types: tuple[str, ...] = (
        "Comment",
        "Editorial",
        "News",
        "Retracted Publication",
    )


@dataclass
class ChunkConfig:
    """Structure-aware chunking of corpus documents."""

    # "section" splits on document structure then packs to a token budget;
    # "passage" is a flat sliding window; "abstract" keeps one chunk per doc.
    strategy: str = "section"
    target_tokens: int = 320
    max_tokens: int = 512
    overlap_tokens: int = 64
    # Never emit a chunk below this many tokens; it merges into the previous one.
    min_tokens: int = 48
    # Prepend these fields to each chunk's embedding text, in order.
    context_fields: tuple[str, ...] = ("title", "section", "journal", "year")
    # Drop documents with no abstract when chunking abstracts-only records.
    require_abstract: bool = True


@dataclass
class EmbedConfig:
    """Dense bi-encoder embedding settings."""

    model_name: str = "BAAI/bge-small-en-v1.5"
    # bge-* expects this instruction prefix on the query side only.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    pooling: str = "cls"
    normalize: bool = True
    batch_size: int = 64
    # Rows embedded per encode() call. Sentence-transformers tokenises the whole input
    # list before running any batch, so a single call over 554,069 texts at
    # max_seq_length=512 materialises every token sequence before one embedding
    # exists. Inputs larger than this are embedded in blocks. 20,000 rows is
    # comfortably inside memory on a 16 GB card and measured identical output.
    block_size: int = 20_000
    max_seq_length: int = 512
    device: str = "auto"
    fp16: bool = True
    # Embedding dimension; verified against the model at load time.
    dim: int = 384


@dataclass
class SparseConfig:
    """Lexical (BM25) retrieval settings."""

    k1: float = 1.2
    b: float = 0.75
    # Persist the inverted index so startup does not re-tokenize the corpus.
    persist: bool = True
    # Stopword handling is delegated to the tokenizer; keep this list minimal.
    min_token_length: int = 2


@dataclass
class RerankConfig:
    """Cross-encoder reranking of first-stage candidates."""

    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-base"
    # Candidates handed to the cross-encoder after first-stage fusion.
    depth: int = 100
    top_k: int = 10
    batch_size: int = 32
    max_seq_length: int = 512
    device: str = "auto"
    fp16: bool = True


@dataclass
class FusionConfig:
    """Late-interaction fusion of the dense and sparse candidate lists."""

    # Reciprocal Rank Fusion is the default: scale-free and robust to
    # mis-calibrated score distributions.
    method: str = "rrf"
    rrf_k: int = 60
    # Candidate depth per retriever before fusion.
    dense_top_k: int = 100
    sparse_top_k: int = 100
    # Learned linear blend, only used when method == "weighted".
    dense_weight: float = 0.6
    sparse_weight: float = 0.4

    # --- contextual-contamination components (see medrag/index/sibling_bleed.py)
    #
    # Sibling bleed mixes a document-conditioned score into a chunk's own score:
    #     s'(c) = (1 - alpha) * s(c) + alpha * max_{c' in same doc} s(c')
    # alpha=0 recovers the base retriever exactly, so a sweep isolates the
    # mechanism. "max" rather than "sum" so a document is not rewarded merely for
    # having many chunks, which is a property of the chunker and not of relevance.
    bleed_alpha: float = 0.3
    bleed_mode: str = "max"
    # Page-level conditioning: s'(c) = s(c) * (1 + alpha * page(doc(c))). 0.0
    # reproduces the base ranking exactly, so the ablation measures the mechanism
    # rather than a different candidate set. Additive rather than multiplicative so
    # a document can promote a passage but never resurrect one whose own text does
    # not match.
    page_alpha: float = 0.5
    # Context expansion: how many top hits act as seeds whose dependency
    # neighbours supply the interpreting passages, and how many per seed.
    context_seeds: int = 5
    context_per_seed: int = 2


@dataclass
class QueryConfig:
    """Query rewriting and decomposition for multi-hop questions."""

    # "none", "llm", or "heuristic".
    rewrite: str = "heuristic"
    # "none" | "decompose" -> fan out sub-queries and merge results.
    mode: str = "single"
    max_subqueries: int = 4
    # Runs of agentic retrieval when the first answer looks unsupported.
    max_hops: int = 2


@dataclass
class ServeConfig:
    """FastAPI retrieval service."""

    host: str = "127.0.0.1"
    port: int = 8080
    # Prefetch model weights into memory before serving.
    warmup: bool = True
    # Cross-origin policy for local agent clients.
    cors_origins: tuple[str, ...] = ("*",)


@dataclass
class EvalConfig:
    """Retrieval-quality evaluation."""

    # k values for recall@k / nDCG@k / MRR.
    metrics_at_k: tuple[int, ...] = (1, 3, 5, 10, 20, 100)
    sample_size: int = 1000
    seed: int = 13


@dataclass
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    acquire: AcquireConfig = field(default_factory=AcquireConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    sparse: SparseConfig = field(default_factory=SparseConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path | None = None) -> "PipelineConfig":
        """Load YAML config; later files deep-merge over earlier ones."""
        cfg = cls()
        if path is None:
            return cfg._finalize()
        paths = [Path(path)] if not isinstance(path, (list, tuple)) else [Path(p) for p in path]
        merged: dict[str, Any] = {}
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"config not found: {p}")
            with p.open("r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"config root must be a mapping: {p}")
            merged = _deep_update(merged, loaded)
        return cfg.from_dict(merged)._finalize()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        kwargs: dict[str, Any] = {}
        section_types = {f.name: f.type for f in fields(cls)}
        for key, value in data.items():
            if key not in section_types:
                raise ValueError(f"unknown config section: {key!r} (expected one of {sorted(section_types)})")
            sub = getattr(cls(), key)
            sub_data = asdict(sub) if is_dataclass(sub) else {}
            sub_data = _deep_update(sub_data, value or {})
            kwargs[key] = type(sub)(**{k: v for k, v in sub_data.items() if k in {f.name for f in fields(sub)}})
        return cls(**kwargs)

    def _finalize(self) -> "PipelineConfig":
        self.paths.resolve()
        self.acquire.api_key = self.acquire.api_key or os.environ.get("NCBI_API_KEY")
        self.acquire.email = self.acquire.email or os.environ.get("NCBI_EMAIL")
        return self

    def to_dict(self) -> dict[str, Any]:
        def enc(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return list(obj)
            raise TypeError(type(obj))

        return asdict(self)  # type: ignore[arg-type]

    def dump(self, path: str | Path) -> Path:
        import json

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=enc), encoding="utf-8")
        return p
