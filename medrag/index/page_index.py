"""Document-level ("page") vectors stored alongside chunk vectors.

What this is
------------
Every chunk gets its own embedding, and so does its whole parent document. At
query time a chunk's score is blended with its document's score, so a passage
from a document that matches the query *as a whole* is preferred over an isolated
passage that happens to share a few terms.

Prior art, cited not claimed
----------------------------
This is Anthropic's Contextual Retrieval (2024) and LangChain's Contextual
Embeddings: prepend document-level context to each chunk before embedding. The
variants differ in where the context enters:

- **Contextual retrieval** rewrites the chunk *text*, so the corpus must be
  re-embedded. Measured cost at our scale: ~64 hours for section-context
  re-embedding across 20M documents.
- **This module** embeds the document once and blends scores at query time. The
  corpus is not modified, the index grows by one row per document rather than one
  re-embedding per chunk, and the whole mechanism can be switched off at query
  time.

That makes it weaker in expressiveness -- it cannot tell a chunk *how* its
document frames it, only *whether* the document as a whole matches -- and much
cheaper. Both properties are why it is worth having as a separate ablation rather
than only implementing the stronger version.

Cost
----
One extra row per document. At 384 dims that is 1.5 KB/document: 1.4 GB for 930K
documents, ~30 GB for 20M. Cheap against 76M chunk rows.

The document vector is built from title, section headings and the first
(eBackground) chunk, because a document's *frame* is carried by its structure and
its opening, not by its full text.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _rss_mb() -> float:
    """Resident set size in MiB, or -1.0."""
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return -1.0

# Characters of document text used to build the page vector. Long enough to carry
# title + headings + opening, short enough that the embedding is cheap.
#: Cap on page text. 400 characters covers the framing fields and is the largest
#: value whose 20,000-row encode completes on this host; at 1,506 characters
#: the same call aborts. See _text_from_chunk.
DEFAULT_PAGE_CHARS = 400


def page_text(doc: Any, *, max_chars: int = DEFAULT_PAGE_CHARS) -> str:
    """The document-level text a page vector is built from.

    Section headings matter more than body text here: a document's framing is
    declared by its structure ("Methods", "Limitations"), which is exactly the
    signal a chunk embedding cannot see.
    """
    parts: list[str] = []
    title = (getattr(doc, "title", "") or "").strip()
    if title:
        parts.append(title)
    journal = (getattr(doc, "journal", "") or "").strip()
    year = getattr(doc, "year", 0) or 0
    if journal or year:
        parts.append(f"{journal} {year}".strip())
    headings: list[str] = []
    for sec in getattr(doc, "sections", ()) or ():
        h = (getattr(sec, "title", "") or "").strip()
        if h:
            headings.append(h)
    if headings:
        parts.append("Sections: " + "; ".join(headings))
    opening = ""
    for sec in getattr(doc, "sections", ()) or ():
        t = (getattr(sec, "text", "") or "").strip()
        if t:
            opening = t
            break
    if not opening:
        opening = (getattr(doc, "abstract", "") or "").strip()
    if opening:
        parts.append(opening)
    return " | ".join(parts)[:max_chars]


@dataclass
class PageIndex:
    """One dense vector per document, with row ids aligned to the chunk store.

    ``row_ids`` are chunk-store row ids: the first chunk of each document. Using
    the store's own ids rather than a separate numbering keeps the two indexes
    joinable without a lookup table, and means a page hit can be projected onto
    chunks the same way the MeSH graph is.
    """

    dim: int = 0
    matrix: Any = None
    row_ids: list[int] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    index_dir: Path | None = None
    #: row id -> doc id, built lazily by doc_of_row().
    _row_to_doc: dict[int, str] | None = None
    #: doc id -> page matrix row, built lazily by position_of().
    _pos_of: dict[str, int] | None = None

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        cfg: Any,
        rows: Sequence[str],
        *,
        kind_of: Any = None,
        doc_of: Any = None,
        first_row_of_doc: dict[str, int] | None = None,
        store: Any = None,
        max_docs: int | None = None,
    ) -> "PageIndex":
        """Embed one vector per document.

        Documents are enumerated from the chunk store rather than the corpus, so
        this runs against an existing index without a second corpus pass. The
        page text is read from the store's opening chunk plus the document's
        title, which the store denormalises onto every chunk.

        The opening chunk of each document comes from
        :meth:`ChunkStore.document_rows`, which reads the in-memory row map, so only
        554,069 blob rows are touched -- one per document -- instead of all
        16,133,551. That is the difference between a few minutes and 2.2 hours on a
        46.56 GiB blob over NFS.
        """
        from .embeddings import DenseEmbedder

        docs: list[tuple[str, int, str]] = []

        if store is not None and hasattr(store, "document_rows"):
            t0 = time.perf_counter()
            doc_ids, first_rows = store.document_rows()
            if max_docs:
                doc_ids, first_rows = doc_ids[:max_docs], first_rows[:max_docs]
            log.info("page index: %d documents located from the row map in %.1fs; "
                     "fetching opening chunks", len(doc_ids), time.perf_counter() - t0)
            t0 = time.perf_counter()
            opening = store.chunks_bulk(first_rows)
            log.info("page index: read %d opening chunks in %.1fs",
                     len(opening), time.perf_counter() - t0)
            for doc_id, row_id in zip(doc_ids, first_rows):
                chunk = opening.get(row_id)
                if chunk is None:
                    continue
                docs.append((doc_id, row_id, _text_from_chunk(chunk)))
        else:
            seen: set[str] = set()
            for row_id in range(len(rows)):
                chunk = store.get(row_id) if store is not None else None
                if chunk is None:
                    continue
                doc_id = chunk.doc_id
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                docs.append((doc_id, row_id, _text_from_chunk(chunk)))
                if max_docs and len(docs) >= max_docs:
                    break
                if row_id % 2_000_000 == 0 and row_id:
                    log.info("page index: scanned %d/%d rows, %d documents, rss=%.0f MiB",
                             row_id, len(rows), len(docs), _rss_mb())
        if not docs:
            return cls()

        embedder = DenseEmbedder(cfg.embed).load()
        texts = [t for _, _, t in docs]
        vecs = embedder.embed(texts, is_query=False, batch_size=cfg.embed.batch_size)
        matrix = np.asarray(vecs, dtype=np.float32)
        # Normalise so a document score is a cosine, directly comparable with the
        # chunk dense scores the retriever already produces.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        log.info("page index: %d documents x %d dims (%.1f MB)",
                 matrix.shape[0], matrix.shape[1], matrix.nbytes / 2**20)
        return cls(
            dim=int(matrix.shape[1]),
            matrix=matrix,
            row_ids=[r for _, r, _ in docs],
            doc_ids=[d for d, _, _ in docs],
        )

    # ----------------------------------------------------------------- search
    def search_rows(self, query_vec: np.ndarray, top_k: int = 100) -> list[tuple[int, float]]:
        """Documents most similar to ``query_vec``, as (first_row_id, score)."""
        if self.matrix is None or self.n_docs == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(q)
        if n:
            q = q / n
        k = min(top_k, self.n_docs)
        scores = self.matrix @ q
        pos = np.argpartition(-scores, k - 1)[:k]
        pos = pos[np.argsort(-scores[pos])]
        return [(self.row_ids[int(i)], float(scores[int(i)])) for i in pos]

    def doc_of_row(self, row_id: int) -> str | None:
        """The document a chunk row belongs to.

        Backed by a dict, not ``list.index``. The original linear search was
        O(n) over 554,069 entries per call, and page-level scoring asks this once
        per candidate -- so a single top-40 page-blended query would have spent
        about 22 million comparisons before returning anything.
        """
        if self._row_to_doc is None:
            self._row_to_doc = {r: d for r, d in zip(self.row_ids, self.doc_ids)}
        return self._row_to_doc.get(row_id)

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        import json
        import pickle

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(
                {
                    "dim": self.dim,
                    "matrix": None if self.matrix is None else self.matrix,
                    "row_ids": self.row_ids,
                    "doc_ids": self.doc_ids,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp.replace(p)
        log.info("saved page index -> %s (%.1f MB)", p, p.stat().st_size / 2**20)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "PageIndex":
        import pickle

        with open(Path(path), "rb") as fh:
            d = pickle.load(fh)
        matrix = d["matrix"]
        if d.get("matrix_file"):
            # Written by assemble_page_blocks: the vectors stay on disk, and the
            # retriever gathers only the rows of candidate documents.
            matrix = np.memmap(Path(path).parent / d["matrix_file"], dtype="<f4", mode="r",
                               shape=(len(d["doc_ids"]), int(d["dim"])))
        return cls(
            dim=int(d["dim"]), matrix=matrix,
            row_ids=list(d["row_ids"]), doc_ids=list(d["doc_ids"]),
        )

    def position_of(self) -> dict[str, int]:
        """doc id -> row of the page matrix, built once.

        The retriever used to rebuild this dict on every query, which is free at
        554,069 documents and about ten seconds per query at 20M.
        """
        if self._pos_of is None:
            self._pos_of = {d: i for i, d in enumerate(self.doc_ids)}
        return self._pos_of

    def stats(self) -> dict[str, Any]:
        return {
            "n_documents": self.n_docs,
            "dim": self.dim,
            "bytes": int(self.matrix.nbytes) if self.matrix is not None else 0,
            "bytes_per_doc": round(float(self.matrix.nbytes) / max(1, self.n_docs), 1)
            if self.matrix is not None
            else 0,
        }


def _text_from_chunk(chunk: Any) -> str:
    """Page text: the document's frame, not its prose.

    Title, venue, year, and the section kinds. Deliberately no body text, for two
    reasons and the second only became clear under measurement.

    The first is the one :func:`page_text` already argues: a document's framing is
    declared by its structure -- "Methods", "Limitations" -- which is exactly the
    signal a chunk embedding cannot see. The opening chunk's body is the opposite: it
    *is* chunk text, already represented by the chunk vector. Page-conditioning a
    passage with its own document's prose blurs the distinction the component exists
    to draw.

    The second is what this build measured. With the body included the text averages
    1,506 characters; framing only, it averages 396. At 1,506 characters an encode
    over 20,000 rows aborted in all three attempts, at batch 64 and batch 32, after
    backoff to batch 8 and six whole-block retries -- 10.24M tokens materialised before
    the first batch runs, which is what exhausts the allocator on a host whose NVML is
    broken. At 396 characters the same 20,000 rows complete in 80s, and at 251 rows/s
    the full 554,069-document build is 37 minutes rather than the 2.8 hours the longer
    text implies. Dropping the body is what made this component buildable; it just
    happens to be the right text anyway.
    """
    parts = []
    title = (getattr(chunk, "title", "") or "").strip()
    if title:
        parts.append(title)
    journal = (getattr(chunk, "journal", "") or "").strip()
    year = getattr(chunk, "year", 0) or 0
    if journal or year:
        parts.append(f"{journal} {year}".strip())
    kinds = getattr(chunk, "categories", ()) or ()
    if kinds:
        parts.append("; ".join(str(k) for k in kinds)[:300])
    text = " | ".join(parts)
    return text[:DEFAULT_PAGE_CHARS] if len(text) > DEFAULT_PAGE_CHARS else text


# ------------------------------------------------------------- blocked build
#
# PageIndex.build embeds every document in one process and holds the matrix in RAM:
# 20.3M documents x 384 dims is 31 GB, which no host here has free. The blocked
# build splits the same computation into three resumable steps:
#
#   1. write_page_text_blocks  (CPU, one sequential pass over the store blob)
#   2. embedding each block    (scripts/remote_embed.py --kind page, any GPU host)
#   3. assemble_page_blocks    (concatenate, normalise, write a memmap)
#
# The text, the embedder and the normalisation are PageIndex.build's own, so the
# result is the same page index, stored on disk instead of in a pickle.

PAGE_TEXTS_DIR = "page_texts"
PAGE_BLOCKS_DIR = "page_blocks"
PAGE_BLOCK_DOCS = 50_000


def page_text_block_path(index_dir: str | Path, b: int) -> Path:
    return Path(index_dir) / PAGE_TEXTS_DIR / f"texts-{b:05d}.jsonl.gz"


def page_vector_block_path(index_dir: str | Path, b: int) -> Path:
    return Path(index_dir) / PAGE_BLOCKS_DIR / f"page-block{b:05d}.npz"


def write_page_text_blocks(index_dir: str | Path, *, block_docs: int = PAGE_BLOCK_DOCS,
                           read_rows: int = 20_000) -> dict[str, Any]:
    """Write each document's page text, in first-appearance order, as gz JSONL blocks.

    Documents are found the way ChunkStore.document_rows finds them -- the chunk id
    up to its last '#' -- but from the records themselves while streaming the blob
    once, so neither the 1.25 GB row map nor a per-row seek is needed. The text is
    _text_from_chunk of the document's first chunk, as PageIndex.build uses.

    ``pages.json`` is written last; its presence means every block is complete.
    """
    import gzip
    import json

    from ..data.schema import Chunk

    index_dir = Path(index_dir)
    store_dir = index_dir / "store"
    out = index_dir / PAGE_TEXTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    n_rows = (store_dir / "offsets.bin").stat().st_size // 16
    seen: set[str] = set()
    buf: list[str] = []
    n_blocks = n_docs = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal n_blocks, buf
        final = page_text_block_path(index_dir, n_blocks)
        tmp = final.with_name(final.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1) as fh:
            fh.write("".join(buf))
        tmp.replace(final)
        n_blocks += 1
        buf = []

    with open(store_dir / "blob.jsonl", "rb") as blob:
        for a in range(0, n_rows, read_rows):
            b = min(n_rows, a + read_rows)
            offs = np.fromfile(store_dir / "offsets.bin", dtype="<u8",
                               count=2 * (b - a), offset=16 * a).reshape(-1, 2)
            start, end = int(offs[0, 0]), int(offs[-1, 0] + offs[-1, 1])
            blob.seek(start)
            raw = blob.read(end - start)
            for i, (off, length) in enumerate(offs):
                rec = json.loads(raw[off - start: off - start + length])
                key = rec.get("chunk_id") or ""
                doc_id = key.rsplit("#", 1)[0] if "#" in key else key
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                chunk = Chunk.from_dict({k: v for k, v in rec.items() if k != "row"})
                buf.append(json.dumps({"doc": doc_id, "row": a + i,
                                       "text": _text_from_chunk(chunk)}) + "\n")
                n_docs += 1
                if len(buf) >= block_docs:
                    flush()
            if (b // read_rows) % 100 == 0:
                log.info("page texts: %d/%d rows, %d documents, %d blocks (%.0fs)",
                         b, n_rows, n_docs, n_blocks, time.time() - t0)
    if buf:
        flush()
    manifest = {"n_docs": n_docs, "n_blocks": n_blocks, "block_docs": block_docs,
                "n_rows": n_rows,
                "text": "page_index._text_from_chunk of each document's first chunk"}
    (out / "pages.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("page texts: %d documents in %d blocks in %.0fs", n_docs, n_blocks, time.time() - t0)
    return manifest


def read_page_text_block(index_dir: str | Path, b: int) -> list[dict[str, Any]]:
    import gzip
    import json

    with gzip.open(page_text_block_path(index_dir, b), "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def assemble_page_blocks(index_dir: str | Path) -> Path:
    """Concatenate the embedded blocks into page_vectors.f32 and write page_index.pkl."""
    import json
    import pickle

    index_dir = Path(index_dir)
    manifest = json.loads((index_dir / PAGE_TEXTS_DIR / "pages.json").read_text(encoding="utf-8"))
    n, n_blocks = manifest["n_docs"], manifest["n_blocks"]
    missing = [b for b in range(n_blocks) if not page_vector_block_path(index_dir, b).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} page blocks missing, first {missing[:5]}")
    dim = int(np.load(page_vector_block_path(index_dir, 0))["embeddings"].shape[1])
    vec_path = index_dir / "page_vectors.f32"
    tmp_vec = vec_path.with_name(vec_path.name + ".tmp")
    mat = np.memmap(tmp_vec, dtype="<f4", mode="w+", shape=(n, dim))
    row_ids: list[int] = []
    doc_ids: list[str] = []
    pos = 0
    t0 = time.time()
    for b in range(n_blocks):
        recs = read_page_text_block(index_dir, b)
        v = np.asarray(np.load(page_vector_block_path(index_dir, b))["embeddings"],
                       dtype=np.float32)
        if v.shape != (len(recs), dim):
            raise ValueError(f"page block {b}: vectors {v.shape} for {len(recs)} documents")
        # PageIndex.build's normalisation, unchanged.
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat[pos:pos + len(recs)] = v / norms
        row_ids.extend(int(r["row"]) for r in recs)
        doc_ids.extend(r["doc"] for r in recs)
        pos += len(recs)
    if pos != n:
        raise ValueError(f"assembled {pos} page vectors, manifest says {n}")
    mat.flush()
    del mat
    tmp_vec.replace(vec_path)
    path = index_dir / "page_index.pkl"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"dim": dim, "matrix": None, "matrix_file": vec_path.name,
                     "row_ids": row_ids, "doc_ids": doc_ids}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    log.info("page index: %d documents x %d dims assembled in %.0fs -> %s",
             n, dim, time.time() - t0, path)
    return path
