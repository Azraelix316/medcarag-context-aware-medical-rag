"""Chunk a corpus in parallel by input shard, then assemble the chunk store from it.

``medrag chunk`` runs at ~124 documents/s in one process (pmcsample), so 4,044,247
full-text documents (pmcexpanded) would take 9 hours, and build_pmc_store.py's
store build another 9 at ~3,500 rows/s. Chunking is per document, so it splits by
input shard; the store is a blob of ``{"row":N,<chunk>}`` lines, so it is assembled
by prefixing each pre-serialised chunk with its global row once all shards are done.

  chunk     claim input shards (mkdir claims) and write, per shard,
            <out>/<shard>.chunks.jsonl.gz, <shard>.ids.txt and <shard>.done
  assemble  concatenate every shard in sorted order into <store>/blob.jsonl,
            offsets.bin, rowmap.json and store.json

The assembled store is byte-identical to build_store_from_chunks over the same
chunks in the same order (tests/test_parallel_store.py), which is what
``medrag chunk`` + build_pmc_store.py produce: shards are read in sorted order and
every chunk goes through the same Chunk.from_dict/to_dict round trip.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.config import PipelineConfig  # noqa: E402
from medrag.data.chunk import chunk_document  # noqa: E402
from medrag.data.schema import Chunk, Document  # noqa: E402
from medrag.data.shards import atomic_write_json, read_shards  # noqa: E402
from medrag.index.store import _BLOB, _META, _OFFSET_STRUCT, _OFFSETS, _ROWMAP  # noqa: E402

log = logging.getLogger("parallel_store")


def chunk_line(chunk: Chunk) -> bytes:
    """A chunk serialised as ChunkStore.add writes it, minus the leading row field."""
    d = Chunk.from_dict(chunk.to_dict()).to_dict()
    return json.dumps(d, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def chunk_shard(shard: Path, out_dir: Path, chunk_cfg) -> dict:
    stem = shard.name.removesuffix(".jsonl.gz")
    tmp = out_dir / f"{stem}.chunks.jsonl.gz.tmp"
    ids_tmp = out_dir / f"{stem}.ids.txt.tmp"
    n_docs = n_chunks = n_bad = 0
    with gzip.open(tmp, "wb", compresslevel=1) as fh, open(ids_tmp, "w", encoding="utf-8") as ih:
        for record in read_shards([shard]):
            try:
                doc = Document.from_dict(record)
            except Exception:  # noqa: BLE001  (iter_documents skips these too)
                n_bad += 1
                continue
            n_docs += 1
            for chunk in chunk_document(doc, chunk_cfg):
                fh.write(chunk_line(chunk) + b"\n")
                ih.write(chunk.chunk_id + "\n")
                n_chunks += 1
    tmp.replace(out_dir / f"{stem}.chunks.jsonl.gz")
    ids_tmp.replace(out_dir / f"{stem}.ids.txt")
    rec = {"shard": shard.name, "n_docs": n_docs, "n_chunks": n_chunks, "n_malformed": n_bad}
    (out_dir / f"{stem}.done").write_text(json.dumps(rec), encoding="utf-8")
    return rec


def cmd_chunk(args: argparse.Namespace) -> int:
    cfg = PipelineConfig.load(args.config)._finalize()
    src = cfg.paths.corpus_dir / cfg.acquire.source
    shards = sorted(src.glob(f"{cfg.acquire.source}-*.jsonl.gz"))
    out_dir = Path(args.out)
    claims = out_dir / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        stem = shard.name.removesuffix(".jsonl.gz")
        if (out_dir / f"{stem}.done").exists():
            continue
        try:
            os.mkdir(claims / stem)
        except FileExistsError:
            continue
        t0 = time.time()
        rec = chunk_shard(shard, out_dir, cfg.chunk)
        print(f"{args.host} {rec} {time.time() - t0:.0f}s", flush=True)
    return 0


def assemble(chunk_dir: Path, store_dir: Path, shard_names: list[str], cache_size: int,
             page_texts: bool = False) -> dict:
    """Write the store; with ``page_texts``, also write_page_text_blocks' output.

    The page texts come from the same pass instead of a second read of the blob
    (345 GB at pmcexpanded): each document's first chunk, in first-appearance order,
    in the same blocks, records and pages.json (tests/test_parallel_store.py).
    """
    from medrag.index import page_index as pi

    store_dir.mkdir(parents=True, exist_ok=True)
    index_dir = store_dir.parent
    rows: list[str] = []
    cursor = 0
    seen: set[str] = set()
    buf: list[str] = []
    n_blocks = 0

    def flush() -> None:
        nonlocal n_blocks, buf
        final = pi.page_text_block_path(index_dir, n_blocks)
        tmp = final.with_name(final.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1) as fh:
            fh.write("".join(buf))
        tmp.replace(final)
        n_blocks += 1
        buf = []

    if page_texts:
        (index_dir / pi.PAGE_TEXTS_DIR).mkdir(parents=True, exist_ok=True)
    with open(store_dir / _BLOB, "wb", buffering=1 << 24) as blob, \
            open(store_dir / _OFFSETS, "wb", buffering=1 << 22) as offs:
        for name in shard_names:
            stem = name.removesuffix(".jsonl.gz")
            with open(chunk_dir / f"{stem}.ids.txt", encoding="utf-8") as ih:
                ids = ih.read().splitlines()
            n_before = len(rows)
            with gzip.open(chunk_dir / f"{stem}.chunks.jsonl.gz", "rb") as fh:
                for line in fh:
                    row = len(rows)
                    out = b'{"row":%d,' % row + line[1:]
                    blob.write(out)
                    offs.write(_OFFSET_STRUCT.pack(cursor, len(out)))
                    cursor += len(out)
                    key = ids[row - n_before]
                    rows.append(key)
                    if page_texts:
                        doc_id = key.rsplit("#", 1)[0] if "#" in key else key
                        if doc_id not in seen:
                            seen.add(doc_id)
                            chunk = Chunk.from_dict(json.loads(line))
                            buf.append(json.dumps({"doc": doc_id, "row": row,
                                                   "text": pi._text_from_chunk(chunk)}) + "\n")
                            if len(buf) >= pi.PAGE_BLOCK_DOCS:
                                flush()
            if len(rows) - n_before != len(ids):
                raise ValueError(f"{stem}: {len(rows) - n_before} chunks but {len(ids)} ids")
        blob.flush()
        os.fsync(blob.fileno())
    if page_texts:
        if buf:
            flush()
        manifest = {"n_docs": len(seen), "n_blocks": n_blocks, "block_docs": pi.PAGE_BLOCK_DOCS,
                    "n_rows": len(rows),
                    "text": "page_index._text_from_chunk of each document's first chunk"}
        (index_dir / pi.PAGE_TEXTS_DIR / "pages.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
    atomic_write_json(store_dir / _ROWMAP, {"n": len(rows), "rows": rows})
    meta = {"n_chunks": len(rows), "blob_bytes": cursor, "cache_size": cache_size, "version": 2}
    atomic_write_json(store_dir / _META, meta)
    return meta


def cmd_assemble(args: argparse.Namespace) -> int:
    cfg = PipelineConfig.load(args.config)._finalize()
    src = cfg.paths.corpus_dir / cfg.acquire.source
    names = [p.name for p in sorted(src.glob(f"{cfg.acquire.source}-*.jsonl.gz"))]
    chunk_dir = Path(args.out)
    missing = [n for n in names if not (chunk_dir / f"{n.removesuffix('.jsonl.gz')}.done").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} shards not chunked yet, e.g. {missing[:3]}")
    t0 = time.time()
    meta = assemble(chunk_dir, cfg.paths.index_dir / "store", names, args.store_cache,
                    page_texts=True)
    done = [json.loads((chunk_dir / f"{n.removesuffix('.jsonl.gz')}.done").read_text()) for n in names]
    summary = {**meta, "n_documents": sum(d["n_docs"] for d in done),
               "n_malformed": sum(d["n_malformed"] for d in done),
               "shards": len(names), "elapsed_s": round(time.time() - t0, 1)}
    atomic_write_json(chunk_dir / "manifest.json", summary)
    print(json.dumps(summary))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["chunk", "assemble"])
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--out", required=True, help="per-shard chunk directory")
    ap.add_argument("--host", default=os.uname().nodename if hasattr(os, "uname") else "local")
    ap.add_argument("--store-cache", type=int, default=500_000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    return cmd_chunk(args) if args.step == "chunk" else cmd_assemble(args)


if __name__ == "__main__":
    raise SystemExit(main())
