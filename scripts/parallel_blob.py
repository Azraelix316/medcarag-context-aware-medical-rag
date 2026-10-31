"""Assemble parallel_store.py's chunk shards into the store with many writers.

parallel_store.py assemble writes the 118M-row pmcexpanded store in one process at
~11,000 rows/s, about 3 hours, limited by its own CPU. Every row's bytes are known
from the shard files alone (``{"row":N,`` + the pre-serialised chunk), so each shard's
byte range in the blob can be computed first and the shards then written into one
preallocated file by any number of hosts:

  size    (parallel) bytes and rows per shard -> <chunks>/blob/<stem>.size
  prep    (single)   layout, preallocated blob.jsonl / offsets.bin, rowmap.json,
                     store.json under <store>
  write   (parallel) each shard's blob range and offsets, plus its documents' page
                     text records -> <chunks>/blob/<stem>.pages
  verify  (parallel) re-read every shard's range from the server and compare
  pages   (single)   page_texts blocks and pages.json from the .pages files

The result is byte-identical to parallel_store.assemble (tests/test_parallel_store.py).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.data.schema import Chunk  # noqa: E402
from medrag.data.shards import atomic_write_json  # noqa: E402
from medrag.index import page_index as pi  # noqa: E402
from medrag.index.store import _BLOB, _META, _OFFSET_STRUCT, _OFFSETS, _ROWMAP  # noqa: E402


def stems_of(chunk_dir: Path) -> list[str]:
    return sorted(p.name.removesuffix(".done") for p in chunk_dir.glob("*.done"))


def row_starts(chunk_dir: Path, stems: list[str]) -> dict[str, int]:
    out, r = {}, 0
    for s in stems:
        out[s] = r
        r += json.loads((chunk_dir / f"{s}.done").read_text())["n_chunks"]
    return out


def _lines(chunk_dir: Path, stem: str, row0: int):
    with gzip.open(chunk_dir / f"{stem}.chunks.jsonl.gz", "rb") as fh:
        for i, line in enumerate(fh):
            yield row0 + i, line, b'{"row":%d,' % (row0 + i) + line[1:]


def _claim(d: Path, name: str) -> bool:
    try:
        os.mkdir(d / name)
        return True
    except FileExistsError:
        return False


def do_size(chunk_dir: Path, stem: str, row0: int) -> None:
    n = size = 0
    for _, _, out in _lines(chunk_dir, stem, row0):
        n += 1
        size += len(out)
    atomic_write_json(chunk_dir / "blob" / f"{stem}.size", {"rows": n, "bytes": size})


def layout(chunk_dir: Path, stems: list[str]) -> dict[str, tuple[int, int]]:
    out, r, b = {}, 0, 0
    for s in stems:
        z = json.loads((chunk_dir / "blob" / f"{s}.size").read_text())
        out[s] = (r, b)
        r += z["rows"]
        b += z["bytes"]
    out["__end__"] = (r, b)
    return out


def do_prep(chunk_dir: Path, store_dir: Path, stems: list[str], cache_size: int) -> dict:
    lay = layout(chunk_dir, stems)
    n, total = lay["__end__"]
    store_dir.mkdir(parents=True, exist_ok=True)
    for name, size in ((_BLOB, total), (_OFFSETS, n * _OFFSET_STRUCT.size)):
        with open(store_dir / name, "wb") as fh:
            fh.truncate(size)
    rows: list[str] = []
    for s in stems:
        rows.extend((chunk_dir / f"{s}.ids.txt").read_text(encoding="utf-8").splitlines())
    if len(rows) != n:
        raise ValueError(f"{len(rows)} chunk ids for {n} rows")
    atomic_write_json(store_dir / _ROWMAP, {"n": n, "rows": rows})
    meta = {"n_chunks": n, "blob_bytes": total, "cache_size": cache_size, "version": 2}
    atomic_write_json(store_dir / _META, meta)
    return meta


def _expected(chunk_dir: Path, stem: str, row0: int, byte0: int):
    """The shard's blob bytes, offsets bytes and page records."""
    blob, offs, pages = bytearray(), bytearray(), []
    seen: set[str] = set()
    cursor = byte0
    for row, line, out in _lines(chunk_dir, stem, row0):
        blob += out
        offs += _OFFSET_STRUCT.pack(cursor, len(out))
        cursor += len(out)
        chunk = None
        key = json.loads(line).get("chunk_id") or ""
        doc_id = key.rsplit("#", 1)[0] if "#" in key else key
        if doc_id not in seen:
            seen.add(doc_id)
            chunk = Chunk.from_dict(json.loads(line))
            pages.append(json.dumps({"doc": doc_id, "row": row,
                                     "text": pi._text_from_chunk(chunk)}))
    return bytes(blob), bytes(offs), pages


def do_write(chunk_dir: Path, store_dir: Path, stem: str, row0: int, byte0: int) -> None:
    blob, offs, pages = _expected(chunk_dir, stem, row0, byte0)
    for name, at, data in ((_BLOB, byte0, blob), (_OFFSETS, row0 * _OFFSET_STRUCT.size, offs)):
        fd = os.open(store_dir / name, os.O_WRONLY)
        try:
            view, pos = memoryview(data), 0
            while pos < len(view):
                pos += os.pwrite(fd, view[pos:pos + (1 << 24)], at + pos)
            os.fsync(fd)
        finally:
            os.close(fd)
    tmp = chunk_dir / "blob" / f"{stem}.pages.{os.getpid()}.{os.uname().nodename}.tmp"
    tmp.write_text("".join(p + "\n" for p in pages), encoding="utf-8")
    tmp.replace(chunk_dir / "blob" / f"{stem}.pages")


def do_verify(chunk_dir: Path, store_dir: Path, stem: str, row0: int, byte0: int) -> bool:
    blob, offs, _ = _expected(chunk_dir, stem, row0, byte0)
    with open(store_dir / _BLOB, "rb") as fh:
        fh.seek(byte0)
        ok = fh.read(len(blob)) == blob
    with open(store_dir / _OFFSETS, "rb") as fh:
        fh.seek(row0 * _OFFSET_STRUCT.size)
        ok = ok and fh.read(len(offs)) == offs
    return ok


def do_pages(chunk_dir: Path, index_dir: Path, stems: list[str], n_rows: int,
             texts_dir: str = pi.PAGE_TEXTS_DIR) -> dict:
    out = index_dir / texts_dir
    out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    buf: list[str] = []
    n_blocks = 0

    def flush() -> None:
        nonlocal n_blocks, buf
        final = out / pi.page_text_block_path(index_dir, n_blocks).name
        tmp = final.with_name(final.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1) as fh:
            fh.write("".join(buf))
        tmp.replace(final)
        n_blocks += 1
        buf = []

    for s in stems:
        for line in (chunk_dir / "blob" / f"{s}.pages").read_text(encoding="utf-8").splitlines():
            doc = json.loads(line)["doc"]
            if doc in seen:
                continue
            seen.add(doc)
            buf.append(line + "\n")
            if len(buf) >= pi.PAGE_BLOCK_DOCS:
                flush()
    if buf:
        flush()
    manifest = {"n_docs": len(seen), "n_blocks": n_blocks, "block_docs": pi.PAGE_BLOCK_DOCS,
                "n_rows": n_rows,
                "text": "page_index._text_from_chunk of each document's first chunk"}
    (out / "pages.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _mark(path: Path, text: str) -> None:
    """Write a marker; a marker another host already wrote for the same shard counts."""
    try:
        path.write_text(text)
    except PermissionError:
        if not path.exists():
            raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["size", "prep", "write", "verify", "pages"])
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--store-name", default="store_par")
    ap.add_argument("--texts-name", default="page_texts_par")
    ap.add_argument("--host", default="")
    ap.add_argument("--store-cache", type=int, default=500_000)
    args = ap.parse_args()
    chunk_dir, index_dir = Path(args.chunks), Path(args.index_dir)
    store_dir = index_dir / args.store_name
    stems = stems_of(chunk_dir)
    work = chunk_dir / "blob"
    (work / f"claims_{args.step}").mkdir(parents=True, exist_ok=True)
    if args.step == "size":
        starts = row_starts(chunk_dir, stems)
        for s in stems:
            if not (work / f"{s}.size").exists() and _claim(work / "claims_size", s):
                do_size(chunk_dir, s, starts[s])
        return 0
    if args.step == "prep":
        print(json.dumps(do_prep(chunk_dir, store_dir, stems, args.store_cache)))
        return 0
    lay = layout(chunk_dir, stems)
    if args.step == "pages":
        print(json.dumps(do_pages(chunk_dir, index_dir, stems, lay["__end__"][0], args.texts_name)))
        return 0
    for s in stems:
        mark = work / f"{s}.{'written' if args.step == 'write' else 'verified'}"
        if mark.exists() or not _claim(work / f"claims_{args.step}", s):
            continue
        t0 = time.time()
        if args.step == "write":
            do_write(chunk_dir, store_dir, s, *lay[s])
            _mark(mark, args.host)
        elif do_verify(chunk_dir, store_dir, s, *lay[s]):
            _mark(mark, args.host)
        else:
            _mark(work / f"{s}.MISMATCH", args.host)
            print(f"{args.host} {s} MISMATCH", flush=True)
        print(f"{args.host} {args.step} {s} {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
