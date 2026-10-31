"""Chunk store: the chunk payloads addressed by retrieval row id.

Retrieval indexes only deal in integer row ids. This module maps row ids back to
chunk text and metadata.

Design constraint that drives everything here: at 7M chunks the payloads are
~10 GB, so they cannot be held as Python objects, and the earlier design (keep
chunks in an LRU, re-materialise on demand) silently **loses** every evicted
chunk. So the store is write-once and streaming:

* chunks are serialised straight into ``blob.jsonl`` as they are added;
* ``offsets.bin`` holds a packed (offset, length) pair per row, enabling true
  ``seek`` + ``read`` random access with no decompression and no shard guessing;
* ``rowmap.json`` maps row id -> chunk id and stays small enough to be resident;
* an LRU of decoded chunks sits in front of the blob so hot documents never hit
  the disk.

Peak memory is therefore O(cache_size), independent of corpus size.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..data.schema import Chunk
from ..data.shards import atomic_write_json

log = logging.getLogger(__name__)

_OFFSET_STRUCT = struct.Struct("<QQ")  # (byte offset, byte length)
_BLOB = "blob.jsonl"
_OFFSETS = "offsets.bin"
_ROWMAP = "rowmap.json"
_META = "store.json"


class _CompactRows(Sequence[str]):
    """Read-only ``row_id -> chunk_id`` in one fixed-width bytes array.

    A loaded store's row map as a list of str is ~85 bytes per row; at 118M rows
    (pmcexpanded) that is 10 GB before anything else is loaded. Same values, same
    order, decoded on access.
    """

    def __init__(self, ids: list[str]) -> None:
        enc = [i.encode("utf-8") for i in ids]
        width = max((len(e) for e in enc), default=1)
        self._a = np.array(enc, dtype=f"S{max(1, width)}")
        if any(e.endswith(b"\x00") for e in enc):
            raise ValueError("chunk id ending in NUL cannot be stored compactly")

    def __len__(self) -> int:
        return len(self._a)

    def __getitem__(self, i: Any) -> Any:
        if isinstance(i, slice):
            return [b.decode("utf-8") for b in self._a[i]]
        return self._a[i].decode("utf-8")

    def __iter__(self) -> Iterator[str]:
        for b in self._a:
            yield b.decode("utf-8")


class ChunkStore:
    """Row-id -> Chunk mapping over an append-only blob with an LRU cache."""

    def __init__(
        self,
        cache_size: int = 200_000,
        *,
        dir_path: str | Path | None = None,
        append: bool = False,
    ) -> None:
        self.rows: list[str] = []                      # row_id -> chunk_id
        self.cache: OrderedDict[str, Chunk] = OrderedDict()
        self.cache_size = max(1, cache_size)
        self.dir: Path | None = Path(dir_path) if dir_path else None
        # Appending continues an existing store at its current end offset.
        self.append = append
        self._offsets: list[tuple[int, int]] = []
        self._cursor: int = 0
        self._wfh: Any = None                          # write handle on the blob
        self._rfh: Any = None                          # read handle on the blob
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __contains__(self, row_id: int) -> bool:
        return 0 <= row_id < len(self.rows)

    # ------------------------------------------------------------------ build
    def _writer(self) -> Any:
        """Open the blob for writing on first write (streaming mode).

        Truncates by default. Appending is only ever correct when the caller
        explicitly resumes an *existing* store, because offsets are absolute
        positions in the blob: appending after a fresh (non-resumed) build would
        leave every new offset pointing into the previous file's bytes.
        """
        if self._wfh is None:
            if self.dir is None:
                raise RuntimeError(
                    "ChunkStore needs a directory to persist into. Pass dir_path=... "
                    "when building, so chunks can stream to disk instead of RAM."
                )
            self.dir.mkdir(parents=True, exist_ok=True)
            existing = self.dir / _BLOB
            mode = "ab" if (self.append and existing.exists()) else "wb"
            if existing.exists() and mode == "wb":
                log.info("truncating existing chunk store blob at %s", existing)
            self._wfh = open(existing, mode)
        return self._wfh

    def add(self, chunk: Chunk) -> int:
        """Append one chunk. Streams to disk immediately; the LRU is only a cache.

        This is what makes a multi-million chunk corpus buildable in bounded
        memory: nothing depends on a chunk still being resident.
        """
        row_id = len(self.rows)
        fh = self._writer()
        line = json.dumps(
            {"row": row_id, **chunk.to_dict()}, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        fh.write(line)
        self._offsets.append((self._cursor, len(line)))
        self._cursor += len(line)
        self.rows.append(chunk.chunk_id)
        self._put(row_id, chunk)
        return row_id

    def extend(self, chunks: Iterable[Chunk]) -> list[int]:
        return [self.add(c) for c in chunks]

    def _put(self, row_id: int, chunk: Chunk) -> None:
        key = str(row_id)
        self.cache[key] = chunk
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def finalize_build(self) -> "ChunkStore":
        """Close the writer and persist the row map plus offset table."""
        self.close_writer()
        if self.dir is None:
            log.warning("finalize_build with no directory; store is memory-only")
            return self
        with open(self.dir / _OFFSETS, "wb") as fh:
            for offset, length in self._offsets:
                fh.write(_OFFSET_STRUCT.pack(offset, length))
        atomic_write_json(self.dir / _ROWMAP, {"n": len(self.rows), "rows": self.rows})
        atomic_write_json(self.dir / _META, {
            "n_chunks": len(self.rows),
            "blob_bytes": self._cursor,
            "cache_size": self.cache_size,
            "version": 2,
        })
        log.info("chunk store finalized: %d chunks, %.1f MiB blob",
                 len(self.rows), self._cursor / 2**20)
        return self

    # ------------------------------------------------------------------- read
    def get(self, row_id: int) -> Chunk | None:
        if not (0 <= row_id < len(self.rows)):
            return None
        key = str(row_id)
        chunk = self.cache.get(key)
        if chunk is not None:
            self.hits += 1
            self.cache.move_to_end(key)
            return chunk
        self.misses += 1
        chunk = self._read_blob(row_id)
        if chunk is not None:
            self._put(row_id, chunk)
        return chunk

    def get_many(self, row_ids: Sequence[int]) -> list[Chunk]:
        out: list[Chunk] = []
        for r in row_ids:
            chunk = self.get(r)
            if chunk is not None:
                out.append(chunk)
        return out

    def text(self, row_id: int) -> str:
        chunk = self.get(row_id)
        return chunk.text if chunk else ""

    def texts(self, row_ids: Sequence[int]) -> dict[int, str]:
        out: dict[int, str] = {}
        for r in row_ids:
            chunk = self.get(r)
            if chunk is not None:
                out[r] = chunk.text
        return out

    def _reader(self) -> Any:
        if self._rfh is None:
            if self.dir is None or not (self.dir / _BLOB).exists():
                raise RuntimeError(f"chunk store blob missing under {self.dir}")
            self._rfh = open(self.dir / _BLOB, "rb")
        return self._rfh

    def _read_blob(self, row_id: int) -> Chunk | None:
        if row_id >= len(self._offsets):
            return None
        offset, length = (int(x) for x in self._offsets[row_id])
        try:
            fh = self._reader()
            fh.seek(offset)
            raw = fh.read(length)
            payload = json.loads(raw.decode("utf-8"))
            return Chunk.from_dict({k: v for k, v in payload.items() if k != "row"})
        except (OSError, EOFError, ValueError, json.JSONDecodeError) as exc:
            log.warning("blob read failed for row %d: %s", row_id, exc)
            return None

    def close_writer(self) -> None:
        if self._wfh is not None:
            try:
                self._wfh.flush()
                os.fsync(self._wfh.fileno())
            except (OSError, ValueError):
                pass
            self._wfh.close()
            self._wfh = None

    def close(self) -> None:
        self.close_writer()
        if self._rfh is not None:
            try:
                self._rfh.close()
            finally:
                self._rfh = None

    def __enter__(self) -> "ChunkStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "n_chunks": len(self.rows),
            "cache_size": len(self.cache),
            "hit_rate": round(self.hits / total, 4) if total else None,
            "blob_bytes": self._cursor,
        }

    # -------------------------------------------------------------- persistence
    @classmethod
    def load(cls, dir_path: str | Path, *, cache_size: int = 200_000, append: bool = False) -> "ChunkStore":
        """Open an existing store, read-only by default or ready to append."""
        dir_path = Path(dir_path)
        if not (dir_path / _ROWMAP).exists():
            raise FileNotFoundError(f"chunk store rowmap missing under {dir_path}")
        store = cls(cache_size=cache_size, dir_path=dir_path)
        rowmap = json.loads((dir_path / _ROWMAP).read_text(encoding="utf-8"))
        # Appending needs real lists to grow; a read-only load keeps both tables compact.
        store.rows = list(rowmap["rows"]) if append else _CompactRows(rowmap["rows"])
        del rowmap
        offsets_path = dir_path / _OFFSETS
        blob_path = dir_path / _BLOB
        if not offsets_path.exists() or not blob_path.exists():
            raise FileNotFoundError(
                f"chunk store at {dir_path} is incomplete: expected both {_OFFSETS} and "
                f"{_BLOB}. This usually means the index was built by an older version "
                f"(or the build was interrupted). Re-run `medrag build-index --force`."
            )
        raw = offsets_path.read_bytes()
        n = len(raw) // _OFFSET_STRUCT.size
        if append:
            store._offsets = [
                _OFFSET_STRUCT.unpack_from(raw, i * _OFFSET_STRUCT.size) for i in range(n)
            ]
        else:
            # (offset, length) pairs as one int64 array: 16 bytes a row instead of ~116
            # for a list of tuples. Readers take int() of each pair.
            store._offsets = np.frombuffer(raw, dtype="<u8").reshape(n, 2).astype(np.int64)
        del raw
        store._cursor = blob_path.stat().st_size
        if len(store._offsets) != len(store.rows):
            raise ValueError(
                f"chunk store is inconsistent: {len(store.rows)} rows but "
                f"{len(store._offsets)} offsets. Rebuild with `medrag build-index --force`."
            )
        log.info("loaded chunk store: %d rows, %d offsets, cache=%d",
                 len(store.rows), len(store._offsets), cache_size)
        if append:
            store.append = True
        return store

    # ------------------------------------------------------------------ iterate
    def iter_chunks(self, *, limit: int | None = None) -> Iterator[Chunk]:
        n = len(self.rows) if limit is None else min(limit, len(self.rows))
        for row_id in range(n):
            chunk = self.get(row_id)
            if chunk is not None:
                yield chunk

    def iter_texts(self, *, use_embed_text: bool = False) -> Iterator[tuple[int, str]]:
        """Yield ``(row_id, text)`` for every chunk; used to build both indexes."""
        for row_id in range(len(self.rows)):
            chunk = self.get(row_id)
            if chunk is None:
                continue
            yield row_id, (chunk.embed_text or chunk.text) if use_embed_text else chunk.text

    def document_rows(self) -> tuple[list[str], list[int]]:
        """``(doc_ids, first_row_ids)`` for every document, in first-appearance order.

        Derived from ``self.rows``, which is already resident after :meth:`load`, so
        this costs no blob reads at all -- the row map carries the document id on
        every row (``pmcid:PMC13640969#0000``).

        This replaced a walk that called :meth:`get` on all 16,133,551 rows purely to
        discover which one opened each document. That walk read 46.56 GiB of blob at
        ~5.8 MB/s over NFS -- about 2.2 hours before the first page vector could
        exist -- to arrive at an answer this returns in seconds. The cost was not
        inherent to the task; it was an artifact of finding out what was already
        known.

        Correct whether or not rows are grouped by document, since "first
        appearance" does not assume grouping.
        """
        doc_ids: list[str] = []
        first_rows: list[int] = []
        seen: set[str] = set()
        for row_id, row_key in enumerate(self.rows):
            doc_id = row_key.rsplit("#", 1)[0] if "#" in row_key else row_key
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc_ids.append(doc_id)
            first_rows.append(row_id)
        log.info("document rows: %d documents from %d chunk rows (no blob reads)",
                 len(doc_ids), len(self.rows))
        return doc_ids, first_rows

    def chunks_bulk(self, row_ids: Sequence[int]) -> dict[int, Chunk]:
        """Fetch many whole chunks by positional read, without the row cache.

        Returns the parsed chunk rather than just its text, because document-level
        text needs the denormalised title, journal and section headings as well.

        Point reads beat clustered reads by a wide margin here, which was not the
        expectation. Measured over 20,000 opening rows sampled from disjoint regions
        of the 46.56 GiB blob, so no region was read twice and the page cache could
        not flatter one case over another:

            clustered, 256 KiB gap, 1 MiB span cap    303.0s  45.02x amplification
            clustered, 256 KiB gap, 8 MiB span cap     49.3s   7.03x
            point reads, no fadvise                    11.7s   1.66x
            point reads + POSIX_FADV_RANDOM            10.6s   1.46x

        Clustering looks like it should win -- reading a contiguous span and picking
        the wanted rows out of it -- and it loses by 29x. Opening chunks sit one per
        ~84 KB, densely enough that the merge is transitive and a 1 MiB span cap
        still covers most of the file, while every large read pulls a readahead
        window far bigger than itself.

        ``os.pread`` rather than seek-then-read so the offset is carried per call and
        the handle stays usable for other work, and a dedicated handle so
        POSIX_FADV_RANDOM applies to the scattered path only and never slows the
        sequential point-read path that ``get`` uses.
        """
        def _fallback() -> dict[int, Chunk]:
            return {r: c for r, c in zip(row_ids, self.get_many(row_ids)) if c is not None}

        wanted = sorted({r for r in row_ids if 0 <= r < len(self._offsets)})
        if not wanted:
            return {}
        if self.dir is None or not (self.dir / _BLOB).exists():
            return _fallback()
        try:
            fd = os.open(self.dir / _BLOB, os.O_RDONLY)
        except OSError:
            return _fallback()
        try:
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            except (AttributeError, OSError):
                pass
            out: dict[int, Chunk] = {}
            for row_id in wanted:
                offset, length = (int(x) for x in self._offsets[row_id])
                try:
                    raw = os.pread(fd, length, offset)
                    payload = json.loads(raw.decode("utf-8"))
                except (OSError, EOFError, ValueError, UnicodeDecodeError) as exc:
                    log.warning("bulk blob read failed for row %d: %s", row_id, exc)
                    continue
                out[row_id] = Chunk.from_dict(
                    {k: v for k, v in payload.items() if k != "row"}
                )
        finally:
            os.close(fd)

        for row_id in wanted:
            if row_id not in out:
                chunk = self.get(row_id)
                if chunk is not None:
                    out[row_id] = chunk
        return out

    def texts_bulk(self, row_ids: Sequence[int]) -> list[str]:
        """Fetch many chunk texts with a single forward-sorted pass over the blob."""
        if len(self._offsets) == 0:
            return [self.text(r) for r in row_ids]
        try:
            fh = self._reader()
        except RuntimeError:
            return [self.text(r) for r in row_ids]
        result: dict[int, str] = {}
        for row_id in sorted({r for r in row_ids if 0 <= r < len(self._offsets)}):
            offset, length = (int(x) for x in self._offsets[row_id])
            fh.seek(offset)
            try:
                payload = json.loads(fh.read(length).decode("utf-8"))
                result[row_id] = payload.get("text", "")
            except (ValueError, json.JSONDecodeError):
                result[row_id] = ""
        return [result.get(r, "") for r in row_ids]


def build_chunk_store(
    chunks: Iterable[Chunk],
    *,
    dir_path: str | Path | None = None,
    cache_size: int = 200_000,
    finalize: bool = True,
    append: bool = False,
) -> ChunkStore:
    """Build a store, streaming chunks to ``dir_path`` when one is given.

    Passing ``dir_path`` is what keeps memory bounded at corpus scale; without
    it the store is memory-only and appropriate for tests and small indexes.
    ``append=True`` continues an existing store instead of replacing it.
    """
    store = ChunkStore(cache_size=cache_size, dir_path=dir_path, append=append)
    for chunk in chunks:
        store.add(chunk)
    if finalize and dir_path is not None:
        store.finalize_build()
    return store


def store_stats(store: ChunkStore) -> dict[str, Any]:
    return store.stats()
