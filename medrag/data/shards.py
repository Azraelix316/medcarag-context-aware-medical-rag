"""Gzip-sharded JSONL I/O for corpus and chunk files.

Everything the pipeline produces is append-only sharded JSONL so that a run
killed midway leaves valid, resumable files. A manifest records what was built
so downstream stages never have to guess which shards belong together.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class ShardWriter:
    """Write records to ``<prefix>-00000.jsonl.gz`` and roll over at ``shard_size``.

    Tracks the shard list and byte count so the caller can emit a manifest
    without walking the output directory again.
    """

    def __init__(self, out_dir: Path, prefix: str, shard_size: int = 10_000, resume: bool = True) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_size = max(1, shard_size)
        self.resume = resume
        self.shard_files: list[Path] = []
        #: Shards this run actually opened and wrote. Distinct from shard_files,
        #: which under resume also holds every pre-existing shard.
        self.written_shards: list[Path] = []
        self.n_records = 0
        self.n_bytes = 0
        self._buffer: list[str] = []
        self._shard_index = self._next_free_index()
        self._fh: gzip.GzipFile | None = None
        self._raw: Any = None
        self.started = time.time()

    def _next_free_index(self) -> int:
        if not self.resume:
            return 0
        existing = sorted(self.out_dir.glob(f"{self.prefix}-*.jsonl.gz"))
        self.shard_files = existing
        return len(existing)

    def _open(self) -> None:
        path = self.out_dir / f"{self.prefix}-{self._shard_index:05d}.jsonl.gz"
        # mtime=0 keeps shard bytes deterministic for reproducible corpus hashes.
        self._raw = open(path, "wb")
        self._fh = gzip.GzipFile(fileobj=self._raw, mode="wb", compresslevel=6, mtime=0)
        self.shard_files.append(path)
        self.written_shards.append(path)
        log.debug("opened shard %s", path)

    def _close_current(self) -> None:
        if self._fh is not None:
            self._flush_buffer()
            self._fh.close()
            assert self._raw is not None
            self._raw.close()
            self._fh = None
            self._raw = None

    def write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            self._open()
        assert self._fh is not None
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        self._buffer.append(line)
        self.n_records += 1
        self.n_bytes += len(line.encode("utf-8"))
        if len(self._buffer) >= self.shard_size:
            # Roll over: the full shard is closed and the next write opens the
            # following one. Without this every record would land in one file.
            self._close_current()
            self._shard_index += 1

    def extend(self, records: Iterable[dict[str, Any]]) -> None:
        for record in records:
            self.write(record)

    def _flush_buffer(self) -> None:
        if self._fh is None or not self._buffer:
            return
        self._fh.write("".join(self._buffer).encode("utf-8"))
        self._buffer.clear()

    def flush(self) -> None:
        """Push buffered records into the open shard without rolling over."""
        self._flush_buffer()

    def close(self) -> list[Path]:
        self._close_current()
        return self.shard_files

    def report(self) -> dict[str, Any]:
        """Distinguish what this run wrote from what was already on disk.

        Under ``resume=True`` the constructor seeds ``shard_files`` with every
        pre-existing shard, so returning that list as *this run's* output is a lie
        whenever a previous run left anything behind. It is not a cosmetic lie: it
        is how a 1,325,000-chunk duplicated prefix went unnoticed. The manifest
        claimed 7,387,592 chunks while the directory held 8,712,592, and every
        downstream number -- chunks embedded, rows indexed, candidates retrieved --
        would have been inflated by 15% with nothing reporting a discrepancy.

        Callers that record a corpus size must record ``written_shards``, and a
        corpus whose manifest disagrees with its own directory must fail rather
        than proceed.
        """
        return {
            "written": [str(p) for p in self.written_shards],
            "resumed_over": [str(p) for p in self.shard_files
                             if p not in set(self.written_shards)],
        }

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def elapsed(self) -> float:
        return time.time() - self.started


def read_shards(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    """Stream records from gzip or plain JSONL shards, skipping torn lines.

    A shard truncated by a hard kill still yields every complete record, which
    is what makes interrupted crawls resumable without data loss.
    """
    for path in paths:
        p = Path(path)
        if not p.exists():
            log.warning("shard missing, skipping: %s", p)
            continue
        opener = gzip.open if p.suffix == ".gz" else open
        try:
            with opener(p, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        log.warning("skipping corrupt record %s:%d (%s)", p, lineno, exc)
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            log.warning("truncated or unreadable shard %s: %s", p, exc)


def count_records(paths: Iterable[str | Path]) -> int:
    return sum(1 for _ in read_shards(paths))


def disk_bytes(paths: Iterable[str | Path]) -> int:
    return sum(Path(p).stat().st_size for p in paths if Path(p).exists())


class _Progress:
    """Minimal throttled progress logger (tqdm may be absent at runtime)."""

    def __init__(self, total: int, label: str, every: float = 30.0) -> None:
        self.total = total
        self.label = label
        self.every = every
        self.n = 0
        self._last = time.time()
        self._start = self._last

    def update(self, n: int = 1) -> None:
        self.n += n
        now = time.time()
        if now - self._last >= self.every:
            self._last = now
            rate = self.n / max(1e-9, now - self._start)
            pct = f"{100.0 * self.n / self.total:.1f}%" if self.total else "?"
            log.info("%s: %d/%s (%s) @ %.0f/s", self.label, self.n, self.total, pct, rate)

    def close(self) -> None:
        log.info("%s: done %d records in %.1fs", self.label, self.n, time.time() - self._start)


def atomic_write_json(path: Path, payload: Any) -> Path:
    """Write JSON via a temp file + rename so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)
    return path
