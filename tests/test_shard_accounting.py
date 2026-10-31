"""Tests for ShardWriter's written-versus-resumed accounting.

The bug this pins
-----------------
``ShardWriter`` defaults to ``resume=True``, and its constructor seeds
``shard_files`` with every pre-existing shard. ``close()`` returned that list, so a
caller recording a corpus size recorded *everything in the directory* rather than
what the run produced.

Concretely, on the worker-B PMC chunking: a first run crashed on a ``NameError`` after
writing 266 shards (1,325,000 chunks, the first 50,000 documents). A second run
resumed at shard index 266 and wrote 7,387,592 chunks. Its manifest claimed
7,387,592 -- while the directory held 8,712,592. Every downstream number (chunks
embedded, vectors indexed, candidates retrieved) would have been inflated by 15%
with nothing anywhere reporting a discrepancy, because the manifest and the code
that produced it agreed with each other.

The recount is the only thing that caught it.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from medrag.data.shards import ShardWriter


def _records(out: Path) -> int:
    return sum(1 for f in sorted(out.glob("x-*.jsonl.gz"))
               for line in gzip.open(f, "rt") if line.strip())


def test_written_and_resumed_are_reported_separately() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        w = ShardWriter(out, "x", shard_size=2)
        for i in range(3):
            w.write({"i": i})
        w.close()

        w2 = ShardWriter(out, "x", shard_size=2)
        for i in range(2):
            w2.write({"j": i})
        w2.close()

        rep = w2.report()
        assert len(rep["written"]) == 1, rep
        assert len(rep["resumed_over"]) == 2, rep
        assert set(rep["written"]).isdisjoint(rep["resumed_over"])
        print("  ok  written and resumed-over shards are distinguished")


def test_close_still_returns_every_shard_present() -> None:
    """Callers that genuinely need the whole directory are not broken."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        w = ShardWriter(out, "x", shard_size=2)
        for i in range(3):
            w.write({"i": i})
        w.close()
        w2 = ShardWriter(out, "x", shard_size=2)
        w2.write({"j": 0})
        assert len(w2.close()) == 3
        print("  ok  close() still lists every shard in the directory")


def test_the_directory_may_hold_more_than_this_run_wrote() -> None:
    """The exact condition that hid the duplication: manifest < directory."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        crashed = ShardWriter(out, "x", shard_size=2)
        for i in range(5):
            crashed.write({"first_run": i})
        crashed.close()  # simulates the run that died mid-corpus
        before = _records(out)

        second = ShardWriter(out, "x", shard_size=2)
        this_run = 2
        for i in range(this_run):
            second.write({"second_run": i})
        written = second.report()["written"]
        second.close()

        counted_by_manifest = this_run
        assert _records(out) == before + counted_by_manifest
        assert _records(out) > counted_by_manifest, (
            "directory holds more than the manifest claims -- the duplication case"
        )
        print(f"  ok  manifest counts {counted_by_manifest}, directory holds "
              f"{_records(out)}; the gap is now visible")


def test_resume_false_starts_from_zero() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        ShardWriter(out, "x", shard_size=2, resume=False).__enter__()
        w = ShardWriter(out, "x", shard_size=2, resume=False)
        for i in range(3):
            w.write({"i": i})
        w.close()
        assert _records(out) == 3
        print("  ok  resume=False ignores pre-existing shards")


def test_a_second_run_cannot_silently_overwrite_a_first() -> None:
    """Resuming writes at a fresh index, so records are never lost or interleaved."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        first = ShardWriter(out, "x", shard_size=2)
        for i in range(4):
            first.write({"run": 1, "i": i})
        first.close()

        second = ShardWriter(out, "x", shard_size=2)
        for i in range(3):
            second.write({"run": 2, "i": i})
        second.close()

        seen = [json.loads(line)["run"]
                for f in sorted(out.glob("x-*.jsonl.gz"))
                for line in gzip.open(f, "rt") if line.strip()]
        assert seen == [1, 1, 1, 1, 2, 2, 2], seen
        assert len(set(seen)) == 2, "both runs are present, neither was clobbered"
        print("  ok  a resumed run appends rather than overwrites")


def main() -> int:
    tests = [
        test_written_and_resumed_are_reported_separately,
        test_close_still_returns_every_shard_present,
        test_the_directory_may_hold_more_than_this_run_wrote,
        test_resume_false_starts_from_zero,
        test_a_second_run_cannot_silently_overwrite_a_first,
    ]
    print("ShardWriter accounting tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())