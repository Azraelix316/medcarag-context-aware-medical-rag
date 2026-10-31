"""Tests for post-crawl field backfill.

A crawl costs hours; a parser fix arrives after it. Backfilling one field from a
re-fetch is the cheap alternative, and the whole thing is worthless if it attaches
values to the wrong documents.

The failure this module has to survive
--------------------------------------
Patches arrive in **fetch order**; shards are in **crawl order**. Those differ. A
positional list would give each document the citations of whichever document
happened to sit at the same offset in the file. Nothing would raise: the field
would be populated, coverage would look excellent, and every citation edge would
point at the wrong paper. That is the failure mode this project has hit six times
already -- plausible numbers instead of errors.

So lookups here are by identifier, and the tests shuffle aggressively.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from medrag.data.backfill import apply_patches, backfill_field, iter_corpus_ids


def _write_shard(path: Path, docs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")
    return path


def _docs(n: int, *, with_citations: bool = False) -> list[dict]:
    return [
        {
            "doc_id": f"pmcid:PMC{i:06d}", "pmcid": f"PMC{i:06d}", "pmid": str(1000 + i),
            "source": "pmc", "title": f"trial {i}", "abstract": "", "sections": [],
            "citations": [1, 2, 3] if with_citations else [],
        }
        for i in range(n)
    ]


def test_identifiers_are_collected_only_where_the_field_is_missing() -> None:
    with tempfile.TemporaryDirectory() as d:
        a = _write_shard(Path(d) / "a.jsonl.gz", _docs(3))
        b = _write_shard(Path(d) / "b.jsonl.gz", _docs(2, with_citations=True))
        todo = iter_corpus_ids([a, b], field="citations")
        assert len(todo) == 3, todo  # the two already-populated ones are excluded
        assert "PMC000000" in todo
        print("  ok  only documents lacking the field are candidates")


def test_patches_are_matched_by_identifier_not_position() -> None:
    """THE test: shuffled patches must still land on the right documents."""
    with tempfile.TemporaryDirectory() as d:
        shard = _write_shard(Path(d) / "a.jsonl.gz", _docs(6))
        # Deliberately reversed relative to shard order, with distinct values.
        patches = {f"PMC{i:06d}": [900 + i] for i in reversed(range(6))}
        n = apply_patches([shard], "citations", patches)
        assert n == 6, n
        got = [json.loads(l)["citations"] for l in gzip.open(shard, "rt")]
        want = [[900 + i] for i in range(6)]
        assert got == want, f"misaligned: {got} != {want}"
        print("  ok  reversed patches still attach to the correct documents")


def test_a_document_with_no_patch_is_left_alone() -> None:
    with tempfile.TemporaryDirectory() as d:
        shard = _write_shard(Path(d) / "a.jsonl.gz", _docs(3))
        n = apply_patches([shard], "citations", {"PMC000001": [77]})
        assert n == 1, n
        rows = [json.loads(l) for l in gzip.open(shard, "rt")]
        assert rows[0]["citations"] == [], rows[0]
        assert rows[1]["citations"] == [77], rows[1]
        assert rows[2]["citations"] == [], rows[2]
        print("  ok  documents without a fetched value keep the field empty")


def test_an_existing_value_is_never_overwritten() -> None:
    with tempfile.TemporaryDirectory() as d:
        shard = _write_shard(Path(d) / "a.jsonl.gz", _docs(2, with_citations=True))
        n = apply_patches([shard], "citations", {"PMC000000": [42]})
        assert n == 0, n
        assert json.loads(gzip.open(shard, "rt").readline())["citations"] == [1, 2, 3]
        print("  ok  an already-populated field is left untouched")


def test_dry_run_spends_no_api_budget() -> None:
    with tempfile.TemporaryDirectory() as d:
        shard = _write_shard(Path(d) / "a.jsonl.gz", _docs(4))

        class ExplodingClient:
            def efetch(self, *a, **k):  # noqa: ANN002, ANN003
                raise AssertionError("dry_run must not fetch")

        stats = backfill_field([shard], "citations", client=ExplodingClient(),
                               source="pmc", dry_run=True)
        assert stats == {"candidates": 4, "fetched": 0, "updated": 0, "still_empty": 0}, stats
        assert json.loads(gzip.open(shard, "rt").readline())["citations"] == []
        print("  ok  dry_run reports candidates without fetching or writing")


def test_an_unbackfillable_field_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as d:
        shard = _write_shard(Path(d) / "a.jsonl.gz", _docs(1))

        class UnusedClient:
            def efetch(self, *a, **k):  # noqa: ANN002, ANN003
                raise AssertionError("must reject before fetching")

        try:
            backfill_field([shard], "title", client=UnusedClient(), source="pmc")
        except ValueError as exc:
            assert "not backfillable" in str(exc), exc
        else:
            raise AssertionError("expected ValueError for a non-backfillable field")
        print("  ok  an unknown field is rejected before any API call")


def test_limit_bounds_the_work() -> None:
    with tempfile.TemporaryDirectory() as d:
        shard = _write_shard(Path(d) / "a.jsonl.gz", _docs(50))

        class CountingClient:
            def efetch(self, ids, **k):  # noqa: ANN001, ANN003
                return "<pmc-articleset></pmc-articleset>"

        stats = backfill_field([shard], "citations", client=CountingClient(),
                               source="pmc", limit=10)
        assert stats["candidates"] == 10, stats
        print("  ok  limit bounds the number of documents considered")


def main() -> int:
    tests = [
        test_identifiers_are_collected_only_where_the_field_is_missing,
        test_patches_are_matched_by_identifier_not_position,
        test_a_document_with_no_patch_is_left_alone,
        test_an_existing_value_is_never_overwritten,
        test_dry_run_spends_no_api_budget,
        test_an_unbackfillable_field_is_rejected,
        test_limit_bounds_the_work,
    ]
    print("Backfill tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())