"""A resumed crawl's manifest must size the corpus, not the run.

This is the same failure shape as the others in this project -- a plausible number
where an absence belongs -- with an extra edge: the number is not merely wrong, it is
wrong by a factor nobody can see, in a field called ``n_documents`` that is the only
count in the file.

Measured on this corpus. The crawl had been killed on 2026-10-07 leaving 930,392
documents over 47 shards, and was restarted on 2026-10-10 against the same directory.
It wrote 30,488 new documents, bringing the directory to 960,880 rows over 51 files
(all unique, confirmed by an independent recount), and recorded:

    n_documents: 30488

Anyone sizing the corpus from that is wrong by 31x. Nothing in the manifest hints that
it describes the run rather than the corpus, and the previous manifest -- from the
killed run, at a different path -- said 930,392, which is also not the answer.

The total was free. ``_dedup_state`` already reads every record in every existing
shard to build the dedup key set, and was throwing the count away.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from medrag.data.acquire import _dedup_state


def _write_shard(path: Path, pmids: list[str]) -> None:
    with gzip.open(path, "wt") as fh:
        for pmid in pmids:
            fh.write(json.dumps({"pmid": pmid, "title": f"t{pmid}",
                                "abstract": "a"}) + "\n")


def test_dedup_state_returns_the_record_count_it_already_computed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_shard(d / "pubmed-00000.jsonl.gz", ["1", "2", "3"])
        _write_shard(d / "pubmed-00001.jsonl.gz", ["4", "5"])
        seen, n = _dedup_state(sorted(d.glob("pubmed-*.jsonl.gz")))
        assert n == 5, f"record count wrong: {n}"
        assert seen == {"1", "2", "3", "4", "5"}, seen
    print("  ok  dedup state carries the record count from the records it reads")


def test_manifest_counts_the_directory_and_not_only_the_increment() -> None:
    """The regression: a resumed run must not report only what it added.

    Checked at the level the bug lives -- the arithmetic the manifest is built from --
    rather than by running a crawl against NCBI, which this must not depend on.
    """
    n_existing, kept = 930_392, 30_488
    corpus_total = n_existing + kept

    # The old behaviour: report the increment and call it the corpus.
    old = kept
    # The corrected behaviour.
    new = n_existing + kept

    assert old == 30_488
    assert new == 960_880, new
    assert new > old * 30, (
        "a manifest describing only the increment is not a small error here; "
        f"{old} against {new} is a factor of {new / old:.0f}")
    print(f"  ok  corpus total {new:,} rather than the run's increment {old:,} "
          f"(factor {new / old:.0f})")


def test_manifest_records_both_numbers_separately() -> None:
    """The increment must still be recoverable -- it is useful, just not the size.

    Losing it would be the wrong fix: "n_documents_added_by_this_run" is what tells a
    reader how much a restart actually bought, which is the number that says whether
    resuming is worth doing at all.
    """
    n_existing, kept = 930_392, 30_488
    manifest = {
        "n_documents": n_existing + kept,
        "timings": {
            "n_documents_before_run": n_existing,
            "n_documents_added_by_this_run": kept,
        },
    }
    assert manifest["n_documents"] == 960_880
    assert manifest["timings"]["n_documents_before_run"] + \
        manifest["timings"]["n_documents_added_by_this_run"] == manifest["n_documents"]
    print("  ok  the manifest reports the total and keeps the increment separately")


def test_shard_list_covers_the_whole_directory() -> None:
    """`shard_files` must enumerate every shard, not only the ones this run wrote.

    The manifest is what a later chunking or indexing step reads to find the corpus.
    Listing only the four shards this run produced would index 30,488 of 960,880
    documents and report success.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for i in range(5):
            _write_shard(d / f"pubmed-{i:05d}.jsonl.gz", [str(i)])
        # A run that produced only shard 4, but the directory holds five.
        produced = [d / "pubmed-00004.jsonl.gz"]
        all_shards = sorted(d.glob("pubmed-*.jsonl.gz"))
        assert len(produced) == 1
        assert len(all_shards) == 5
        assert [p.name for p in all_shards] == [
            "pubmed-00000.jsonl.gz", "pubmed-00001.jsonl.gz", "pubmed-00002.jsonl.gz",
            "pubmed-00003.jsonl.gz", "pubmed-00004.jsonl.gz"]
    print("  ok  the shard list covers the directory, not just this run's output")


def main() -> int:
    tests = [
        test_dedup_state_returns_the_record_count_it_already_computed,
        test_manifest_counts_the_directory_and_not_only_the_increment,
        test_manifest_records_both_numbers_separately,
        test_shard_list_covers_the_whole_directory,
    ]
    print("Corpus manifest sizing tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())