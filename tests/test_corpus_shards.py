"""Tests that read paths do not silently crawl.

The bug this pins
-----------------
``corpus_shards`` used to call ``crawl_corpus`` when it found no shards. Every
read path goes through it -- ``chunk``, ``build-index``, the dependency-graph and
co-citation stages -- so "no shards found" silently became "start crawling".

It fired here. The merged PMC corpus holds 113 shards named
``pmcoa2_c7__pmc-00000.jsonl.gz`` and so on, while the glob looks for
``pmc-*.jsonl.gz``. Nothing matched, so ``medrag build-index`` began a fresh crawl
of the entire PMC Open Access corpus. It wrote 30,000 documents into six new
``pmc-*.jsonl.gz`` shards in the corpus directory between 03:18 and 03:30, spent
the API budget, and would have run for hours. Nothing errored. The only symptom was
a build that never finished.

Two properties are now enforced: a read path raises instead of crawling, and the
error names the mismatch so the fix is obvious.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from medrag.config import AcquireConfig, PipelineConfig
from medrag.data.acquire import corpus_shards


def _cfg(root: Path) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.paths.data_root = root
    cfg.acquire = AcquireConfig(source="pmc")
    return cfg._finalize()  # resolves corpus_dir under data_root


def _shard(directory: Path, name: str, n: int = 1) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with gzip.open(path, "wt") as fh:
        for i in range(n):
            fh.write(json.dumps({"doc_id": f"pmcid:PMC{i}", "title": "t"}) + "\n")
    return path


def test_missing_shards_raise_instead_of_crawling() -> None:
    """The whole point: a read must not become a multi-hour crawl."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "corpus" / "pmc").mkdir(parents=True)
        try:
            corpus_shards(_cfg(root))
        except FileNotFoundError as exc:
            assert "acquire" in str(exc), exc
        else:
            raise AssertionError(
                "corpus_shards() must raise when no shards exist, not crawl"
            )
        print("  ok  a missing corpus raises rather than crawling")


def test_the_error_names_the_naming_mismatch() -> None:
    """The failure that actually happened: shards present, names unexpected."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _shard(root / "corpus" / "pmc", "pmcoa2_c7__pmc-00000.jsonl.gz", 2)
        try:
            corpus_shards(_cfg(root))
        except FileNotFoundError as exc:
            msg = str(exc)
            assert "pmcoa2_c7__pmc-00000.jsonl.gz" in msg, (
                f"the error must name the file that is actually there: {msg}"
            )
            assert "pmc-*.jsonl.gz" in msg, msg
        else:
            raise AssertionError("expected FileNotFoundError")
        print("  ok  the error names both the expected pattern and the real file")


def test_correctly_named_shards_are_found() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _shard(root / "corpus" / "pmc", "pmc-00000.jsonl.gz")
        _shard(root / "corpus" / "pmc", "pmc-00001.jsonl.gz")
        _shard(root / "corpus" / "pmc", "other-00000.jsonl.gz")
        found = corpus_shards(_cfg(root))
        assert len(found) == 2, [p.name for p in found]
        assert all(p.name.startswith("pmc-") for p in found)
        print("  ok  correctly named shards are found and foreign names ignored")


def test_crawl_if_missing_is_opt_in() -> None:
    """The pipeline still needs crawl-then-read; it must ask for it explicitly."""
    import inspect

    sig = inspect.signature(corpus_shards)
    assert sig.parameters["crawl_if_missing"].default is False
    print("  ok  crawling from a read path is opt-in")


def main() -> int:
    tests = [
        test_missing_shards_raise_instead_of_crawling,
        test_the_error_names_the_naming_mismatch,
        test_correctly_named_shards_are_found,
        test_crawl_if_missing_is_opt_in,
    ]
    print("Corpus shard resolution tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())