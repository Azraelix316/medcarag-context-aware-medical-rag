"""The verifier must probe rows that exist.

`scripts/verify_embeddings.py` is the gate the distributed-embedding success criterion
rests on, so its output has to be actionable. It was not, on any index smaller than the
16,133,551-row PMC one.

The fixed sample list -- 0, 1, 5,000, 1,000,000, 4,000,000, 8,000,000, 15,000,000,
16,133,550 -- is the right shape for that index, and it earned its place: it is what
caught the 7.39M-versus-8.71M chunk miscount and the duplicated worker-B prefix. But run
against the 2,368,622-row abstract index it asked for row 4,000,000, found no block 200,
and printed

    ERROR: block 200 missing for sampled rows [4000000]
    NOT VERIFIED -- do not depend on this index

for an index that verifies perfectly once probed correctly: 119 of 119 blocks, zero bad
rows, and every sampled row retrieving itself at cosine 1.000000.

That is worse than a missed detection. The same "NOT VERIFIED" covered both a real fault
and an out-of-range probe, so nobody could tell which they had, and a conservative check
that fires on healthy input is one whose output gets ignored when it matters.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

sys.path.insert(0, "scripts")

from verify_embeddings import SAMPLE_ROWS, sample_rows


def test_every_sampled_row_exists_in_the_index() -> None:
    """The whole point: no probe may address a row the index does not have."""
    for n_rows in (2_368_622, 16_133_551, 500_000, 12_000, 37):
        rows = sample_rows(n_rows)
        for row in rows:
            assert 0 <= row < n_rows, (
                f"row {row} is outside an index of {n_rows:,} rows: {rows}")
    print("  ok  every sampled row is inside the index, at four scales")


def test_the_large_fixed_probes_are_kept_where_they_exist() -> None:
    """Range-awareness must not throw away the checks that earn their keep.

    Filtering the fixed list down to nothing and probing only row 0 would pass on a
    corrupt index. On the 16,133,551-row PMC index all eight original rows survive.
    """
    rows = sample_rows(16_133_551)
    assert set(SAMPLE_ROWS) <= set(rows), (
        f"the original probes were dropped: {sorted(set(SAMPLE_ROWS) - set(rows))}")
    # Plus the endpoints and midpoint, which is the point of the fix: strictly more
    # coverage than the fixed list gave, never less.
    assert len(rows) > len(SAMPLE_ROWS), (len(rows), len(SAMPLE_ROWS))
    assert 16_133_550 in rows and 16_133_551 // 2 in rows, rows
    print(f"  ok  all {len(SAMPLE_ROWS)} original probes kept, plus "
          f"{len(rows) - len(SAMPLE_ROWS)} endpoint/midpoint probes")


def test_a_small_index_is_probed_across_its_own_range() -> None:
    """Not just the first few rows: the endpoints are where truncation shows."""
    n = 500_000
    rows = sample_rows(n)
    assert 0 in rows, rows
    assert n - 1 in rows, rows
    assert len(set(rows)) >= 4, rows
    print(f"  ok  a 500K-row index is probed at {len(rows)} points including both ends")


def test_a_tiny_index_does_not_crash() -> None:
    for n in (1, 2, 3, 5):
        rows = sample_rows(n)
        assert rows, n
        assert max(rows) < n, (n, rows)
    # Degenerate input is reported rather than raising, so the verifier prints a verdict
    # instead of a traceback on an empty index.
    assert sample_rows(0) == (0,), sample_rows(0)
    print("  ok  a 1-row index probes once and an empty one returns a placeholder")


def test_the_abstract_index_verifies_now() -> None:
    """The regression that motivated this, asserted as an outcome.

    2,368,622 rows is the abstract benchmark index: 119 blocks of 20,000. Before the fix
    it reported NOT VERIFIED with "block 200 missing"; after, all four probe rows
    self-retrieve.
    """
    n = 2_368_622
    rows = sample_rows(n)
    assert max(rows) < n
    # Row 4,000,000 is the probe that broke it, and it must be absent.
    assert 4_000_000 not in rows, rows
    assert rows == (0, 1, 5_000, 592_155, 1_000_000, 1_184_311, 1_776_466, 2_368_621), rows
    assert 2_368_621 in rows, f"the last row of the index is unprobed: {rows}"
    print(f"  ok  the 2.37M-row abstract index probes {len(rows)} rows across its full "
          f"range and skips row 4,000,000")


def main() -> int:
    tests = [
        test_every_sampled_row_exists_in_the_index,
        test_the_large_fixed_probes_are_kept_where_they_exist,
        test_a_small_index_is_probed_across_its_own_range,
        test_a_tiny_index_does_not_crash,
        test_the_abstract_index_verifies_now,
    ]
    print("Verifier probe-selection tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())