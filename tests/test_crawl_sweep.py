"""The date-partitioned sweep, and the ceiling it lifts.

Why this exists
---------------
The objective puts the corpus crawl at ~20M documents and calls it the long pole.
Measured, the crawl as configured could never get there: ``max_records`` is applied
**per query**, ``DEFAULT_QUERIES["pubmed"]`` has **six** publication-type slices, and the
default cap is 200,000, so the ceiling is 6 x 200,000 = **1.2M raw records** however long
it runs. It reached 1,022,707 and stopped. PubMed holds roughly 37M.

``AcquireConfig.query`` was documented as 'empty means sweep everything', which is the
kind of sentence that reads as a feature and is not one: with ``query=""`` the code
takes ``DEFAULT_QUERIES`` exactly as before. The documented mode did not exist.

Date partitioning is also the only route past NCBI's own ceiling -- esearch will not page
beyond roughly 10,000,000 records for a single query -- so the sweep has to be sliced
anyway.

What these tests hold
---------------------
- **Windows tile the range with no gap and no overlap.** A document falls in exactly one
  slice, which is what lets two nodes shard the slices without both fetching it. The
  first two-node attempt ran identical windows on both machines and returned 99.8%
  overlapping document sets; that is the failure this property prevents.
- **Every month has a real last day.** "2015/04/31" is not a date. A filter naming one
  is rejected or silently narrowed by NCBI, either of which quietly drops records from a
  sweep whose whole purpose is not dropping records.
- **The ceiling is computed, not assumed.** The number that makes 20M reachable is
  slices x cap, and a test asserts it rather than leaving it to arithmetic in prose.
"""

from __future__ import annotations

import re
import sys

sys.path.insert(0, ".")

from medrag.data.acquire import DEFAULT_QUERIES, sweep_queries


def _years(queries: tuple[str, ...]) -> list[str]:
    return [re.search(r'"(\d{4})/', q).group(1) for q in queries]


def test_yearly_windows_tile_the_range() -> None:
    q = sweep_queries("2015/01/01", "2016/12/31", granularity="yearly")
    assert len(q) == 2, q
    assert _years(q) == ["2015", "2016"], _years(q)
    assert q[0] == '("2015/01/01"[dp] : "2015/12/31"[dp])', q[0]
    print("  ok  yearly slices tile the range, one per year, no gap or overlap")


def test_monthly_windows_use_real_last_days() -> None:
    q = sweep_queries("2015/01/01", "2015/12/31", granularity="monthly")
    assert len(q) == 12, len(q)
    assert '"2015/02/28"' in q[1], q[1]          # 2015 February
    assert '"2015/04/30"' in q[3], q[3]          # 2015 April
    # Leap year: 2016 February has 29 days.
    assert '"2016/02/29"' in sweep_queries("2016/01/01", "2016/12/31",
                                          granularity="monthly")[1]
    for bad in ("31", "32"):
        for query in q:
            month = int(re.search(r'"(\d{4})/(\d{2})/', query).group(2))
            if bad == "32":
                continue
            # No April/June/September/November slice may name a 31st.
            if month in (4, 6, 9, 11):
                assert f"/{bad}\"" not in query, (month, query)
    print("  ok  monthly slices name each month's real last day, including leap years")


def test_slices_are_disjoint_so_two_nodes_cannot_double_fetch() -> None:
    """Round-robin sharding only works if a document belongs to one slice."""
    q = sweep_queries("1996/01/01", "2026/12/31", granularity="yearly")
    assert len(q) == 31, len(q)
    assert len(set(q)) == len(q), "duplicate slices would double-fetch under sharding"
    years = _years(q)
    assert years == sorted(years), years
    assert len(set(years)) == len(years), "a year appears in more than one slice"
    # Round-robin over 2 nodes must leave each year on exactly one node.
    c7 = {y for i, y in enumerate(years) if i % 2 == 0}
    c1 = {y for i, y in enumerate(years) if i % 2 == 1}
    assert c7.isdisjoint(c1) and (c7 | c1) == set(years)
    print(f"  ok  {len(q)} disjoint slices; a two-node round-robin splits them cleanly")


def test_the_first_and_last_windows_respect_the_configured_bounds() -> None:
    """Monthly partitioning must not widen the range it was given.

    Partitioning is by month, so without clamping the day-level bounds a sweep of
    2016/02/01..2016/02/28 emits all twelve months of 2016 -- the caller gets records
    outside the range it asked for, and since the ceiling is slices x cap, it also gets
    a ceiling larger than the one it computed.
    """
    q = sweep_queries("2016/02/01", "2016/02/28", granularity="monthly")
    assert len(q) == 1, q
    assert q[0] == '("2016/02/01"[dp] : "2016/02/28"[dp])', q[0]

    r = sweep_queries("2015/11/15", "2016/02/10", granularity="monthly")
    assert len(r) == 4, r
    assert r[0] == '("2015/11/15"[dp] : "2015/11/30"[dp])', r[0]
    assert r[-1] == '("2016/02/01"[dp] : "2016/02/10"[dp])', r[-1]
    print("  ok  the first and last windows are clamped to the configured range")


def test_a_base_filter_is_carried_into_every_slice() -> None:
    base = 'AND ("randomized controlled trial"[pt])'
    q = sweep_queries("2015/01/01", "2016/12/31", base=base)
    assert len(q) == 2
    for query in q:
        assert base.strip() in query, query
    print("  ok  a publication-type filter is preserved in every date slice")


def test_bad_input_is_rejected_rather_than_silently_narrowed() -> None:
    for kwargs in (
        {"granularity": "weekly"},
        {"min_date": "not-a-date"},
    ):
        try:
            if "granularity" in kwargs:
                sweep_queries("2015/01/01", "2016/12/31", **kwargs)
            else:
                sweep_queries(kwargs["min_date"], "2016/12/31")
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid sweep input: {kwargs}")
    try:
        sweep_queries("2020/01/01", "2015/01/01")
    except ValueError:
        pass
    else:
        raise AssertionError("accepted max_date before min_date")
    print("  ok  unknown granularity, unparseable dates and reversed ranges are rejected")


def test_the_sweep_lifts_the_ceiling_the_default_queries_impose() -> None:
    """The number that makes the 20M target reachable, asserted rather than asserted-about.

    The default configuration is bounded by len(DEFAULT_QUERIES) * max_records, which is
    the problem this whole change exists to fix. The sweep multiplies that by the number
    of date windows instead.
    """
    cap = 200_000
    n_default = len(DEFAULT_QUERIES["pubmed"])
    default_ceiling = n_default * cap

    yearly = len(sweep_queries("1950/01/01", "2026/12/31", granularity="yearly"))
    monthly = len(sweep_queries("1950/01/01", "2026/12/31", granularity="monthly"))

    assert n_default == 6, n_default
    assert default_ceiling == 1_200_000, default_ceiling
    assert yearly == 77, yearly
    assert monthly == 924, monthly
    assert yearly * cap > default_ceiling * 10, (yearly, default_ceiling)
    print(f"  ok  ceiling {default_ceiling:,} -> {yearly * cap:,} (yearly) / "
          f"{monthly * cap:,} (monthly), a {yearly * cap / default_ceiling:.0f}x lift")


def main() -> int:
    tests = [
        test_yearly_windows_tile_the_range,
        test_monthly_windows_use_real_last_days,
        test_slices_are_disjoint_so_two_nodes_cannot_double_fetch,
        test_the_first_and_last_windows_respect_the_configured_bounds,
        test_a_base_filter_is_carried_into_every_slice,
        test_bad_input_is_rejected_rather_than_silently_narrowed,
        test_the_sweep_lifts_the_ceiling_the_default_queries_impose,
    ]
    print("Date-partitioned sweep tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())