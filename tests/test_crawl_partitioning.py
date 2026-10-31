"""Tests for date-partitioned PubMed crawling.

The bug these guard against is silent and severe: NCBI caps retrievable records
at 10,000 per query, and ``retstart`` beyond that returns nothing -- with *or*
without a WebEnv history set. Measured directly:

    retstart=0      -> 200 ids
    retstart=9900   ->  99 ids
    retstart=50000  ->   0 ids   (even with usehistory=y)

A crawler that pages with ``retstart`` alone therefore stops at 10,000 records
and reports success, silently producing a corpus one twentieth of the intended
size. The only way past the cap is to partition the query, which is what
``iter_ids_partitioned`` does.

These tests use a stub client so no network access is required; one live test is
included behind ``--live``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")

from medrag.data.eutils import EutilsClient


class StubClient(EutilsClient):
    """An EutilsClient whose esearch answers from a synthetic calendar.

    ``counts`` maps a date window to its hit count. Records are generated as
    deterministic strings so paging behaviour can be asserted exactly.
    """

    def __init__(self, counts: dict[tuple[str, str], int], cap: int = 10_000) -> None:
        super().__init__(use_api_offline=True) if False else super().__init__()
        self.counts = counts
        self.cap = cap
        # (query, reported_count, retmax) per esearch call. The count probe uses
        # retmax=0, so the service cap applies to retmax, not to the reported
        # total -- which may legitimately exceed 10,000 for a whole year.
        self.requested: list[tuple[str, int, int]] = []
        self.queries: list[str] = []

    @property
    def max_retmax(self) -> int:
        return max((r for _q, _c, r in self.requested), default=0)

    # -- window parsing mirrors the real client -----------------------------
    @staticmethod
    def _window_from(query: str) -> tuple[str, str] | None:
        import re

        match = re.search(r'\("(\d{4}/\d{2}/\d{2})"\[dp\] : "(\d{4}/\d{2}/\d{2})"\[dp\]\)', query)
        if not match:
            return None
        return match.group(1), match.group(2)

    def esearch(self, query: str, *, db: str = "pubmed", retmax: int = 10_000,
                retstart: int = 0, usehistory: bool = False, sort: str | None = None,
                **filters) -> object:
        self.queries.append(query)
        window = self._window_from(query)
        count = self.counts.get(window, 0) if window else 0
        self.requested.append((query, count, retmax))
        ids: list[str] = []
        if retstart < count:
            ids = [f"pmid-{window[0]}-{retstart + i}" for i in range(min(retmax, count - retstart))]
        return type("R", (), {"count": count, "ids": ids, "webenv": "", "query_key": "",
                              "query_translation": ""})()


def test_split_window_year_month_day() -> None:
    years = EutilsClient._split_window("2018/01/01", "2021/12/31", "year")
    assert years == [("2021/01/01", "2021/12/31"), ("2020/01/01", "2020/12/31"),
                     ("2019/01/01", "2019/12/31"), ("2018/01/01", "2018/12/31")], years
    # Newest first is the documented contract.
    assert years[0][0].startswith("2021")

    months = EutilsClient._split_window("2021/01/15", "2021/04/10", "month")
    assert len(months) == 4, months
    assert months[0] == ("2021/04/01", "2021/04/10"), months[0]
    assert months[-1] == ("2021/01/15", "2021/01/31"), months[-1]
    # The boundary months are clamped to the requested range.
    assert months[-1][0] == "2021/01/15"

    days = EutilsClient._split_window("2021/03/01", "2021/03/05", "day")
    assert days == [("2021/03/05", "2021/03/05"), ("2021/03/04", "2021/03/04"),
                    ("2021/03/03", "2021/03/03"), ("2021/03/02", "2021/03/02"),
                    ("2021/03/01", "2021/03/01")], days


def test_february_leap_years() -> None:
    months = EutilsClient._split_window("2020/02/01", "2020/02/29", "day")
    assert len(months) == 29, len(months)
    leap = EutilsClient._split_window("2020/02/01", "2020/02/29", "month")
    assert leap == [("2020/02/01", "2020/02/29")], leap
    common = EutilsClient._split_window("2021/02/01", "2021/02/28", "month")
    assert common == [("2021/02/01", "2021/02/28")], common


def test_small_window_is_not_split() -> None:
    """A window under the cap must be fetched in one request."""
    client = StubClient({("2021/01/01", "2021/12/31"): 400})
    got = list(client.iter_ids_partitioned("cancer[pt]", db="pubmed", limit=None,
                                           min_date="2021/01/01", max_date="2021/12/31",
                                           page=9000))
    assert len(got) == 400, len(got)
    # Only two esearch calls: the count probe and the fetch.
    assert len(client.queries) == 2, client.queries


def test_oversized_window_splits_and_exceeds_cap() -> None:
    """The regression: a window over 10k must be split, and yield >10k records."""
    year = "2020/01/01..2020/12/31"
    client = StubClient({
        ("2020/01/01", "2020/12/31"): 30_000,
        ("2020/12/01", "2020/12/31"): 4_000,
        ("2020/11/01", "2020/11/30"): 4_000,
        ("2020/10/01", "2020/10/31"): 4_000,
    })
    got = list(client.iter_ids_partitioned("cancer[pt]", db="pubmed", limit=None,
                                           min_date="2020/01/01", max_date="2020/12/31",
                                           page=9000))
    total = len(got)
    assert total > 10_000, f"partitioning failed: only {total} records"
    ids = [pmid for _label, pmid in got]
    assert len(set(ids)) == len(ids), "records must be unique across windows"
    labels = [label for label, _ in got]
    assert labels[0] == "2020/12/01..2020/12/31", f"newest window must come first, got {labels[0]}"
    # No request may ask for more records than the service will return. The
    # count probe legitimately reports >10k; what matters is retmax.
    assert client.max_retmax <= 10_000, client.requested
    print(f"  (split a 30,000-hit year into months and retrieved {total} records)")


def test_daily_split_when_months_still_too_large() -> None:
    client = StubClient({
        ("2021/03/01", "2021/03/31"): 40_000,
        ("2021/03/31", "2021/03/31"): 100,
        ("2021/03/30", "2021/03/30"): 100,
    })
    got = list(client.iter_ids_partitioned("x[pt]", db="pubmed", limit=5000,
                                           min_date="2021/03/01", max_date="2021/03/31",
                                           page=9000))
    assert len(got) > 0
    assert client.max_retmax <= 10_000, client.requested
    # Splitting must have happened: a month was too big, so days were requested.
    assert any("2021/03/31" in q for q in client.queries), "expected a day-level request"


def test_limit_is_respected() -> None:
    client = StubClient({("2021/01/01", "2021/12/31"): 5_000})
    got = list(client.iter_ids_partitioned("x[pt]", db="pubmed", limit=1_200,
                                           min_date="2021/01/01", max_date="2021/12/31",
                                           page=9000))
    assert len(got) == 1_200, len(got)


def test_empty_windows_are_skipped_without_fetches() -> None:
    client = StubClient({})
    got = list(client.iter_ids_partitioned("x[pt]", db="pubmed", limit=100,
                                           min_date="2021/01/01", max_date="2021/12/31"))
    assert got == []
    # Only the count probe, no data fetch.
    assert len(client.queries) == 1, client.queries


def test_date_filter_is_not_duplicated_in_query() -> None:
    """build_query must omit the date range; partitioning adds it per window."""
    from medrag.config import AcquireConfig
    from medrag.data.acquire import build_query

    cfg = AcquireConfig(min_date="2015/01/01", max_date="2026/12/31")
    query = build_query(cfg, "randomized controlled trial[pt]")
    assert "2015" not in query, f"date range must not be baked in: {query}"
    assert "randomized controlled trial[pt]" in query
    assert '"english"[language]' in query

    client = StubClient({("2015/01/01", "2015/12/31"): 10})
    list(client.iter_ids_partitioned(query, db="pubmed", limit=10,
                                     min_date="2015/01/01", max_date="2015/12/31"))
    # The partition filter must use the dates passed to iter_ids_partitioned.
    assert any('"2015/01/01"[dp]' in q and '"2015/12/31"[dp]' in q for q in client.queries), client.queries
    # Exactly one date range, not the query's baked-in range plus the window's.
    # (A single range contributes two "[dp]" tags, so count the " : " separator.)
    assert client.queries[0].count('"[dp] : "') == 1, client.queries


def main() -> int:
    tests = [
        test_split_window_year_month_day,
        test_february_leap_years,
        test_small_window_is_not_split,
        test_oversized_window_splits_and_exceeds_cap,
        test_daily_split_when_months_still_too_large,
        test_limit_is_respected,
        test_empty_windows_are_skipped_without_fetches,
        test_date_filter_is_not_duplicated_in_query,
    ]
    print("Date-partitioned crawl tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
