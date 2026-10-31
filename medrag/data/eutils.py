"""NCBI E-utilities client: polite, resumable, rate-limited acquisition.

PubMed and PMC are the backbone of the corpus. NCBI asks unauthenticated
clients for <=3 requests/second and up to 10 with an ``api_key``, so the client
enforces a token-bucket rate limit, uses exponential backoff on 429/5xx, and
never issues concurrent requests. That keeps a multi-day crawl inside the
service's stated limits.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import requests

log = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def parse_ncbi_json(text: str) -> Any:
    """Parse NCBI's JSON, which is not always strictly RFC-8259 compliant.

    E-utilities frequently emits raw control characters (newlines, tabs) inside
    JSON string values -- notably in ``querytranslation``. ``requests.json()``
    and strict ``json.loads`` both reject those with a JSONDecodeError, which
    would abort a multi-hour crawl on the first such response. ``strict=False``
    accepts them, which is what we want for a read-only API response.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text, strict=False)


class RateLimiter:
    """Simple thread-safe token bucket enforcing a requests-per-second cap."""

    def __init__(self, rate: float, burst: int | None = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.capacity = float(burst if burst is not None else max(1, int(rate)))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait = deficit / self.rate
            time.sleep(min(wait, 5.0))


@dataclass
class ESearchResult:
    count: int
    ids: list[str]
    webenv: str = ""
    query_key: str = ""
    query_translation: str = ""


class EutilsClient:
    """Thin, dependency-light wrapper over E-utilities."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        email: str | None = None,
        tool: str = "medrag",
        rate: float = 3.0,
        timeout: int = 120,
        max_retries: int = 5,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or None
        self.email = email or None
        self.tool = tool
        self.timeout = timeout
        self.max_retries = max_retries
        self.limiter = RateLimiter(rate)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self._user_agent()})
        self.n_requests = 0
        self.n_retries = 0
        self.n_bytes = 0

    def _user_agent(self) -> str:
        bits = [self.tool]
        if self.email:
            bits.append(f"contact={self.email}")
        return "medrag/0.1 (" + "; ".join(bits) + ")"

    def _params(self, extra: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {"tool": self.tool}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        params.update(extra)
        return params

    def _get(self, endpoint: str, params: dict[str, Any], *, stream: bool = False) -> requests.Response:
        url = f"{EUTILS_BASE}/{endpoint}"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            try:
                resp = self.session.get(url, params=self._params(params), timeout=self.timeout, stream=stream)
                self.n_requests += 1
                if resp.status_code == 429 or resp.status_code >= 500:
                    # Retry-After wins when present; otherwise exponential backoff.
                    delay = float(resp.headers.get("Retry-After", 2.0**attempt))
                    log.warning("%s -> HTTP %d, retry in %.1fs", endpoint, resp.status_code, delay)
                    self.n_retries += 1
                    time.sleep(min(delay, 60.0))
                    continue
                resp.raise_for_status()
                self.n_bytes += len(resp.content or b"")
                return resp
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                self.n_retries += 1
                if attempt == self.max_retries:
                    break
                time.sleep(min(2.0**attempt, 60.0))
        raise RuntimeError(f"E-utilities {endpoint} failed after {self.max_retries + 1} attempts") from last_exc

    # ------------------------------------------------------------------ APIs
    def esearch(
        self,
        query: str,
        *,
        db: str = "pubmed",
        retmax: int = 10_000,
        retstart: int = 0,
        usehistory: bool = True,
        sort: str = "pub_date",
        **filters: str,
    ) -> ESearchResult:
        params: dict[str, Any] = {
            "db": db,
            "term": query,
            "retmax": retmax,
            "retstart": retstart,
            "retmode": "json",
            "sort": sort,
        }
        if usehistory:
            params["usehistory"] = "y"
        params.update(filters)
        payload = parse_ncbi_json(self._get("esearch.fcgi", params).text)["esearchresult"]
        return ESearchResult(
            count=int(payload.get("count", 0)),
            ids=list(payload.get("idlist", [])),
            webenv=payload.get("webenv", ""),
            query_key=payload.get("querykey", ""),
            query_translation=payload.get("querytranslation", ""),
        )

    # Maximum retrievable records per query. NCBI caps retrievable records at
    # 10,000 per query and does not raise it for WebEnv history sets -- measured
    # directly: retstart=9900 returns 99 ids, retstart=50000 returns 0. Any crawl
    # of more than 10k records must therefore partition the query, which is what
    # :meth:`iter_ids_partitioned` does via publication-date windows.
    MAX_RECORDS_PER_QUERY = 10_000

    def _date_window(self, db: str, lo: str, hi: str, field: str) -> str:
        return f'("{lo}"[{field}] : "{hi}"[{field}])'

    @staticmethod
    def _split_window(lo: str, hi: str, granularity: str) -> list[tuple[str, str]]:
        """Split an inclusive ``YYYY/MM/DD`` window at the requested granularity.

    Sub-windows come back NEWEST FIRST, and the caller pushes them onto the
    front of the queue, so a crawl walks from the present backwards.
    """
        def parts(value: str) -> tuple[int, int, int]:
            nums = value.split("/")
            return (int(nums[0]), int(nums[1]) if len(nums) > 1 else 1, int(nums[2]) if len(nums) > 2 else 1)

        y0, m0, d0 = parts(lo)
        y1, m1, d1 = parts(hi)
        out: list[tuple[str, str]] = []
        if granularity == "year":
            for year in range(y1, y0 - 1, -1):
                start = lo if year == y0 else f"{year}/01/01"
                end = hi if year == y1 else f"{year}/12/31"
                out.append((start, end))
        elif granularity == "month":
            spans: list[tuple[int, int]] = []
            year, month = y0, m0
            while (year, month) <= (y1, m1):
                spans.append((year, month))
                month += 1
                if month > 12:
                    month = 1
                    year += 1
            for year, month in reversed(spans):
                if month in (4, 6, 9, 11):
                    last = 30
                elif month == 2:
                    last = 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
                else:
                    last = 31
                start = lo if (year, month) == (y0, m0) else f"{year}/{month:02d}/01"
                end = hi if (year, month) == (y1, m1) else f"{year}/{month:02d}/{last:02d}"
                out.append((start, end))
        else:  # "day"
            import datetime

            start_date = datetime.date(y0, m0, d0)
            end_date = datetime.date(y1, m1, d1)
            step = datetime.timedelta(days=1)
            dates: list[datetime.date] = []
            while start_date <= end_date:
                dates.append(start_date)
                start_date += step
            for day in reversed(dates):
                out.append((day.strftime("%Y/%m/%d"), day.strftime("%Y/%m/%d")))
        return out

    def iter_ids_partitioned(
        self,
        query: str,
        *,
        db: str = "pubmed",
        limit: int | None = None,
        min_date: str = "1900/01/01",
        max_date: str = "2100/12/31",
        date_field: str | None = None,
        page: int = 9_000,
        granularity: str = "year",
        sort: str | None = None,
        **filters: str,
    ) -> Iterator[tuple[str, str]]:
        """Yield ``(window_label, pmid)`` pairs, never asking for >10k at once.

        The query is partitioned by publication date and each window is
        recursively narrowed (year -> month -> day) when it still exceeds the
        per-query cap. Windows are walked NEWEST FIRST, which biases the corpus
        toward recent evidence -- the more useful end of the literature for a
        clinical RAG resource, and the end that guideline-style queries target.

        This is the only way to retrieve more than 10,000 records for a single
        PubMed query, and it keeps every esearch well inside the service limits.
        """
        if date_field is None:
            date_field = "dp" if db == "pubmed" else "edat"

        remaining = limit
        queue: list[tuple[str, str, str]] = [(granularity, min_date, max_date)]
        while queue:
            gran, lo, hi = queue.pop(0)
            window_query = f"({query}) AND {self._date_window(db, lo, hi, date_field)}"
            kwargs: dict[str, Any] = dict(filters)
            if sort:
                kwargs["sort"] = sort
            head = self.esearch(window_query, db=db, retmax=0, **kwargs)
            if head.count <= 0:
                continue
            label = f"{lo}..{hi}"
            if head.count > self.MAX_RECORDS_PER_QUERY and gran != "day":
                # Too many for one request: split and retry, newest first.
                finer = {"year": "month", "month": "day"}.get(gran)
                if finer:
                    log.info("partition %s has %d hits; splitting to %s", label, head.count, finer)
                    # The queue is drained FIFO from the front, so the
                    # newest-first sub-window list must be inserted back-to-front
                    # for the queue itself to come out newest-first.
                    for sub_lo, sub_hi in reversed(self._split_window(lo, hi, finer)):
                        queue.insert(0, (finer, sub_lo, sub_hi))
                    continue
            take = head.count if remaining is None else min(head.count, remaining)
            take = min(take, self.MAX_RECORDS_PER_QUERY)
            if take <= 0:
                continue
            page_kwargs: dict[str, Any] = dict(kwargs)
            if sort:
                page_kwargs["sort"] = sort
            got = self.esearch(
                window_query, db=db, retmax=take, retstart=0, **page_kwargs
            )
            for pmid in got.ids:
                yield label, pmid
            if remaining is not None:
                remaining -= len(got.ids)
                if remaining <= 0:
                    return

    def esearch_iter(
        self,
        query: str,
        *,
        db: str = "pubmed",
        batch: int = 10_000,
        limit: int | None = None,
        usehistory: bool = False,
        sort: str = "pub_date",
        **filters: str,
    ) -> Iterator[str]:
        """Yield PMIDs in batches. Capped at 10,000 by the service.

        Prefer :meth:`iter_ids_partitioned` for anything larger than a single
        page; this method exists for small queries and tests.
        """
        head = self.esearch(query, db=db, retmax=0, usehistory=usehistory, sort=sort, **filters)
        total = head.count if limit is None else min(head.count, limit)
        total = min(total, self.MAX_RECORDS_PER_QUERY)
        if usehistory and head.webenv and head.query_key:
            retstart = 0
            while retstart < total:
                page = self.esearch(
                    query,
                    db=db,
                    retmax=min(batch, total - retstart),
                    retstart=retstart,
                    usehistory=usehistory,
                    sort=sort,
                    **filters,
                    WebEnv=head.webenv,
                    query_key=head.query_key,
                )
                if not page.ids:
                    break
                yield from page.ids
                retstart += len(page.ids)
            return
        # No history: page with retstart alone.
        retstart = 0
        while retstart < total:
            page = self.esearch(
                query, db=db, retmax=min(batch, total - retstart), retstart=retstart,
                usehistory=False, sort=sort, **filters,
            )
            if not page.ids:
                break
            yield from page.ids
            retstart += len(page.ids)

    def efetch(
        self,
        ids: list[str],
        *,
        db: str = "pubmed",
        retmode: str = "xml",
        **extra: str,
    ) -> str:
        """Fetch XML for up to ``ids``. Large batches are chunked internally."""
        if not ids:
            return ""
        out: list[str] = []
        # GET URLs get long fast; keep batches modest so we stay well under limits.
        step = 200
        for i in range(0, len(ids), step):
            params: dict[str, Any] = {"db": db, "id": ",".join(ids[i : i + step]), "retmode": retmode}
            params.update(extra)
            out.append(self._get("efetch.fcgi", params).text)
        return "".join(out)

    def esummary(self, ids: list[str], *, db: str = "pubmed") -> dict[str, Any]:
        if not ids:
            return {}
        params = {"db": db, "id": ",".join(ids[:500]), "retmode": "json"}
        return parse_ncbi_json(self._get("esummary.fcgi", params).text).get("result", {})

    def health_check(self) -> bool:
        """Confirm E-utilities is reachable and responding."""
        try:
            resp = self._get("einfo.fcgi", {"retmode": "json"})
            return resp.status_code == 200
        except Exception as exc:  # pragma: no cover - network dependent
            log.error("E-utilities health check failed: %s", exc)
            return False

    def build_query(self, base: str = "", filters: tuple[str, ...] = (), **kw: str) -> str:
        """Assemble an E-utilities query string from reusable fragments."""
        parts = [base.strip()] if base.strip() else []
        parts += [f.strip() for f in filters if f and f.strip()]
        parts += [f"{k}:{v}" for k, v in kw.items() if v]
        return " AND ".join(f"({p})" if " " in p else p for p in parts)

    def url_for(self, endpoint: str, **params: str) -> str:
        return f"{EUTILS_BASE}/{endpoint}?{urlencode(self._params(params))}"
