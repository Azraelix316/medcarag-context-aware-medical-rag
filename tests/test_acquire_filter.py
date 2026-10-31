"""Tests for the crawl's keep/drop decision.

This is the single most consequential filter in the project: it decides what enters
the corpus, and a bug here does not crash -- it quietly returns 22% of what was
fetched while every downstream number looks perfectly reasonable.

Two real bugs are pinned here.

1. PMC and PubMed do not share a publication-type vocabulary. Measured on 360 live
   PMC OA RCT records, "research article" is 75% and "review article" 14%, while
   "journal article" does not occur at all. A PubMed-shaped whitelist therefore kept
   22% of live PMC records and discarded the 78% that were ordinary research
   articles.

2. The no-pubtype fallback required a 200-character abstract, which PMC full text
   essentially never has (100% of a live 60-record sample had no abstract, while
   93% carried real body text). That fallback silently discarded the corpus the
   contextual-contamination add-ons were specifically aimed at.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from medrag.config import AcquireConfig
from medrag.data.acquire import (
    DEFAULT_QUERIES,
    _KEEP_PT_PREFIXES_BY_SOURCE,
    _KEEP_PT_PREFIXES_PMC,
    _KEEP_PT_PREFIXES_PUBMED,
    _keep_document,
    _section_chars,
)
from medrag.data.schema import Document, Section

PMC = AcquireConfig(source="pmc")
PUBMED = AcquireConfig(source="pubmed")


def _doc(**kw) -> Document:
    base = {"doc_id": "x", "title": "A trial title that is long enough",
            "source": "pmc", "abstract": ""}
    base.update(kw)
    return Document(**base)  # type: ignore[arg-type]


BIG_DICT_SECTIONS = [
    {"title": "Methods", "kind": "methods", "text": "x" * 3000},
    {"title": "Results", "kind": "results", "text": "y" * 3000},
]


def test_a_typed_record_with_no_text_is_dropped() -> None:
    """A publication type is not evidence of content.

    This is the single most consequential ordering bug in the keep filter, and it was
    invisible for the whole life of the project because both existing PMC tests used
    ``publication_types=None`` -- they exercised the fallback path, never the early
    return that this bug lives in.

    The filter used to read:

        if doc.publication_types:
            return any(pt.startswith(prefixes) for pt in lowered)   # <- returns here

    so for any record carrying a publication type, control left the function *before*
    any content check ran. PMC's "open access" filter marks the PDF as free, not the
    JATS body as present -- probed live across nine eras at 40 records each, only
    86.4% of open-access records carry a usable ``<body>``, and only 45.0% in 1995.
    An all-OA sweep therefore draws roughly one record in seven that is empty: no
    abstract, no sections, no text. ``"Research Article"`` is a perfectly good type
    for an article containing nothing, so every one of them was kept.

    Measured on a live 50,000-record sweep corpus: **65.4% of kept documents had no
    text and no abstract.**

    The same latent bug is present but harmless in the pre-existing corpus, which is
    worth stating because it is why this went unnoticed: ``pmcmerged`` measures 0.1%,
    because its RCT-restricted query happens to correlate with full-text availability.
    A bug can be real, severe in one place and invisible in another, and the second
    fact is not evidence against the first.
    """
    empty = _doc(sections=[], abstract="",
                 publication_types=["Research Article"])
    assert not _keep_document(empty, PMC, require_abstract=False), (
        "a typed record with no text was kept -- this admits ~65% empty records on an "
        "all-OA sweep"
    )
    # And the boundary, so the fix is a floor rather than a blanket rejection.
    stub = _doc(sections=[{"title": "A", "kind": "", "text": "x" * 4999}],
                abstract="", publication_types=["Research Article"])
    assert not _keep_document(stub, PMC, require_abstract=False), (
        "4,999 characters is under the 5,000 floor and must be dropped"
    )
    real = _doc(sections=BIG_DICT_SECTIONS, abstract="",
                publication_types=["Research Article"])
    assert _keep_document(real, PMC, require_abstract=False), (
        "a typed record with real full text must still be kept -- the content check "
        "filters, it does not replace, the publication-type check"
    )
    print("  ok  a publication type does not substitute for content")


def test_pmc_default_query_is_not_a_pubmed_only_filter() -> None:
    """The default PMC query must not contain filters PMC does not implement.

    `has_text_ml[filter]` is a PubMed-side filter. In the PMC database it matches
    nothing -- measured against the live API:

        '"open access"[filter] AND has_text_ml[filter]'   count = 0
        '"has_text_ml"[filter]'                          count = 0
        '"open access"[filter]'                           count = 8,298,989

    and NCBI echoes the term back *untranslated*, so esearch succeeds with count 0 and
    a crawl run against this default acquires nothing and exits clean. No exception, no
    warning, no empty-shard complaint -- just a corpus of zero documents. Every PMC
    config in this repo overrode the default, which is the only reason it survived: the
    bug is invisible to any test that uses a config, and only reachable by the code
    path a new user is most likely to take.

    Pinned here as a static property because the failure is a silent zero, not a crash.
    """
    for source, queries in DEFAULT_QUERIES.items():
        for query in queries:
            assert "has_text_ml" not in query, (
                f"DEFAULT_QUERIES[{source!r}] contains has_text_ml, which resolves to "
                f"zero records in the {source!r} database: {query!r}"
            )
    # And the PMC default is the one thing that does resolve there.
    assert DEFAULT_QUERIES["pmc"] == ('"open access"[filter]',), (
        "the PMC default changed; re-measure its NCBI count before trusting it"
    )


def test_pmc_default_query_is_not_capped_by_publication_type() -> None:
    """The default must not silently cap the corpus at one publication type.

    pmcmerged was acquired with '"open access"[filter] AND "randomized controlled
    trial"[pt]', a query whose NCBI ceiling is 590,762 records. The corpus reached
    555,142 -- 94% of everything that query can ever return -- and stopped growing for
    a reason that had nothing to do with wall-clock time or crawl progress. Reading
    that as "the crawl stalled" would send someone to debug the crawler, which is
    working correctly, when the actual constraint is the query string.

    Measured ceilings, so the ratio is a fact rather than an intuition:

        OA AND randomized controlled trial[pt]     590,762
        OA AND clinical trial[pt]                   919,173
        OA                                          8,298,989   (14.0x the RCT query)
    """
    for query in DEFAULT_QUERIES["pmc"]:
        assert "[pt]" not in query, (
            f"the PMC default restricts publication type and so caps the corpus at "
            f"that type's ceiling: {query!r}"
        )
        assert "randomized controlled trial" not in query.lower(), (
            f"the PMC default is capped at 590,762 records by the RCT publication "
            f"type; use 'open access'[filter] alone (8,298,989 measured): {query!r}"
        )


def test_pmc_and_pubmed_whitelists_are_separate() -> None:
    """The bug's root cause: one list served two incompatible vocabularies."""
    assert "journal article" in _KEEP_PT_PREFIXES_PUBMED
    assert "journal article" not in _KEEP_PT_PREFIXES_PMC, (
        "if PMC ever emits 'journal article' this test should be revisited, "
        "but the two lists must not be silently merged"
    )
    assert _KEEP_PT_PREFIXES_BY_SOURCE["pmc"] is _KEEP_PT_PREFIXES_PMC
    assert _KEEP_PT_PREFIXES_BY_SOURCE["pubmed"] is _KEEP_PT_PREFIXES_PUBMED
    print("  ok  PubMed and PMC whitelists are distinct")


def test_research_article_is_kept_for_pmc() -> None:
    """75% of live PMC OA RCT records. Dropping these cost 78% of the corpus.

    The fixture carries real body text. It did not when this test was written -- it was
    a bare publication type with no sections and no abstract, which is an empty
    record, so the test was asserting that the filter keeps nothing. It passed, and
    passed for a year, while the filter admitted 65.4% empty records, because the
    fixture and the bug agreed that a typed empty document should be kept. A test whose
    fixture encodes the bug will not catch the bug; the inverse case is pinned in
    ``test_a_typed_record_with_no_text_is_dropped``.
    """
    d = _doc(sections=BIG_DICT_SECTIONS, publication_types=["Research Article"])
    assert _keep_document(d, PMC, require_abstract=False), (
        "'research article' is the most common PMC publication type and must be kept"
    )
    print("  ok  'research article' is kept for PMC")


def test_review_article_is_kept_for_pmc() -> None:
    """14% of live PMC records."""
    d = _doc(sections=BIG_DICT_SECTIONS, publication_types=["Review Article"])
    assert _keep_document(d, PMC, require_abstract=False)
    print("  ok  'review article' is kept for PMC")


def test_editorial_and_news_are_dropped() -> None:
    """The fix must not become a keep-everything filter.

    Each fixture carries full body text, so this measures the publication-type filter
    and nothing else. Without it every case below would be dropped by the content
    check, and the test would still be green while testing nothing at all -- the
    mirror image of the fixture bug above.
    """
    for pt in ("editorial", "news", "correction", "retraction", "discussion",
               "other", "abstract", "meeting report", "product review"):
        d = _doc(sections=BIG_DICT_SECTIONS, publication_types=[pt.title()])
        assert not _keep_document(d, PMC, require_abstract=False), pt
    print("  ok  editorials, news and corrections are still dropped")


def test_pmc_full_text_substitutes_for_a_missing_abstract() -> None:
    """PMC records have no abstract but do have body text.

    Measured: 100% of a live 60-record sample had an empty abstract; 93% carried
    more than 5,000 characters of body text.
    """
    d = _doc(sections=BIG_DICT_SECTIONS, publication_types=None)
    assert not d.abstract.strip(), "fixture must have no abstract"
    assert _keep_document(d, PMC, require_abstract=False), (
        "full text with no abstract was discarded -- this is the bug"
    )
    print("  ok  PMC full text is accepted in place of an abstract")


def test_a_stub_of_text_is_still_dropped() -> None:
    """The substitute signal must have a floor, or it keeps empty records."""
    d = _doc(sections=[{"title": "A", "kind": "", "text": "x" * 200}],
             publication_types=None)
    assert not _keep_document(d, PMC, require_abstract=False)
    print("  ok  a 200-character stub with no pubtype is dropped")


def test_section_chars_handles_objects_and_dicts() -> None:
    """Shards store dicts; the parser produces Section objects.

    Reading only the object form returns 0 for anything loaded from disk, which is
    where a re-crawl reads from -- the filter would then drop real documents.
    """
    objs = [Section(title="Methods", kind="methods", text="x" * 3000),
            Section(title="Results", kind="results", text="y" * 3000)]
    assert _section_chars(_doc(sections=objs)) == 6000
    assert _section_chars(_doc(sections=BIG_DICT_SECTIONS)) == 6000
    print("  ok  section character count works for objects and dicts")


def test_pubmed_behaviour_is_unchanged() -> None:
    """The PubMed path must not move; this fix is scoped to PMC."""
    keep = _doc(source="pubmed", abstract="a" * 250)
    drop_short = _doc(source="pubmed", abstract="short")
    drop_empty = _doc(source="pubmed", abstract="")
    assert _keep_document(keep, PUBMED, require_abstract=True)
    assert not _keep_document(drop_short, PUBMED, require_abstract=True)
    assert not _keep_document(drop_empty, PUBMED, require_abstract=True)
    print("  ok  the PubMed keep/drop rule is untouched")


def test_short_titles_are_always_dropped() -> None:
    assert not _keep_document(_doc(title="hi", publication_types=["research article"]),
                              PMC, require_abstract=False)
    print("  ok  a trivially short title is dropped regardless of pubtype")


def test_live_sample_keep_rate_is_reasonable() -> None:
    """A regression guard on the *rate*, using the observed live distribution.

    Before the fix this was 22%. If a future change drops the rate back toward that,
    it has reintroduced a vocabulary mismatch.
    """
    distribution = [("research article", 269), ("review article", 51),
                    ("other", 15), ("editorial", 6), ("systematic review", 5),
                    ("case report", 3), ("brief report", 2), ("methods article", 2),
                    ("retraction", 1), ("correction", 1), ("abstract", 1),
                    ("news", 1), ("discussion", 1), ("product review", 1),
                    ("meeting report", 1)]
    cfg = PMC
    # Real full text on every fixture, so this measures the publication-type
    # whitelist and nothing else. See the fixture note in
    # test_research_article_is_kept_for_pmc.
    kept = sum(n for pt, n in distribution
               if _keep_document(_doc(sections=BIG_DICT_SECTIONS,
                                      publication_types=[pt.title()]), cfg,
                                 require_abstract=False))
    rate = kept / sum(n for _, n in distribution)
    assert rate > 0.85, f"keep rate fell to {rate:.2f} on the live distribution"
    print(f"  ok  keep rate on the live 2022/2018/2005 distribution is {rate:.1%}")


def main() -> int:
    tests = [
        test_pmc_and_pubmed_whitelists_are_separate,
        test_research_article_is_kept_for_pmc,
        test_review_article_is_kept_for_pmc,
        test_editorial_and_news_are_dropped,
        test_pmc_full_text_substitutes_for_a_missing_abstract,
        test_a_stub_of_text_is_still_dropped,
        test_section_chars_handles_objects_and_dicts,
        test_pubmed_behaviour_is_unchanged,
        test_short_titles_are_always_dropped,
        test_live_sample_keep_rate_is_reasonable,
        test_a_typed_record_with_no_text_is_dropped,
        test_pmc_default_query_is_not_a_pubmed_only_filter,
        test_pmc_default_query_is_not_capped_by_publication_type,
    ]
    print("Crawl keep/drop filter tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())