"""Tests for reference extraction and graph-ready fields.

Regression guard. The original parser read citations from
``MedlineCitation/CommentsCorrectionsList/Ref``, which is an errata/"cited-by"
list, not a bibliography. That element is empty for essentially every record, so
the ``citations`` field was populated for 0.0% of a 200,000-document sample while
the code and the schema both claimed otherwise -- a field that looked live and
was dead.

PubMed's actual bibliography is ``PubmedData/ReferenceList/Reference``. Fixing
this changes measured behaviour from 0% to era-dependent coverage:

    2019-2026   72% of records, ~33 references each
    2000-2009   20% of records,  ~7 references each

The era dependence is a property of NLM's data, not of the parser, and the
graph code must account for it rather than assume uniform density.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from medrag.data.pubmed_parser import parse_pmid_xml
from medrag.data.schema import Document

WITH_REFERENCES = """<?xml version="1.0"?>
<PubmedArticleSet><PubmedArticle>
 <MedlineCitation Status="MEDLINE">
  <PMID>40000001</PMID>
  <Article PubModel="Print-Electronic">
   <Journal><Title>Test Journal</Title>
    <JournalIssue><Volume>10</Volume><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
   <ArticleTitle>A trial of therapy X versus placebo in adults.</ArticleTitle>
   <Abstract><AbstractText>Patients were randomised. Outcomes improved.</AbstractText></Abstract>
   <AuthorList><Author><LastName>Smith</LastName><Initials>A</Initials></Author></AuthorList>
   <Language>eng</Language>
   <PublicationTypeList><PublicationType>Journal Article</PublicationType>
    <PublicationType>Randomized Controlled Trial</PublicationType></PublicationTypeList>
   <GrantList CompleteYN="Y">
    <Grant><Agency>NIGMS NIH HHS</Agency><GrantID>R35 GM133510</GrantID>
     <Country>United States</Country></Grant>
   </GrantList>
  </Article>
  <CommentsCorrectionsList>
    <Ref RefType="ErratumIn"><PMID>39999999</PMID><Citation>Correction notice.</Citation></Ref>
  </CommentsCorrectionsList>
 </MedlineCitation>
 <PubmedData>
  <ReferenceList>
   <Reference><Citation>Smith 2000</Citation>
    <ArticleIdList><ArticleId IdType="pubmed">11111111</ArticleId></ArticleIdList></Reference>
   <Reference><Citation>Doe 2001</Citation>
    <ArticleIdList><ArticleId IdType="pubmed">22222222</ArticleId></ArticleIdList></Reference>
   <Reference><Citation>No PMID here</Citation>
    <ArticleIdList><ArticleId IdType="doi">10.1/x</ArticleId></ArticleIdList></Reference>
   <Reference><Citation>Duplicate</Citation>
    <ArticleIdList><ArticleId IdType="pubmed">11111111</ArticleId></ArticleIdList></Reference>
  </ReferenceList>
  <ArticleIdList><ArticleId IdType="pubmed">40000001</ArticleId></ArticleIdList>
 </PubmedData>
</PubmedArticle></PubmedArticleSet>
"""

# The failure mode that hid for so long: an erratum list with no bibliography.
# Note the real element is <CommentsCorrections> with a <PMID> child -- not
# <Ref> with a PMID attribute, which is what the original parser looked for.
ONLY_ERRATA = """<?xml version="1.0"?>
<PubmedArticleSet><PubmedArticle>
 <MedlineCitation>
  <PMID>40000002</PMID>
  <Article>
   <Journal><Title>J</Title><JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue></Journal>
   <ArticleTitle>A study with only an erratum record.</ArticleTitle>
   <Abstract><AbstractText>Body text long enough to keep this document.</AbstractText></Abstract>
   <AuthorList><Author><LastName>X</LastName><Initials>Y</Initials></Author></AuthorList>
   <Language>eng</Language>
   <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
  </Article>
  <CommentsCorrectionsList>
   <CommentsCorrections RefType="ErratumFor">
    <RefSource>J Example. 2022;1:1-2.</RefSource>
    <PMID Version="1">38888888</PMID>
   </CommentsCorrections>
  </CommentsCorrectionsList>
 </MedlineCitation>
 <PubmedData><ArticleIdList><ArticleId IdType="pubmed">40000002</ArticleId></ArticleIdList></PubmedData>
</PubmedArticle></PubmedArticleSet>
"""

NO_REFERENCES = """<?xml version="1.0"?>
<PubmedArticleSet><PubmedArticle>
 <MedlineCitation>
  <PMID>40000003</PMID>
  <Article>
   <Journal><Title>J</Title><JournalIssue><PubDate><Year>2001</Year></PubDate></JournalIssue></Journal>
   <ArticleTitle>An older record with no deposited reference list.</ArticleTitle>
   <Abstract><AbstractText>Body text long enough to keep this document.</AbstractText></Abstract>
   <AuthorList><Author><LastName>Z</LastName><Initials>W</Initials></Author></AuthorList>
   <Language>eng</Language>
   <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
  </Article>
 </MedlineCitation>
 <PubmedData><ArticleIdList><ArticleId IdType="pubmed">40000003</ArticleId></ArticleIdList></PubmedData>
</PubmedArticle></PubmedArticleSet>
"""


def test_references_come_from_reference_list() -> None:
    """The core regression: ReferenceList, not CommentsCorrectionsList."""
    docs = parse_pmid_xml(WITH_REFERENCES)
    assert len(docs) == 1
    d = docs[0]
    # Both bibliography entries present...
    assert 11111111 in d.citations, d.citations
    assert 22222222 in d.citations, d.citations
    # The DOI-only reference has no PMID and must be dropped, not coerced.
    assert len(d.citations) == 2, d.citations
    # Duplicates collapsed.
    assert list(d.citations).count(11111111) == 1, d.citations
    print("  ok  citations read from PubmedData/ReferenceList/Reference")


def test_erratum_list_is_not_treated_as_a_bibliography() -> None:
    """CommentsCorrections carries errata, not citations, and must be ignored.

    Measured on 250 RCT records: only 2.8% have a CommentsCorrectionsList, its
    elements are ``<CommentsCorrections RefType="CommentIn|ErratumFor">`` with a
    PMID child, and it never contains citation edges. Folding it in would
    fabricate a graph edge that does not exist.
    """
    docs = parse_pmid_xml(ONLY_ERRATA)
    assert len(docs) == 1
    assert docs[0].citations == (), (
        f"erratum/CommentIn PMIDs must not become citation edges: {docs[0].citations}"
    )
    print("  ok  CommentsCorrections is not mistaken for a bibliography")


def test_missing_reference_list_is_empty_not_an_error() -> None:
    docs = parse_pmid_xml(NO_REFERENCES)
    assert len(docs) == 1
    assert docs[0].citations == (), docs[0].citations
    print("  ok  records without a deposited reference list parse cleanly")


def test_grants_are_captured() -> None:
    docs = parse_pmid_xml(WITH_REFERENCES)
    g = docs[0].grants
    assert g, "grants were never populated before this change"
    assert "NIH" in g[0] and "R35 GM133510" in g[0], g
    print(f"  ok  grants captured: {g}")


def test_citations_survive_serialisation() -> None:
    """The graph builder reads documents back from JSONL, so the field must
    round-trip as a tuple of ints rather than degrading to a list of strings."""
    docs = parse_pmid_xml(WITH_REFERENCES)
    payload = docs[0].to_dict()
    back = Document.from_dict(payload)
    assert back.citations == docs[0].citations, (back.citations, docs[0].citations)
    assert all(isinstance(c, int) for c in back.citations), back.citations
    assert isinstance(payload["citations"], list)
    print("  ok  citations round-trip through JSON as ints")


def test_reference_ids_are_plausible_pmids() -> None:
    """Guard against picking up DOIs or garbage that would corrupt the graph."""
    docs = parse_pmid_xml(WITH_REFERENCES)
    for c in docs[0].citations:
        assert isinstance(c, int)
        assert 1 <= c < 100_000_000, f"implausible PMID {c}"
    print("  ok  every citation id is a plausible PMID")


# --------------------------------------------------------------- PMC / JATS
#
# PMC does not use PubMed's ReferenceList. Before _parse_citations_pmc existed,
# citations were populated for 0.0% of PMC documents -- which meant the co-citation
# edge type could only be built on the abstract-only PubMed subset, exactly the
# corpus that has no section structure to depend on.

PMC_MIXED = """<article>
  <front><article-meta>
    <article-id pub-id-type="pmc">PMC123</article-id>
    <title-group><article-title>A trial</article-title></title-group>
  </article-meta></front>
  <back><ref-list>
    <ref id="cit1"><mixed-citation>Smith J. A thing. J Clin 2020;
      <pub-id pub-id-type="doi">10.1/x</pub-id>
      <pub-id pub-id-type="pmid">111</pub-id></mixed-citation></ref>
    <ref id="cit2"><mixed-citation>Jones A. Another. Lancet 2019;
      <pub-id pub-id-type="pmid">222</pub-id></mixed-citation></ref>
    <ref id="cit3"><mixed-citation>A book chapter with no PMID at all.</mixed-citation></ref>
    <ref id="cit4"><mixed-citation>Duplicate;
      <pub-id pub-id-type="pmid">111</pub-id></mixed-citation></ref>
  </ref-list></back>
</article>"""

PMC_STRUCTURED = """<article>
  <front><article-meta>
    <article-id pub-id-type="pmc">PMC456</article-id>
    <title-group><article-title>Another trial</article-title></title-group>
  </article-meta></front>
  <back><ref-list>
    <ref><element-citation publication-type="journal">
      <article-title>Structured ref</article-title>
      <pub-id pub-id-type="pmid">333</pub-id>
    </element-citation></ref>
    <ref><element-citation><pub-id pub-id-type="pmid-elink">444</pub-id></element-citation></ref>
    <ref><element-citation><pub-id pub-id-type="doi">10.9/z</pub-id></element-citation></ref>
  </ref-list></back>
</article>"""


def test_pmc_mixed_citation_pmids_are_extracted() -> None:
    import xml.etree.ElementTree as ET
    from medrag.data.pubmed_parser import parse_pmc_article

    doc = parse_pmc_article(ET.fromstring(PMC_MIXED))
    assert doc is not None
    assert doc.citations == (111, 222), doc.citations
    print("  ok  PMC mixed-citation PMIDs are parsed")


def test_pmc_refs_without_a_pmid_are_skipped() -> None:
    """Books and non-indexed works cannot be represented in a PMID-keyed graph."""
    import xml.etree.ElementTree as ET
    from medrag.data.pubmed_parser import parse_pmc_article

    doc = parse_pmc_article(ET.fromstring(PMC_MIXED))
    assert doc is not None
    assert 999 not in doc.citations
    assert len(doc.citations) == 2, doc.citations
    print("  ok  references without a PMID are skipped")


def test_pmc_duplicate_pmids_are_deduplicated() -> None:
    import xml.etree.ElementTree as ET
    from medrag.data.pubmed_parser import parse_pmc_article

    doc = parse_pmc_article(ET.fromstring(PMC_MIXED))
    assert doc is not None
    assert doc.citations.count(111) == 1, doc.citations
    print("  ok  a reference cited twice yields one edge")


def test_pmc_structured_citations_are_extracted() -> None:
    """element-citation is the other common shape, and uses pmid-elink too."""
    import xml.etree.ElementTree as ET
    from medrag.data.pubmed_parser import parse_pmc_article

    doc = parse_pmc_article(ET.fromstring(PMC_STRUCTURED))
    assert doc is not None
    assert doc.citations == (333, 444), doc.citations
    print("  ok  PMC element-citation PMIDs are parsed, including pmid-elink")


def test_a_doi_is_not_mistaken_for_a_pmid() -> None:
    import xml.etree.ElementTree as ET
    from medrag.data.pubmed_parser import parse_pmc_article

    doc = parse_pmc_article(ET.fromstring(PMC_STRUCTURED))
    assert doc is not None
    assert all(c > 0 for c in doc.citations)
    assert len(doc.citations) == 2, doc.citations
    print("  ok  DOIs are not collected as PMIDs")


def test_pubmed_and_pmc_parsers_do_not_cross_talk() -> None:
    """Each parser must read its own bibliography and ignore the other's."""
    import xml.etree.ElementTree as ET
    from medrag.data.pubmed_parser import _parse_citations, _parse_citations_pmc

    pubmed = ET.fromstring(
        "<PubmedArticle><PubmedData><ReferenceList>"
        "<Reference><ArticleIdList>"
        "<ArticleId IdType='pubmed'>555</ArticleId></ArticleIdList></Reference>"
        "</ReferenceList></PubmedData></PubmedArticle>")
    pmc = ET.fromstring(PMC_MIXED)
    assert _parse_citations(pubmed) == (555,)
    assert _parse_citations_pmc(pmc) == (111, 222)
    assert _parse_citations(pmc) == (), "the PubMed reader must not see JATS"
    assert _parse_citations_pmc(pubmed) == (), "the JATS reader must not see PubMed"
    print("  ok  the two bibliography parsers do not cross-talk")


def main() -> int:
    tests = [
        test_references_come_from_reference_list,
        test_erratum_list_is_not_treated_as_a_bibliography,
        test_missing_reference_list_is_empty_not_an_error,
        test_grants_are_captured,
        test_citations_survive_serialisation,
        test_reference_ids_are_plausible_pmids,
        test_pmc_mixed_citation_pmids_are_extracted,
        test_pmc_refs_without_a_pmid_are_skipped,
        test_pmc_duplicate_pmids_are_deduplicated,
        test_pmc_structured_citations_are_extracted,
        test_a_doi_is_not_mistaken_for_a_pmid,
        test_pubmed_and_pmc_parsers_do_not_cross_talk,
    ]
    print("Citation extraction tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
