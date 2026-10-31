"""PubMed/PMC XML -> :class:`Document` parsing.

The parser is deliberately tolerant: PubMed's XML is produced by ~10k journals
with inconsistent markup, so every field is optional and failures degrade to
empty values rather than dropping the record. Structured abstracts (with
labelled BACKGROUND/METHODS/RESULTS sections) are preserved as ``Section``
objects, which the chunker later uses for structure-aware splitting.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from xml.etree import ElementTree as ET

from .schema import Author, Document, MeshTerm, Section

log = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")
# Inline markup such as <sup>, <i>, or <sub> is often used purely as a visual
# separator with no surrounding whitespace, e.g. "...in the control
# group.<sup>a</sup>Conclusions". Converting a tag to "" would weld the two words
# together, so tags become a space and then get collapsed.
_SPLIT_AFTER_PUNCT = re.compile(r"([.!?:;])([A-Z][a-z])")

# Section-label -> canonical kind, used for chunk provenance.
_SECTION_KINDS = {
    "background": "background",
    "introduction": "background",
    "context": "background",
    "objective": "background",
    "purpose": "background",
    "aim": "background",
    "aims": "background",
    "objectives": "background",
    "rationale": "background",
    "methods": "methods",
    "method": "methods",
    "materials and methods": "methods",
    "patients and methods": "methods",
    "design": "methods",
    "data and methods": "methods",
    "study design": "methods",
    "results": "results",
    "findings": "results",
    "result": "results",
    "conclusions": "conclusions",
    "conclusion": "conclusions",
    "interpretation": "conclusions",
    "discussion": "discussion",
    "limitations": "discussion",
    "references": "references",
}

_ABSTRACT_LABELS = tuple(_SECTION_KINDS) + (
    "important negative results",
    "significance",
    "study selection",
    "data extraction",
    "main outcomes",
    "research design",
    "abstract",
    "summary",
    "problem",
)


def clean_text(value: str | None) -> str:
    """Collapse XML whitespace and strip residual markup."""
    if not value:
        return ""
    # Tags become spaces: inline markup is often the only visual separator
    # between two adjacent words.
    text = _TAG_RE.sub(" ", value)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&apos;", "'")
    # Restore a space where punctuation is directly followed by a capitalised
    # word, which indicates a lost element boundary.
    for _ in range(3):
        patched = _SPLIT_AFTER_PUNCT.sub(r"\1 \2", text)
        if patched == text:
            break
        text = patched
    text = _WS_RE.sub(" ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def _text_of(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return clean_text("".join(node.itertext()))


def _sub(node: ET.Element | None, path: str) -> ET.Element | None:
    if node is None:
        return None
    return node.find(path)


def _first_text(node: ET.Element | None, *paths: str) -> str:
    for p in paths:
        found = _sub(node, p)
        if found is not None:
            txt = _text_of(found)
            if txt:
                return txt
    return ""


def _year_from(value: str) -> int:
    match = re.search(r"(1[6-9]\d{2}|20\d{2})", value or "")
    return int(match.group(1)) if match else 0


def section_kind(title: str) -> str:
    """Map a raw section heading to a small closed vocabulary.

    Unrecognised headings return "" rather than the raw text, so downstream
    aggregation over ``kind`` stays meaningful.
    """
    key = (title or "").strip().lower().rstrip(":")
    if not key:
        return ""
    if key in _SECTION_KINDS:
        return _SECTION_KINDS[key]
    # "Methods." / "METHODS:" style decorations.
    stripped = key.rstrip(". ")
    if stripped in _SECTION_KINDS:
        return _SECTION_KINDS[stripped]
    for label, kind in _SECTION_KINDS.items():
        if label in key and len(key) <= len(label) + 20:
            return kind
    return ""


def _parse_authors(article: ET.Element) -> list[Author]:
    authors: list[Author] = []
    author_list = _sub(article, "MedlineCitation/Article/AuthorList")
    if author_list is None:
        return authors
    for node in author_list.findall("Author"):
        collective = _first_text(node, "CollectiveName")
        if collective:
            authors.append(Author(collective=collective))
            continue
        family = _first_text(node, "LastName")
        initials = _first_text(node, "Initials")
        if not family and not initials:
            suffix = _first_text(node, "Suffix")
            if suffix:
                authors.append(Author(collective=suffix))
            continue
        authors.append(Author(family=family, initials=initials))
    return authors


def _parse_mesh(article: ET.Element) -> list[MeshTerm]:
    terms: list[MeshTerm] = []
    mesh_listing = _sub(article, "MedlineCitation/MeshHeadingList")
    if mesh_listing is None:
        return terms
    for node in mesh_listing.findall("MeshHeading"):
        descriptor = _first_text(node, "DescriptorName")
        if not descriptor:
            continue
        qualifiers = tuple(
            _text_of(q) for q in node.findall("QualifierName") if _text_of(q)
        )
        terms.append(
            MeshTerm(
                descriptor=descriptor,
                ui=node.find("DescriptorName").get("UI", "") if node.find("DescriptorName") is not None else "",
                major=node.find("DescriptorName").get("MajorTopicYN", "N") == "Y"
                if node.find("DescriptorName") is not None
                else False,
                qualifiers=qualifiers,
            )
        )
    return terms


def _parse_structured_abstract(abstract: ET.Element) -> tuple[str, list[Section]]:
    """Return (flat abstract text, labelled sections).

    Unlabelled abstracts collapse to a single section so the chunker still sees
    one consistent unit.
    """
    labelled: list[Section] = []
    for part in abstract.findall("AbstractText"):
        label = part.get("Label") or part.get("NlmCategory") or ""
        text = _text_of(part)
        if not text:
            continue
        kind = section_kind(label) if label else ""
        if label and kind not in ("body",):
            labelled.append(Section(title=label.strip().title(), text=text, level=0, kind=kind))
        else:
            labelled.append(Section(title="", text=text, level=0, kind=""))
    if not labelled:
        return "", []
    flat = "\n\n".join(f"{s.title}: {s.text}" if s.title else s.text for s in labelled)
    return flat, labelled


def _parse_plain_abstract(abstract: ET.Element) -> tuple[str, list[Section]]:
    """Parse an abstract whose ``AbstractText`` parts carry no labels.

    A record may have several ``AbstractText`` children with no ``Label``. Using
    ``itertext()`` on the whole element concatenates them with no separator, which
    welds words together across part boundaries ("control group.Conclusions").
    Each part is therefore joined explicitly.
    """
    parts = abstract.findall("AbstractText")
    if len(parts) > 1:
        texts = [clean_text(_text_of(p)) for p in parts]
        texts = [t for t in texts if t]
        if texts:
            return "\n\n".join(texts), [Section(title="Abstract", text="\n\n".join(texts),
                                                level=0, kind="background")]
    text = _text_of(abstract)
    if not text:
        return "", []
    return text, [Section(title="Abstract", text=text, level=0, kind="background")]


def _parse_keywords(article: ET.Element) -> tuple[str, ...]:
    kws: list[str] = []
    kw_list = _sub(article, "MedlineCitation/KeywordList")
    if kw_list is not None:
        for node in kw_list.findall("Keyword"):
            txt = _text_of(node)
            if txt:
                kws.append(txt)
    return tuple(dict.fromkeys(kws))


def _parse_grants(article: ET.Element) -> tuple[str, ...]:
    """Funding agency + grant id, e.g. 'NIH R01CA123456'.

    Grant funding is a useful bias signal (industry-sponsored work behaves
    differently from publicly funded work), and it is present on ~37% of records.
    """
    out: list[str] = []
    for grant in article.findall("MedlineCitation/Article/GrantList/Grant"):
        agency = _text_of(grant.find("Agency"))
        grant_id = _text_of(grant.find("GrantID"))
        country = _text_of(grant.find("Country"))
        if agency and grant_id:
            out.append(f"{agency} {grant_id}" + (f" ({country})" if country else ""))
        elif agency:
            out.append(agency)
        elif grant_id:
            out.append(grant_id)
    return tuple(dict.fromkeys(out))


def _parse_chemicals(article: ET.Element) -> tuple[str, ...]:
    chem_list = _sub(article, "MedlineCitation/ChemicalList")
    if chem_list is None:
        return ()
    names = (
        _first_text(node, "NameOfSubstance") or _text_of(node.find("RegistryNumber"))
        for node in chem_list.findall("Chemical")
    )
    return tuple(dict.fromkeys(name for name in names if name))


def _parse_citations(article: ET.Element) -> tuple[int, ...]:
    """Extract outbound reference PMIDs from PubMed's deposited bibliography.

    The bibliography is ``PubmedData/ReferenceList/Reference``, with each
    reference carrying ``ArticleIdList/ArticleId[@IdType='pubmed']``.

    Two traps this function previously fell into, both of which made the field
    silently empty (0.0% coverage over a 200k-document sample) while the schema
    implied it was populated:

    * ``MedlineCitation/CommentsCorrectionsList`` is not a bibliography. It holds
      ``<CommentsCorrections RefType="CommentIn|ErratumFor|RefIn">`` elements
      with a ``<PMID>`` *child*, not ``<Ref>`` elements with a ``PMID``
      *attribute*. Measured on 250 RCT records: only 2.8% have the list at all,
      and it is absent of citation edges.
    * Coverage of the real ``ReferenceList`` is era-dependent, because NLM only
      deposits bibliographies for a subset of records: ~72% of 2019-2026 RCTs
      (33 refs each) but ~20% of 2000-2009 RCTs (7 refs each). Graph code must
      treat citation density as a biased signal rather than a uniform one.
    """
    out: list[int] = []
    for ref in article.findall("PubmedData/ReferenceList/Reference"):
        art_id = ref.find('ArticleIdList/ArticleId[@IdType="pubmed"]')
        raw = ((art_id.text if art_id is not None else "") or "").strip()
        if not raw:
            continue
        try:
            out.append(int(raw))
        except ValueError:
            continue
    return tuple(dict.fromkeys(out))


def parse_pubmed_article(node: ET.Element, taxonomy: Any = None) -> Document | None:
    """Parse one ``<PubmedArticle>`` node into a Document."""
    pmid = _first_text(node, "MedlineCitation/PMID")
    title = _first_text(node, "MedlineCitation/Article/ArticleTitle")
    abstract_el = _sub(node, "MedlineCitation/Article/Abstract")

    if abstract_el is not None:
        if len(abstract_el.findall("AbstractText")) > 1:
            abstract, sections = _parse_structured_abstract(abstract_el)
        else:
            abstract, sections = _parse_plain_abstract(abstract_el)
    else:
        abstract, sections = "", []

    pub_types = tuple(
        t for t in (
            _text_of(n) for n in node.findall("MedlineCitation/Article/PublicationTypeList/PublicationType")
        ) if t
    )
    languages = [
        _text_of(n) for n in node.findall("MedlineCitation/Article/Language")
    ]
    language = next((lang.lower() for lang in languages if lang), "english")

    pub_date = _first_text(
        node,
        "MedlineCitation/Article/Journal/JournalIssue/PubDate/Year",
        "MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate",
        "MedlineCitation/Article/ArticleDate/Year",
        "MedlineCitation/Article/ArticleDate/MedlineDate",
    )
    # Prefer the electronic ArticleDate over the issue PubDate: a paper published
    # online in Dec 2026 often carries a Jan 2027 issue date, and a date filter
    # keyed on the issue date will wrongly exclude it.
    article_date = _first_text(
        node,
        "MedlineCitation/Article/ArticleDate/Year",
        "MedlineCitation/Article/ArticleDate/MedlineDate",
    )
    year = _year_from(article_date) or _year_from(pub_date) or _year_from(
        _first_text(node, "PubmedData/History/PubMedPubDate[@PubStatus='entrez']/Year")
    )

    mesh = _parse_mesh(node)
    citations = _parse_citations(node)

    doc = Document(
        doc_id=f"pmid:{pmid}" if pmid else f"doi:{_first_text(node, 'PubmedData/ArticleIdList/ArticleId[@IdType=\"doi\"]')}",
        source="pubmed",
        pmid=pmid,
        pmcid=_first_text(node, "PubmedData/ArticleIdList/ArticleId[@IdType='pmc']"),
        doi=_first_text(node, "PubmedData/ArticleIdList/ArticleId[@IdType='doi']"),
        title=title,
        abstract=abstract,
        sections=sections,
        authors=_parse_authors(node),
        journal=_first_text(node, "MedlineCitation/Article/Journal/Title")
        or _first_text(node, "MedlineCitation/Article/Journal/ISOAbbreviation"),
        year=year,
        publication_date=pub_date,
        volume=_first_text(node, "MedlineCitation/Article/Journal/JournalIssue/Volume"),
        issue=_first_text(node, "MedlineCitation/Article/Journal/JournalIssue/Issue"),
        pages=_first_text(node, "MedlineCitation/Article/Pagination/MedlinePgn"),
        issn=_first_text(node, "MedlineCitation/Article/Journal/ISSN"),
        language=language,
        publication_types=pub_types,
        keywords=_parse_keywords(node),
        mesh_terms=mesh,
        chemicals=_parse_chemicals(node),
        citations=citations,
        grants=_parse_grants(node),
        categories=derive_categories(mesh, pub_types, taxonomy),
    )
    return doc


def derive_categories(
    mesh: list[MeshTerm],
    pub_types: tuple[str, ...],
    taxonomy: Any = None,
) -> tuple[str, ...]:
    """Map MeSH descriptors onto coarse clinical specialty facets.

    The tree position of a descriptor cannot be inferred from its UI: every
    descriptor UI starts with ``D`` regardless of whether it is a disease
    (``D003924`` -> tree C19) or a drug (``D008968`` -> tree D02). When a
    :class:`~medrag.data.mesh_taxonomy.MeshTaxonomy` is supplied its resolved
    tree roots are authoritative; otherwise a label-pattern fallback is used.

    Publication-type facets (rct, evidence-synthesis, case-report) are added
    independently of taxonomy availability.
    """
    cats: list[str] = []
    for term in mesh:
        if taxonomy is not None:
            name = taxonomy.category(term.ui, term.descriptor)
        else:
            from .mesh_taxonomy import TREE_CATEGORIES, MeshTaxonomy

            root = MeshTaxonomy.infer_root(term.descriptor)
            name = TREE_CATEGORIES.get(root, "")
        if name and name not in cats:
            cats.append(name)

    if any("Randomized Controlled Trial" in t for t in pub_types):
        cats.append("rct")
    if any("Systematic Review" in t or "Meta-Analysis" in t for t in pub_types):
        cats.append("evidence-synthesis")
    if any("Case Reports" in t for t in pub_types):
        cats.append("case-report")
    return tuple(cats)


def _parse_citations_pmc(article: ET.Element) -> tuple[int, ...]:
    """Extract outbound reference PMIDs from a PMC JATS bibliography.

    PMC does not use PubMed's ``PubmedData/ReferenceList/Reference``. A JATS article
    carries ``back/ref-list/ref``, and the identifier lives inside the citation
    element as ``<pub-id pub-id-type="pmid">`` rather than as a ``<PMID>`` attribute.

    This matters more than a missing field. Measured on 25 live 2023 PMC OA RCT
    articles: 1,293 references, of which **1,165 (90.1%) carry a PMID**, and 25/25
    articles have a ``<ref-list>``. Before this parser, ``citations`` was populated
    for 0.0% of PMC documents, so the co-citation graph -- one of the three edge
    types in the section-dependency graph -- could not be built on the full-text
    corpus at all. It could only be built on the abstract-only PubMed subset, which
    is exactly the corpus that has no section structure.

    Two shapes occur and both are handled: ``<mixed-citation>`` (older free-form
    markup) and ``<element-citation>`` (structured). Both hold the PMID as a
    ``<pub-id>`` child, so one lookup covers them. ``pub-id-type="pmid-elink"`` and
    ``"pmid-print"`` are accepted as well, since NLM emits them for some records.

    References without a PMID (books, reports, anything not MEDLINE-indexed) are
    skipped: the graph is keyed on PubMed ids and cannot represent them.
    """
    out: list[int] = []
    for ref in article.findall("back/ref-list/ref"):
        for pid in ref.iter("pub-id"):
            kind = (pid.get("pub-id-type") or "").strip().lower()
            if kind not in {"pmid", "pmid-elink", "pmid-print"}:
                continue
            raw = (pid.text or "").strip()
            if not raw.isdigit():
                continue
            out.append(int(raw))
            break
    return tuple(dict.fromkeys(out))


def parse_pmc_article(node: ET.Element, taxonomy: Any = None) -> Document | None:
    """Parse a ``<article>`` from PMC JATS into a Document."""
    meta = _sub(node, "front/article-meta")
    if meta is None:
        return None
    pmcid = _first_text(meta, "article-id[@pub-id-type='pmc']")
    if not pmcid:
        pmcid = _first_text(meta, "article-id[@pub-id-type='pmcid']")
    pmid = _first_text(meta, "article-id[@pub-id-type='pmid']")
    doi = _first_text(meta, "article-id[@pub-id-type='doi']")

    title = _first_text(meta, "title-group/article-title")
    if not title:
        return None

    abstract_el = _sub(meta, "abstract")
    if abstract_el is not None:
        abstract, sections = _parse_structured_abstract(abstract_el)
    else:
        abstract, sections = "", []

    # In JATS, journal-meta is a sibling of article-meta under <front>, so the
    # journal title must be looked up from the front element, not from meta.
    front = _sub(node, "front")
    journal = _first_text(front, "journal-meta/journal-title-group/journal-title") or \
        _first_text(front, "journal-meta/journal-title") or \
        _first_text(meta, "journal-title")
    year = _year_from(_first_text(meta, "pub-date[@pub-type='epub']/year")) or \
        _year_from(_first_text(meta, "pub-date/year"))
    body = _sub(node, "body")
    if body is not None:
        sections = sections + _parse_pmc_sections_flat(body)

    authors: list[Author] = []
    contrib_group = _sub(meta, "contrib-group")
    if contrib_group is not None:
        for contrib in contrib_group.findall("contrib[@contrib-type='author']"):
            family = _first_text(contrib, "name/surname")
            given = _first_text(contrib, "name/given-names")
            collab = _first_text(contrib, "collab-alternative")
            if collab and not family:
                authors.append(Author(collective=collab))
            elif family:
                authors.append(Author(family=family, initials=given))

    keywords: list[str] = []
    for kwg in meta.findall("kwd-group"):
        for kwd in kwg.findall("kwd"):
            txt = _text_of(kwd)
            if txt:
                keywords.append(txt)

    mesh: list[MeshTerm] = []
    for kwd_group in meta.findall("kwd-group"):
        for kwd in kwd_group.findall("kwd"):
            txt = _text_of(kwd)
            if txt.lower().startswith("mesh:"):
                mesh.append(MeshTerm(descriptor=txt[5:].strip()))

    pub_types: list[str] = []
    article_type = node.get("article-type", "")
    if article_type:
        pub_types.append(article_type.replace("-", " ").title())

    return Document(
        doc_id=f"pmcid:{pmcid}" if pmcid else f"pmid:{pmid}",
        source="pmc",
        pmid=pmid,
        pmcid=pmcid,
        doi=doi,
        title=title,
        abstract=abstract,
        sections=sections,
        authors=authors,
        journal=journal,
        year=year,
        publication_date=_first_text(meta, "pub-date"),
        language=_first_text(meta, "lang-group/language") or "english",
        publication_types=tuple(pub_types),
        keywords=tuple(dict.fromkeys(keywords)),
        mesh_terms=mesh,
        citations=_parse_citations_pmc(node),
        categories=derive_categories(mesh, tuple(pub_types), taxonomy),
    )


def _parse_pmc_sections_flat(node: ET.Element, level: int = 0, inherited: str = "") -> list[Section]:
    """Walk nested ``<sec>`` elements in document order, tracking the enclosing title.

    ``node`` may be a container (``<body>``) or a ``<sec>``. When it is a ``<sec>``
    that section is emitted before its own ``<sec>`` children are visited.

    JATS nests sections arbitrarily deep and publishers vary in whether a ``<sec>``
    carries a title, so the nearest enclosing *classified* title is threaded down
    through ``inherited``. Only direct ``<p>`` and ``<list>`` children contribute
    text, so a parent's prose is not duplicated into its children.

    Three earlier versions were wrong in ways that produced plausible numbers
    rather than errors, all measured on live PMC OA RCT XML:

    * ``node.iter("sec")`` flattens the tree, so ``inherited`` was never populated:
      23.9% of sections classified.
    * Keeping that flat walk *and* adding recursion double-processed nested
      sections, so 216 of 10,000 documents carried duplicated text.
    * Emitting children instead of the node skipped every non-leaf section.
    """
    sections: list[Section] = []
    # When called on a container we emit its <sec> children; when called on a <sec>
    # we must emit THAT section before descending. Emitting children only was a
    # real bug: the recursive call on a child <sec> looked for that child's own
    # children, so every non-leaf section was skipped and a Methods section
    # containing "Search strategy" and "Eligibility criteria" returned only
    # "Methods".
    for sec in ([node] if node.tag == "sec" else node.findall("sec")):
        title = _text_of(_sub(sec, "title"))
        own: list[str] = []
        for child in sec:
            if child.tag == "p":
                txt = _text_of(child)
                if txt:
                    own.append(txt)
            elif child.tag == "list":
                for li in child.iter("list-item"):
                    txt = _text_of(li)
                    if txt:
                        own.append(txt)
        text = "\n\n".join(own)
        if title or text:
            sections.append(
                Section(
                    title=title or inherited,
                    text=text,
                    level=min(level, 4),
                    kind=_kind_with_inheritance(title, inherited),
                )
            )
        # Descend with this section's own title as context, but only when it was
        # classifiable -- otherwise an unrecognised heading should not mask the
        # enclosing one.
        here = title if (title and section_kind(title)) else inherited
        for child_sec in sec.findall("sec"):
            sections.extend(_parse_pmc_sections_flat(child_sec, level + 1, here))
    return sections


def _kind_with_inheritance(title: str, inherited_title: str) -> str:
    """Section kind, falling back to the enclosing section's kind.

    Full-text articles nest heavily, and the inner headings are usually specific
    rather than coarse: "Search strategy", "Eligibility criteria", "Study
    selection" sit under "Methods". Measured on PMC Open Access RCTs, 205,046 of
    240,000 parsed sections had no recognised kind -- overwhelmingly these
    sub-headings.

    That matters because an unclassified section gets no dependency edges. The
    dependency graph would then be blind to exactly the Methods detail a Results
    chunk needs, which is the structure the graph exists to model.

    Inheritance is only from the immediately enclosing heading and only when that
    heading was itself classified, so a "Limitations" sub-heading still reads as
    discussion rather than inheriting results from an outer section.
    """
    kind = section_kind(title) if title else ""
    if kind:
        return kind
    return section_kind(inherited_title) if inherited_title else ""


def parse_pmid_xml(xml_text: str, taxonomy: Any = None) -> list[Document]:
    """Parse an efetch ``pubmed`` response into Documents."""
    if not xml_text.strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.warning("failed to parse efetch XML: %s", exc)
        return []
    docs: list[Document] = []
    for node in root.iter("PubmedArticle"):
        try:
            doc = parse_pubmed_article(node, taxonomy)
        except Exception as exc:  # noqa: BLE001 - never let one record kill a batch
            log.debug("skipping malformed PubmedArticle: %s", exc)
            continue
        if doc and (doc.title or doc.abstract):
            docs.append(doc)
    # Bookshelf/PubMedBookArticle records reuse most of the same shape.
    for node in root.iter("PubmedBookArticle"):
        try:
            doc = parse_pubmed_article(node, taxonomy)
        except Exception as exc:  # noqa: BLE001
            log.debug("skipping malformed PubmedBookArticle: %s", exc)
            continue
        if doc and (doc.title or doc.abstract):
            docs.append(doc)
    return docs


def parse_pmc_xml(xml_text: str, taxonomy: Any = None) -> list[Document]:
    """Parse an efetch ``pmc`` response into Documents."""
    if not xml_text.strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.warning("failed to parse PMC XML: %s", exc)
        return []
    docs: list[Document] = []
    for node in root.iter("article"):
        try:
            doc = parse_pmc_article(node, taxonomy)
        except Exception as exc:  # noqa: BLE001
            log.debug("skipping malformed PMC article: %s", exc)
            continue
        if doc:
            docs.append(doc)
    return docs


def parse_efetch(xml_text: str, source: str = "pubmed", taxonomy: Any = None) -> list[Document]:
    """Dispatch to the PMC or PubMed parser based on ``source``."""
    return parse_pmc_xml(xml_text, taxonomy) if source == "pmc" else parse_pmid_xml(xml_text, taxonomy)
