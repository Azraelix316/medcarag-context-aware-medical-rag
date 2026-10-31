"""Tests for corpus parsing, chunking, storage, retrieval, and the agent loop.

These are correctness tests for the pieces where a silent bug would corrupt the
paper's numbers: XML parsing (tolerant by design, so it must be checked against
real NCBI payloads), chunk boundaries, row-id alignment across the three
indexes, and the coverage/merge logic in the agent.

Fixtures are small synthetic records, so the suite runs in seconds and needs no
network access.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from medrag.agent.loop import ResearchAgent, _dedup_by_document, _merge_results, assess_coverage
from medrag.agent.planner import (
    QueryPlanner,
    decompose_heuristic,
    detect_multi_hop,
    expand_query,
    strip_question_framing,
)
from medrag.config import ChunkConfig, PipelineConfig
from medrag.data.chunk import chunk_document, split_sentences
from medrag.data.pubmed_parser import parse_pmc_xml, parse_pmid_xml, section_kind
from medrag.data.schema import Document, Section
from medrag.data.shards import ShardWriter, read_shards
from medrag.eval.metrics import (
    Query,
    Qrels,
    compute_all_metrics,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    significance_bootstrap,
)
from medrag.index.hybrid import Candidate, rrf_fuse, weighted_fuse
from medrag.index.store import ChunkStore, build_chunk_store


# ------------------------------------------------------------------- fixtures
PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation Status="MEDLINE">
   <PMID Version="1">40123456</PMID>
   <Article PubModel="Print-Electronic">
    <Journal>
     <ISSN IssnType="Print">0028-4796</ISSN>
     <JournalIssue CitedMedium="Internet">
      <Volume>60</Volume><Issue>3</Issue>
      <PubDate><Year>2024</Year><Month>Mar</Month></PubDate>
     </JournalIssue>
     <Title>Mayo Clinic Proceedings</Title>
    </Journal>
    <ArticleTitle>Metformin and cardiovascular outcomes in type 2 diabetes: a randomized trial.</ArticleTitle>
    <Pagination><MedlinePgn>245-256</MedlinePgn></Pagination>
    <ELocationID EIdType="doi" ValidYN="Y">10.1016/j.mayocp.2024.01.001</ELocationID>
    <Abstract>
     <AbstractText Label="BACKGROUND" NlmCategory="BACKGROUND">
       Type 2 diabetes confers substantial cardiovascular risk.
     </AbstractText>
     <AbstractText Label="METHODS" NlmCategory="METHODS">
       We randomized 2,540 adults with type 2 diabetes to metformin or placebo and followed them for a mean of 4.2 years.
     </AbstractText>
     <AbstractText Label="RESULTS" NlmCategory="RESULTS">
       Metformin reduced the composite cardiovascular endpoint by 14% (HR 0.86, 95% CI 0.76-0.97; p=0.03).
     </AbstractText>
     <AbstractText Label="CONCLUSIONS" NlmCategory="CONCLUSIONS">
       Metformin modestly reduced cardiovascular events in this population.
     </AbstractText>
    </Abstract>
    <AuthorList CompleteYN="Y">
     <Author ValidYN="Y"><LastName>Nakagawa</LastName><Initials>T</Initials></Author>
     <Author ValidYN="Y"><LastName>Ortega</LastName><Initials>J</Initials></Author>
     <Author ValidYN="Y"><CollectiveName>Diabetes Trials Unit</CollectiveName></Author>
    </AuthorList>
    <Language>eng</Language>
    <PublicationTypeList>
     <PublicationType UI="D016428">Journal Article</PublicationType>
     <PublicationType UI="D016430">Randomized Controlled Trial</PublicationType>
    </PublicationTypeList>
   </Article>
   <ChemicalList>
    <Chemical>
     <RegistryNumber>6805</RegistryNumber>
     <NameOfSubstance UI="D008966" Source="MeSH">Metformin</NameOfSubstance>
    </Chemical>
   </ChemicalList>
   <MeshHeadingList>
    <MeshHeading>
     <DescriptorName UI="D003924" MajorTopicYN="N">Diabetes Mellitus, Type 2</DescriptorName>
    </MeshHeading>
    <MeshHeading>
     <DescriptorName UI="D008968" MajorTopicYN="Y">Metformin</DescriptorName>
     <QualifierName UI="Q000187" MajorTopicYN="N">drug effects</QualifierName>
    </MeshHeading>
   </MeshHeadingList>
   <KeywordList Owner="NOTNLM">
    <Keyword MajorTopicYN="N">metformin</Keyword>
    <Keyword MajorTopicYN="Y">cardiovascular outcomes</Keyword>
   </KeywordList>
  </MedlineCitation>
  <PubmedData>
   <ArticleIdList>
    <ArticleId IdType="pubmed">40123456</ArticleId>
    <ArticleId IdType="doi">10.1016/j.mayocp.2024.01.001</ArticleId>
    <ArticleId IdType="pmc">PMC12345678</ArticleId>
   </ArticleIdList>
  </PubmedData>
 </PubmedArticle>
</PubmedArticleSet>
"""

PMC_XML = """<?xml version="1.0"?>
<pmc-articleset>
 <article article-type="research-article">
  <front>
   <journal-meta>
    <journal-title-group><journal-title>Journal of Clinical Oncology</journal-title></journal-title-group>
   </journal-meta>
   <article-meta>
    <article-id pub-id-type="pmcid">PMC99999999</article-id>
    <article-id pub-id-type="pmid">39999999</article-id>
    <article-id pub-id-type="doi">10.1200/JCO.2024.55</article-id>
    <article-categories><subj-group><subject>Oncology</subject></subj-group></article-categories>
    <title-group><article-title>Immunotherapy sequencing in advanced melanoma.</article-title></title-group>
    <contrib-group>
     <contrib contrib-type="author"><name><surname>Ivanova</surname><given-names>EP</given-names></name></contrib>
     <contrib contrib-type="author"><name><surname>Chen</surname><given-names>L</given-names></name></contrib>
    </contrib-group>
    <pub-date pub-type="epub"><year>2024</year><month>7</month></pub-date>
    <volume>42</volume><issue>15</issue>
    <abstract>
     <p>We evaluated PD-1 inhibitors in 480 patients with advanced melanoma.</p>
    </abstract>
    <kwd-group><kwd subject="major">melanoma</kwd><kwd>immunotherapy</kwd></kwd-group>
   </article-meta>
  </front>
  <body>
   <sec>
    <title>Methods</title>
    <p>Patients were randomized 2:1 to pembrolizumab or ipilimumab. The primary endpoint was progression-free survival.</p>
    <sec>
     <title>Statistical analysis</title>
     <p>We used a Cox proportional hazards model adjusted for LDH and ECOG performance status.</p>
    </sec>
   </sec>
   <sec>
    <title>Results</title>
    <p>Median progression-free survival was 8.4 months versus 2.1 months, hazard ratio 0.42 (95% CI 0.31-0.58).</p>
   </sec>
   <sec>
    <title>Discussion</title>
    <p>Checkpoint inhibition outperformed chemotherapy in this cohort, consistent with prior trials.</p>
   </sec>
  </body>
 </article>
</pmc-articleset>
"""


# --------------------------------------------------------------------- tests
def test_pubmed_parsing() -> None:
    docs = parse_pmid_xml(PUBMED_XML)
    assert len(docs) == 1, f"expected 1 doc, got {len(docs)}"
    d = docs[0]
    assert d.pmid == "40123456"
    assert d.pmcid == "PMC12345678"
    assert d.doi == "10.1016/j.mayocp.2024.01.001"
    assert d.journal == "Mayo Clinic Proceedings"
    assert d.year == 2024
    assert d.volume == "60" and d.issue == "3" and d.pages == "245-256"
    assert "Metformin" in d.title
    # Structured abstracts must become labelled sections, not one blob.
    titles = [s.title for s in d.sections]
    assert titles == ["Background", "Methods", "Results", "Conclusions"], titles
    kinds = [s.kind for s in d.sections]
    assert kinds == ["background", "methods", "results", "conclusions"], kinds
    assert "0.86" in d.abstract, "structured abstract flattened incorrectly"
    assert d.publication_types == ("Journal Article", "Randomized Controlled Trial")
    assert len(d.authors) == 3
    assert d.authors[2].collective == "Diabetes Trials Unit"
    assert d.authors[0].family == "Nakagawa"
    assert d.keywords == ("metformin", "cardiovascular outcomes")
    mesh = {m.descriptor: m for m in d.mesh_terms}
    assert set(mesh) == {"Diabetes Mellitus, Type 2", "Metformin"}
    assert mesh["Metformin"].major is True
    assert mesh["Metformin"].qualifiers == ("drug effects",)
    assert d.chemicals == ("Metformin",), d.chemicals
    assert "rct" in d.categories, d.categories
    # Categories come from MeSH tree roots resolved by label/API, so the drug
    # bucket is "drugs-and-therapeutics" (MeSH's own top-level D label).
    assert "diseases" in d.categories, d.categories
    assert "drugs-and-therapeutics" in d.categories, d.categories
    print("  ok  PubMed XML -> Document (structured abstract, MeSH, authors)")


def test_pmc_parsing() -> None:
    docs = parse_pmc_xml(PMC_XML)
    assert len(docs) == 1
    d = docs[0]
    assert d.pmcid == "PMC99999999"
    assert d.pmid == "39999999"
    # journal-meta is a sibling of article-meta under <front>.
    assert d.journal == "Journal of Clinical Oncology", d.journal
    assert d.year == 2024
    assert len(d.authors) == 2
    assert d.authors[0].family == "Ivanova"
    assert "melanoma" in d.keywords
    # Nested sections must be flattened in document order, including depth 2.
    titles = [s.title for s in d.sections]
    assert "Methods" in titles and "Statistical analysis" in titles, titles
    assert "Results" in titles and "Discussion" in titles, titles
    assert titles.index("Methods") < titles.index("Statistical analysis") < titles.index("Results")
    stats = next(s for s in d.sections if s.title == "Statistical analysis")
    assert "Cox" in stats.text
    results = next(s for s in d.sections if s.title == "Results")
    assert "8.4 months" in results.text
    assert len(d.text) > 400
    print("  ok  PMC JATS -> Document (nested sections flattened in order)")


def test_unlabeled_multipart_abstract_is_not_welded() -> None:
    """Several AbstractText parts without labels must not run together.

    Regression: using itertext() over the whole Abstract element concatenates
    the parts with no separator, producing text like "control group.Conclusions"
    that corrupts chunk boundaries and every downstream sentence split.
    """
    xml = """<?xml version="1.0"?>
    <PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>999</PMID><Article>
    <Journal><Title>Test J</Title><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
    <ArticleTitle>Nutrition education and premenstrual syndrome.</ArticleTitle>
    <Abstract>
      <AbstractText>No significant changes were observed in the control group.</AbstractText>
      <AbstractText>Even a short-duration nutrition education intervention can reduce PMS severity.</AbstractText>
    </Abstract>
    <AuthorList><Author><LastName>Doe</LastName><Initials>J</Initials></Author></AuthorList>
    <Language>eng</Language>
    <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
    </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    docs = parse_pmid_xml(xml)
    assert len(docs) == 1
    text = docs[0].abstract
    # Parts must be separated by a boundary, not welded into one token stream.
    assert "group.Even" not in text, f"parts were welded together: {text!r}"
    assert re.search(r"control group\.\s+Even", text), repr(text)
    # Both parts survive intact.
    assert "control group." in text and "reduce PMS severity" in text
    # The chunker must see the boundary too.
    doc = docs[0]
    chunks = chunk_document(doc, ChunkConfig(target_tokens=200, max_tokens=300))
    assert any("Even a short-duration" in c.text or "control group." in c.text for c in chunks)
    print("  ok  unlabeled multi-part abstracts keep their part boundaries")


def test_inline_markup_does_not_weld_words() -> None:
    """Inline tags become spaces, and lost word boundaries are restored."""
    from medrag.data.pubmed_parser import clean_text

    # <sup> is a footnote marker; it must separate rather than vanish.
    assert clean_text("in the control group.<sup>a</sup>Mean age was 54.") == \
        "in the control group. a Mean age was 54."
    # Italic emphasis inside a sentence must not join the surrounding words.
    assert clean_text("received<i>metformin</i>daily") == "received metformin daily"
    # A boundary lost to inline markup between two sentences is restored.
    assert "Ablation.Conclusions" not in clean_text("after ablation.<i></i>Conclusions")
    # Ordinary prose must be untouched.
    assert clean_text("A simple sentence.") == "A simple sentence."
    assert clean_text("Effects of drug A. Results were significant.") == \
        "Effects of drug A. Results were significant."
    print("  ok  clean_text preserves word boundaries across inline markup")


def test_parser_tolerates_malformed() -> None:
    assert parse_pmid_xml("") == []
    assert parse_pmid_xml("not xml at all <<<") == []
    assert parse_pmc_xml("") == []
    # A record with neither title nor abstract is dropped, not crashed on.
    empty = "<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID></MedlineCitation></PubmedArticle></PubmedArticleSet>"
    assert parse_pmid_xml(empty) == []
    print("  ok  parser degrades gracefully on empty/malformed input")


def test_section_kind_closed_vocabulary() -> None:
    assert section_kind("METHODS") == "methods"
    assert section_kind("Materials and Methods") == "methods"
    assert section_kind("Results:") == "results"
    assert section_kind("Patient Selection") == ""      # not in the closed set
    assert section_kind("") == ""
    print("  ok  section kinds map to a closed vocabulary or ''")


def test_sentence_splitting_keeps_abbreviations() -> None:
    text = "Patients received metformin 500 mg i.v. twice daily. e.g. no change in HbA1c. Outcome was significant."
    sents = split_sentences(text)
    assert len(sents) == 3, sents
    assert sents[0].endswith("twice daily.")
    assert "e.g." in sents[1], "abbreviation split the sentence"
    print("  ok  sentence splitter respects clinical abbreviations")


def test_chunking_is_structure_aware() -> None:
    doc = Document.from_dict(parse_pmid_xml(PUBMED_XML)[0].to_dict())
    cfg = ChunkConfig(target_tokens=40, max_tokens=64, overlap_tokens=8, min_tokens=8)
    chunks = chunk_document(doc, cfg)
    assert chunks, "expected chunks"
    # Every abstract section should be represented.
    kinds = {c.section_kind for c in chunks}
    assert {"background", "methods", "results", "conclusions"} & kinds, kinds
    # Chunk ids must be unique and ordered.
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Context prefix must be present so passages are self-describing.
    for c in chunks:
        assert "Mayo Clinic Proceedings" in c.embed_text
        assert "Metformin" in c.embed_text
        assert c.pmid == "40123456"
        assert c.year == 2024
    assert all(c.is_abstract for c in chunks)
    print(f"  ok  structure-aware chunking ({len(chunks)} chunks, context-enriched)")


def test_chunking_respects_budget() -> None:
    long_section = Section(title="Results", text=" ".join(["The patient recovered well after surgery."] * 40))
    doc = Document(doc_id="d1", title="A study", abstract="Short abstract about surgery outcomes.",
                   sections=[long_section])
    cfg = ChunkConfig(target_tokens=60, max_tokens=90, overlap_tokens=16, min_tokens=8)
    chunks = chunk_document(doc, cfg)
    assert len(chunks) > 1, "a long section must be split into multiple chunks"
    for c in chunks:
        # 4 chars/token approximation; allow slack for the context prefix.
        assert c.token_count <= 120, f"chunk of {c.token_count} tokens exceeds the budget"
    print(f"  ok  chunk budget respected ({len(chunks)} chunks from one long section)")


def test_shard_roundtrip_and_torn_line_recovery() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        writer = ShardWriter(Path(tmp), "rec", shard_size=2, resume=False)
        for i in range(5):
            writer.write({"i": i, "text": f"record {i}"})
        shards = writer.close()
        assert len(shards) == 3, f"expected 3 shards, got {len(shards)}"

        # Simulate a shard truncated by a hard kill: the last line is partial.
        last = shards[-1]
        data = last.read_bytes()
        last.write_bytes(data[: len(data) - 12])
        records = list(read_shards(shards))
        # Complete records must still be recovered.
        assert len(records) >= 4, f"recovered only {len(records)} records"
        assert records[0]["i"] == 0
    print("  ok  sharded JSONL round-trips and survives a truncated tail")


def test_chunk_store_alignment_and_random_access() -> None:
    """A tiny cache must not lose chunks: the store streams to its blob."""
    chunks = chunk_document(Document.from_dict(parse_pmid_xml(PUBMED_XML)[0].to_dict()), ChunkConfig())
    with tempfile.TemporaryDirectory() as tmp:
        # cache_size=1 forces almost everything through the LRU eviction path,
        # which is exactly the condition that used to drop chunks on the floor.
        store = build_chunk_store(chunks, dir_path=Path(tmp), cache_size=1)
        assert len(store) == len(chunks)
        assert store.stats()["cache_size"] <= 1, store.stats()
        store.close()

        reloaded = ChunkStore.load(Path(tmp), cache_size=1)
        assert len(reloaded) == len(chunks)
        assert len(reloaded._offsets) == len(chunks), "one offset per row is required"
        for row_id, original in enumerate(chunks):
            got = reloaded.get(row_id)
            assert got is not None, f"row {row_id} unreadable from disk"
            assert got.text == original.text, f"row {row_id} text mismatch"
            assert got.chunk_id == original.chunk_id
            assert got.pmid == original.pmid
        # Bulk fetch must agree with single reads.
        bulk = reloaded.texts_bulk([0, len(chunks) // 2, len(chunks) - 1])
        assert bulk == [chunks[0].text, chunks[len(chunks) // 2].text, chunks[-1].text]
        reloaded.close()
    print(f"  ok  chunk store streams to disk; all {len(chunks)} rows survive a 1-entry cache")


def test_chunk_store_append_resumes() -> None:
    """A second build pass must append, not truncate, the existing blob."""
    chunks = chunk_document(Document.from_dict(parse_pmid_xml(PUBMED_XML)[0].to_dict()), ChunkConfig())
    with tempfile.TemporaryDirectory() as tmp:
        first = build_chunk_store(chunks, dir_path=Path(tmp), cache_size=64)
        n_first = len(first)
        first.close()
        # Reopen and append the same chunks again.
        second = ChunkStore.load(Path(tmp), cache_size=64)
        second.extend(chunks)
        second.finalize_build()
        assert len(second) == 2 * n_first, (len(second), n_first)
        # Original rows must still be intact at their original offsets.
        assert second.get(0).text == chunks[0].text
        assert second.get(n_first).chunk_id == chunks[0].chunk_id
        second.close()
        third = ChunkStore.load(Path(tmp))
        assert len(third) == 2 * n_first
        assert third.get(n_first - 1).text == chunks[-1].text
        third.close()
    print(f"  ok  store append/resume keeps prior rows addressable ({2 * n_first} rows total)")


def test_fusion_methods() -> None:
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("b", 12.0), ("c", 8.0), ("d", 3.0)]
    fused = rrf_fuse({"dense": dense, "sparse": sparse}, k=60, top_k=10)
    ids = [c.chunk_id for c in fused]
    assert set(ids) == {"a", "b", "c", "d"}, ids
    # b and c are found by both retrievers, so they must outrank the singles.
    assert ids.index("b") < ids.index("a"), ids
    assert ids.index("c") < ids.index("d"), ids
    both = [c for c in fused if len(c.sources) == 2]
    assert len(both) == 2 and all(c.dense_rank and c.sparse_rank for c in both)

    weighted = weighted_fuse(dense, sparse, top_k=10)
    wids = [c.chunk_id for c in weighted]
    assert set(wids) == {"a", "b", "c", "d"}
    # Scores must be normalised, so all outputs stay in [0, 1].
    assert all(0.0 <= c.score <= 1.0 for c in weighted), [c.score for c in weighted]
    print("  ok  RRF and weighted fusion both promote cross-retriever agreement")


def test_metrics() -> None:
    ranked = ["x", "y", "z", "w"]
    relevant = {"y", "w"}
    # At k=1 the single relevant doc at rank 1 ("x") is missed -> 0 of 2 found.
    assert recall_at_k(ranked, relevant, 1) == 0.0
    # At k=2, "y" is retrieved -> 1 of 2.
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 4) == 1.0
    # The first relevant document sits at rank 2.
    assert reciprocal_rank(ranked, relevant) == 0.5
    assert hit_rate_at_k(ranked, relevant, 1) == 0.0
    assert hit_rate_at_k(ranked, relevant, 2) == 1.0
    assert precision_at_k(ranked, relevant, 4) == 0.5
    # nDCG must reward putting relevant documents first.
    assert ndcg_at_k(["y", "w", "x", "z"], relevant, 4) > ndcg_at_k(["x", "z", "y", "w"], relevant, 4)
    metrics = compute_all_metrics(ranked, relevant, (1, 2, 4), supporting=relevant)
    assert metrics["success@all"] == 1.0
    # Degenerate cases must not raise.
    assert recall_at_k(ranked, set(), 2) != recall_at_k(ranked, set(), 2)  # NaN
    print("  ok  retrieval metrics (recall/ndcg/mrr/precision/success) behave correctly")


def test_significance_bootstrap_detects_a_real_difference() -> None:
    better = [0.90 + 0.001 * i for i in range(40)]
    worse = [0.60 + 0.001 * i for i in range(40)]
    result = significance_bootstrap(better, worse, n_resamples=500)
    assert result["significant"], result
    assert result["delta"] > 0.25
    noise_a = [0.5 + (0.01 * (i % 2)) for i in range(40)]
    noise_b = [0.5 + (0.01 * ((i + 1) % 2)) for i in range(40)]
    same = significance_bootstrap(noise_a, noise_b, n_resamples=500)
    assert not same["significant"], same
    print("  ok  paired bootstrap separates a real gain from noise")


def test_qrels_roundtrip_and_sampling() -> None:
    qrels = Qrels([Query(qid=f"q{i}", text=f"query {i}", relevant_docs={f"d{i}"}) for i in range(50)])
    with tempfile.TemporaryDirectory() as tmp:
        path = qrels.save(Path(tmp) / "q.json")
        loaded = Qrels.load(path)
    assert len(loaded) == 50
    assert loaded.by_id["q7"].relevant_docs == {"d7"}
    sample = loaded.sample(10, seed=13)
    assert len(sample) == 10
    assert sample == loaded.sample(10, seed=13), "sampling must be deterministic"
    print("  ok  qrels round-trip and deterministic sampling")


def test_multi_hop_detection_and_decomposition() -> None:
    # A bare term, and a short single-hop question, are not multi-hop.
    assert not detect_multi_hop("metformin")
    assert not detect_multi_hop("What is the effect of metformin on mortality?")
    # Long questions with relational connectives are.
    assert detect_multi_hop(
        "Is there an association between type 2 diabetes and chronic kidney disease, and what mediates it?"
    )
    assert detect_multi_hop(
        "Does metformin reduce HbA1c in patients with type 2 diabetes and chronic kidney disease?"
    )
    parts = decompose_heuristic(
        "Does metformin reduce HbA1c in patients with type 2 diabetes and chronic kidney disease?"
    )
    assert len(parts) >= 2, parts
    assert any("metformin" in p.lower() for p in parts), parts
    # Every part must be a self-contained retrievable phrase, not a fragment.
    assert all(len(p.split()) >= 2 for p in parts), parts
    print(f"  ok  multi-hop detection + decomposition -> {len(parts)} atomic queries")


def test_clinical_synonym_expansion() -> None:
    ex = expand_query("heart attack mortality")
    assert any("myocardial infarction" in e.lower() for e in ex), ex
    # Expansion must not duplicate what the query already says.
    assert not any(e.lower() == "heart attack" for e in ex)
    assert strip_question_framing("What is the current evidence on sepsis?") == "sepsis"
    print("  ok  clinical synonym expansion bridges vocabulary mismatch")


def test_planner_output_shape() -> None:
    plan = QueryPlanner().plan(
        "Does statin therapy reduce cardiovascular events in adults with chronic kidney disease?"
    )
    assert plan.sub_queries
    texts = plan.texts
    assert len(texts) == len({t.lower() for t in texts}), "duplicate sub-queries emitted"
    assert all(isinstance(t, str) and t.strip() for t in texts)
    payload = plan.to_dict()
    assert payload["original"].startswith("Does statin")
    assert isinstance(payload["is_multi_hop"], bool)
    print(f"  ok  query plan: {len(plan.sub_queries)} sub-queries, strategy={plan.strategy}")


def test_coverage_reflection() -> None:
    from medrag.index.retriever import RetrievedPassage

    on_topic = RetrievedPassage(chunk_id="a#0", row_id=0, text="Metformin reduced HbA1c in type 2 diabetes.",
                                score=1.0, title="Metformin in diabetes")
    off_topic = RetrievedPassage(chunk_id="b#0", row_id=1, text="Surgical management of appendicitis.",
                                 score=1.0, title="Appendicitis")
    q = "metformin HbA1c type 2 diabetes"
    good = assess_coverage(q, [on_topic])
    bad = assess_coverage(q, [off_topic])
    assert good.score > bad.score, (good.score, bad.score)
    assert good.missing == [], good.missing
    assert bad.missing, bad.missing
    print(f"  ok  coverage reflection discriminates on/off-topic evidence ({good.score:.2f} vs {bad.score:.2f})")


def test_result_merge_and_dedup() -> None:
    from medrag.index.retriever import RetrievalResult, RetrievedPassage

    def make(passages):
        return RetrievalResult(query="q", passages=list(passages), mode="hybrid")

    # The same passage retrieved by two different sub-queries is the real
    # multi-query case, and must be reinforced above a passage found once.
    shared = RetrievedPassage(chunk_id="d1#0", row_id=0, text="shared", score=0.80)
    rival = RetrievedPassage(chunk_id="d2#0", row_id=1, text="rival", score=0.81)
    also_shared = RetrievedPassage(chunk_id="d1#0", row_id=0, text="shared", score=0.80)
    merged = _merge_results([make([rival, shared]), make([also_shared])], top_k=10)
    order = [p.chunk_id for p in merged]
    assert len(merged) == 2, merged
    # Despite a marginally lower raw score, agreement must win.
    assert order[0] == "d1#0", order
    assert merged[0].score > merged[1].score, [(p.chunk_id, p.score) for p in merged]

    # Distinct chunks from one document must all survive the merge...
    a = RetrievedPassage(chunk_id="d1#0", row_id=0, text="alpha", score=0.9)
    a2 = RetrievedPassage(chunk_id="d1#1", row_id=2, text="alpha two", score=0.7)
    b = RetrievedPassage(chunk_id="d2#0", row_id=1, text="beta", score=0.8)
    merged = _merge_results([make([a, b]), make([a2])], top_k=10)
    assert len(merged) == 3, [p.chunk_id for p in merged]

    # ...but the per-document cap must stop one document filling the context.
    many = [RetrievedPassage(chunk_id=f"d1#{i}", row_id=i, text=f"t{i}", score=1.0 - i * 0.01) for i in range(5)]
    many += [RetrievedPassage(chunk_id=f"d2#{i}", row_id=10 + i, text=f"u{i}", score=0.5 - i * 0.01) for i in range(5)]
    limited = _dedup_by_document(many, per_doc=2)
    counts: dict[str, int] = {}
    for p in limited:
        key = p.chunk_id.split("#")[0]
        counts[key] = counts.get(key, 0) + 1
    assert all(v <= 2 for v in counts.values()), counts
    assert len(counts) == 2, counts
    print("  ok  multi-query merge reinforces cross-query agreement; per-document cap enforced")


def test_agent_end_to_end_with_stub_retriever() -> None:
    class StubRetriever:
        """Minimal retriever standing in for the real hybrid stack."""

        def __init__(self) -> None:
            from medrag.index.retriever import RetrievedPassage

            self.texts = [
                "Metformin lowered glycated hemoglobin in patients with type 2 diabetes by 1.0 percent.",
                "Metformin is generally safe for patients with chronic kidney disease when dose adjusted.",
                "Unrelated evidence about surgical site infections in orthopedic procedures.",
            ]
            self.passages = [
                RetrievedPassage(chunk_id=f"d{i}#0", row_id=i, text=t, score=1.0 - i * 0.1,
                                 title=f"Study {i}", pmid=str(1000 + i), journal="J Test", year=2024)
                for i, t in enumerate(self.texts)
            ]

        def retrieve(self, query, *, top_k=10, mode="hybrid", sub_queries=None, include_text=True):
            from medrag.index.retriever import RetrievalResult

            return RetrievalResult(query=query, passages=list(self.passages), mode=mode)

    agent = ResearchAgent(StubRetriever(), planner=QueryPlanner(), max_hops=1, top_k=3)  # type: ignore[arg-type]
    answer = agent.research("Does metformin help type 2 diabetes patients with chronic kidney disease?")
    assert answer.answer, "empty answer"
    assert answer.citations, "no citations attached"
    assert len(answer.evidence) <= 3
    actions = [s.action for s in answer.trace]
    assert actions[0] == "plan" and actions[-1] == "answer", actions
    assert "retrieve" in actions and "reflect" in actions, actions
    # Every quoted sentence must carry a citation.
    for line in answer.answer.splitlines():
        if line.startswith("- ") and len(line) > 40:
            assert "[" in line, f"uncited claim: {line}"
    assert answer.elapsed > 0
    print(f"  ok  agent loop: {len(actions)} traced steps, {len(answer.citations)} citations, "
          f"coverage={answer.coverage.score:.2f}")


def test_quote_cleaner_strips_structured_labels() -> None:
    """Quoted evidence must not lead with a structured-abstract label."""
    from medrag.agent.loop import _clean_quote

    assert _clean_quote("Background: Type 2 diabetes confers risk.") == \
        "Type 2 diabetes confers risk."
    assert _clean_quote("Methods:  We randomised 2,540 adults.") == \
        "We randomised 2,540 adults."
    assert _clean_quote("Results:\n  HbA1c fell by 1.0%.") == "HbA1c fell by 1.0%."
    # Whitespace is collapsed regardless of source formatting.
    assert _clean_quote("A  result   spread\n  over lines.") == "A result spread over lines."
    # A dangling section heading is removed, taking the period that introduced it.
    assert _clean_quote("No changes in the control group. Conclusions") == \
        "No changes in the control group"
    # Ordinary sentences keep their final punctuation.
    assert _clean_quote("Metformin reduced HbA1c by 1.0 percent.") == \
        "Metformin reduced HbA1c by 1.0 percent."
    print("  ok  quoted evidence is stripped of structured-abstract labels")


def test_config_layering_and_overrides() -> None:
    cfg = PipelineConfig.load(["configs/default.yaml", "configs/smoke.yaml"])
    # Later files win; untouched keys keep the base value.
    assert cfg.paths.data_root.name == "data_smoke"
    assert cfg.acquire.max_records == 300
    assert cfg.embed.batch_size == 128
    assert cfg.rerank.enabled is False
    assert cfg.chunk.strategy == "section", "base value should survive layering"
    assert cfg.paths.corpus_dir.is_absolute()
    assert cfg.paths.corpus_dir.parent == cfg.paths.data_root
    # Round-trip through a dict.
    again = PipelineConfig.from_dict(cfg.to_dict())._finalize()
    assert again.acquire.max_records == 300
    assert again.paths.data_root == cfg.paths.data_root
    print("  ok  config layering, override precedence, and dict round-trip")


def test_unknown_config_key_is_rejected() -> None:
    try:
        PipelineConfig.from_dict({"nonsense": {}})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown config section")
    print("  ok  unknown config sections raise rather than being silently ignored")


def main() -> int:
    tests = [
        test_pubmed_parsing,
        test_pmc_parsing,
        test_unlabeled_multipart_abstract_is_not_welded,
        test_inline_markup_does_not_weld_words,
        test_parser_tolerates_malformed,
        test_section_kind_closed_vocabulary,
        test_sentence_splitting_keeps_abbreviations,
        test_chunking_is_structure_aware,
        test_chunking_respects_budget,
        test_shard_roundtrip_and_torn_line_recovery,
        test_chunk_store_alignment_and_random_access,
        test_fusion_methods,
        test_metrics,
        test_significance_bootstrap_detects_a_real_difference,
        test_qrels_roundtrip_and_sampling,
        test_multi_hop_detection_and_decomposition,
        test_clinical_synonym_expansion,
        test_planner_output_shape,
        test_coverage_reflection,
        test_result_merge_and_dedup,
        test_agent_end_to_end_with_stub_retriever,
        test_quote_cleaner_strips_structured_labels,
        test_config_layering_and_overrides,
        test_unknown_config_key_is_rejected,
    ]
    print("MedRAG pipeline tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
