"""Tests for MeSH taxonomy resolution and benchmark construction.

The specific failure this guards against is subtle and was found the hard way:
MeSH descriptor UIs all begin with ``D``, so ``ui[0]`` is NOT the tree category.
Diabetes Mellitus Type 2 (D003924, tree C19 = diseases) and Metformin
(D008687, tree D02 = drugs) share the same ``D`` prefix. Any code that infers
specialty from the UI silently mislabels the whole corpus.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from medrag.data.mesh_taxonomy import (
    CLINICAL_ROOTS,
    GENERIC_ROOTS,
    TREE_CATEGORIES,
    MeshTaxonomy,
)
from medrag.data.schema import Document, MeshTerm
from medrag.eval.benchmark import (
    _clean_topic,
    _descriptor_phrase,
    build_clinical_qrels,
    build_multi_hop_qrels,
)


def test_ui_prefix_is_not_the_tree_category() -> None:
    """The core regression: identical UI prefixes, different tree categories."""
    disease_ui, disease_label = "D003924", "Diabetes Mellitus, Type 2"
    drug_ui, drug_label = "D008687", "Metformin"
    # Ground truth from the NLM MeSH API, cached to avoid network dependency.
    cache = {
        disease_ui: "C",   # C19.246.300 - diseases
        drug_ui: "D",      # D02.078.370.141.450 - drugs & therapeutics
    }
    taxonomy = MeshTaxonomy(use_api=False)
    taxonomy._cache.update(cache)

    assert disease_ui[0] == drug_ui[0] == "D", "both UIs must start with D"
    assert taxonomy.category(disease_ui, disease_label) == "diseases"
    assert taxonomy.category(drug_ui, drug_label) == "drugs-and-therapeutics"
    assert taxonomy.category(disease_ui, disease_label) != taxonomy.category(drug_ui, drug_label)
    print("  ok  tree category comes from the taxonomy, not the UI prefix")


def test_tree_category_names_are_valid_mesh_levels() -> None:
    """Category labels must follow MeSH's own top-level taxonomy, not guesses."""
    assert TREE_CATEGORIES["E"] == "diagnostic-and-therapeutic-techniques", TREE_CATEGORIES["E"]
    assert TREE_CATEGORIES["M"] == "persons-population"
    assert TREE_CATEGORIES["G"] == "phenomena-and-processes"
    assert set(TREE_CATEGORIES) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    # Diseases, drugs, psychiatry, anatomy, biological sciences are the clinical cores.
    assert CLINICAL_ROOTS == {"A", "C", "D", "F", "G", "P"}
    # Adult / Aged / Humans live in M, so M must be excluded from queries.
    assert "M" in GENERIC_ROOTS
    assert "C" not in GENERIC_ROOTS and "D" not in GENERIC_ROOTS
    print("  ok  MeSH top-level category names are accurate")


def test_cache_roundtrip_and_offline_mode() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mesh.json"
        tax = MeshTaxonomy(path, use_api=False)
        tax._cache.update({"D003924": "C", "D008687": "D"})
        tax.flush()
        assert json.loads(path.read_text()) == {"D003924": "C", "D008687": "D"}

        reloaded = MeshTaxonomy(path, use_api=False)
        assert reloaded.tree_root("D003924", "Diabetes Mellitus, Type 2") == "C"
        assert reloaded.n_cache_hits == 1
        # Uncached UIs must not trigger network calls in offline mode.
        assert reloaded.tree_root("D999999", "Something Obscure") == MeshTaxonomy.infer_root("Something Obscure")
    print("  ok  taxonomy cache round-trips and offline mode never calls the API")


def test_corrupt_cache_degrades_gracefully() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mesh.json"
        path.write_text("{not valid json", encoding="utf-8")
        tax = MeshTaxonomy(path, use_api=False)  # must not raise
        assert tax.tree_root("D003924", "Diabetes Mellitus, Type 2") == MeshTaxonomy.infer_root(
            "Diabetes Mellitus, Type 2"
        )
    print("  ok  a corrupt taxonomy cache degrades to label heuristics instead of raising")


def test_label_heuristic_fallback() -> None:
    """With no API, common clinical labels must still land in the right bucket."""
    tax = MeshTaxonomy(use_api=False)
    assert tax.category("", "Diabetes Mellitus, Type 2") == "diseases"
    assert tax.category("", "breast neoplasms") == "diseases"
    assert tax.category("", "Antineoplastic Agents") == "drugs-and-therapeutics"
    assert tax.category("", "Depressive Disorder") == "psychiatry-and-psychology"
    # An unrecognised label yields no category rather than a wrong one.
    assert tax.category("", "Zzyzx Widget") == ""
    print("  ok  label heuristics correctly bucket common clinical descriptors")


def test_descriptor_phrase_inversion() -> None:
    assert _descriptor_phrase("Diabetes Mellitus, Type 2") == "type 2 diabetes mellitus"
    assert _descriptor_phrase("Neoplasms") == "neoplasms"
    # Single-word and non-clinical descriptors make poor benchmark queries and
    # must be rejected: "Humans" or "Adult" would retrieve most of the corpus.
    assert _clean_topic("Humans") is None, "single-word descriptors are too generic"
    assert _clean_topic("Adult") is None
    assert _clean_topic("") is None
    assert _clean_topic("Unknown") is None, "non-clinical descriptors must be rejected"
    assert _clean_topic("Diabetes Mellitus, Type 2") == "type 2 diabetes mellitus"
    assert _clean_topic("Chronic Kidney Disease") == "chronic kidney disease"
    print("  ok  MeSH 'Head, Qualifier' labels become natural phrases; generic ones rejected")


def _fake_docs(n_per_desc: dict[str, int]) -> list[Document]:
    """Build synthetic documents carrying one MeSH descriptor each."""
    docs: list[Document] = []
    for desc, count in n_per_desc.items():
        for i in range(count):
            docs.append(
                Document(
                    doc_id=f"pmid:{abs(hash((desc, i))) % 10**9}",
                    title=f"{desc} study number {i}",
                    abstract=f"This trial investigated {desc.lower()} in a randomised cohort of patients. "
                             f"Primary outcome was assessed at 12 months.",
                    mesh_terms=(MeshTerm(descriptor=desc, ui="D003924", major=True),),
                    publication_types=("Journal Article", "Randomized Controlled Trial"),
                )
            )
    return docs


def test_clinical_qrels_exclude_generic_roots() -> None:
    docs = _fake_docs({
        "Diabetes Mellitus, Type 2": 40,
        "Hypertension": 30,
        # Generic descriptors that must never become benchmark queries.
        "Adult": 50,
        "Aged": 50,
        "Humans": 50,
    })
    tax = MeshTaxonomy(use_api=False)
    tax._cache.update({
        "D003924": "C",
    })
    qrels = build_clinical_qrels(docs, n_queries=100, min_gold=10, seed=5, taxonomy=tax)
    topics = {q.answer for q in qrels.queries}
    assert topics, "expected some clinical queries"
    # Generic roots are only resolvable via cache/API; with an unknown UI the
    # label fallback must still keep Adults and Humans out of the query set.
    assert "diabetes mellitus, type 2" in topics or "type 2 diabetes mellitus" in topics, topics
    assert not any(t in {"adult", "aged", "humans"} for t in topics), topics
    for q in qrels.queries:
        assert len(q.relevant_docs) >= 10, q.qid
        assert q.difficulty == "single-hop"
    print(f"  ok  clinical qrels exclude generic roots ({len(qrels.queries)} queries, topics={sorted(topics)})")


def test_multi_hop_qrels_require_both_hops() -> None:
    docs: list[Document] = []
    # 6 documents carrying BOTH descriptors -> forms a multi-hop intersection.
    for i in range(6):
        docs.append(
            Document(
                doc_id=f"pmid:both{i}",
                title=f"Both condition study {i}",
                abstract="Study spanning two conditions.",
                mesh_terms=(MeshTerm(descriptor="Diabetes Mellitus, Type 2", ui="D1"),
                            MeshTerm(descriptor="Chronic Kidney Disease", ui="D2")),
            )
        )
    # Support-only documents for each side, to make the hops separable.
    for i in range(4):
        docs.append(
            Document(
                doc_id=f"pmid:a{i}",
                title=f"Diabetes only {i}",
                abstract="Diabetes-only study.",
                mesh_terms=(MeshTerm(descriptor="Diabetes Mellitus, Type 2", ui="D1"),),
            )
        )
        docs.append(
            Document(
                doc_id=f"pmid:b{i}",
                title=f"CKD only {i}",
                abstract="CKD-only study.",
                mesh_terms=(MeshTerm(descriptor="Chronic Kidney Disease", ui="D2"),),
            )
        )
    qrels = build_multi_hop_qrels(docs, n_queries=10, min_gold=4, seed=5)
    assert len(qrels.queries) >= 1, "expected at least one multi-hop query"
    q = qrels.queries[0]
    assert q.difficulty == "multi-hop"
    assert q.category == "multi-hop"
    # Gold must be the documents satisfying BOTH hops.
    assert all(d.startswith("pmid:both") for d in q.relevant_docs), q.relevant_docs
    # Supporting set must be a superset so success@all can credit finding either hop.
    assert q.supporting_docs >= q.relevant_docs
    assert any(d.startswith("pmid:a") for d in q.supporting_docs)
    assert any(d.startswith("pmid:b") for d in q.supporting_docs)
    print(f"  ok  multi-hop qrels separate the two hops ({len(qrels.queries)} queries, gold={len(q.relevant_docs)})")


def test_mesh_sidecar_labels_a_fulltext_corpus() -> None:
    """A PMC corpus carries no MeSH; without the sidecar it yields no queries at all."""
    import gzip

    from medrag.config import AcquireConfig, PipelineConfig
    from medrag.eval.benchmark import build_benchmark_from_corpus

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = PipelineConfig()
        cfg.paths.data_root = root
        cfg.acquire = AcquireConfig(source="pmc")
        cfg = cfg._finalize()
        shard_dir = cfg.paths.corpus_dir / "pmc"
        shard_dir.mkdir(parents=True)
        docs = [Document(doc_id=f"pmcid:PMC{i}", source="pmc", pmid=str(i),
                         title=f"doc {i}", abstract="text") for i in range(40)]
        with gzip.open(shard_dir / "pmc-00000.jsonl.gz", "wt") as fh:
            for d in docs:
                fh.write(json.dumps(d.to_dict()) + "\n")
        sidecar = root / "pmc_mesh.json.gz"
        with gzip.open(sidecar, "wt") as fh:
            json.dump({"ui_by_descriptor": {"Diabetes Mellitus, Type 2": "D003924"},
                       "by_doc_id": {f"pmcid:PMC{i}": ["Diabetes Mellitus, Type 2"] for i in range(20)}}, fh)
        taxonomy = MeshTaxonomy(use_api=False)
        taxonomy._cache.update({"D003924": "C"})

        bare = build_benchmark_from_corpus(cfg, root / "bare", n_clinical=10, n_multihop=0,
                                           taxonomy=taxonomy)
        assert bare["families"]["clinical"] == 0, bare["families"]

        out = root / "labelled"
        built = build_benchmark_from_corpus(cfg, out, n_clinical=10, n_multihop=0,
                                            taxonomy=taxonomy, mesh_sidecar=sidecar)
        assert built["families"]["clinical"] == 1, built["families"]
        q = json.loads((out / "qrels_clinical.json").read_text())
        q = q[0] if isinstance(q, list) else (q.get("queries") or list(q.values()))[0]
        assert len(q["relevant_docs"]) == 20, q
        # The UI from the sidecar decides the category; the name alone would not.
        assert q["category"] == "diseases", q["category"]
    print("  ok  mesh sidecar labels a full-text corpus with the right category")


def main() -> int:
    tests = [
        test_ui_prefix_is_not_the_tree_category,
        test_tree_category_names_are_valid_mesh_levels,
        test_cache_roundtrip_and_offline_mode,
        test_corrupt_cache_degrades_gracefully,
        test_label_heuristic_fallback,
        test_descriptor_phrase_inversion,
        test_clinical_qrels_exclude_generic_roots,
        test_multi_hop_qrels_require_both_hops,
        test_mesh_sidecar_labels_a_fulltext_corpus,
    ]
    print("MeSH taxonomy and benchmark tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
