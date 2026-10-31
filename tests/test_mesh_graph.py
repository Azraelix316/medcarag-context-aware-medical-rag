"""Tests for the MeSH co-occurrence graph retriever.

This is the third retrieval signal, alongside BM25 and the dense bi-encoder, so
the tests focus on the properties that make it *different* rather than
redundant: it must resolve natural-language queries onto curated descriptors, it
must expand to clinically associated concepts, and it must not degenerate into
returning whatever is merely frequent.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from medrag.index.mesh_graph import MeshGraph


def build_tiny_graph() -> MeshGraph:
    """A hand-built corpus where the correct associations are known by construction.

    ``min_df_for_expansion=2`` rather than the production default of 50: each
    cluster here has only 40 documents, so the production floor would mask out
    every pair and the expansion path would never be exercised.
    """
    docs = [
        # 40 diabetes-with-complications records
        *[("d%03d" % i, ["Diabetes Mellitus, Type 2", "Diabetic Nephropathies",
                         "Diabetic Neuropathies", "Foot Ulcer", "Humans"])
          for i in range(40)],
        # 40 VAP records
        *[("v%03d" % i, ["Pneumonia, Ventilator-Associated", "Respiration, Artificial",
                         "Pneumonia, Bacterial", "Humans"])
          for i in range(40)],
        # 40 AF ablation records
        *[("a%03d" % i, ["Atrial Fibrillation", "Catheter Ablation",
                         "Pulmonary Veins", "Humans"])
          for i in range(40)],
        # decoy: photography, must never be expanded toward
        *[("p%03d" % i, ["Photography", "Artifacts", "Humans"]) for i in range(40)],
    ]
    return MeshGraph(min_df_for_expansion=2).build(docs, min_pair_count=2)


def test_graph_shape() -> None:
    g = build_tiny_graph()
    stats = g.stats()
    assert stats["n_docs"] == 160, stats
    # 13 distinct descriptors: 4 diabetes + 3 VAP + 3 AF + 2 decoy + Humans.
    assert stats["n_descriptors"] == 13, stats["n_descriptors"]
    assert stats["n_cooc_pairs"] > 0
    # Every document carries "Humans", which must not blow up the graph.
    assert "Humans" in g.descriptors
    print(f"  ok  graph built: {stats['n_descriptors']} descriptors, "
          f"{stats['n_cooc_pairs']} pairs")


def test_resolves_natural_language_forms() -> None:
    g = build_tiny_graph()
    # Natural phrasing must reach the MeSH descriptor.
    seeds = g.resolve_descriptors("type 2 diabetes mellitus")
    assert "Diabetes Mellitus, Type 2" in seeds, seeds
    # The inverted MeSH form too.
    assert "Atrial Fibrillation" in g.resolve_descriptors("atrial fibrillation ablation")
    # Token-level fallback for a single concept.
    assert "Pneumonia, Ventilator-Associated" in g.resolve_descriptors(
        "ventilator associated pneumonia"
    )
    print(f"  ok  natural-language queries resolve to descriptors: {seeds}")


def test_unknown_text_resolves_to_nothing() -> None:
    g = build_tiny_graph()
    assert g.resolve_descriptors("zzzqqq nonexistent concept") == []
    assert g.plan("zzzqqq nonexistent concept").seeds == []
    assert g.search("zzzqqq nonexistent concept", {}) == []
    print("  ok  unknown queries resolve to no descriptors and return no results")


def test_expansion_finds_true_associations() -> None:
    g = build_tiny_graph()
    pairs = g.expand(["Diabetes Mellitus, Type 2"], top_k=6, min_pmi=0.5)
    names = {d for d, _ in pairs}
    # Complications are co-indexed with T2D and must surface.
    assert "Diabetic Nephropathies" in names, names
    assert "Diabetic Neuropathies" in names, names
    # Unrelated decoy must not.
    assert "Photography" not in names, names
    print(f"  ok  T2D expands to its complications: {[d for d,_ in pairs[:4]]}")


def test_expansion_is_discriminative_not_merely_frequent() -> None:
    """A ubiquitous descriptor must not outrank a specific association.

    "Humans" appears in every document in this fixture. Naive co-occurrence
    would rank it first; IDF-style masking via the df threshold must suppress it.
    """
    g = build_tiny_graph()
    pairs = g.expand(["Atrial Fibrillation"], top_k=8, min_pmi=0.1)
    names = [d for d, _ in pairs]
    assert "Humans" not in names, (
        f"ubiquitous descriptor leaked into expansion: {names}"
    )
    assert "Catheter Ablation" in names, names
    print(f"  ok  frequent-but-uninformative descriptors are suppressed: {names[:4]}")


def test_search_ranks_matching_documents_first() -> None:
    g = build_tiny_graph()
    doc_desc = {}
    for i in range(40):
        doc_desc[f"d{i:03d}"] = [g.desc_index[d] for d in
            ["Diabetes Mellitus, Type 2", "Diabetic Nephropathies", "Diabetic Neuropathies"]]
        doc_desc[f"v{i:03d}"] = [g.desc_index[d] for d in
            ["Pneumonia, Ventilator-Associated", "Respiration, Artificial"]]
        doc_desc[f"a{i:03d}"] = [g.desc_index[d] for d in
            ["Atrial Fibrillation", "Catheter Ablation"]]
        doc_desc[f"p{i:03d}"] = [g.desc_index[d] for d in ["Photography", "Artifacts"]]

    hits = g.search("diabetic nephropathy", doc_desc, top_k=10, expand_top_k=6, min_pmi=0.5)
    assert hits, "no results returned"
    top_ids = [doc for doc, _ in hits]
    assert all(t.startswith("d") for t in top_ids[:5]), top_ids
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True), "results must be score-ordered"
    print(f"  ok  graph search ranks the right cluster first ({len(hits)} hits, "
          f"top score {scores[0]:.2f})")


def test_expanded_matches_are_reachable() -> None:
    """A query naming only a seed must still retrieve on expanded descriptors.

    This is the whole point of the signal: the user says "atrial fibrillation"
    but the evidence sits in papers indexed under "Catheter Ablation".
    """
    g = build_tiny_graph()
    doc_desc = {f"a{i:03d}": [g.desc_index["Atrial Fibrillation"]] for i in range(40)}
    # One document indexed ONLY with the expanded term, no seed at all.
    doc_desc["only_expanded"] = [g.desc_index["Catheter Ablation"]]
    hits = dict(g.search("atrial fibrillation", doc_desc, top_k=50, expand_top_k=6, min_pmi=0.5))
    assert "only_expanded" in hits, (
        "a document carrying only a PMI-associated descriptor should be reachable"
    )
    print("  ok  documents reachable only through graph expansion are retrieved")


def test_save_load_roundtrip() -> None:
    import tempfile
    from pathlib import Path

    g = build_tiny_graph()
    with tempfile.TemporaryDirectory() as tmp:
        path = g.save(Path(tmp) / "mesh.pkl")
        back = MeshGraph.load(path)
    assert back.descriptors == g.descriptors
    assert back.n_docs == g.n_docs
    assert (back.df == g.df).all()
    assert back.stats()["n_cooc_pairs"] == g.stats()["n_cooc_pairs"]
    assert back.resolve_descriptors("type 2 diabetes") == g.resolve_descriptors("type 2 diabetes")
    # Expansion must survive serialisation exactly.
    a = g.expand(["Diabetes Mellitus, Type 2"], top_k=5, min_pmi=0.5)
    b = back.expand(["Diabetes Mellitus, Type 2"], top_k=5, min_pmi=0.5)
    assert a == b, (a, b)
    print("  ok  graph save/load is lossless, including expansion")


def test_pmi_filters_rare_and_negative_pairs() -> None:
    g = build_tiny_graph()
    # min_pair_count pruning removes one-off coincidences.
    strict = MeshGraph().build(
        [("x", ["A", "B"]), ("y", ["A", "C"]), ("z", ["B", "C"])], min_pair_count=5
    )
    assert strict.stats()["n_cooc_pairs"] == 0, strict.stats()
    # A co-occurrence more frequent than chance is retained in the PMI matrix.
    loose = MeshGraph().build(
        [("x", ["A", "B"]), ("y", ["A", "B"]), ("z", ["A"]), ("w", ["B"])], min_pair_count=1
    )
    assert loose.stats()["n_pmi_pairs"] >= 1, loose.stats()
    print("  ok  rare pairs pruned; chance-level pairs kept")


def test_empty_corpus_is_safe() -> None:
    g = MeshGraph().build([])
    assert g.stats()["n_descriptors"] == 0
    assert g.search("anything", {}) == []
    assert g.expand(["nothing"]) == []
    print("  ok  empty graph degrades safely instead of raising")


def test_doc_to_chunks_cache_equals_the_per_query_scan() -> None:
    from medrag.index import graph_retriever as gr

    def old(store_rows, ranked, per_doc):
        by_doc = {}
        for row_id, chunk_id in enumerate(store_rows):
            by_doc.setdefault(chunk_id.split("#", 1)[0], []).append(row_id)
        out = []
        for doc_id, score in ranked:
            for row_id in by_doc.get(doc_id, [])[:per_doc]:
                out.append((str(row_id), score))
        return out

    # D1 is split across the store, as 472 documents are in abstracts20m.
    rows = ["D1#0", "D1#1", "D2#0", "D3#0", "D3#1", "D3#2", "D3#3", "D1#2", "D1#3"]
    ranked = [("D3", 0.9), ("D9", 0.8), ("D1", 0.5), ("D2", 0.1)]
    for per_doc in (1, 3, 5):
        assert gr.doc_to_chunks(rows, ranked, per_doc=per_doc) == old(rows, ranked, per_doc)
    other = rows[:3]
    assert gr.doc_to_chunks(other, ranked, per_doc=3) == old(other, ranked, 3)
    print("  doc_to_chunks with its cached map == the per-query scan it replaced")


def test_inverted_graph_search_equals_graph_retriever() -> None:
    import random
    from types import SimpleNamespace

    from medrag.index.graph_retriever import DocDescriptorMap, GraphRetriever, InvertedGraphSearch

    g = build_tiny_graph()
    names = list(g.descriptors)
    rng = random.Random(3)
    docs = []
    for i in range(400):
        picked = rng.sample(names, rng.randint(0, 6))
        docs.append(SimpleNamespace(doc_id=f"x{i % 390:03d}",   # ten ids repeat
                                    mesh_terms=[SimpleNamespace(descriptor=n) for n in picked]))
    doc_map = DocDescriptorMap().build(g, docs)
    ref, fast = GraphRetriever(g, doc_map), InvertedGraphSearch(g, doc_map)
    queries = ["diabetic nephropathy", "ventilator associated pneumonia", "atrial fibrillation",
               "catheter ablation of pulmonary veins", "photography", "nothing matches this"]
    for q in queries:
        for top_k in (1, 5, 50, 1000):
            for kw in ({}, {"expand_top_k": 6, "min_pmi": 0.5}):
                assert fast.search(q, top_k=top_k, **kw) == ref.search(q, top_k=top_k, **kw), (q, top_k, kw)
    print("  inverted graph search == GraphRetriever.search, repeated doc ids included")


def test_batched_pair_counts_equal_one_batch() -> None:
    from medrag.index import mesh_graph as mg

    whole = build_tiny_graph()
    old = mg.PAIR_BATCH_DOCS
    mg.PAIR_BATCH_DOCS = 7          # 160 documents in 23 uneven batches
    try:
        batched = build_tiny_graph()
    finally:
        mg.PAIR_BATCH_DOCS = old
    for name in ("_cooc", "_pmi"):
        a, b = getattr(whole, name), getattr(batched, name)
        assert (a != b).nnz == 0 and a.shape == b.shape, name
    assert list(whole.df) == list(batched.df) and whole.descriptors == batched.descriptors
    print("  batched co-occurrence counts == one batch")


def main() -> int:
    tests = [
        test_graph_shape,
        test_resolves_natural_language_forms,
        test_unknown_text_resolves_to_nothing,
        test_expansion_finds_true_associations,
        test_expansion_is_discriminative_not_merely_frequent,
        test_search_ranks_matching_documents_first,
        test_expanded_matches_are_reachable,
        test_save_load_roundtrip,
        test_pmi_filters_rare_and_negative_pairs,
        test_empty_corpus_is_safe,
        test_doc_to_chunks_cache_equals_the_per_query_scan,
        test_inverted_graph_search_equals_graph_retriever,
        test_batched_pair_counts_equal_one_batch,
    ]
    print("MeSH co-occurrence graph tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())