"""Page-level conditioning must behave as an ablation, not as a new candidate set.

The objective lists page-level vectors stored alongside chunk vectors as the first
component of the contextual-contamination system, after Anthropic Contextual Retrieval
and LangChain Contextual Embeddings. Both of those *rewrite* a chunk's text with its
document context and so require re-embedding the corpus; at 16,133,551 chunks that
costs hours, which is why they are cited rather than claimed.

This is the same signal computed at query time from vectors that already exist:
``s'(c) = s(c) * (1 + alpha * page(doc(c)))``.

Two properties make it a usable ablation, and both are easy to break:

* **At alpha=0 the ranking is the base ranking, exactly.** Otherwise ``page`` and
  ``sparse`` differ in their candidate sets and any difference in the results is
  confounded with which passages were available at all.
* **A missing page vector demotes nothing.** A hit whose document has no page score
  keeps its base score. Dropping it would silently shrink the candidate set, which is
  the same confound by another route.

The third test records the reason this mode existed at all: the page index was built
over 554,069 documents and *nothing consumed it*. An artifact nobody reads is not a
retrieval mode.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from medrag.index.page_scoring import PageConfig, apply_page_scores, normalise
from medrag.index.retriever import ALL_MODES, MODE_PAGE, MODE_PAGE_CONTEXT


def _hits():
    return [(10, 5.0), (11, 4.0), (12, 3.0), (13, 2.0)]


def _docs():
    return {10: "dA", 11: "dB", 12: "dC", 13: "dZ"}


def test_alpha_zero_reproduces_the_base_ranking() -> None:
    hits = _hits()
    out = apply_page_scores(hits, {"dA": 1.0, "dB": 0.0}, _docs().get, PageConfig(0.0))
    assert out == hits, out
    print("  ok  alpha=0 returns the base ranking byte-identically")


def test_a_better_document_promotes_its_passage() -> None:
    out = dict(apply_page_scores(_hits(), {"dA": 1.0, "dB": 0.0},
                                 _docs().get, PageConfig(0.5)))
    assert out[10] == 7.5, out          # 5.0 * (1 + 0.5*1.0)
    assert out[11] == 4.0, out          # 4.0 * (1 + 0.5*0.0)
    assert out[10] > out[11], "the better document must rank higher"
    print("  ok  a better-matched document promotes its passage")


def test_a_missing_page_vector_keeps_the_base_score() -> None:
    """dZ has no entry in page_scores; it must not be dropped or zeroed."""
    out = dict(apply_page_scores(_hits(), {"dA": 1.0}, _docs().get, PageConfig(0.5)))
    assert out[13] == 2.0, f"a hit with no page vector was altered: {out}"
    assert 13 in out, "the hit was dropped from the candidate set"
    print("  ok  a hit whose document has no page vector is left untouched")


def test_page_scoring_can_only_promote_never_override() -> None:
    """Additive, not multiplicative: a strong document cannot resurrect a passage
    whose own text does not match at all."""
    weak_doc_high, weak_doc_low = 1.0, 0.0
    base_a, base_b = 0.2, 10.0
    out = dict(apply_page_scores([(1, base_a), (2, base_b)],
                                 {"dHi": weak_doc_high, "dLo": weak_doc_low},
                                 {1: "dHi", 2: "dLo"}.get, PageConfig(0.5)))
    assert out[1] < out[2], (
        f"page scoring overrode the text signal: {out[1]} beat {out[2]}")
    print(f"  ok  a 0.2 passage in a perfect document stays below a 10.0 one "
          f"({out[1]:.2f} < {out[2]:.2f})")


def test_ordering_is_invariant_to_the_base_score_scale() -> None:
    """BM25 is unbounded and cosine is not; a scale-sensitive formula would rank
    differently depending on which retriever produced the base scores, so the same
    hits rescaled 1000x must come out in the same order."""
    hits = [(10, 5.0), (11, 4.0), (12, 3.0), (13, 2.0)]
    scaled = [(r, s * 1000.0) for r, s in hits]
    pages = {"dA": 1.0, "dB": 0.0, "dC": 0.5}
    doc = _docs()
    a = [r for r, _ in apply_page_scores(hits, pages, doc.get, PageConfig(0.5))]
    b = [r for r, _ in apply_page_scores(scaled, pages, doc.get, PageConfig(0.5))]
    assert a == b, f"ranking depends on score scale: {a} vs {b}"
    print(f"  ok  ranking unchanged when every base score is multiplied by 1000 ({a})")


def test_page_scoring_actually_reorders() -> None:
    """Rescaling without re-sorting returns the baseline ranking unchanged.

    The `page` arm reported byte-identical rankings to `sparse` on 400 of 400 PMC
    queries while the scores it produced genuinely differed -- 23.531 became 30.604
    and 22.586 became 32.028. Every downstream consumer ranks on list position, so
    recomputing a score without reordering the list is not a scoring function at all,
    it is a no-op with extra arithmetic.

    Here row 10 has the best page score but the worst base score, so a ranking
    function must move it to first place.
    """
    hits = _hits()                      # [(10, 5.0), (11, 4.0), (12, 3.0), (13, 2.0)]
    docs = _docs()                      # 10 -> dA, 11 -> dB, 12 -> dC, 13 -> dZ
    pages = {"dA": 1.0, "dB": 0.0, "dC": 0.0, "dZ": 0.0}
    out = apply_page_scores(hits, pages, docs.get, PageConfig(1.0))
    assert out[0][0] == 10, (
        f"best page score did not reach the top: {[r for r, _ in out]}")
    assert [s for _, s in out] == sorted((s for _, s in out), reverse=True), out

    # Deterministic: equal scores keep a fixed order, not dict-iteration order.
    tie_pages = {"dA": 0.5, "dB": 0.5, "dC": 0.5}
    first = apply_page_scores(hits, tie_pages, docs.get, PageConfig(1.0))
    second = apply_page_scores(list(reversed(hits)), tie_pages, docs.get, PageConfig(1.0))
    assert first == second, (first, second)
    print("  ok  page scoring reorders, and orders ties deterministically")


def test_empty_page_scores_is_a_no_op() -> None:
    hits = _hits()
    assert apply_page_scores(hits, {}, _docs().get, PageConfig(0.9)) == hits
    print("  ok  no page scores returns the base ranking")


def test_normalise_bounds_the_discount() -> None:
    got = normalise([0.2, 0.5, 0.9])
    assert got[0] == 0.0 and got[2] == 1.0, got
    assert abs(got[1] - 3 / 7) < 1e-12, got
    assert normalise([5.0, 5.0, 5.0]) == [0.5, 0.5, 0.5], "constant scores must not divide by zero"
    assert normalise([]) == []
    print("  ok  normalisation maps to [0,1] and survives a constant list")


def test_the_page_modes_are_registered() -> None:
    assert MODE_PAGE in ALL_MODES and MODE_PAGE_CONTEXT in ALL_MODES
    print(f"  ok  modes {MODE_PAGE!r} and {MODE_PAGE_CONTEXT!r} are selectable "
          f"({len(ALL_MODES)} modes total)")


def test_page_scores_are_addressed_by_position_not_by_chunk_row_id() -> None:
    """PageIndex.row_ids are chunk-store row ids; they are not matrix positions.

    The bug this pins is silent by construction. A page vector attaches to a
    document, and the matrix is addressed by position in the page index, while
    PageIndex.row_ids holds chunk-store row ids. Indexing the matrix with the latter
    worked while every candidate came from an early chunk and raised IndexError the
    moment a candidate came from row 7,843,341 of a 554,069-row matrix -- so on a
    smaller index it would have kept running and scored most documents against
    somebody else's vector. Plausible cosines, wrong papers.
    """
    import numpy as np

    from medrag.index.page_index import PageIndex
    from medrag.index.retriever import MedRAGRetriever

    # Three documents. Their page-matrix positions are 0, 1, 2 -- deliberately far
    # from their chunk-store row ids of 7,843,341, 12 and 99.
    doc_ids = ["docA", "docB", "docC"]
    chunk_rows = [7843341, 12, 99]
    page = PageIndex(
        dim=3,
        # docA matches the query direction best, docC worst.
        matrix=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                          dtype=np.float32),
        row_ids=chunk_rows,
        doc_ids=doc_ids,
    )

    class _Chunk:
        def __init__(self, d):
            self.doc_id = d

    class _Store:
        def get(self, row):
            return _Chunk(doc_ids[chunk_rows.index(row)])

    class _Embedder:
        def embed(self, texts, is_query=False):
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    import types

    cfg = types.SimpleNamespace(
        fusion=types.SimpleNamespace(page_alpha=0.5),
        rerank=types.SimpleNamespace(enabled=False),
    )
    r = MedRAGRetriever(store=_Store(), page=page, cfg=cfg)
    r.dense_embedder = _Embedder()

    # docC's passage has the WORST page score and must be demoted below docA's,
    # even though its chunk row id (99) is far larger than docA's (7,843,341).
    hits = [(99, 10.0), (7843341, 9.0)]
    out = dict(r._apply_page(hits, ["query"]))
    assert out[7843341] > out[99], (
        f"page scores addressed by chunk row id instead of position: {out}")
    assert abs(out[7843341] - 9.0 * 1.5) < 1e-9, out
    assert abs(out[99] - 10.0 * 1.0) < 1e-9, out
    print("  ok  page scores addressed by matrix position, not by chunk row id")


def test_context_modes_return_their_context_passages_without_text() -> None:
    """Context arms must not degrade to sparse on the include_text=False path.

    include_text=False is the cheap path every offline driver and metric script takes,
    because the passages only need row ids to be scored. It was also where the
    context arms were silently equivalent to the baseline: the candidate window was
    sliced to top_k before the framing passages were appended, and since those
    passages sit after the seeds, `context` returned byte-identical rankings to
    `sparse` on 400 of 400 queries in the first PMC ablation. Only the
    include_text=True path widened the window, so the bug hid precisely where the
    arms were being measured.

    A context arm that returns exactly its seeds is not a weak context arm; it is no
    context arm, and it reports as indistinguishable from the baseline.
    """
    import types

    import numpy as np

    from medrag.index.retriever import MedRAGRetriever, MODE_CONTEXT, MODE_SPARSE

    class _Chunk:
        def __init__(self, doc, kind):
            self.doc_id, self.section_kind, self.text, self.title = doc, kind, "body", "T"
            self.journal, self.year, self.categories = "J", 2020, (kind,)
            self.chunk_id = f"{doc}#{kind}"
            self.section_title, self.is_abstract = kind, False
            self.pmid = self.pmcid = self.doi = ""
            self.publication_types = ()

    # Row 0 is a Methods chunk BM25 ranks first. Row 1 is the Results chunk that
    # qualifies it -- present in the document and reachable through the dependency
    # graph, but not something the lexical query matches, so it is not in the BM25
    # results and can only arrive by expansion.
    chunks = {0: _Chunk("docA", "methods"), 1: _Chunk("docA", "results")}

    class _Store:
        def get(self, row):
            return chunks.get(row)

        def __len__(self):
            return 2

    class _Dep:
        """The Methods chunk (row 0) has the Results chunk (row 1) as its
        interpreting neighbour -- exactly the section-typed edge the dependency
        graph is built from."""

        def context_pairs(self, seeds, *, top_k=10, max_per_seed=6):
            return [(seed, 1, 0.9) for seed in seeds if int(seed) == 0]

    class _BM25:
        def search(self, q, top_k=10):
            return [(0, 10.0)]

    cfg = types.SimpleNamespace(
        fusion=types.SimpleNamespace(page_alpha=0.5, context_seeds=1,
                                     context_per_seed=1, sparse_top_k=10,
                                     bleed_alpha=0.3, bleed_mode="max"),
        rerank=types.SimpleNamespace(enabled=False, depth=10),
    )
    r = MedRAGRetriever(bm25=_BM25(), store=_Store(), dependency=_Dep(), cfg=cfg)

    base = r.retrieve("q", top_k=1, mode=MODE_SPARSE, include_text=False)
    ctx = r.retrieve("q", top_k=1, mode=MODE_CONTEXT, include_text=False)
    assert len(base.passages) == 1, base.passages
    assert len(ctx.passages) > len(base.passages), (
        f"the context arm returned no more than the baseline: "
        f"{[p.chunk_id for p in ctx.passages]}")
    assert "context" in ctx.timings, f"the context path never ran: {ctx.timings}"
    assert [p.chunk_id for p in ctx.passages][:1] == [p.chunk_id for p in base.passages], \
        "the context arm must still lead with the seed it was given"
    print(f"  ok  context mode returns {len(ctx.passages)} passages where sparse "
          f"returns {len(base.passages)}, with text omitted")


def test_context_expansion_honours_its_configured_budget() -> None:
    """seeds x per_seed must actually control how many passages expansion adds.

    Regression, and one that was invisible rather than absent. ``_apply_context`` did not
    pass ``context_budget``, so ``expand_with_context`` defaulted it to ``top_k`` and
    capped expansion at ten passages whatever the configuration said.

    It looked correct at the shipped default, because ``5 * 2 == top_k == 10`` -- the
    configuration happened to agree with the cap. The moment it was changed to 10 x 12,
    ``retrieve`` computed a 130-passage output window while the expansion could supply
    20. A whole falsifier was run through that path and compared two ten-passage arms
    while believing it compared two budgets.
    """
    import types

    from medrag.index.retriever import MedRAGRetriever

    class _Chunk:
        def __init__(self, doc, kind, text="body"):
            self.doc_id, self.section_kind, self.text = doc, kind, text
            self.title, self.journal, self.year = "T", "J", 2020
            self.categories = (kind,)
            self.chunk_id = f"{doc}#{kind}"
            self.section_title, self.is_abstract = kind, False
            self.pmid = self.pmcid = self.doi = ""
            self.publication_types = ()

    # 10 seeds, each with 4 contextual neighbours, plus 3 unrelated docs as distractors.
    chunks = {}
    for s in range(10):
        chunks[s] = _Chunk(f"doc{s}", "results")
        for c in range(4):
            chunks[100 + s * 10 + c] = _Chunk(f"doc{s}", "discussion")

    class _Store:
        def get(self, row):
            return chunks.get(int(row))

        def __len__(self):
            return len(chunks)

    class _Dep:
        def context_pairs(self, seeds, *, top_k=10, max_per_seed=6):
            out = []
            for seed in seeds:
                s = int(seed)
                for c in range(4):
                    out.append((s, 100 + s * 10 + c, 0.9))
            return out

    def run(seeds_n, per_n):
        cfg = types.SimpleNamespace(
            fusion=types.SimpleNamespace(page_alpha=0.5, bleed_alpha=0.3,
                                         bleed_mode="max", context_seeds=seeds_n,
                                         context_per_seed=per_n),
            rerank=types.SimpleNamespace(enabled=False),
        )
        r = MedRAGRetriever(store=_Store(), dependency=_Dep(), cfg=cfg)
        hits = [(row, 10.0 - row * 0.1) for row in range(10)]
        return r._apply_context(hits, top_k=10)

    five_by_two = run(5, 2)
    ten_by_twelve = run(10, 12)
    # 10 hits + 5*2 added = 20. Before the fix both of these returned 20, because the
    # missing context_budget defaulted to top_k.
    assert len(five_by_two) == 20, len(five_by_two)
    assert len(ten_by_twelve) == 50, (
        f"10 x 12 should add 40 passages, not {len(ten_by_twelve) - 10}; the budget is "
        f"still being capped at top_k")
    assert len(run(2, 1)) == 12, "2 x 1 should add exactly 2"
    print(f"  ok  the configured budget is honoured: 5x2 adds "
          f"{len(five_by_two) - 10}, 10x12 adds {len(ten_by_twelve) - 10}")


def main() -> int:
    tests = [
        test_alpha_zero_reproduces_the_base_ranking,
        test_a_better_document_promotes_its_passage,
        test_a_missing_page_vector_keeps_the_base_score,
        test_page_scoring_can_only_promote_never_override,
        test_ordering_is_invariant_to_the_base_score_scale,
        test_page_scoring_actually_reorders,
        test_empty_page_scores_is_a_no_op,
        test_normalise_bounds_the_discount,
        test_page_scores_are_addressed_by_position_not_by_chunk_row_id,
        test_context_modes_return_their_context_passages_without_text,
        test_context_expansion_honours_its_configured_budget,
        test_the_page_modes_are_registered,
    ]
    print("Page-level conditioning tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())