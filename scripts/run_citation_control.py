"""Circularity control on the full-text corpus: citation-derived gold and bpref.

Why this exists
---------------
The MeSH-derived gold labels and the graph retriever share their signal: both come
from MeSH descriptors. A gain measured against MeSH gold is therefore partly a
measurement of the retriever agreeing with itself. The control is to derive relevance
from a source the retriever never touches -- bibliographies -- and re-run the
comparison.

Citation-derived gold: every seed document indexed under a descriptor votes for the
documents it cites, and a document becomes gold once ``min_seeds`` distinct seeds
vote for it. bpref (Buckley & Voorhees 2004) is the headline metric because these
labels are incomplete by construction -- exactly the regime where that metric exists.

This existed as a library with no committed driver, which made the control
irreproducible: the abstract-corpus numbers came from an ad-hoc script. Written out
here so the control can be re-run on any corpus.

The ``judged`` pool, which is the subtle part
---------------------------------------------
bpref's ``judged`` set must be the pool the rankings were drawn from, not the gold.
Passing the MeSH gold as the judged pool -- which happened here once -- makes bpref
degenerate to 0.0000 for every arm, because every ranked document is "relevant" and
there is nothing to prefer, inventing a reversal that does not exist. For a complete
pool, pass ``judged=None``. For an explicitly incomplete pool, pass it.

Both regimes are reported side by side, because the point of the control is the
*ordering* of arms, not the absolute level: levels from two different label spaces are
not comparable to each other, only the direction of a difference is.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.ERROR)

from medrag.config import PipelineConfig
from medrag.data.acquire import iter_documents
from medrag.eval.citation_labels import (
    build_citation_labels,
    is_seed_document,
    label_agreement,
    select_seeds,
)
from medrag.eval.metrics import Qrels, bpref, ndcg_at_k
from medrag.index.retriever import load_index
from medrag.index.sibling_bleed import expand_with_context

log = logging.getLogger("citation_control")

#: doc_id -> citation tuple, and descriptor -> set of doc_ids carrying it.
citations: dict[str, tuple[int, ...]] = {}
mesh_by_desc: dict[str, set[str]] = {}


_n_seen = 0


#: doc_id -> MeSH descriptors borrowed from the PubMed record with the same PMID.
#:
#: PMC JATS records carry no MeSH at all -- 6 descriptors across 555,142 documents --
#: so seed selection produced zero seeds and the control was reported as structurally
#: inapplicable. PMC-OA articles are indexed in PubMed under the same PMID, so
#: ``scripts/attach_mesh_to_pmc.py`` joins the two and attaches MeSH to 97,069 PMC
#: documents (17.49%, 14,612 distinct descriptors).
#:
#: That coverage is stated rather than presented as the whole corpus. It is limited by
#: the abstract corpus holding 1.02M of roughly 37M PubMed records, crawled over six
#: publication-type slices, so the joined subset is enriched for RCTs, systematic
#: reviews, meta-analyses and guidelines. A control measured here describes that slice
#: of PMC and not PMC as a whole.
MESH_SIDECAR: dict[str, list[str]] = {}

#: bare PMID -> corpus document id.
#:
#: Needed because ``build_citation_labels`` defaults to emitting ``pmid:<ref>`` for
#: cited references, while PMC documents are keyed ``pmcid:PMC...``. With the default,
#: the citation labels and the retrieved documents share zero identifiers, so agreement
#: measured as Jaccard 0.0000 across all 139 descriptors and bpref came back 0.0000 on
#: both arms over 139 queries with citation gold. That is an artefact of the key format
#: -- the exact failure the `id_of` docstring in citation_labels.py warns about -- and it
#: printed as a clean-looking "INCONSISTENT" verdict rather than as an error.
DOC_ID_BY_PMID: dict[str, str] = {}


def load_mesh_sidecar(path: str | None) -> dict[str, list[str]]:
    """Read the doc_id -> MeSH sidecar, or nothing if no path was given."""
    if not path:
        return {}
    import gzip as _gzip

    with _gzip.open(path, "rt") as fh:
        payload = json.load(fh)
    return payload.get("by_doc_id", {})


def _cite_pass(batch, citations, mesh_by_desc):
    """Keep only what the control needs from a batch of Documents.

    ``list(iter_documents(cfg))`` over 555,142 PMC documents reached 22 GB of RSS --
    every Document carries its sections, authors, MeSH terms and citations -- and was
    killed twice while competing for a 31 GiB box. Nothing here needs the objects:
    voting and seed selection only want the identifier, the citation list and the
    MeSH descriptors.
    """
    global _n_seen
    for d in batch:
        did = d.doc_id
        _n_seen += 1
        pmid = getattr(d, "pmid", "") or ""
        if pmid:
            DOC_ID_BY_PMID.setdefault(str(pmid), did)
        cits = tuple(getattr(d, "citations", ()) or ())
        if cits:
            citations[did] = cits
        # The sidecar wins where present, because on PMC it is the only source of MeSH;
        # where it is absent the document's own terms are used, which is the abstract
        # corpus path and unchanged.
        borrowed = MESH_SIDECAR.get(did)
        if borrowed:
            for desc in borrowed:
                mesh_by_desc.setdefault(desc, set()).add(did)
            continue
        for m in getattr(d, "mesh_terms", ()) or ():
            desc = (m.descriptor or "").strip()
            if desc:
                mesh_by_desc.setdefault(desc, set()).add(did)


def build_seeds(citations, mesh_by_desc, max_seeds):
    """Seed documents from the streamed pairs.

    Mirrors :func:`select_seeds`, including its requirement that a seed carry at
    least one MeSH descriptor -- seeds vote *per descriptor*, so a seed with no
    descriptor cannot vote at all. On a corpus without MeSH that yields zero seeds,
    which is the honest result rather than a control silently restricted to nothing.
    """
    from medrag.eval.citation_labels import CitationSeed

    with_mesh = {d for d, docs in mesh_by_desc.items() if docs} if mesh_by_desc else set()
    out = []
    for did, cits in citations.items():
        if len(cits) < 5:
            continue
        descs = frozenset(d for d in with_mesh if did in mesh_by_desc[d])
        if not descs:
            continue
        out.append(CitationSeed(doc_id=did, pmid="", year=0, mesh=descs,
                                citations=cits))
        if len(out) >= max_seeds:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--seed-count", type=int, default=10)
    ap.add_argument("--context-per-seed", type=int, default=1)
    ap.add_argument("--min-seeds", type=int, default=2)
    ap.add_argument("--max-seeds", type=int, default=40_000)
    ap.add_argument("--hold", type=int, default=5_000,
                    help="Documents held in memory at once while streaming the corpus")
    ap.add_argument("--mesh-sidecar", default=None,
                    help="gzipped JSON of doc_id -> MeSH descriptors, from "
                         "scripts/attach_mesh_to_pmc.py; required on corpora whose "
                         "records carry no MeSH of their own")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    global MESH_SIDECAR
    MESH_SIDECAR = load_mesh_sidecar(args.mesh_sidecar)
    if MESH_SIDECAR:
        print(f"  MeSH sidecar: {len(MESH_SIDECAR):,} documents", flush=True)

    cfg = PipelineConfig.load(args.config)._finalize()
    print(f"  index dir: {cfg.paths.index_dir}", flush=True)
    print(f"  corpus root: {cfg.paths.data_root}", flush=True)
    r = load_index(cfg, with_reranker=False, with_graph=False, with_dependency=True,
                   with_dense=False)
    if r.dependency is None:
        print("  dependency graph missing; run `medrag dependency-graph` first")
        return 1

    # ---- citation-derived gold -------------------------------------------------
    #
    # Streamed, not materialised. `list(iter_documents(cfg))` over 555,142 PMC
    # documents reached 22 GB of RSS -- every Document carries its sections, authors,
    # MeSH terms and citations -- and was killed twice while competing with another
    # job for a 31 GiB box. Nothing here needs the objects: seed selection and
    # descriptor voting only want the identifier, the citation list and the MeSH
    # descriptors, which together are a few hundred bytes per document.
    t0 = time.time()
    docs_ref: list = []
    for doc in iter_documents(cfg):
        docs_ref.append(doc)
        if len(docs_ref) >= args.hold:
            _cite_pass(docs_ref, citations, mesh_by_desc)
            docs_ref.clear()
    if docs_ref:
        _cite_pass(docs_ref, citations, mesh_by_desc)
        docs_ref.clear()
    # `citations` only holds documents that HAVE a bibliography, so it cannot also
    # be the document count -- summing its values summed the citation tuples.
    n_docs = _n_seen
    have_cit = len(citations)
    print(f"  corpus {n_docs:,} documents, {have_cit:,} with bibliographies "
          f"({100 * have_cit / max(1, n_docs):.1f}%) in {time.time() - t0:.0f}s",
          flush=True)

    seeds = build_seeds(citations, mesh_by_desc, args.max_seeds)
    if not seeds:
        # Seed selection requires a MeSH descriptor because seeds vote per
        # descriptor. PMC JATS records carry none -- 0.0% over a 20,000-document
        # sample -- so the citation-derived control cannot be constructed on this
        # corpus at all.
        print("  NO CITATION SEEDS: seed selection requires MeSH descriptors and this\n"
              "  corpus has none, so the circularity control cannot be built here.\n"
              "  That is the finding: the control guards against the retriever sharing\n"
              "  signal with MeSH-derived labels, and on full text there are no MeSH\n"
              "  labels to share signal with.")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"n_documents": n_docs, "docs_with_bibliographies": have_cit,
                       "mesh_descriptors": len(mesh_by_desc),
                       "citation_seeds": 0, "control_possible": False,
                       "reason": "seed selection requires MeSH descriptors; "
                                 "PMC JATS records carry none (0.0% coverage)"}, fh, indent=2)
        print(f"  -> {args.out}")
        return 0
    print(f"  {len(seeds):,} seed documents", flush=True)

    bm = Qrels.load(args.qrels)
    qs = [q for q in bm.queries if q.relevant_docs]
    random.Random(7).shuffle(qs)
    qs = qs[: args.n_queries]
    # Each query's descriptor is the MeSH term in its qid, NOT its category.
    #
    # The benchmark's qids are `clinical::<MeSH descriptor>` -- "clinical::Delayed Graft
    # Function", "clinical::Peptic Ulcer" -- because the gold was built from MeSH. The
    # previous code passed `q.category`, whose values are coarse group names
    # ("diseases", "drugs-and-therapeutics"), and handed them to build_citation_labels as
    # restrict_descriptors. Those are not MeSH descriptors, so every label was filtered
    # away: 40,000 seeds were built and then discarded because not one carried a
    # descriptor named "diseases". The result printed as
    # "arm ordering under citation gold: NOT MEASURED -- 0 queries have citation gold",
    # which is honest about the number and completely silent about the cause.
    def _descriptor_of(q):
        qid = q.qid or ""
        if "::" in qid:
            return qid.split("::", 1)[1].strip()
        # No namespace: the qid may already be the descriptor.
        return qid.strip() or (q.category or "")

    descriptors_by_query = {q.qid: _descriptor_of(q) for q in qs}
    known_desc = {d for d in descriptors_by_query.values() if d in mesh_by_desc}
    print(f"  query descriptors: {len(set(descriptors_by_query.values()))} distinct; "
          f"{len(known_desc)} of them present in this corpus's MeSH", flush=True)
    labels = build_citation_labels(
        seeds, min_seeds=args.min_seeds,
        restrict_descriptors=set(descriptors_by_query.values()),
        id_of=lambda ref: DOC_ID_BY_PMID.get(str(ref), f"pmid:{ref}"),
    )
    print(f"  citation labels: {labels.summary()}", flush=True)

    # ---- MeSH labels over the same corpus, for the agreement measurement -------
    #
    # Agreement has to be measured per *descriptor*, which is what the citation
    # labels are keyed by. The qrels only carry a coarse `category`, so comparing
    # those to citation labels would compare seven groups and say nothing. Instead
    # the MeSH side is read straight off the corpus: the documents carrying each
    # descriptor. That is the same pseudo-labelling scheme the retriever's advantage
    # would be measured against, so it is the right thing to compare to.
    print(f"  MeSH descriptors in corpus: {len(mesh_by_desc):,}", flush=True)

    # Descriptors where both regimes have an opinion, so Jaccard is meaningful.
    shared_desc = [d for d in labels.by_descriptor if d in mesh_by_desc]
    print(f"  descriptors with both regimes: {len(shared_desc):,}", flush=True)
    if MESH_SIDECAR:
        n_side = sum(1 for d in mesh_by_desc if True)
        print(f"  NOTE: MeSH on this corpus is BORROWED from the PubMed record with the "
              f"same PMID, over {len(MESH_SIDECAR):,} documents "
              f"({100 * len(MESH_SIDECAR) / max(n_docs, 1):.2f}% of the corpus). The "
              f"subset is enriched for RCTs, systematic reviews, meta-analyses and "
              f"guidelines, because the abstract corpus was crawled over six "
              f"publication-type slices. Agreement and ordering measured here describe "
              f"that slice of PMC, not PMC as a whole.")
    elif not mesh_by_desc:
        # Measured on this corpus: PMC JATS records carry no MeSH terms at all
        # (0.0% over a 20,000-document sample), and the crawl logged
        # `mesh taxonomy stats: {'cached': 0, 'api_calls': 0}` because there was
        # nothing to resolve. So without a sidecar the control cannot be built here.
        print("  NOTE: no MeSH terms in this corpus and no sidecar supplied; label "
              "agreement cannot be measured. Run scripts/attach_mesh_to_pmc.py and "
              "pass --mesh-sidecar to borrow MeSH from the PubMed records.")

    # ---- rank both arms once ---------------------------------------------------
    ranked: dict[str, dict[str, list[str]]] = {q.qid: {"sparse": [], "context": []}
                                               for q in qs}
    mesh_gold: dict[str, set[str]] = {q.qid: set(q.relevant_docs) for q in qs}
    judged_pool: set[str] = set()

    t0 = time.time()
    for i, q in enumerate(qs):
        pool = r._sparse_search(q.text, args.pool)
        seeds_rows = [(int(x), s) for x, s in pool[: args.top_k]]
        arms = {
            "sparse": [x for x, _ in seeds_rows],
            "context": [int(m[0]) for m in expand_with_context(
                seeds_rows, r.store, r.dependency, top_k=args.top_k,
                seed_count=args.seed_count,
                context_per_seed=args.context_per_seed)[: args.top_k + args.seed_count * args.context_per_seed]],
        }
        for name, rows in arms.items():
            ids: list[str] = []
            for row in rows:
                c = r.store.get(int(row))
                if c is None:
                    continue
                ids.append(c.doc_id)
                judged_pool.add(c.doc_id)
            ranked[q.qid][name] = ids
        if (i + 1) % 50 == 0:
            print(f"  ranked {i + 1}/{len(qs)} | {time.time() - t0:.0f}s", flush=True)

    # ---- evaluate under both regimes ------------------------------------------
    out: dict[str, object] = {
        "n_queries": len(qs),
        "judged_pool_size": len(judged_pool),
        "min_seeds": args.min_seeds,
        "citation_labels": labels.summary(),
        "arms": {},
    }

    print("\n  MeSH-derived gold (complete pool, judged=None):", flush=True)
    mesh: dict[str, dict[str, float]] = {}
    for name in ("sparse", "context"):
        bp = [bpref(ranked[q.qid][name], mesh_gold[q.qid]) for q in qs
              if mesh_gold[q.qid]]
        nd = [ndcg_at_k(ranked[q.qid][name], mesh_gold[q.qid], 10) for q in qs
              if mesh_gold[q.qid]]
        mesh[name] = {
            "bpref": round(sum(bp) / len(bp), 4) if bp else 0.0,
            "ndcg@10": round(sum(nd) / len(nd), 4) if nd else 0.0,
            "n_queries": len(bp),
        }
        print(f"    {name:9s} bpref {mesh[name]['bpref']:.4f}  ndcg@10 "
              f"{mesh[name]['ndcg@10']:.4f}  ({len(bp)} queries)")

    print("\n  citation-derived gold (incomplete pool, judged=pool):", flush=True)
    cite: dict[str, dict[str, float]] = {}
    for name in ("sparse", "context"):
        bp, nd, used = [], [], 0
        for q in qs:
            gold = labels.relevant(descriptors_by_query[q.qid])
            if not gold:
                continue
            used += 1
            b = bpref(ranked[q.qid][name], gold, judged=judged_pool)
            n = ndcg_at_k(ranked[q.qid][name], gold, 10)
            if b == b:
                bp.append(b)
            if n == n:
                nd.append(n)
        cite[name] = {
            "bpref": round(sum(bp) / len(bp), 4) if bp else 0.0,
            "ndcg@10": round(sum(nd) / len(nd), 4) if nd else 0.0,
            "n_queries": used,
        }
        print(f"    {name:9s} bpref {cite[name]['bpref']:.4f}  ndcg@10 "
              f"{cite[name]['ndcg@10']:.4f}  ({used} queries with citation gold)")

    if shared_desc:
        agree = label_agreement({d: mesh_by_desc[d] for d in shared_desc},
                                {d: set(labels.relevant(d)) for d in shared_desc})
    else:
        agree = {"mean_jaccard": None, "n_descriptors": 0,
                 "note": "no MeSH terms in this corpus; agreement not measurable"}
    out["arms"] = {"mesh_gold": mesh, "citation_gold": cite}
    out["label_agreement"] = agree
    out["ordering_mesh"] = mesh["context"]["bpref"] > mesh["sparse"]["bpref"]
    # Comparing two zeros is not an ordering. When no query has citation gold, both
    # arms report bpref 0.0000 and `0.0 > 0.0` is False -- which prints as
    # "arm ordering under citation gold: context > sparse = False", i.e. the control
    # reversed the ranking. Nothing was measured. Reporting that as a reversal is
    # exactly the failure this project keeps hitting: a plausible number where an
    # honest "not measured" belongs.
    _cite_measured = cite["context"]["n_queries"] > 0
    out["ordering_citation"] = (
        cite["context"]["bpref"] > cite["sparse"]["bpref"] if _cite_measured else None
    )
    out["citation_gold_measured"] = _cite_measured

    print("\n  CONTROLLING FOR PSEUDO-LABEL CIRCULARITY", flush=True)
    jac = agree.get("mean_jaccard")
    _ = jac  # used below
    print(f"    label agreement (mean Jaccard): "
          f"{'not measurable (no MeSH terms in this corpus)' if jac is None else f'{jac:.4f}'}")
    print(f"    arm ordering under MeSH gold:      context > sparse = "
          f"{out['ordering_mesh']}")
    # A citation label set that shares no identifier with anything the arms retrieve
    # cannot order the arms, and `0.0 > 0.0` prints as False -- a reversal where nothing
    # was discriminated. Say so instead of computing a comparison that has no meaning.
    _cite_identifiers = {d for v in labels.by_descriptor.values() for d in v}
    _cite_overlap = len(_cite_identifiers & judged_pool)
    out["citation_label_identifiers"] = len(_cite_identifiers)
    out["citation_labels_in_retrieved_pool"] = _cite_overlap
    _cite_degenerate = _cite_measured and _cite_overlap == 0
    if _cite_degenerate:
        out["ordering_citation"] = None
        out["ordering_consistent"] = None
        print(f"    arm ordering under citation gold:  NOT MEASURED -- the citation "
              f"label set holds {len(_cite_identifiers):,} identifiers and none of them "
              f"appears in the {len(judged_pool):,}-document retrieved pool. Both arms "
              f"score 0.0000 because neither can reach the labels, not because they "
              f"rank equally.")
    elif _cite_measured:
        print(f"    arm ordering under citation gold:  context > sparse = "
              f"{out['ordering_citation']}")
    else:
        print("    arm ordering under citation gold:  NOT MEASURED -- 0 queries have "
              "citation gold, so both arms are 0.0000 and the comparison is vacuous")
    # With no MeSH labels here, "consistent" is not testable; only the citation
    # ordering exists. Reporting consistency on one arm's ordering would be circular.
    if jac is None:
        consistent = None
        out["ordering_consistent"] = None
    else:
        consistent = out["ordering_mesh"] == out["ordering_citation"]
        out["ordering_consistent"] = consistent
    if _cite_degenerate:
        consistent = None
        print("    -> NOT TESTABLE: the citation-derived labels are unreachable from "
              "the retrieved pool, so they cannot order the arms.")
    elif consistent is None:
        print("    -> NOT TESTABLE here: one label regime only. The cross-corpus "
              "consistency check belongs to the abstract corpus.")
    else:
        print(f"    -> {'CONSISTENT' if consistent else 'INCONSISTENT'}: the arm "
              f"ranking {'survives' if consistent else 'does not survive'} replacing "
              f"MeSH labels with bibliographies")
    out["ordering_consistent"] = consistent

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())