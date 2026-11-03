# MedCARAG: Context-Aware Medical RAG Benchmark and Full-Text Index

**MedCARAG (Medical Context-Aware Retrieval-Augmented Generation)** is an open benchmark,
full-text index and toolkit for **context-aware medical RAG**. It tests whether a medical
RAG system retrieves the context that makes a clinical finding safe to repeat (the
population, the comparator and the caveats from the same study), as well as the passage
that matches the question. It covers **4,044,247 PubMed Central Open Access full-text
articles (PMC-4M)** and **20.5M PubMed abstracts (PubMed-20M)**.

Use it to:

- **benchmark medical RAG retrievers** on whether they return qualified evidence, with
  *readiness* metrics that check the retrieved passages for a rate, a population, a
  caveat and a benefit;
- **compare context-aware retrieval with direct RAG** (BM25 top-k) under passage-count
  and random-retrieval controls, on the same 2,000 MeSH-labelled clinical queries;
- **build biomedical retrieval indexes larger than RAM** (BM25, dense vectors, section
  graphs) across several machines.

A paper describing the resource is under review; a link will be added here.

## What you get

| Resource | Size | Licence |
|---|---|---|
| Section-typed full-text index of the PubMed Central Open Access Subset (**PMC-4M**) | 4,044,247 articles, 109.6M chunks | CC BY 4.0 (derived indexes) |
| Section-typed index of PubMed abstracts (**PubMed-20M**) | 20.5M abstracts, 48.3M chunks | CC BY 4.0 (derived indexes) |
| Two smaller full-text indexes (PMC-RCT, PMC-300K) | 555K and 300K articles | CC BY 4.0 |
| BM25 (segmented, exact), document vectors, chunk vectors (PubMed-20M), a same-article section graph | on the [`indexes-1`](https://github.com/Azraelix316/medcarag-context-aware-medical-rag/releases/tag/indexes-1) release (229 GB, article text excluded); rebuildable with the builders | CC BY 4.0 |
| MeSH-derived clinical benchmark: queries, qrels, PMID-to-MeSH mapping, PMID/PMCID manifests | `benchmark/` (large files on the GitHub release); 2,539 PMC-4M and 2,459 PubMed-20M queries | CC0 |
| **Readiness** metrics: does the retrieved evidence state a rate, a population, a caveat, a benefit? | `medrag/eval/readiness.py` | Apache-2.0 |
| Per-query result rows, significance tests, 2,700 generated answers | `docs/runs/` | CC0 |
| Multi-host build tools for indexes larger than RAM | `medrag/index/`, `scripts/` | Apache-2.0 |

No article text is redistributed. You rebuild the text from NCBI with the acquisition
scripts, and the derived indexes line up with it row for row. See
[`DATA_LICENSE.md`](DATA_LICENSE.md).

## Key findings

On 2,000 clinical queries over 4M full-text articles:

- **Direct RAG is poor at qualified evidence.** BM25 top-10 finds an article on the right
  topic for 60.6% of queries, but returns a rate together with its population for 14.2%.
- **Context-aware retrieval roughly doubles it** (27.9%) by conditioning on the whole
  document and returning more of it. It stays 5.8 points ahead of BM25 given the same
  number of passages, so the gain is not just more text.
- **The section has little influence.** Random passages of the retrieved articles
  do as well as passages chosen by section type.
- **Part of the gain reaches the answer.** With a Gemma 4 26B-A4B generator, answers
  citing a rate from the right study rise from 21% to 30% (significant after
  correction). Caveats are almost never passed on, whatever the retrieval.

## Quick start

```bash
pip install -e ".[index]"
medrag -c configs/smoke.yaml doctor          # environment and NCBI connectivity check
medrag -c configs/smoke.yaml acquire         # 300 recent RCT abstracts from PubMed
medrag -c configs/smoke.yaml chunk
medrag -c configs/smoke.yaml build-index
medrag -c configs/smoke.yaml page-index        # document vectors, for page / pagectx
medrag -c configs/smoke.yaml dependency-graph  # same-article graph, for the context modes
medrag -c configs/smoke.yaml query "statin myopathy risk" --mode pagectx
```

Retrieval modes (`--mode`): `sparse` (direct BM25 RAG), `page` (document conditioning),
`bleed` (sibling re-scoring), `context` (same-document expansion), `pagectx`,
`bleedctx`, plus `dense`, `hybrid` and `reranked`.

Score readiness on any retrieved set:

```python
from medrag.eval.readiness import assess_readiness
verdict = assess_readiness(passages, gold_doc_ids)   # passages: RetrievedPassage list
verdict.ready, verdict.has_rate, verdict.has_population, verdict.has_caveat
```

## Reproducing the results

| Result | Command | Output |
|---|---|---|
| Six retrieval arms on four corpora | `scripts/run_mode_ablation.py`, then `scripts/paired_bootstrap.py` | `docs/runs/mode_ablation_*.json`, `significance_*.json` |
| Passage-count controls | `scripts/matched_controls.py` | `docs/runs/matched_controls_*.json` |
| Random-retrieval floor | `scripts/random_floor.py` | `docs/runs/random_floor_*.json` |
| Answer-side read test | `scripts/run_read_test_rows.py` with an OpenAI-compatible endpoint (llama.cpp `llama-server`) | `docs/runs/read_test_*.json`, `read_answers_*.jsonl` |
| Index verification and SHA-256 manifests | `scripts/release_manifest.py`, `scripts/verify_*.py` | `docs/runs/verify_*`, `release_manifest_*` |

Every result is stored in `docs/runs/` as written by these scripts. Seeds are fixed (7).

## Building at scale

PMC-4M's chunk store is 320 GB. The builders split chunking, store assembly, BM25
segments, document vectors and graph construction into units that any number of hosts
claim from a shared directory. Each parallel path is tested to produce byte-identical
output to the single-process build (`tests/`).

## FAQ

**What is context-aware medical RAG?**
Medical retrieval-augmented generation (RAG) that retrieves the documents behind a match,
not only the best-matching passage. A clinical result such as "the complication rate was
4%" is only safe to repeat with whom it applies to and what limits it, and those usually
sit in the same article's Methods, Discussion or Conclusions, which a direct RAG
retriever does not return.

**How is MedCARAG different from MedRAG and MIRAGE?**
MIRAGE and the MedRAG toolkit benchmark medical question answering over PubMed titles and
abstracts and score the final answer. MedCARAG is a separate resource built on full text:
it scores whether the *retrieved evidence* carries its qualifying context, and releases
full-text indexes, controls and per-query rows so a new retriever can be compared on the
same queries.

**What retrieval methods are included?**
Direct BM25 RAG (`sparse`), document conditioning (`page`), sibling re-scoring (`bleed`),
same-document expansion (`context`) and their combinations (`pagectx`, `bleedctx`), plus
dense (bge-small-en-v1.5), hybrid and re-ranked modes.

**Does it include article text?**
No. PMC articles carry mixed licences, so the release has derived indexes, identifiers and
labels only; the text is rebuilt from NCBI with the acquisition scripts and lines up with
the indexes row for row.

**Where are the indexes?**
On the [`indexes-1`](https://github.com/Azraelix316/medcarag-context-aware-medical-rag/releases/tag/indexes-1) release (229 GB, split into
parts under 1.9 GB, with SHA-256 manifests). Identifier lists, the PMID-to-MeSH mapping and
large qrels are on [`data-1`](https://github.com/Azraelix316/medcarag-context-aware-medical-rag/releases/tag/data-1).

## Citation

```bibtex
@misc{medcarag2026,
  title  = {MedCARAG: Context-Aware Medical RAG Benchmark and Full-Text Index},
  author = {Chang, Jared and Yeah, Sophia},
  year   = {2026},
  note   = {Code, benchmark and indexes},
  url    = {https://github.com/Azraelix316/medcarag-context-aware-medical-rag}
}
```

## Licence

Code: Apache-2.0 (`LICENSE`). Data: CC0 for the benchmark and results, CC BY 4.0 for
derived indexes (`DATA_LICENSE.md`). PubMed data courtesy of the U.S. National Library
of Medicine; NLM does not endorse this resource.
