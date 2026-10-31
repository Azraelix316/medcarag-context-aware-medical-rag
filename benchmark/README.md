# MedCARAG benchmark files

The clinical benchmark used in the paper, for both corpora. Released under CC0.

| file | what |
|---|---|
| `queries_clinical2000.jsonl` | the 2,000 clinical queries every experiment uses (sampled with seed 7); one `{"qid", "query"}` per line |
| `qrels_clinical.json` | gold sets for all clinical queries (PMC-4M: 2,539; PubMed-20M: 2,459), keyed by `qid`; a document is relevant if NLM indexed it under the query's MeSH descriptor |
| `benchmark_summary.json` | query counts, mean gold-set sizes and MeSH categories |
| `RELEASE_MANIFEST.json` | SHA-256 sums and sizes of every index file, for checking a download or a rebuild |

`pmc4m/` uses PMC-4M document ids (`pmcid:PMC...`), and `pubmed20m/` uses PubMed-20M
ids (`pmid:...`).

These larger files are attached to the GitHub release instead, because they exceed
GitHub's file-size limit:

- `pmc4m_ids.tsv.gz`, `pubmed20m_ids.tsv.gz`: the PMID/PMCID manifests (doc_id, pmid,
  pmcid, doi, year) that the corpora are rebuilt from;
- `pmc_mesh_20m.json.gz`: the PMID-to-MeSH mapping that gives PMC-4M articles their
  descriptors;
- `qrels_mesh_*.json.gz`: gold sets for the descriptor-only queries (30,684 for PMC-4M,
  20,373 for PubMed-20M; not used in the paper's results).

MeSH descriptors and PubMed identifiers are courtesy of the U.S. National Library of
Medicine, which does not endorse this resource; they reflect the snapshot the corpora
were built from and may not match NLM's current data.
