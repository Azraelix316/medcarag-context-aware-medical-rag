# Data licences

The **code** in this repository is under the Apache License 2.0 (`LICENSE`).
The **data** we release is licensed by component, because some of it is derived from
third-party article text:

| Component | Licence | Notes |
|---|---|---|
| Benchmark query files, qrels, the PMID-keyed MeSH sidecar, PMID/PMCID manifests | **CC0 1.0** | Identifiers and NLM index terms only, with no article text. MeSH is courtesy of the U.S. National Library of Medicine. |
| Per-query result rows, readiness flags, read-test answers, significance files (`docs/runs/`) | **CC0 1.0** | Experiment outputs. The read-test answers are model output, not article text. |
| Derived indexes: BM25 segments, document (page) vectors, chunk vectors, section-dependency graph, store offsets | **CC BY 4.0** | See "Source licences" below. |
| Article text and chunk text | **Not redistributed** | Rebuild it from NCBI with the released acquisition scripts. |

## Source licences

The indexes are built from:

- **PMC Open Access Subset** articles. Each article carries its own licence: CC0, CC BY,
  CC BY-SA or CC BY-ND (commercial use allowed); CC BY-NC variants (non-commercial
  only); or custom/unspecified terms. Our licence on the derived indexes does not
  replace those terms. If your use depends on them (for example, commercial use),
  check the licence of each article you rely on, or rebuild the indexes from the
  commercial-use subset with the released scripts.
- **PubMed abstracts.** Courtesy of the U.S. National Library of Medicine. NLM does not
  endorse this resource, and the data may not reflect NLM's latest records. Abstracts
  may be under publisher copyright, which is why no abstract text is shipped.

Bulk retrieval from NCBI must follow the E-utilities usage policy (rate limits,
registered `tool` and `email`). The acquisition scripts respect it by default.

CC0 1.0: https://creativecommons.org/publicdomain/zero/1.0/
CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
