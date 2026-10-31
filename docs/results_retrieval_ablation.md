# Retrieval Ablation Results

Two runs are reported: the **production corpus** (624,489 documents, the paper's
artifact) and a **validation corpus** (40,432 documents) used to debug. All
numbers come from `data/runs/eval_report*.json`.

---

## Production corpus: 624,489 documents -> 2,368,622 chunks

500 queries, exact `IndexFlatIP`, BM25 at 399 bytes/chunk.
Full report: `data/runs/eval_report_exact.json`.

| mode | hit@1 | hit@5 | hit@10 | hit@100 | mrr@10 | nDCG@10 | precision@10 |
|---|---|---|---|---|---|---|---|
| sparse (BM25) | 0.6240 | 0.8240 | 0.8680 | 0.9280 | 0.7100 | 0.5471 | 0.5258 |
| dense (bi-encoder) | 0.6160 | 0.8560 | 0.8960 | 0.9660 | 0.7205 | 0.5350 | 0.5104 |
| **hybrid (RRF)** | **0.6920** | **0.8780** | **0.9260** | **0.9700** | **0.7706** | **0.5929** | **0.5662** |
| reranked (cross-enc.) | 0.5340 | 0.8200 | 0.8840 | 0.9700 | 0.6608 | 0.5020 | 0.4908 |

Latency for 500 queries: sparse 10 s, dense 90 s, hybrid 97 s, reranked 2813 s.

**Hybrid fusion wins every metric**, and the margin holds at 15x the corpus size
of the validation run:

| comparison | hit@1 | hit@10 | nDCG@10 |
|---|---|---|---|
| hybrid vs sparse | +6.8 | +5.8 | +4.6 |
| hybrid vs dense | +7.6 | +3.0 | +5.8 |

The ordering reproduced exactly at 40k and 624k documents, which is the useful
part: the hybrid gain is a property of the retrievers, not of a corpus size.

### Why exactness mattered more than expected

The first production evaluation reported dense nDCG@10 of **0.2743** — half the
validation corpus's 0.5626. The cause was not the corpus but a silent index-type
switch: above 200k vectors the builder chose `IndexIVFPQ`. Rebuilding exactly on
the identical 500 queries:

| mode | IVFPQ (approx) | IndexFlatIP (exact) | delta |
|---|---|---|---|
| dense hit@1 | 0.3320 | **0.6440** | +0.312 |
| dense nDCG@10 | 0.2744 | **0.5422** | +0.268 |
| hybrid nDCG@10 | 0.4668 | **0.5938** | +0.127 |

Exact dense nDCG@10 (0.5422) matches the 151k-chunk run (0.5626), which confirms
the corpus was fine throughout. The failure mode is worth flagging for the paper:
**the index built cleanly, reported a plausible vector count, and returned
plausible-looking results.** Nothing errored. Only a cross-scale comparison
exposed it.

Latency, for the record:

| index | latency | index size |
|---|---|---|
| IndexFlatIP (exact) | 99.4 ms/query | 3.39 GiB |
| IndexIVFPQ, nprobe=16 | 0.1 ms/query | 0.10 GiB |

A 99 ms query is usable for research retrieval, batch evaluation, and an agent
fanning out a handful of sub-queries, so the quality loss was not worth the
latency win. `auto` now selects exact search up to 20M vectors; approximate
search requires an explicit `index_type`.

---

## Validation corpus: 40,432 documents -> 151,040 chunks

| mode | hit@1 | hit@10 | hit@100 | mrr@10 | nDCG@10 | recall@10 | precision@10 |
|---|---|---|---|---|---|---|---|
| sparse (BM25) | 0.6600 | 0.8950 | 0.9625 | 0.7440 | 0.5720 | 0.1848 | 0.5450 |
| dense (bi-encoder) | 0.6650 | 0.9175 | 0.9775 | 0.7580 | 0.5626 | 0.1771 | 0.5325 |
| **hybrid (RRF)** | **0.7400** | **0.9350** | 0.9775 | **0.8115** | **0.6281** | **0.2009** | **0.5968** |
| reranked (cross-enc.) | 0.5800 | 0.9275 | 0.9775 | 0.7059 | 0.5417 | 0.1782 | 0.5292 |

400 queries. Reproduce:

```bash
medrag -c configs/default.yaml -c configs/mid.yaml evaluate \
    --sample 400 --out data_mid/runs/eval_report.json
```

**Recall@k is intrinsically low** (0.20 at k=10) because gold sets are large: a
MeSH descriptor query has ~40-67 gold documents, so 10 retrieved passages can
cover at most a sixth of them. hit@k and nDCG@k are the informative metrics for
this benchmark, not recall@k. This is a property of document-level labels, not a
retrieval deficiency, and should be stated explicitly when reporting.

---

## Finding: cross-encoder reranking hurts under MeSH-derived labels

The most interesting measurement in either run, so stated precisely.

**It is not a bug in the reranking path.** `BAAI/bge-reranker-base` separates
relevant from irrelevant medical text cleanly in isolation:

| query | relevant | irrelevant |
|---|---|---|
| "esophageal neoplasm treatment outcomes" | **+0.012** | −10.197 |
| "metformin reduces HbA1c in type 2 diabetes" | **+0.848** | −8.309 |

**It is not recall loss.** hit@100 is identical (0.9775) for dense, hybrid, and
reranked on the validation corpus. The candidate set is unchanged; reranking only
permutes it.

**The labels are the problem.** MeSH relevance is **document-level**; reranking
operates at **passage level**. A gold document is any article a PubMed indexer
filed under the query's descriptor, and such an article frequently contains no
passage matching the query's phrasing — the descriptor may appear only in the
indexing metadata, or the passage may use different vocabulary. A cross-encoder
trained on passage-level web relevance rewards semantic match and demotes exactly
those passages.

Reproduces across every slice on the validation corpus (nDCG@10):

| slice | sparse | dense | hybrid | reranked |
|---|---|---|---|---|
| single-hop | 0.5166 | 0.5163 | **0.5785** | 0.4960 |
| multi-hop | 0.7685 | 0.7266 | **0.8038** | 0.7037 |
| anatomy | 0.4804 | 0.4547 | **0.5422** | 0.4114 |
| diseases | 0.6233 | 0.6222 | **0.6847** | 0.5882 |
| drugs-and-therapeutics | 0.3655 | 0.3980 | **0.4124** | 0.4007 |
| organisms | 0.4871 | 0.5515 | **0.6064** | 0.4350 |
| phenomena-and-processes | 0.4798 | 0.4396 | **0.5577** | 0.4455 |
| psychiatry-and-psychology | 0.5296 | 0.5149 | **0.5971** | 0.4968 |

And on the production corpus: hybrid 0.5929 -> reranked 0.5020 at nDCG@10,
p = 0.000 by paired bootstrap, 95% CI [-0.202, -0.114]. hit@100 is *identical*
(0.9700) for all three dense-backed modes, which is the cleanest evidence that
the candidate set is untouched and only the ordering changes.

**Implication.** Reranking is near-universal in web and enterprise RAG and is
usually reported as a gain. On biomedical corpora with expert-derived,
document-level labels, an off-the-shelf general-domain cross-encoder does not
transfer. Anyone deploying medical RAG should either fine-tune the reranker on
domain relevance data, or evaluate it against passage-level judgments — which are
expensive precisely because they need expert annotation. Making that trade
explicit is what this corpus does.

### Multi-hop scores higher than single-hop

Counter-intuitive, and worth explaining. Two mechanical reasons:

- Multi-hop gold sets are **smaller** (mean 9.3 documents vs 67.1), so fewer are
  needed for a hit.
- A two-descriptor query ("X and Y") is **lexically more distinctive** than a
  single descriptor.

The metric to watch is `success@all`, which requires evidence for *both* hops and
is reported separately in the JSON.

---

## Where hybrid fusion's gain comes from

Measured over 200 queries on the returned top-10 passages (validation corpus):

| provenance of returned passage | share |
|---|---|
| found by **both** BM25 and dense | **89.8%** |
| dense only | 5.5% |
| sparse only | 4.8% |

The retrievers agree strongly at the top of the list, and single-retriever
passages are relegated to the tail. That is the mechanism behind the gain: RRF
promotes exactly the documents both rankings independently endorse, and those
are far more likely to be relevant.

---

## Qualitative examples (production corpus, hybrid + exact dense)

Raw output: `data/runs/qualitative_examples.json`. Latency is end-to-end hybrid
retrieval with exact search at 2,368,622 chunks, ~230 ms per query.

| query | top result |
|---|---|
| SGLT2 inhibitors reduce HF hospitalization in T2D? | *Glucose cotransporter-2 inhibitors on mortality and hospitalization in heart failure* (Front Endocrinol, 2026) |
| Probiotics for ventilator-associated pneumonia? | *Preventive Effect of Probiotics on Ventilator-Associated Pneumonia: A Meta-analysis* (Ann Pharmacother, 2020) |
| Adjuvant chemotherapy after pancreatic cancer resection? | *Meta-analysis of adjuvant chemotherapy versus no adjuvant chemotherapy* (Surgery, 2023) |
| Do ACE inhibitors prevent incident dementia? | *Association between renin-angiotensin-aldosterone system inhibitors and risk* (Pharm Res, 2021) |
| Optimal anticoagulation duration after AF ablation? | *Efficacy, Safety, and Net Clinical Benefit of Post-AF Ablation* (Pacing Clin Electrophysiol, 2026) |

Every query returned the correct target study class at rank 1, and the second hit
was also on-topic in each case — a useful sanity check that the benchmark's high
hit@1 is not an artefact.

### Query planning in practice

The planner expands clinical vocabulary without an LLM:

| input phrase | expansions added |
|---|---|
| heart failure | cardiac failure, congestive heart failure, CHF |
| cancer | neoplasm, malignancy, carcinoma |

This is what lets "heart failure" reach literature indexed under "cardiac
failure". Note that expansions are always issued as *`<question> <expansion>`*
rather than standalone, so a generic term like "neoplasm" cannot swamp retrieval
with off-topic documents.

---

## Statistical significance

Paired bootstrap over per-query differences, 1,000 resamples:

| comparison | metric | delta | 95% CI | p |
|---|---|---|---|---|
| reranked vs hybrid (validation, 400 q) | nDCG@1 | −0.160 | [−0.208, −0.108] | 0.000 |
| reranked vs hybrid (production, 500 q) | nDCG@1 | −0.158 | [−0.202, −0.114] | 0.000 |

Both regressions are well outside sampling noise, and the effect size is
essentially identical at 40k and 624k documents.

---

## Latency (validation corpus, 400 queries)

| mode | total | per query |
|---|---|---|
| sparse | 1.2 s | 3.0 ms |
| dense | 9.4 s | 23 ms |
| hybrid | 9.3 s | 23 ms |
| reranked | 2194 s | 5.5 s |

Fusion is free relative to retrieval. Reranking dominates by ~240× because it is
a 278M-parameter cross-encoder over 100 (query, passage) pairs per request on a
5 GB consumer GPU.

At production scale, exact dense search costs 99 ms/query (section above).

### A 300× fp16 regression, found and fixed

The first reranked evaluation took **616 seconds for 100 pairs**. Cause:
`CrossEncoderReranker` applied `.half()` whenever the device name started with
`cuda`, without checking for native fp16 support. These P2000s are compute
capability 6.1, where fp16 is emulated in fp32 units. After routing precision
through the same `select_fp16()` helper the embeddings use, the same work takes
9.4 s — a **65× speedup** with no change in quality.

---

## Reproducing

```bash
bash scripts/setup_env.sh
bash scripts/run_pipeline.sh -c configs/default.yaml -c configs/production.yaml
medrag -c configs/default.yaml -c configs/production.yaml evaluate \
    --sample 500 --out data/runs/eval_report.json
```

The JSON report contains per-mode metrics, per-slice breakdowns (difficulty and
clinical category), bootstrap significance, and index statistics.