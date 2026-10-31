# Corpus scale: what is reachable, and what the number in the paper should be

## The problem

The plan targets ~20,000,000 documents. Measured against the live NCBI index, the
six query slices currently in use return:

| query slice | records |
|---|---|
| `randomized controlled trial[pt]` | 672,380 |
| `systematic review[pt]` | 347,720 |
| `meta-analysis[pt]` | 244,892 |
| `clinical practice guideline[pt]` | 19,032 |
| `randomized controlled trial[pt] AND ("2020"[dp]:"2026"[dp])` | 163,504 |
| `multicenter study[pt] AND ("2018"[dp]:"2026"[dp])` | 172,136 |
| **sum before cross-query dedup** | **1,619,664** |

That is **3.94%** of PubMed's 41.1M records. The crawl has fetched 1,260,150
documents, so it is roughly 78% through everything these queries can ever return
and will exhaust them within the hour, after which it plateaus.

**The 20M figure is therefore not a time budget to be optimised. It is a
different corpus definition.** No amount of additional compute or additional
machines reaches it with these queries. A paper claiming a 20M-document corpus
built from these slices would be refuted by anyone who counts.

## Why the slices are publication-type restricted, and what that costs

Publication-type filtering is what makes this corpus appropriate for evidence
synthesis. Every document is a trial, systematic review, meta-analysis or
guideline, which is why:

- benchmark labels derived from MeSH indexing are meaningful (a document indexed
  under a descriptor *and* restricted to a trusted design);
- the citation-seed construction has something to seed from (2,908 seeds in the
  1990s, 292,996 in the 2020s);
- the reranking negative result is about evidence-synthesis literature rather than
  about incidental mentions.

Broadening to topical queries would multiply the corpus size and **destroy that
property**: MeSH pseudo-labels over all of PubMed are much closer to "documents
that mention this topic", and the whole evaluation weakens.

## Options, with the trade-off stated

| option | reachable size | keeps design validity | cost |
|---|---|---|---|
| A. Complete the current slices | ~1.4M unique | yes | ~1 h remaining crawl; done |
| B. Add PMC OA full text | +589K RCTs with full sections | yes, and strengthens it | ~1 day crawl + page embeddings |
| C. Broaden to topical slices | 10-20M | **no** — labels weaken | ~40 h crawl + ~50 h embedding |
| D. Crawl all of PubMed by date | 41M | no | ~85 h crawl + ~130 h embedding |

C and D buy a bigger number and cost the reason the corpus is worth building.

## Recommendation

**A, then B.** Complete the evidence-synthesis corpus (option A), then add PMC
Open Access full text (option B), which is the only addition that *increases*
what the system can do rather than diluting it: full Methods/Results/Discussion
structure is what the section-dependency graph and the context-provision
evaluation actually operate on.

Under the objective's own cost discipline — concentrate the expensive add-ons on
the PMC Open Access articles where real section structure exists rather than
paying for abstract-only documents that cannot support dependency structure —
option A+B is the coherent choice. It yields roughly 2.0M documents of which
589K carry full text and dense section structure.

## The number the paper should state

Not "20M documents". The defensible claims, in descending order of strength:

1. **~1.4M evidence-synthesis documents** — the complete set of PubMed-indexed
   RCTs, systematic reviews, meta-analyses and practice guidelines in English
   from 1996, which is a census rather than a sample of that design class.
2. **44.6% of all RCTs ever indexed**, from the earlier 624K measurement.
3. **1.51% of all PubMed**, or **100% of the target design class.**

Claim 1 is the strong one and it is checkable. A reviewer can run the same query
and get the same ceiling. That is a better position than a number they cannot
reproduce but can easily disprove.

## Measurement to back the claim

Verify completion by counting distinct PMIDs per publication type in the corpus
and comparing against the live index counts above. The gap is documents lost to
dedup or to query exhaustion, and it should be reported as a percentage rather
than assumed.

```
python - <<'PY'
import gzip, glob, json, collections
types = collections.Counter(); seen = set()
for f in glob.glob('data/data_scale/corpus/pubmed/*.jsonl.gz') + \
         glob.glob('data/data_cst1/corpus/pubmed/*.jsonl.gz'):
    with gzip.open(f, 'rt') as fh:
        for line in fh:
            d = json.loads(line)
            if d['doc_id'] in seen: continue
            seen.add(d['doc_id'])
            for t in d.get('publication_types') or (): types[t] += 1
print('unique documents:', len(seen))
for t, n in types.most_common(8): print(f'  {n:>10,}  {t}')
PY
```
## The full-text corpus: measured, and why it is the one that matters

The abstract corpus above is a census of a design class. The contextual-contamination
work needs something it cannot use, so the PMC Open Access full-text corpus was
crawled alongside it. These are measured numbers, not projections from the abstract
side.

### The full-text corpus was capped by its query, not by the crawl

This is the most consequential scale fact in the project, and it was invisible for a
long time because the crawl was working perfectly the entire time.

`pmcmerged` was acquired with

```
"open access"[filter] AND "randomized controlled trial"[pt]
```

That query has an NCBI ceiling of **590,762 records**. It is not a rate limit, not a
pagination limit and not a time limit -- it is the total number of PMC OA articles
carrying the randomised-controlled-trial publication type, full stop. The corpus holds
**555,142** of them: **94.0% of everything that query can ever return**.

So the corpus did not finish early and did not stall. It was *done*, in the sense that
another month of crawling would return 35,620 documents and then stop permanently.
Every diagnostic that treats corpus size as a function of elapsed time or crawler
health points the wrong way here: the crawler is correct, the schedule is correct, and
the ceiling is a property of the query string.

Measured ceilings for the candidate queries, so the size of this is a number and not an
intuition:

| PMC query | NCBI count | vs pmcmerged |
|---|---|---|
| `OA AND "randomized controlled trial"[pt]` | 590,762 | 1.00x -- the corpus's own ceiling |
| `OA AND "clinical trial"[pt]` | 919,173 | 1.56x |
| `"open access"[filter]` | **8,298,989** | **14.05x** |

Dropping the publication-type restriction is not a marginal widening, it is a 14x one,
and it is the only change that makes the full-text corpus limited by disk and wall-clock
rather than by NCBI.

### ...but 8.3M is the number of *records*, not of full texts

Stating 8,298,989 as the expansion would overstate what is reachable, because PMC's
`"open access"` filter marks the **PDF** as free, not the **JATS body** as present. Some
fraction of open-access records have no machine-readable body at all. Probed live, 40
records per era, counting characters inside `<body>`:

| era | OA records | >=200 chars | >=5,000 chars | section-typed |
|---|---|---|---|---|
| 1995 | 3,762 | 45.0% | 30.0% | 12.5% |
| 2000 | 5,536 | 85.0% | 77.5% | 35.0% |
| 2005 | 16,724 | 67.5% | 60.0% | 50.0% |
| 2010 | 82,931 | 97.5% | 80.0% | 92.5% |
| 2015 | 255,549 | 97.5% | 92.5% | 92.5% |
| 2018 | 354,927 | 100.0% | 95.0% | 100.0% |
| 2021 | 737,191 | 97.5% | 92.5% | 97.5% |
| 2024 | 792,631 | 100.0% | 95.0% | 97.5% |
| 2026 | 678,936 | 100.0% | 92.5% | 97.5% |
| **sampled** | | | **79.4%** | |

79.4% of open-access records clear the filter's 5,000-character floor, so the expansion
is:

| quantity | documents |
|---|---|
| `pmcmerged` acquired | 555,142 |
| `pmcmerged` query ceiling | 590,762 |
| all PMC OA records | 8,298,989 |
| **achievable full-text documents** | **~6,593,085** |

**11.2x the query ceiling, 11.9x what is held.** Not the 14.05x that the record count
alone suggests, and the paper should quote the 11.9x. Section structure -- what the
dependency graph actually consumes -- is scarcer still and strongly era-skewed: 12.5% in
1995, 35% in 2000, 50% in 2005, 90%+ from 2010 on.

This is the direct, measured answer to the era-bias question. `pmcmerged`'s skew toward
2021-2026 is **not** an artefact of the RCT filter and **not** something a 14x wider
crawl fixes; it is a property of when PMC acquired machine-readable full text. Widening
the query widens *type* coverage, which was a choice. Widening *era* coverage was never
available, because the full text does not exist to be had.

The same bug sits one level down, in `DEFAULT_QUERIES["pmc"]`, whose default was
`'"open access"[filter] AND has_text_ml[filter]'`. `has_text_ml` is a PubMed-side filter
that does not exist in the PMC database and matches **zero** records there:

```
"open access"[filter] AND has_text_ml[filter]"   count = 0
"has_text_ml"[filter]"                          count = 0
"open access"[filter]"                           count = 8,298,989
```

NCBI echoes the term back *untranslated*, so esearch succeeds, returns count 0, and a
crawl against that default exits clean having acquired nothing. Every PMC config in the
repo overrode it, which is exactly why it survived: the failure is invisible to any test
that goes through a config and only reachable by the path a new user is most likely to
take. Both facts are now pinned as static properties in `tests/test_acquire_filter.py`,
with a mutation check confirming the old default fails them.

### What the expansion is, and what it is not

`configs/pmcsweep_c7.yaml` sweeps the whole PMC OA subset into a **new sidecar root**,
`data/pmcsweep`. `pmcmerged` is not touched, so every number already measured
on it stays reproducible against exactly the bytes it was measured on.

Per-year availability, measured, which is what sets the slice sizes:

| year | PMC OA records | note |
|---|---|---|
| 1996 | 4,001 | era the RCT corpus had only 133 pre-2001 documents of |
| 2000 | 5,536 | |
| 2005 | 16,724 | |
| 2010 | 82,931 | |
| 2015 | 255,549 | |
| 2020 | 597,726 | |
| 2025 | 929,767 | exceeds the 900k per-slice cap the earlier configs used |
| 2026 | 678,936 | |

The cap is 2,000,000 per slice rather than 900,000 because 2024-2026 alone exceed 900k
per year, and a 900k cap would silently discard the most recent literature -- the part
most likely to carry MeSH indexing and section structure. Early years exhaust naturally
(1996 gives 4,001), so the ceiling only binds where the data actually is.

**What this does not fix.** Publication skews hard toward recent years, so acquiring all
of PMC OA does not remove the era bias in `docs/citation_coverage_era_bias.md`. That
skew is a property of what is open access, not of the RCT filter. What the expansion
*does* widen is **type** coverage, from randomised trials alone to observational work,
case reports and reviews. Type bias was the half of this bias that was actually a
choice.

Rate budget: the abstract sweep runs at 3.0 req/s against NCBI's 10 req/s key ceiling,
leaving 7; the PMC sweep takes 5, for 8 total. In practice PMC is parse-bound rather
than request-bound -- 32.8 records/s was measured at an effective 0.16 efetch/s, under
2% of the budget -- so the real NCBI cost of running both crawlers is close to the
3 req/s already being spent. Disk is not a constraint: 8.3M records at the measured
~10.4 KB compressed each is ~86 GB against 3.3 TB free.

### What the crawl actually returned, and what it threw away first

The first pass over 1996-2026 returned 137,313 and 142,166 records on the two nodes
-- a union of 142,186. That looked like the corpus. It was not: the keep filter was
discarding 76% of what it fetched, because `"research article"` is 75% of PMC
publication types and was absent from a PubMed-shaped whitelist
(`medrag/data/acquire.py`; `tests/test_acquire_filter.py`).

After the fix, live keep rates are 86-97% by era. On the re-crawl the observed rate
was 140,072 kept of 147,400 fetched (94.8%).

| corpus | documents | what it supports |
|---|---|---|
| PubMed abstracts | 624,489 (dense index) / 1,118,153 (full) | hybrid retrieval, qrels, bpref |
| PMC OA full text | 589,687 available, ~578K kept | dependency graph, context provision |

### Chunk scale, measured end to end

The earlier projection from a 3,000-document slice (37.0 chunks/doc) was too high,
because that slice came from the 2023-2026 era. The real corpus is era-weighted and
older papers have fewer sections.

| node | era | documents | chunks | chunks/doc |
|---|---|---|---|---|
| c7 | 2023-2026 + 1996-2010 | 282,904 | 8,745,959 | 30.9 |
| worker-B | 2011-2022 | 272,238 | 7,387,592 | 27.1 |
| **total** | 1996-2026 | **555,142** | **16,133,551** | **29.1** |

Every figure verified by an independent recount of the shard files rather than by
reading the manifest, which is what caught a 1,325,000-chunk duplicated prefix on
the worker-B side (see below). The merged store recounts to 16,133,551 rows in 3,228
shards, matching exactly, and random-access rows at 0, 1, 100,000, 4,000,000,
8,000,000 and 16,133,550 all return the correct document and section kind.

Throughput and cost, measured:

- chunking: 130-164 docs/s per node; both nodes in parallel
- store build: 3,611 chunks/s, 5,888 s, 46.6 GB blob
- dense embedding: ~119 rows/s per shard, ~476 rows/s across 4 shards
- **embedding ETA ~9.4 h** for 16,133,551 chunks at 384 dimensions (30 GB of vectors)
- BM25 over 16.1M chunks: ~3,850 chunks/s, ~70 min

Disk is not the constraint: 46.6 GB store, ~30 GB vectors, within the 282 GB free
locally and 3.5 TB on NFS.

### Three accounting bugs this scale exposed

Worth recording, because each produced a plausible number rather than an error.

1. **The chunk manifest counted shards it only resumed over.** `ShardWriter`
   defaults to `resume=True` and seeds `shard_files` from disk, so `close()`
   returned the whole directory. A first chunking run that crashed on a
   `NameError` left 266 shards; the retry resumed at index 266 and wrote 7,387,592
   chunks. Its manifest claimed 7,387,592 while the directory held 8,712,592 -- an
   18% inflation that would have propagated into every embedded-row count, index
   size and retrieval denominator.
2. **A read path silently started a full crawl.** `corpus_shards` called
   `crawl_corpus` when it found no shards, and every read path goes through it. The
   merged shards are prefixed (`pmcoa2_c7__pmc-*.jsonl.gz`) while the glob looks for
   `pmc-*.jsonl.gz`, so `medrag build-index` began re-crawling all of PMC OA,
   writing 30,000 documents and spending the API budget with no error.
3. **All four embedding shards built BM25 over the full corpus.** BM25 is a
   whole-corpus artifact, so the `if not bm25_path.exists()` guard does not help
   when four nodes start together. Four-fold duplicated work, caught by watching
   four processes report identical progress.

None of the three would have been visible in a test that checked return values.
Each was found by comparing an independently recomputed number against a recorded
one.

### Why the add-ons belong only here

The objective's cost-discipline constraint was that page embeddings (~40 extra hours
across 20M documents) and section-context re-embedding (~64 extra hours) must be
spent on the 589K PMC full-text articles rather than on abstract-only documents.
That is now not a preference but a measurement: chunk section kinds on the PMC
slice are dominated by `methods` (33,604), `results` (28,846) and `discussion`
(25,753) against 35,368 unclassified, with `references` essentially excluded (41).
Abstracts contribute almost nothing to a Methods/Results/Discussion dependency
graph, because an abstract has no Methods section to depend on.

### The honest ceiling

**589,687 is the number for PMC OA RCTs, and it is checkable.** It is also recent:
the 1996-2000 window returns zero records, so 8,376 documents cover 1996-2010
against 291,534 for 2023-2026. A claim of "1996-2026 coverage" would be false.

**The 20M target in the original plan is not reachable at document level with these
design-class filters**, and the difference is a corpus-definition problem rather
than a time-budget problem. The nearest defensible reading is that 20M is a
*chunk* count, and the PMC full text reaches 16.1M of it -- short of 20M, so even
that reading does not reach the target and should not be claimed as though it does. Both readings should
be stated explicitly in the paper rather than letting the reader assume documents.


## The section-dependency graph at full scale

Built over all 16,133,551 chunks and verified independently of the builder:

| | |
|---|---|
| nodes (chunks) | 16,133,551 |
| undirected edges | 80,810,064 |
| directed entries in CSR | 161,620,128 |
| documents | 554,069 |
| mean degree | 10.02 |
| max degree | 1,711 |
| isolated nodes | 3,517 |
| persisted size | 1.43 GB |
| build time | 4,306 s |

Mean degree by section kind, which is the structure the motivating claim predicts:

| kind | mean degree |
|---|---|
| conclusions | 21.80 |
| discussion | 19.59 |
| results | 11.57 |
| methods | 10.45 |
| background | 2.62 |
| (unclassified) | 2.62 |
| references | 2.33 |

Conclusions and Discussion sit an order of magnitude above Background, which is the
intended asymmetry: every Methods or Results chunk reaches its document's
interpretive summary, while a Background chunk mostly reaches the chunk that
follows it in the argument. Results outrank Methods because a Results chunk both
draws its Methods qualifier and is itself qualified by the discussion -- two
dependencies where Methods has one.

For scale: the earlier graph over the abstract corpus carried 2,813,321 edges over
2,368,622 chunks, or 1.19 edges per chunk. This corpus carries 5.01. That difference
is the whole justification for concentrating the add-ons on PMC Open Access full
text: an abstract has no Methods section to depend on, so the structure that makes
the mechanism work simply is not present.

Verification was independent of the build: the saved pickle was reloaded and checked
for 16,133,551 nodes, a non-decreasing `indptr` whose final value equals
`len(indices)`, in-range indices, unique document ids, and reciprocity across 3,000
sampled nodes with zero violations.


## The co-citation graph at full scale

Built over the same 555,142 documents, after the PMC JATS bibliography parser made
citations available at all:

| | |
|---|---|
| documents with bibliographies | 552,941 |
| documents entering the graph | 548,700 |
| distinct references | 7,784,100 |
| citation edges | 28,672,183 |
| enumerated pairs (max_citers=200) | 232,992,097 |
| distinct pairs | 170,445,592 |
| **surviving pairs (min_shared=2)** | **30,854,914** |
| mean degree | 112.47 |
| max degree | 6,212 |
| peak RSS | 5,073 MiB |
| build time | 618 s |

Era coverage of documents with bibliographies:

| decade | documents |
|---|---|
| 2020s | 416,083 |
| 2010s | 130,394 |
| 2000s | 6,330 |
| 1990s | 112 |
| 1980s | 19 |
| 1970s | 2 |
| 1960s | 1 |

This is the third graph and the second at full scale. Against the abstract corpus it
is roughly 2.7x the pair count (30,854,914 against 11,219,027) on 1.4x the
documents, which is the co-citation-side counterpart of the dependency graph's 5.01
versus 1.19 edges per chunk: PMC full text is denser on both structural measures.

Verification independent of the builder: 61,709,828 directed entries equal
`indptr[-1]`, `indptr` non-decreasing, indices in range, every `shared` value at or
above `min_shared`, **zero reciprocity violations across 40,762 edges sampled from
1,200 documents**, and the shared-reference count agreeing in both directions for
every edge sampled.

### The memory pattern, twice

Both full-scale graph builds were killed by the OOM killer before being rewritten,
and in both cases the intermediate existed only to be converted to CSR numpy:

| build | accumulator | peak before | after | mechanism |
|---|---|---|---|---|
| dependency graph | `dict[int, list[tuple[int, float]]]` | 21.9 GiB (killed) | 5.4 GiB | `array('i')`/`array('f')`, numpy dedup |
| co-citation | `Counter[tuple[int, int]]` | ~35 GiB (killed) | 5.1 GiB | packed int64 keys, `np.unique` |

The lesson generalises past this repository: when a Python-object structure is an
intermediate for an array you are about to build, the intermediate *is* the memory
cost, and at corpus scale it is the whole cost.
