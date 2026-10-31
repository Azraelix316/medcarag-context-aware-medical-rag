# Results index

Every measurement this project makes, with the artifact it came from. Measured, not
projected. Where a number is not comparable to another, the reason is stated.

## Retrieval ablation: readiness and context provision

| corpus | arm | readiness | quant | population | caveat | bare |
|---|---|---|---|---|---|---|
| PubMed abstracts, 624,489 docs | sparse | 0.3051 | — | — | 0.1060 | — |
| | + context | 0.4902 | — | — | 0.2444 | — |
| **PMC full text, 554,069 docs** | sparse | **0.0625** | 0.1075 | 0.1575 | 0.0450 | 0.2200 |
| | + context | **0.1375** | 0.1650 | 0.2525 | 0.1050 | 0.1625 |
| | delta | **+0.0750** | +0.0575 | +0.0950 | +0.0600 | −0.0575 |

The two corpora are **not** comparable. On PMC, gold was found for only 131 of 400
queries; the other 269 score zero on every dimension by construction. Read the
direction, not the level. `data/runs/pmc_fulltext_eval.json`

## End-to-end: what reaches the reader

120 queries, local Gemma 4 12B, identical prompt, paired per query.

| ingredient | input gain | output gain | transmission |
|---|---|---|---|
| population | +19% | +26% | 136% |
| rate | +69% | +37% | 54% |
| **caveat** | **+180%** | **+25%** | **14%** |

**These answer-side figures come from the concurrent harness and are superseded.** Its
noise range, measured afterwards, was 0.0666 on caveat against effects of 0.0167
(Falsifier 17), so 14% is inside the band and cannot be distinguished from it. The
serial re-measurement at the end of this document gives caveat transmission **+0.33**
against a floor of exactly zero. `data/runs/read_test.json`

## Grounded attribution: closing the caveat gap

120 queries, arms sharing one retrieval pass. Conditional = passages contained a
caveat; unconditional = they did not.

| arm | overall | conditional | unconditional | declined |
|---|---|---|---|---|
| baseline | 0.0333 | 0.0556 | 0.0152 | 0.0000 |
| required | 0.0250 | 0.0556 | 0.0000 | 0.8167 |
| two-pass | 0.1583 | 0.2963 | 0.0455 | 0.1833 |
| **attributed** | **0.4500** | **1.0000** | **0.0000** | 0.5500 |

Asking for a limitation made the model worse. Two-pass bought 4.2× recall with a 4.6%
fabrication rate. Attribution is exact by construction: conditional 1.0000,
unconditional 0.0000. `data/runs/attributed_caveats.json`

## Components

| component | verdict | evidence |
|---|---|---|
| dependency-graph expansion | supported | every readiness dimension improves on full text |
| grounded attribution | supported, with a stated limit | conditional 1.0000 / unconditional 0.0000 |
| sibling bleed | **contradicted** | MeSH bpref 0.4059→0.3083, citation bpref 0.2131→0.1615 |
| cross-hop framing | **contradicted** | readiness 0.2262 vs 0.2881, including on multi-hop |
| page-level vectors | built, not yet ablated | 554,069 document vectors |

## Corpus and scale

| corpus | documents | chunks | graphs |
|---|---|---|---|
| PMC OA full text | 555,142 | 16,133,551 | dependency 80,810,064 edges; co-citation 30,854,914 pairs |
| PubMed abstracts | 624,489 (dense) | 3,583,195 | co-citation 11,219,027 pairs |

Citation coverage by era, measured by a full pass over every document in both corpora
(`scripts/measure_era_bias.py`):

| corpus | documents | coverage 1996-2000 | coverage 2021-2026 | refs/doc, all records |
|---|---|---|---|---|
| PMC OA full text | 555,142 | 84.2% | 99.7% | 31.8 → 53.7 |
| PubMed abstracts | 960,880 at measurement | **6.6%** | **55.1%** | **1.42 → 24.53** |

Coverage on PMC is effectively era-flat from 2006 onward, so its co-citation graph is
not era-truncated. On abstracts coverage rises 8.4x across eras and references per
document rise 17x, so that graph is a sample of recent literature. PMC instead carries a
**composition** bias: 68.1% of its documents are 2021-2026 and only 133 predate 2001.

The era bias was traded between corpora, not removed. See
`docs/citation_coverage_era_bias.md`.

## Component ablation through the shipped retrieval path (PMC full text)

400 queries, every arm via `Retriever.retrieve(mode=...)`. Companion to Falsifier 9.

| arm | readiness | caveat | bare | gold found | qualified | qar | bar |
|---|---|---|---|---|---|---|---|
| sparse | 0.0625 | 0.0450 | 0.2200 | 85 | 16 | 0.0400 | 0.1725 |
| bleed | 0.0825 | 0.0525 | 0.1600 | 84 | 24 | 0.0600 | 0.1500 |
| context | 0.1050 | 0.1000 | 0.1875 | 100 | 68 | 0.1700 | 0.0800 |
| bleedctx | 0.1150 | 0.0975 | 0.1425 | 95 | 63 | 0.1575 | 0.0800 |
| page | 0.1025 | 0.0700 | 0.2150 | 95 | 31 | 0.0775 | 0.1600 |
| **pagectx** | **0.1475** | **0.1175** | 0.1800 | **109** | **79** | **0.1975** | **0.0750** |

`pagectx` against the baseline: readiness +0.0850, caveat rate +0.0725 (2.6x), bare
rate -0.0400, qualified answers 16 -> 79 (4.9x), gold found 85 -> 109.

**Page-level vectors are the cheapest component and they work.** `page` alone is
+0.0400 readiness and qar 0.0400 -> 0.0775, at query time and with no corpus
re-embedding. That is the practical difference from Anthropic Contextual Retrieval
and LangChain Contextual Embeddings, both of which rewrite chunk text and re-embed.

**Sibling bleed's verdict is corpus-dependent, not settled.** Contradicted on abstracts
(MeSH bpref 0.4059 -> 0.3083, citation bpref 0.2131 -> 0.1615); helps on full text
(+0.0200 readiness, and `bleedctx` beats `context` on readiness and bare rate). Both
stand.

**The sparse baseline's caveat rate is 0.0450**, against 4-5% of *responses* in
arXiv 2502.14898. Different corpus, different granularity, same figure.

Source: `data/runs/mode_ablation_pmc.json`, driver `scripts/run_mode_ablation.py`.

## Circularity control on PMC (Falsifier 14)

Previously reported as structurally inapplicable. Joining PMC to PubMed on PMID supplies
MeSH to 17.49% of PMC documents (`scripts/attach_mesh_to_pmc.py`), and the control runs:

| arm | MeSH-derived gold | citation-derived gold |
|---|---|---|
| sparse | bpref 0.1548, nDCG@10 0.0578 (150q) | bpref 0.0373, nDCG@10 0.0139 (139q) |
| context | bpref 0.1665, nDCG@10 0.0578 (150q) | bpref 0.0474, nDCG@10 0.0139 (139q) |
| **ordering** | context > sparse | context > sparse |

Label agreement mean Jaccard **0.0175**, median 0.0, over 139 descriptors with opinion
under both regimes. Mean citation gold size 38 documents, median 20.

**The arm ranking is consistent across both label regimes.** `context` beating `sparse`
on PMC is not an artefact of MeSH-derived labels — the regime the graph retriever shares
signal with. Document-level claims remain unlicensed: at 0.0175 agreement, a document
judged relevant under one regime is almost never judged relevant under the other.

The 17.49% MeSH coverage is not a random sample. The abstract corpus holds 1.02M of
roughly 37M PubMed records, crawled over six publication-type slices, so the joined
subset is enriched for RCTs, systematic reviews, meta-analyses and guidelines.

Source: `data/runs/citation_control_pmc.json`.


## Expansion budget, answer-side, inside the generator's capacity (Falsifier 16) -- SUPERSEDED

**Every answer-side number in this section was produced by a harness whose own noise
floor is larger than the effects, and its conclusion has been withdrawn.** It is kept
because it is the evidence for the withdrawal, not because it is a result. The
replacement measurement is the serial run at the bottom of this section.

240 queries, four arms, every arm produced 240/240 answers -- no generation failures.

| arm | input ready | input caveat | output caveat | output benefit | output ready |
|---|---|---|---|---|---|
| sparse | 0.0750 | 0.0542 | **0.0417** | 0.1583 | 0.2833 |
| context 5x2 | 0.1250 | 0.1292 | **0.0333** | 0.1958 | 0.2708 |
| context 10x2 | 0.1875 | 0.1667 | **0.0292** | 0.2083 | 0.2708 |
| context 10x3 | 0.2208 | 0.1833 | **0.0250** | 0.2083 | 0.2792 |

Passage-side caveat provision rises monotonically, 3.4x from baseline to the largest
budget the generator can read. The answer's caveat rate **falls** monotonically over the
same four points. Transmission is -0.11, -0.11 and -0.13.

**This "falls monotonically" reading is wrong, and the total spread it rests on is
0.0167 against a measured noise range of 0.0666 (Falsifier 17).** The monotonicity is
the part that is not noise -- four points in order are not four points at random -- but
four points inside the noise band can order any way they like, so the ordering of a
quantity smaller than the band it lives in carries no information. The passage-side
rise is real and deterministic; the answer-side fall is an artefact of batch
composition in the generator.

Population transmission decays 0.50 -> 0.26 -> 0.22 and benefit 1.00 -> 0.63 -> 0.57;
quantification is negative but improves, -0.67 -> -0.19 -> -0.17.

**How strong:** each comparison alone is underpowered (sign tests p = 0.79, 0.55, 0.39
on caveats; composite ready flat at p = 0.72, 0.74, 1.00), and the cross-tab disagrees
with the rate (0.077, 0.065, 0.100, 0.068). The monotonicity across four nested budgets
is the part that is not noise.

The generator's ceiling bounds what can be tested at all: 30-40 passages fit
`n_ctx_slot = 4096`, so 10 x 2 and 10 x 3 are reachable and 10 x 12 is not.

Source: `data/runs/read_test_pmc_budgets.json`.

## Index coverage: the abstract experiments ran on 55% of the corpus, and it is one decade

Every abstract-corpus measurement here -- the bpref comparisons, the citation-label
agreement, the sibling-bleed contradiction -- was computed against a chunk index holding
**624,394** documents while the corpus beside it holds **1,022,706**. Measured in full,
not sampled (`scripts/measure_index_coverage.py`, `data/runs/index_coverage.json`):

| decade | in corpus | indexed | coverage |
|---|---|---|---|
| 1990s | 32,273 | 348 | **1.1%** |
| 2000s | 159,273 | 1,970 | **1.2%** |
| 2010s | 333,861 | 78,175 | **23.4%** |
| 2020s | 497,299 | 483,197 | **97.2%** |
| **total** | **1,022,706** | **563,690** | **55.1%** |

**459,016 corpus documents are not in the index.** The index verifies clean -- 119 of 119
blocks, zero bad rows, every sampled row retrieving itself at cosine 1.000000 -- so this
is not corruption. The index was built from a narrower corpus than the one now beside it,
and the difference is invisible unless the two are compared by era.

**The two have also diverged in the other direction: 60,704 indexed documents (9.7% of
the index) are no longer in the corpus at all**, having been dropped by the keep filter
when it was fixed. So it is not a subset; the two sets disagree in both directions.

### This compounds the citation era bias

Measured earlier on the same corpus: only **6.6%** of 1996-2000 abstracts carry a
bibliography against **55.1%** of 2021-2026 ones. Put beside the table above:

- old records are largely absent from the **co-citation graph** (6.6% coverage)
- and largely absent from the **index that does the retrieving** (1.1-1.2% coverage)

Both biases point the same way. **Every abstract-corpus retrieval number in this
repository describes recent literature twice over**, and a reader given "hybrid bpref
0.2131 against sparse 0.2131" is being told a fact about 2020s PubMed.

### A candidate explanation for the sibling-bleed contradiction -- tested below, and withdrawn

Sibling bleed was contradicted on abstracts (MeSH bpref 0.4059 -> 0.3083) and helps on
full text. Era skew is a plausible mechanism for that: bleed conditions a chunk on the
importance of its document's other chunks, and document-level importance depends on how
much is in the document to be important about -- 33 references per document in
2019-2026 against 7.3 in 2000-2009. An index that is 97.2% 2020s and 1.2% 2000s
therefore evaluates bleed almost entirely on documents of the kind where it helps.

### The prediction, tested and withdrawn

Recorded above as a specific testable prediction, then tested
(`scripts/measure_bleed_by_era.py`, `data/runs/bleed_by_era.json`). The statistic is
**rank movement** per candidate under bleed, bucketed by the publication year of the
candidate's document -- rank rather than score because `sibling_bleed` min-max
normalises to [0,1] while BM25 scores are ~10-25, so the two are not comparable.

| era | candidates | documents | mean rank delta | median | s.e. |
|---|---|---|---|---|---|
| before 2000 | 5 | 5 | -3.6000 | -2.0 | 1.72 |
| 2000s | 74 | 42 | +1.1351 | -0.5 | 0.76 |
| 2010s | 3,691 | 2,397 | +0.1766 | -1.0 | 0.10 |
| 2020s | 11,151 | 7,271 | -0.0644 | -1.0 | 0.06 |

**The prediction is wrong.** It said bleed is least favourable to the oldest documents,
so the statistic should rise with era. It falls: comparing the two buckets that clear a
2,000-document floor, **2010s - 2020s is -0.241 +/- 0.114 rank positions (t = -2.12)**,
so bleed is if anything marginally *more* favourable to the older of the two.

**Withdrawn.** The era explanation for the sibling-bleed contradiction is not supported.
The contradiction on abstracts and the help on full text are both still measured facts
and both still stand; what is withdrawn is the mechanism I proposed for them between
those two findings.

Two things about how the measurement was made, recorded because both were wrong first:

- **The first version compared raw scores and every bucket came out at about -0.99.**
  Not a finding: it divided bled scores in [0,1] by BM25 scores around 10-25, which is
  close to 0.1/10 whatever the era. An apparently uniform, large, era-independent effect
  was a unit mismatch.
- **The second version issued `SUPPORTS` from a 5-document bucket**, and its verdict
  string printed "moves newer documents up by -0.241" -- accepting a magnitude and a
  direction from opposite ends. Verdict now requires a 2,000-document floor per bucket
  and checks the sign, and reports INSUFFICIENT rather than a verdict when the floor is
  not met.

The pre-2000 and 2000s buckets stay in the table with their counts visible rather than
merged away: they are exactly where the index coverage collapses to 1.1% and 1.2%, and
hiding them would hide the reason this test cannot be run where the prediction was most
interesting.


## Benefit provision, and where qualification comes from (PMC, 400 queries)

Both measured from the same ablation pass. `benefit_rate` is the third ingredient of the
motivating paper's taxonomy, added in Falsifier 12.

| arm | readiness | caveat | **benefit** | bare |
|---|---|---|---|---|
| sparse | 0.0625 | 0.0450 | 0.0750 | 0.2200 |
| bleed | 0.0825 | 0.0525 | 0.0700 | 0.1600 |
| context | 0.1050 | 0.1000 | 0.1075 | 0.1875 |
| bleedctx | 0.1150 | 0.0975 | 0.0950 | 0.1425 |
| page | 0.1025 | 0.0700 | 0.0975 | 0.2150 |
| **pagectx** | **0.1475** | **0.1175** | **0.1425** | 0.1800 |

`pagectx` nearly doubles the benefit rate in retrieved passages, 0.0750 to 0.1425, in
the same way it more than doubles the caveat rate. Sibling bleed slightly *reduces* it
(0.0750 to 0.0700), which is consistent with bleed promoting whole documents rather than
the passages that carry the framing.

Section composition of gold-document passages, which is what the objective's
"section-typed gold" reduces to in this implementation:

| section | sparse | pagectx | graph mean degree |
|---|---|---|---|
| background | 30.3% | 24.6% | 2.62 |
| methods | 23.9% | 21.2% | 10.45 |
| discussion | 20.1% | 21.5% | 19.59 |
| results | 13.7% | 19.4% | 11.57 |
| **conclusions** | **3.4%** | **9.0%** | **21.80** |

Background -- the least connected section -- shrinks; Conclusions, the most connected,
grow 2.6x. The ordering of the shift matches the graph's own degree ordering, which is
what expansion along section-typed edges should produce. Conclusions remain the
smallest class at 9.0% in every arm, so the mechanism moves retrieval toward the
interpretive end without reaching it.

Source: `data/runs/mode_ablation_pmc.json` (`readiness`, `section_composition`).

## The motivating paper's failure taxonomy, and a third of it was missing

arXiv 2502.14898 reports three failure modes: complication rates in 4-5% of responses,
**procedure benefits in 6-10%**, and disputed-diagnosis caveats dropping from 100% to
56-69% on presupposing queries. This project's evaluation covers the objective's
instruction to use that taxonomy, and until now it implemented two of the three.
Rates and caveats both had detectors; benefits had none. The word appeared only inside
the caveat patterns, in "risks and benefits must be weighed" -- the language of
declining to take a position. Every table in this repository reported rates and
caveats without saying a third of the taxonomy was absent.

`find_benefit` now exists on both the passage side (`Readiness.has_benefit`,
`ReadinessReport.benefit_rate`) and the answer side (`AnswerVerdict.has_benefit`), and
`benefit_rate` is a reported column in all four drivers. Two properties are load-bearing:

- **A negated finding is not a benefit.** "We did not observe a reduction in
  readmission" contains every token a benefit pattern needs and states the reverse.
  Counting it would make the metric improve on exactly the passages that report no
  effect.
- **Negation is clause-scoped**, so a caveat early in a passage does not veto an
  unqualified result later in the same one.

Measured on 6,000 real PMC section texts, benefit language appears in **15.2%**
overall, and by section kind:

| background | discussion | conclusions | results | methods |
|---|---|---|---|---|
| 43.7% | 33.4% | 22.1% | 11.8% | **2.8%** |

Methods describe what was done rather than what was achieved. A detector that did not
reproduce that ordering would be counting keywords rather than claims, so the ordering
is the check that the detector measures the intended thing.

Note on provenance: the 300-query read test was launched before this change, so its
answer-side figures cover rates and caveats only. The input-side benefit rate is
measured separately from the ablation.

Source: `tests/test_benefit_taxonomy.py`, `medrag/eval/readiness.py`.

## Read test on PMC full text: caveats do not transmit

120 queries, `diseases` and `drugs-and-therapeutics`, arms through
`Retriever.retrieve(mode=...)`, answers from the local Gemma-12B judge. Same seed and
queries as the ablation, so the two are comparable.

| arm | input caveat | output caveat | transmission | output ready |
|---|---|---|---|---|
| sparse | 0.0667 | 0.0417 | — | 0.2583 |
| context | 0.1417 | 0.0500 | 0.11 | 0.2917 |
| pagectx | 0.1417 | 0.0417 | **0.00** | 0.2750 |

Cross-tab, the per-query form of the same question:

| arm | queries whose passages carried a caveat | answers mentioning one | rate |
|---|---|---|---|
| sparse | 8 | 1 | 0.125 |
| context | 17 | 2 | 0.118 |
| pagectx | 17 | 2 | 0.118 |

The context arms more than double the supply of queries that could be answered with a
caveat, 8 to 17, and **fifteen of the seventeen retrieved caveats are dropped at the
generation step**. Population transmits (1.00 to 1.12); quantification does not.

Two runs of the same queries with different generator sampling: population transmission
came out 0.63 then 1.12, so that estimate is not usable at this sample size and both
are reported. Caveat transmission was 0.00 in both, and the cross-tab held at 0.12.

The generator is not the limit -- Falsifier 10 got a conditional caveat rate of 1.0000
from the same model when the caveat was quoted for attribution. It is a salience
failure, not an incapacity.

Source: `data/runs/read_test_pmc.json`.

## page_alpha sweep (PMC full text, 400 queries)

| alpha | readiness | caveat | bare | qar | rankings changed |
|---|---|---|---|---|---|
| 0.0 | 0.0625 | 0.0450 | 0.2200 | 0.0400 | 0 / 400 |
| 0.1 | 0.0825 | 0.0525 | 0.2175 | 0.0625 | 400 |
| 0.25 | 0.1050 | 0.0675 | 0.2150 | 0.0750 | 400 |
| **0.5** (configured) | 0.1025 | 0.0700 | 0.2150 | 0.0775 | 400 |
| 1.0 | **0.1125** | 0.0700 | 0.2075 | 0.0775 | 400 |
| 2.0 | 0.1050 | 0.0650 | 0.2050 | 0.0750 | 400 |
| 4.0 | 0.1025 | 0.0625 | 0.2050 | 0.0800 | 400 |

alpha=0 reproduces the sparse ranking byte-identically on all 400 queries and its
readiness (0.0625) equals the ablation's sparse baseline, computed by a different code
path. All seven weights produce distinct rankings.

The curve rises to 0.25 and is flat thereafter; the configured 0.5 sits on the plateau
and the best weight (1.0) is a tie with it. The gain is not a tuning artefact. qar
rises monotonically to alpha=4.0 while readiness falls, so the two measures disagree
about the far end; readiness was the pre-chosen metric.

Source: `data/runs/alpha_sweep_pmc.json`, driver `scripts/run_alpha_sweep.py`.

## Limits

- The ablation driver reports `identical to sparse on N/400` per arm. Three separate
  bugs surfaced through that line alone -- a mis-addressed page matrix, page scoring
  that never re-sorted, and context arms truncated back to the baseline. Before they
  were fixed, `page` and `context` were byte-identical to the baseline and would have
  been reported as "no effect" rather than as broken.
- `page_alpha` is 0.5 and unswept.
- 85 of 400 queries find any gold under the baseline; the rest score zero on
  provision by construction.

## Attribution contract on PMC full text

150 queries, local Gemma-12B judge, `enable_thinking=False`. 122 of 150 queries (81%)
had a caveat-bearing passage, so the contract is tested where attribution has something
to attribute.

| arm | overall | conditional | unconditional | declined | ready |
|---|---|---|---|---|---|
| baseline | 0.0267 | 0.0328 | 0.0000 | 0.0000 | 0.2200 |
| required | 0.0600 | 0.0738 | 0.0000 | 0.8600 | 0.2133 |
| twopass | 0.2467 | 0.3033 | 0.0000 | 0.2000 | 0.2267 |
| **attributed** | **0.8133** | **1.0000** | **0.0000** | 0.1867 | 0.2867 |

All four contract checks pass: cannot fabricate (unconditional exactly 0.0000), always
reports (conditional 1.0000 against a 0.9500 floor), beats baseline conditionally
(0.0328 to 1.0000), does not cost the stated rate (0.3733 to 0.4133 -- an increase).

Attribution does not displace the numbers it annotates, which was the plausible failure
mode for an append-text step. The baseline conditional rate of 0.0328 here, on a corpus
where a caveat was available 81% of the time, reproduces the motivating paper's 4-5%
figure independently.

Source: `data/runs/attributed_caveats_pmc.json`.

## Honest limits

- **Not comparable across corpora.** The full-text numbers come with a query set
  translated at 9.8% gold retention.
- **Multi-hop is not evaluable on PMC.** 2 of 420 queries survive translation. The
  cross-hop result stands only on abstracts. *(2026-10-27: sidecar-MeSH qrels hold 205
  `pmcmerged` and 223 `pmcsample` multi-hop queries; not yet run.)*
- **Pattern matching measures presence, not correctness.** A percentage can be the
  wrong percentage; caveat language can be boilerplate. `readiness` counts a rate and
  a population, so attaching a caveat cannot move it by construction.
- **Attribution quotes, it does not compose.** A study's own hedge is not a
  qualification transferred to the reader's particular question.
- **20M is not reached.** 16,133,551 full-text chunks; 19,716,746 pooling with the
  abstract chunks, and the two cannot simply be added. *(2026-10-27: 20,535,206 unique
  abstracts are now crawled (`abstracts20m`) but not indexed; indexed full text is
  16,133,551 + 8,086,716 chunks over `pmcmerged` and `pmcsample`.)*

The verdict, the six falsifiers and what must not be claimed are in
`docs/hypothesis_determination.md`. What is built and what is unmeasured is in
`docs/status.md`.
## Caveat transmission, measured against a zero noise floor (Falsifier 18)

The run Falsifier 17's diagnosis called for: serial generation, so batch composition
cannot vary. 80 queries, three arms, `--workers 1`, two full repeat passes of the
baseline arm from identical prompts.

**Noise floor, two identical passes over all 80 queries:**

| ingredient | mean | sd | range |
|---|---|---|---|
| has_rate | 0.5250 | 0.0000 | **0.0000** |
| has_population | 0.3875 | 0.0000 | **0.0000** |
| has_caveat | 0.0625 | 0.0000 | **0.0000** |
| has_benefit | 0.1750 | 0.0000 | **0.0000** |
| ready | 0.2500 | 0.0000 | **0.0000** |

Zero, not merely small. Every one of the 80 queries returned a byte-identical verdict
on both passes. `effects_below_noise: []`.

**Passage side and answer side, same queries:**

| arm | passage ready | passage caveat | answer rate | answer pop | answer caveat | answer benefit | answer ready |
|---|---|---|---|---|---|---|---|
| `sparse` | 0.1000 | 0.0750 | 0.5250 | 0.3875 | 0.0625 | 0.1750 | 0.2500 |
| `context` | 0.1125 | **0.1500** | 0.4125 | 0.4625 | 0.0875 | 0.2500 | 0.2500 |
| `pagectx` | **0.1500** | **0.1500** | 0.4625 | 0.4000 | 0.0625 | 0.1625 | 0.2375 |

**Transmission, output gain over input gain, paired on the same queries:**

| ingredient | passage gain | answer gain | transmission |
|---|---|---|---|
| caveat_rate | +0.0750 | +0.0250 | **0.33** |
| population_rate | +0.0250 | +0.0750 | **3.00** |
| benefit_rate | +0.0250 | +0.0750 | **3.00** |
| quantification_rate | +0.0125 | -0.1125 | **-9.00** |

**Paired, same queries:**

| comparison | ingredient | delta | flips | verdict |
|---|---|---|---|---|
| sparse -> context | has_caveat | +0.0250 | +5/-3 | better |
| sparse -> context | has_population | +0.0750 | +11/-5 | better |
| sparse -> context | has_rate | -0.1125 | +4/-13 | not better |
| sparse -> context | ready | +0.0000 | +6/-6 | not better |
| sparse -> pagectx | has_caveat | +0.0000 | +5/-5 | not better |
| sparse -> pagectx | has_population | +0.0125 | +12/-11 | better |
| sparse -> pagectx | has_rate | -0.0625 | +9/-14 | not better |
| sparse -> pagectx | ready | -0.0125 | +9/-10 | not better |

Caveat cross-tab, of the queries whose passages contained a caveat, how many answers
mention one: `sparse` 1/6 = 0.17, `context` 2/12 = 0.17, `pagectx` 1/12 = 0.08.

**Limits.** 80 queries, one generator, one prompt template, one corpus. A zero floor
certifies that this pipeline produced these numbers twice; it does not certify that a
paired +0.0250 would survive a different query sample or a different reader. Serial
generation costs roughly 4x the wall clock, which is what buys the floor.

## Replication on a disjoint query sample (Falsifier 19)

Same harness, same generator, `--workers 1`, two repeat passes. Queries 80-159 of the
seed-7 shuffle; overlap with the original 80 is **0 of 240**.

| | original (0-79) | replication (80-159) |
|---|---|---|
| queries whose passages contain a caveat | 6 | **3** |
| noise floor, every ingredient | 0.0000 | **0.0000** |
| `context` paired `has_caveat` | **+0.0250** better | **+0.0250** better |
| `context` caveat transmission | **0.33** | **0.40** |
| `context` paired `has_rate` | -0.1125 not better | -0.0375 not better |
| `context` paired `has_population` | +0.0750 better | +0.0500 better |
| `pagectx` paired `has_caveat` | +0.0000 not better | -0.0125 not better |
| `pagectx` paired `has_rate` | -0.0625 not better | -0.0875 not better |
| caveat cross-tab (answers with a caveat / queries whose passages had one) | 1/6, 2/12, 1/12 | **0/0, 0/0, 0/0** |

**Withdrawn:** population and benefit transmit at 3.00, above unity. Replication gives
**0.80 and 0.50**. A ratio above unity means the answer states an ingredient more often
than the passages contained it, and it does not survive a change of sample.

**Withdrawn:** `pagectx` improves population. +0.0125 and better, then -0.0500 and not
better.

**Held:** caveat transmission, at a paired +0.0250 on both samples. **Held:** the bare
rate falls, -0.1125 and -0.0375. **Held:** page conditioning transmits no caveats on
either sample.

**Bounds.** The replication has 3 qualifying queries against 6, and its cross-tab is
empty -- no arm produced a single answer-side caveat on a query whose passages carried
one. A matching paired delta on a weaker test is worth more than the same number on a
stronger one, but with 3 and 6 qualifying queries these are single-digit counts. The
pool is the limit: 240 queries carry full-text gold. Growing it means building qrels
against the expanded PMC corpus, which is now large enough to support that and is not
yet indexed.
