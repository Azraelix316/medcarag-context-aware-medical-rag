# Era bias in citation coverage

The objective requires this reported honestly. It does so -- but the honest answer
differs by corpus, differs from the figures quoted in the original plan, and the
version that is easiest to state is the one that is wrong.

## Measured on the corpora, not on samples

Both figures below come from a **full pass over every document**, with no sampling:
`scripts/measure_era_bias.py`, results in `data/runs/era_bias_pmc.json` and
`data/runs/era_bias_pubmed.json`. The previous version of this file used 150 randomly
sampled live PMC articles per era window and tilde-prefixed figures for abstracts
(`~20% / 7.3` and `~72% / 33`). Both are replaced.

**`refs/doc*` is the mean over documents that carry a bibliography. `all-docs` is the
mean over every document in the window, so the records without a bibliography count as
zero.** The distinction is not pedantry: it is the difference between measuring how
thoroughly a record is cited and measuring how much citation signal a corpus window
actually contains, and the two move in the same direction here, so conflating them
makes the bias look smaller than it is.

## PubMed abstracts -- 960,880 documents, every record counted

| window | documents | coverage | refs/doc* | all-docs | median* |
|---|---|---|---|---|---|
| 1996-2000 | 43,235 | **6.6%** | 21.6 | 1.42 | 19 |
| 2001-2005 | 70,475 | **9.2%** | 26.5 | 2.44 | 23 |
| 2006-2010 | 101,414 | **20.9%** | 32.9 | 6.87 | 29 |
| 2011-2015 | 156,580 | **34.7%** | 37.1 | 12.87 | 32 |
| 2016-2020 | 204,635 | **40.9%** | 41.3 | 16.92 | 35 |
| 2021-2026 | 384,541 | **55.1%** | 44.5 | 24.53 | 38 |

Coverage rises **8.3x** from the oldest window to the newest (6.6% to 55.1%), and
references per document across the corpus rise **17x** (1.42 to 24.53). Both halves of
the original claim hold in direction and both are much sharper than the approximate
figures: the "~72%" was closer to 55%, and the "~33" is `refs/doc*` rather than
`all-docs`, which is 24.53.

Even the newest window is not covered: **44.9% of 2021-2026 abstracts have no
bibliography at all.**

## PMC Open Access full text -- 555,142 documents, every record counted

| window | documents | coverage | refs/doc* | all-docs | median* |
|---|---|---|---|---|---|
| 1996-2000 | 133 | **84.2%** | 37.8 | 31.81 | 26 |
| 2001-2005 | 962 | **96.0%** | 40.9 | 39.27 | 29 |
| 2006-2010 | 8,789 | **99.5%** | 46.8 | 46.58 | 34 |
| 2011-2015 | 44,285 | **99.5%** | 45.6 | 45.43 | 35 |
| 2016-2020 | 122,679 | **99.4%** | 48.3 | 48.02 | 36 |
| 2021-2026 | 378,207 | **99.7%** | 53.9 | 53.72 | 40 |

**Citation coverage on PMC is effectively era-flat**: 99.4-99.7% for every window from
2006 onward. The co-citation graph built here (30,854,914 pairs over 548,700 documents)
is not era-truncated in the way the PubMed one is.

**But PMC has the opposite problem, and it is severe.** The corpus is recent *by
construction*: **68.1% of its documents are from 2021-2026** and only **133 documents
predate 2001**. PMC Open Access coverage of the literature is a function of the paper's
age, so the corpus is not a sample of the literature with a bias -- it is almost
entirely the recent literature.

## The same measurement on the expanded corpus, which partly answers it

The `pmcmerged` era skew above is not only a query artefact, but the query is not
innocent either. Re-measured on the all-OA sweep (`configs/pmcsweep_c7.yaml` and its
worker-B twin), which carries no publication-type restriction, over 910,000 documents
scanned and counted rather than sampled:

| window | documents | coverage | refs/doc (with refs) | median |
|---|---|---|---|---|
| 1990-1994 | 851 | **81.1%** | 9.94 | 6 |
| 1995-1999 | 7,296 | **97.6%** | 40.21 | 41 |
| 2000-2004 | 26,362 | **98.2%** | 32.96 | 29 |
| 2005-2009 | 126,721 | **99.0%** | 35.04 | 30 |
| 2010-2014 | 614,933 | **98.9%** | 36.73 | 31 |
| 2015-2019 | 133,690 | **99.0%** | 36.15 | 30 |

Two things this settles and one it does not.

**It settles the claim that PMC's citation coverage is era-flat.** From 1995 onward it
is 97.6-99.0% in every window, on 910,000 documents rather than 555,142, and with no
publication-type filter. The co-citation graph is not era-truncated.

**It partially answers the era-skew objection.** `pmcmerged` held **133 documents
predate 2001**; the sweep already holds **8,147 pre-2000 documents**, roughly 61x, with
1990-1994 coverage at 81.1%. The early era exists and is citable -- it was simply
invisible behind a query that asked for randomised controlled trials. But the deep
problem remains and is not a query problem: only 1.6% of the expanded corpus is 1990s,
because PMC acquired machine-readable full text mostly after 2010 (see the body-availability
table in `docs/corpus_scale_ceiling.md` -- 45.0% of 1995 open-access records carry a
usable `<body>`, against 100.0% in 2018).

**It does not settle references-per-document.** The expanded corpus shows 33-40
refs/doc where `pmcmerged` shows 46-54, and the expanded corpus is mostly 2010s where
`pmcmerged` is mostly 2021-2026. Both corpora agree the count rises with era; they
disagree on the level because they are differently weighted across eras. Neither
number is wrong and the gap is a composition effect, not a measurement error.

## What this means, stated carefully

**On PubMed abstracts the bias is severe and unavoidable.** NLM deposits a bibliography
for 6.6% of 1996-2000 records and 55.1% of 2021-2026 ones. A co-citation graph built
there is a sample of *recent* literature plus a thin tail, not a sample of the
literature. Any claim resting on that graph is a claim about the last decade.

**On PMC full text there is no coverage bias, and a large composition bias instead.**
The absence of a coverage bias is a real gain and it is what makes the dependency-graph
experiments usable: every document carries its bibliography, so a section-typed edge
is not silently missing for old papers. The composition bias is the price, and it means
the PMC results describe recent medicine.

**The two corpora are not interchangeable and neither is era-neutral.** They fail in
opposite directions. The experiments in this repository run on PMC full text precisely
because the mechanism depends on section structure that abstracts do not have -- and
that choice simultaneously buys citation coverage and accepts a recent-only corpus.
The honest summary is not "the era bias was corrected"; it is "the bias was traded, not
removed".

**What would remove it.** Only a corpus with bibliographies uniformly deposited across
eras -- which PubMed does not provide and PMC cannot reach before ~2001. Until one
exists, era is a confound in every co-citation result reported here, in opposite
directions depending on corpus, and the mitigation is to state the era distribution
alongside any co-citation number rather than to treat the graph as era-neutral.