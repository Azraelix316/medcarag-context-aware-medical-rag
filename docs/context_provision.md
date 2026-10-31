# Context provision: metric, mechanisms, and controls

## The failure being measured

arXiv 2502.14898 documents, against Google AI Overview and Perplexity, that
medical RAG returns answers that are *literally accurate* and *pragmatically
misleading*:

| failure | measured rate |
|---|---|
| response mentions complication rates | 4-5% of queries |
| response mentions procedure benefits | 6-10% of queries |
| disputed diagnosis flagged as disputed, direct query | 100% |
| disputed diagnosis flagged as disputed, presupposing query | 56% (Google) / 69% (Perplexity) |
| Jaccard overlap of sources for "why is X safe" vs "why is X dangerous" | 0.16 (Google) / 0.31 (Perplexity) |

None of this is visible to hit@k, precision@k, nDCG or MRR. A system that lists
every complication of a procedure, cites genuine sources and returns the correct
document scores a *perfect* recall while telling the reader nothing about how
often those complications occur.

## Why the corpus structure makes this likely here

Section distribution over a 30,000-chunk sample of the production store:

| section | share |
|---|---|
| background | 34.5% |
| results | 18.7% |
| methods | 18.4% |
| conclusions | 18.0% |
| discussion | 0.7% |
| unlabelled | 9.8% |

A query about an outcome matches *Results*. The passages that make Results
interpretable — the population and comparator (Methods), the rate and caveats
(Conclusions) — live in different chunks that do not match the query.

## The metric

For a query with known gold documents, a retrieved set is:

- **bare** — contains a gold document's evidential passage (Results, abstract) but
  no framing passage from *that same document*.
- **qualified** — contains the finding *and* a framing passage from the same
  document (Conclusions or Discussion; Methods under the non-strict setting).

Framing is judged **within a document** deliberately. A conclusions chunk from a
different study does not qualify this study's result — different population,
different rate.

| metric | definition |
|---|---|
| `qar@k` | queries whose top-k holds a gold finding *and* its own framing |
| `bar@k` | queries whose top-k holds a gold finding but no framing for it |
| `sabr@k` | of queries that found a gold document, the fraction that found it *with* framing |
| `qual_ratio@k` | mean framing passages per evidential passage |

`qar + bar + not-found = 1`. `qar` is **set-based and order-insensitive**, which
matters below: a mechanism that only reorders cannot change it unless it changes
which chunks make the cut.

## Two mechanisms

**Sibling bleed** (`medrag/index/sibling_bleed.py`) conditions a chunk's score on
the strength of its document's other chunks:

```
s'(c) = (1 - alpha)*s(c) + alpha*max_{c' in same doc, c' != c} s(c')
```

Max, not sum, so a document is not rewarded merely for having been split into
more chunks. Scores are min-max normalised first, so `alpha` means the same thing
for BM25 (unbounded) and cosine ([-1,1]). `alpha=0` is an exact identity.

**Dependency context** appends the interpreting passages of the top-ranked seeds,
drawn from the section-dependency graph (2,813,321 edges over 2,368,622 chunks).

Both are query-time re-scorings of an existing index. Contextual retrieval
(Anthropic 2024) instead rewrites chunk text with document context and requires
re-embedding the corpus, which at our scale is the ~64 hours the plan estimated.
That cost difference is the practical argument for measuring these separately
rather than as an ablation of theirs.

## Result: 400 benchmark queries, 2,368,622 chunks, matched budgets

Top-10 results plus 10 context slots, bleed alpha 0.6, sparse candidate pool 50.

| arm | qar | bar | sabr | qual_ratio | nDCG@10 |
|---|---|---|---|---|---|
| sparse | 0.3800 | 0.4050 | 0.4841 | 0.217 | 0.5350 |
| random-partner bleed *(control)* | 0.2875 | 0.4950 | 0.3674 | 0.266 | 0.4823 |
| **sibling bleed** | 0.6175 | 0.1375 | 0.8179 | 0.440 | **0.5676** |
| random same-doc context *(control)* | 0.3800 | 0.4050 | 0.4841 | 0.217 | 0.5350 |
| **section-typed context** | 0.6475 | 0.1450 | 0.8170 | 0.449 | 0.5350 |
| random same-doc + bleed *(control)* | 0.6525 | 0.1525 | 0.8106 | 0.358 | 0.5676 |
| **bleed + context** | **0.6975** | **0.0675** | **0.9118** | 0.502 | 0.5676 |

Decomposition against matched controls:

```
total gain, bleed + context            +0.3175
document membership (vs random partner) +0.3300
section structure   (vs random same-doc) +0.2675
random controls alone: partner -0.0925, same-doc +0.0000
```

Both controls **do nothing or worse than nothing**. Random partner bleeding is
*worse than the baseline* (nDCG 0.5350 → 0.4823), so smoothing alone is harmful and
knowing which chunks share a document is what makes it work. Random same-document
context adds literally zero QAR, so adding more chunks of an already-relevant
document is worthless unless those chunks are the framing ones.

Sibling bleed also improves ranking, not just composition: nDCG@10 0.5350 →
0.5676. That is a genuine retrieval gain, not a set-composition artefact.

## An earlier ablation reached the opposite conclusion, and why

The first version of this control measured total QAR gain 0.3800 → 0.6825 and
attributed only 26% to structure, concluding that most of the benefit came from
adding any same-document chunk. The matched control above gives the opposite
answer.

The difference is entirely in how the control was built, and it is worth stating
plainly because the first control was changed after it went against the method:

- **First control** sampled "random same-document chunks" from a `doc -> chunks`
  map built by walking the whole store at a stride. Those chunks came from
  documents across the corpus, not from the seeds' documents. It therefore tested
  "add arbitrary relevant-ish chunks", not "add another chunk of what you already
  found".
- **Matched control** samples only from the retrieved seeds' own documents, using
  the same number of slots. That is the control the mechanism claims credit for.

The matched control is the right one: the claim is that section typing selects
*which* chunk of an already-relevant document supplies the framing. The first
control tested a different and much broader claim. Both numbers are reported here
because the first was published in an earlier revision of this document and
replaced without explanation would be the dishonest choice.

Two controls were also built wrong before they were built right, and both failures
were silent because the control produced no change:

1. **Relabelling document ids is mathematically incapable of controlling for
   bleed.** A permutation is a function, so two chunks sharing a document still
   share their permuted label. Group sizes measured identically before and after
   ({1:34, 2:5, 3:2} for a 50-chunk pool), and rankings came out bit-identical
   despite 100% of labels changing. The control now breaks group *membership* by
   drawing partners at random from the pool.
2. A random-reorder control over an already-truncated top-10 is vacuous: the
   resulting set is identical, so no set-based metric can see it.

The general lesson, and the reason both are recorded: **a control that produces no
difference is the most dangerous kind, because it is indistinguishable from a
passing control.** Both of these initially read as "structure contributes 0.0000".

## Per-query-family breakdown: the gain is broad, not concentrated

The motivating paper's failures are concentrated in complication, safety and
disputed-condition queries. If the mechanism only helped there, that would be a
sharper and more believable story. It does not: the gain is broad.

All 2,340 labelled queries, sparse top-10 plus 10 context slots, bleed alpha 0.6.

| category | difficulty | n | qar base | qar +context | delta | bar base | bar +context |
|---|---|---|---|---|---|---|---|
| **ALL** | | 2340 | 0.3654 | 0.6509 | **+0.2855** | 0.3932 | 0.0821 |
| anatomy | single-hop | 220 | 0.4409 | 0.7864 | +0.3455 | 0.4364 | 0.0727 |
| psychiatry-and-psychology | single-hop | 128 | 0.3203 | 0.6484 | +0.3281 | 0.4766 | 0.1094 |
| diseases | single-hop | 687 | 0.4716 | 0.7962 | +0.3246 | 0.3988 | 0.0480 |
| organisms | single-hop | 90 | 0.5222 | 0.8333 | +0.3111 | 0.4333 | 0.1000 |
| drugs-and-therapeutics | single-hop | 598 | 0.3344 | 0.6187 | +0.2843 | 0.3896 | 0.0769 |
| phenomena-and-processes | single-hop | 197 | 0.3096 | 0.5635 | +0.2538 | 0.4010 | 0.1827 |
| multi-hop | multi-hop | 420 | 0.2024 | 0.3905 | +0.1881 | 0.3286 | 0.0905 |

Two things to take from this, one encouraging and one narrowing:

**Every category improves**, by +0.188 to +0.346. There is no family where the
mechanism fails or hurts, which is what you want from a general retrieval
component rather than a trick tuned for one query shape.

**The gain is *not* concentrated in the motivating failure taxonomy.** It is
roughly uniform. That is weaker evidence for the specific thesis — "this fixes
the complication-rate and benefits omission documented in arXiv 2502.14898" —
than a concentrated effect would have been. What the mechanism actually does is
generic: return the framing passages of documents already retrieved. That helps
whatever the question was, which includes but is not limited to the failures the
paper describes.

Multi-hop is the informative outlier. It has both the lowest baseline `qar`
(0.2024) and the smallest gain (+0.1881), and the reason is structural: a
multi-hop query needs evidence from *several* documents, so "the framing of the
document this passage came from" supplies less of what is missing. A
multi-hop-aware version would need to provision context across the supporting
documents, not within one.

Numbers: `data/runs/context_by_family.json`.

## Why document-locality is not a limitation to work around: a negative result

Multi-hop queries have both the lowest baseline `qar` (0.2024) and the smallest
gain (+0.1881), so the obvious reading is that single-document framing is the
wrong unit when evidence spans papers. That reading is wrong, and testing it
produced the sharpest evidence for the mechanism's actual design.

`medrag/index/multihop_context.py` adds two things: cross-hop framing (a passage
may be framed by one from a *different* document) and per-hop seeding (seeds from
each hop's own prefix, so a hop ranked 15th overall still contributes framing).
Measured on all 2,340 queries:

| arm | qar | bar | readiness | quantified | caveat |
|---|---|---|---|---|---|
| sparse | 0.3654 | 0.3932 | 0.3051 | 0.3479 | 0.1060 |
| **single-document context** | **0.6769** | **0.0936** | **0.4902** | **0.5278** | **0.2444** |
| multi-hop context | 0.6538 | 0.1150 | 0.4791 | 0.5162 | 0.2380 |

And on the multi-hop family alone, where the hypothesis predicted the largest gain:

| arm | qar | readiness |
|---|---|---|
| sparse | 0.2024 | 0.1738 |
| **single-document context** | **0.4333** | **0.2881** |
| multi-hop context | 0.3095 | 0.2262 |

**Cross-hop framing is worse than single-document framing, including on the
multi-hop queries it was designed for.** Permitting a passage to be framed by
another paper's Methods fills the same limited context budget with text that does
not qualify this finding, displacing the same-document framing that does. The
context slots are the scarce resource, and locality is what makes them pay.

That reframes the original observation. Multi-hop queries score lower not because
same-document framing is the wrong unit, but because their gold sets are
intersections across papers, so fewer retrieved passages come from a document that
is relevant at all. The fix for that is better *recall* of the supporting
documents, not cross-document framing.

### A bug this experiment caught

The first version of the multi-hop provisioner rebuilt the result set from the
per-hop prefixes, for a two-hop query yielding six rows instead of ten and dropping
the strongest single-hop hits. On the multi-hop family that cost more than the
framing gained: readiness fell to 0.1357, *below* the 0.1738 sparse baseline it was
meant to improve on. Fixed by requiring `base_rankings` (now mandatory, raising if
omitted) so per-hop prefixes select seeds and never results; readiness recovered to
0.2262. Still below single-document framing, which is the substantive result.

## Remaining circularity

This result is still partly definitional. The mechanisms add or promote framing
passages, and the metric counts framing passages. Demonstrating that a mechanism
which supplies qualifying text raises the rate of qualifying text is not evidence
that a reader is better informed.

What the matched controls *do* establish is narrower and real: the gain is not
reproducible by smoothing arbitrarily, and not reproducible by adding arbitrary
chunks of relevant documents. What they do not establish is that a reader given
these passages states rates, populations and limitations correctly.

Ranked by how much each would change the conclusion:

1. **LLM or human judgement on the retrieved sets.** The only measurement that
   breaks the circularity.
2. **Citation-derived gold with bpref.** The MeSH gold used here shares its
   signal with the MeSH-graph retriever. Measured same-universe agreement between
   the two label regimes is weak — mean Jaccard 0.016–0.022 over 3,144–5,654
   descriptors — so a gain under one need not hold under the other. See
   `data/runs/citation_label_agreement.json`.
3. **Per-query-family breakdown.** The motivating paper's failures are
   concentrated in complication, safety and disputed-condition queries; if the
   gain is absent elsewhere, that is the honest result.

## Cost

| mechanism | per query | corpus change |
|---|---|---|
| sibling bleed | 0.0–0.3 ms | none |
| dependency context | 0.1–0.8 ms | none |
| contextual retrieval (Anthropic-style) | same at query time | ~64 h re-embed |

Both mechanisms here are free at query time and require no corpus rebuild, which
is what makes them practical to deploy alongside an index that is still growing.

## Files

- Metric: `medrag/eval/context_provision.py`, 12 tests
- Mechanisms: `medrag/index/sibling_bleed.py`, 12 tests
- Graph: `medrag/index/dependency_graph.py`, 14 tests
- Citation control: `medrag/eval/citation_labels.py`, 18 tests
- Numbers: `data/runs/context_ablation.json`, `data/runs/citation_label_agreement.json`