# abstracts20m: the 48M-chunk index, built without holding it in RAM

Built 27–28 October 2026. Every number below is read from a file under
`data/abstracts20m/index` or `data/logs/embed20m/`, named
beside it. Nothing here is a retrieval or answer-quality result; none has been run
on this corpus yet.

## What exists

| artifact | size | contents | verified by |
|---|---|---|---|
| `store/` | 97.1 GB | 48,255,655 chunk rows (`blob.jsonl` + `offsets.bin` + `rowmap.json`) | row counts below |
| `embeddings.f32` + `embeddings.json` | 74.1 GB | 48,255,655 x 384 float32, row-major, no header; bge-small-en-v1.5, CLS pooling, normalised | `verify_embeddings.json` |
| `bm25_segments/` | 22.4 GB | 16 BM25 segments of 3,016,000 rows (last 3,015,655) | `verify_bm25.json` |
| `page_vectors.f32` + `page_index.pkl` | 31.5 GB + 0.43 GB | 20,532,828 document vectors, 384 dims, normalised | `verify_page.json` |
| `page_texts/` | 1.5 GB | each document's page text and first row, 411 gz blocks | `pages.json` |
| `dependency_graph.pkl` | 1.77 GB | section-dependency graph over all 48,255,655 rows | `verify_graph.json` |

`RELEASE_MANIFEST.json` in the index directory lists every shippable file with its
size and sha256 (`scripts/release_manifest.py`).

Loading `embeddings.f32` needs no code from this repository:

```python
import json, numpy as np
m = json.load(open("embeddings.json"))
X = np.memmap(m["file"], dtype="<f4", mode="r", shape=tuple(m["shape"]))
```

## Counts

| quantity | value | source |
|---|---|---|
| chunk rows | 48,255,655 | `store/offsets.bin` |
| documents (distinct `doc_id`) | 20,532,828 | `page_texts/pages.json`, `graph_inputs/kind_names.json` |
| runs of consecutive rows sharing a `doc_id` | 20,533,300 | `graph_inputs/kind_names.json` |
| documents stored twice | 472 | `graph_inputs/split_docs.json` |
| rows whose `doc_id` differs from their chunk id's prefix | 0 | `graph_inputs/kind_names.json` |
| unique abstracts crawled, as stated in the README before indexing | 20,535,206 | README |

The 472 documents stored twice each have exactly two runs with identical chunk ids;
471 have identical text in both runs and 1 differs
(`logs/embed20m/split_docs_check.json`). They were not removed: doing so would change
row ids under the finished embeddings. The 2,378 difference between documents crawled
and documents indexed has not been investigated.

Section kinds present (`kind_names.json`): results, discussion, background, methods,
conclusions, references, and empty.

## Verification

`scripts/verify_20m.py`, seed 20261028.

**BM25** (`verify_bm25.json`): 16 segments, contiguous, covering 48,255,655 of
48,255,655 rows. 32 sampled rows queried with their own indexed text: 32 at rank 1.
The batched search over all 16 segments took 458 s.

**Page vectors** (`verify_page.json`): 20,532,828 documents, equal to the manifest;
loads as a memmap; 20 random blocks equal to their embedded npz after normalisation,
with document ids and first rows in order; norms over the first 2M rows within
[0.99999988, 1.00000012]; document ids unique.

**Dependency graph** (`verify_graph.json`): 48,255,655 nodes; 39,421,922 undirected
edges (1.63 per node, max degree 22); 6,060,864 isolated nodes; CSR consistent;
4,920 edges from 3,000 sampled nodes, 0 unreciprocated. Mean degree by section kind:

| kind | mean degree |
|---|---|
| conclusions | 3.328 |
| methods | 2.990 |
| discussion | 2.942 |
| results | 2.099 |
| (empty) | 1.720 |
| background | 1.029 |
| references | 1.000 |

**Embeddings** (`verify_embeddings.json`): shape 48,255,655 x 384 and file size equal
to the manifest; 200 random blocks compared in full against their npz, 0 mismatched;
0 all-zero rows; norms within [0.99999982, 1.00000012]; 64 sampled rows, each
searched against all 48,255,655 rows in one streamed pass: 64 retrieve themselves at
rank 1.

## How it was built

| step | where | wall clock | command |
|---|---|---|---|
| chunk embedding | worker-B, worker-A (2x P2000 each); worker-C (GTX 1080 Ti) and worker-D (RTX 3060 Ti) via `scripts/remote_embed.py` | 27 Oct 09:23 to 28 Oct 16:47 UTC | `build-index --block-shard`, `remote_embed.py work` |
| matrix assembly | worker-A, CPU | 16:50 to 19:43 UTC | `build-index --skip-bm25 --no-faiss` |
| BM25 segments | worker-B + worker-A, 8 workers | 16:47 to 17:23 UTC | `bm25-segments --rows-per-segment 3016000` |
| page texts | worker-B, CPU | 15:46 to 16:20 UTC | `page-index --write-texts` |
| page vectors | all four GPU hosts | 16:47 to 18:19 UTC | `remote_embed.py work --kind page` |
| page assembly | worker-B | 18:19 to 19:05 UTC | `page-index --assemble-blocks` |
| graph inputs | worker-A, 5 processes | 16:27 to 16:41 UTC | `dependency-graph --scan-inputs` |
| dependency graph | worker-B | 17:33 to 17:40 UTC | `dependency-graph --from-scan` |

All vectors, chunk and page, were computed in fp32 on every host. Before writing any
block, each remote GPU re-embedded an already published chunk block and was compared
with it: worker-C max abs diff 1.8e-7, worker-D 6e-7.

The first graph build, on worker-A, was OOM-killed in `_finalize` at 220,023,278 directed
edges while the matrix assembly shared the host; the rebuild on worker-B used the
single-key finalize (commit ff5eead), which tests show gives output identical to the
form it replaced.

## Searching it

Neither index fits in RAM on these hosts, so search is batched over a query file and
streamed from disk; both searches are exact.

```bash
medrag -c configs/default.yaml -c configs/abstracts20m.yaml search-segmented \
  --queries queries.jsonl --out run.jsonl --modes sparse,dense
```

`queries.jsonl` has one `{"qid": ..., "query": ...}` per line. The output has one line
per query with `[row, score]` lists per mode; the default top-k is
`fusion.sparse_top_k` / `fusion.dense_top_k` (100).

* **Sparse** scores each segment with the corpus-wide N, average length and document
  frequency, so the scores and ranking equal one monolithic BM25 index
  (`tests/test_segmented.py`). Each run loads all 16 segments once, about 8 minutes,
  regardless of the number of queries.
* **Dense** is the exact inner product against all 48,255,655 rows, one sequential
  pass over `embeddings.f32` per run.

The retriever's sparse-based modes (`sparse`, `bleed`, `context`, `bleedctx`, `page`,
`pagectx`) can run on top of a sparse run unchanged: load the retriever with
`load_index(...)` and set `retriever.bm25 = PrecomputedBM25.from_run(run, queries,
n_docs=48255655)`. A query (or sub-query) that is not in the run raises rather than
returning nothing.

## Related fix

`BM25Index.search` bounded term ids by the row count instead of the vocabulary size
(fixed in 6d55a62). It never affected an index with more rows than terms; pmcsample
(8,086,716 rows, 8,130,690 terms) ignored its 43,974 most recently added terms in
sparse search before that commit. pmcmerged was not affected.
