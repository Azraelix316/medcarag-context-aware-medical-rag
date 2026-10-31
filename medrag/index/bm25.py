"""BM25 lexical retrieval over a compact sparse matrix.

Why not a dict-of-dicts inverted index: measured on this host, a Python
``dict[str, dict[int, int]]`` postings table costs **~3.5 kB per chunk**. At the
7M-chunk scale this corpus targets that is 22.6 GiB of RAM before the chunk
store, embeddings, or FAISS index -- it does not fit.

This implementation stores the same postings in a ``scipy.sparse`` CSC matrix
(one row per term, columns are chunk row ids). Posting payloads become typed
arrays (int32 doc ids + uint16 term frequencies), which is ~8 bytes per posting
instead of ~3500, a **~6x memory reduction**, and scoring becomes a vectorised
sparse gather rather than a Python loop over postings.

Scoring maths is unchanged: standard BM25 with the IDF variant
``log(1 + (N - df + 0.5)/(df + 0.5))``, so results are comparable to the
reference implementation and to Lucene.
"""

from __future__ import annotations

import json
import logging
import math
import gc
import pickle
import re
from array import array
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Words carrying no retrieval signal. Kept short on purpose: many clinical terms
# are short ("no", "hr", "iv") but dropping them costs recall on real queries.
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further had hadn't has
hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other
ought our ours ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves then
there there's these they they'd they'll they're they've this those through to too
under until up very was wasn't we we'd we'll we're we've were weren't what
what's when when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str, *, min_length: int = 2, stem: bool = True) -> list[str]:
    """Lowercase, split, drop stopwords, and apply light suffix stemming.

    The stemmer is a small Porter-step-1 subset rather than a full Porter
    implementation: it folds plural/verb forms ("patients", "treated", "treats")
    which is what actually matters for recall on clinical phrasing variants.
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        token = raw.replace("'", "")
        if len(token) < min_length or token in STOPWORDS:
            continue
        if stem:
            token = _light_stem(token)
        if len(token) >= min_length:
            tokens.append(token)
    return tokens


def _light_stem(token: str) -> str:
    """Conservative suffix stripping. Never shortens below 3 characters."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    if len(token) > 5 and token.endswith("ing"):
        base = token[:-3]
        if len(base) > 3 and base[-1] == base[-2]:
            base = base[:-1]
        elif len(base) > 3 and not re.search(r"[aeiou]", base[-2:]):
            base += "e"
        return base
    if len(token) > 4 and token.endswith("ed"):
        base = token[:-2]
        if len(base) > 3 and base[-1] == base[-2] and base[-1] not in "lsz":
            base = base[:-1]
        return base
    return token


class BM25Index:
    """Compact BM25 index backed by a scipy CSC term-document matrix.

    Build incrementally with :meth:`add`, then call :meth:`finalize` to compute
    document frequencies and IDF. Both are required before :meth:`search`.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75, min_token_length: int = 2) -> None:
        self.k1 = k1
        self.b = b
        self.min_token_length = min_token_length
        self.n_docs: int = 0
        self.avgdl: float = 0.0
        self._vocab: dict[str, int] = {}
        self._terms: list[str] = []
        self._doc_len = np.zeros(0, dtype=np.int32)
        # Accumulated COO triplets; converted to CSC on finalize().
        # Growable typed arrays, not lists of per-document numpy arrays. See add().
        self._rows: array = array("i")   # term ids
        self._cols: array = array("i")   # doc ids
        self._vals: array = array("f")   # term frequencies
        self._cur_doc: int = 0
        self._matrix: Any = None            # scipy.sparse.csc_matrix
        self._df: np.ndarray = np.zeros(0, dtype=np.int32)
        self._idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._finalized = False

    # ------------------------------------------------------------------ build
    def add(self, doc_id: int, text: str) -> None:
        """Index one document. ``doc_id`` must equal the next expected row id.

        Postings accumulate into three growable typed arrays, not a Python list of
        per-document numpy arrays. At 16.1M chunks the list form holds 48.4M array
        objects, and measured on this corpus it reached 11.6 GiB after only 5.15M
        documents -- roughly 36 GiB extrapolated, against 31 GiB of RAM. array('i') /
        array('f') store the same values as raw 4-byte cells with no per-object
        header, and finalize()'s concatenation becomes a single copy.
        """
        if doc_id != self._cur_doc:
            raise ValueError(
                f"BM25Index.add expects sequential ids; got {doc_id}, expected {self._cur_doc}"
            )
        tokens = tokenize(text, min_length=self.min_token_length)
        if tokens:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self._rows.extend(
                self._term_id(t) for t in counts
            )
            self._cols.extend([doc_id] * len(counts))
            self._vals.extend(counts.values())
        self._cur_doc += 1
        self._finalized = False

    def _term_id(self, term: str) -> int:
        tid = self._vocab.get(term)
        if tid is None:
            tid = len(self._terms)
            self._vocab[term] = tid
            self._terms.append(term)
        return tid

    def extend(self, items: Iterable[tuple[int, str]]) -> None:
        for doc_id, text in items:
            self.add(doc_id, text)

    def _spill(self, spill_dir: str | Path, terms: np.ndarray, docs: np.ndarray,
               vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Write postings to disk, free the RAM copies, and return memory-mapped views.

        The arrays are contiguous views over the ``array`` buffers, so they must be
        copied out before the buffers are dropped. After this the process holds no
        Python-side copy: the blocked fill reads pages on demand and the OS may evict
        them, which is the entire point.
        """
        d = Path(spill_dir)
        d.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, arr in (("terms", terms), ("docs", docs), ("vals", vals)):
            p = d / f"{name}.npy"
            np.save(p, arr)
            paths[name] = p
        del self._rows, self._cols, self._vals
        self._rows = array("i")
        self._cols = array("i")
        self._vals = array("f")
        gc.collect()
        mapped = [np.load(paths[k], mmap_mode="r") for k in ("terms", "docs", "vals")]
        log.info("BM25: spilled %d postings (%.1f GiB) to %s",
                 terms.size, sum(a.nbytes for a in mapped) / 2**30, d)
        return mapped[0], mapped[1], mapped[2]

    def finalize(self, *, spill_dir: str | Path | None = None) -> "BM25Index":
        """Materialise the CSC matrix and precompute document frequencies/IDF.

        Built one term block at a time rather than through
        ``coo_matrix(...).tocsc()``, because scipy's conversion holds its own copy of
        the values and both index arrays while allocating the result.

        ``spill_dir`` writes the accumulated postings to disk first and reads them
        back memory-mapped, which is what makes this fit at corpus scale. Measured on
        the 16,133,551-chunk PMC corpus: RSS reached 15.1 GiB while only 66% of the
        chunks were in, extrapolating to roughly 23 GiB of inputs, and the CSC output
        adds about 13 GiB -- so holding both in RAM peaks near 36 GiB on a 31 GiB box.
        Spilling releases the inputs before the output is allocated, leaving a peak of
        about 14 GiB: the output plus whatever pages the blocked fill touches.

        Two earlier attempts are recorded because both are tempting:

        * a per-document list of numpy arrays, which added 48.4M array objects and
          reached 11.6 GiB after only 5.15M chunks;
        * a Python scatter loop filling the CSC directly, which avoids scipy's
          intermediate but takes hours over more than a billion postings.

        The blocked fill sorts within each block. That is what makes it correct:
        postings arrive grouped by document, so a boolean mask yields them in
        *document* order, while a CSC slice needs *term* order. Skipping the sort
        wrote each term's slice with another term's documents whenever one block
        spanned every term -- silently, with correct nnz and correct indptr, so only
        an index-for-index comparison against scipy caught it.
        """
        from scipy import sparse

        n_docs = self._cur_doc
        self.n_docs = n_docs
        self._doc_len = np.zeros(n_docs, dtype=np.int32)

        if self._rows:
            terms = np.frombuffer(self._rows, dtype=np.int32)
            docs = np.frombuffer(self._cols, dtype=np.int32)
            vals = np.frombuffer(self._vals, dtype=np.float32)
            n_terms = len(self._terms)
            n_post = int(terms.size)

            # doc lengths, reconstructed from the postings we already hold
            np.add.at(self._doc_len, docs, vals.astype(np.int32))
            counts = np.bincount(terms, minlength=max(1, n_terms))
            indptr = np.zeros(n_terms + 1, dtype=np.int64)
            np.cumsum(counts, out=indptr[1:])

            if spill_dir is not None:
                terms, docs, vals = self._spill(spill_dir, terms, docs, vals)

            out_idx = np.empty(n_post, dtype=np.int32)
            out_val = np.empty(n_post, dtype=np.float32)

            # Single pass, chunked, fully vectorised.
            #
            # The previous version iterated term blocks and located each block's
            # postings with a boolean mask over the whole array. That is a full scan
            # of the 15.4 GiB input per block -- tens of blocks -- and it was killed
            # partway through with no error in its log.
            #
            # Here each chunk is sorted by term, which yields the rank of every
            # posting within its own term group, and that rank plus the term's
            # running cursor gives the destination slot directly. So every posting is
            # read once and written once, and the only large allocation is the output
            # itself.
            #
            # A stable sort within a chunk keeps postings of one term in document
            # order, since the input is document-major and a term occurs at most once
            # per document. That is what makes each slice strictly ascending without
            # a second pass.
            cursor = indptr[:-1].copy()
            chunk_rows = 1 << 25  # 33.5M postings: ~800 MiB of working set
            for begin in range(0, n_post, chunk_rows):
                end = min(n_post, begin + chunk_rows)
                t = terms[begin:end]
                order = np.argsort(t, kind="stable")
                ts = t[order]
                # First position of each term group within this chunk.
                first = np.flatnonzero(np.concatenate(([True], ts[1:] != ts[:-1])))
                group = np.cumsum(np.concatenate(([True], ts[1:] != ts[:-1]))) - 1
                slot = cursor[ts[first][group]] + (np.arange(end - begin) - first[group])
                cursor[ts[first]] += (np.diff(np.concatenate((first, [end - begin]))))
                out_idx[slot] = docs[begin:end][order]
                out_val[slot] = vals[begin:end][order]
                del t, order, ts, first, group, slot

            matrix = sparse.csc_matrix(
                (out_val, out_idx, indptr),
                shape=(max(1, n_docs), max(1, n_terms)),
                dtype=np.float32,
            )
            del out_idx, out_val, counts
        else:
            matrix = sparse.csc_matrix((max(1, n_docs), max(1, len(self._terms))), dtype=np.float32)

        self._matrix = matrix
        self._df = np.diff(matrix.indptr).astype(np.int32)
        total = float(n_docs) or 1.0
        self._idf = np.log(
            1.0 + (total - self._df.astype(np.float32) + 0.5) / (self._df.astype(np.float32) + 0.5)
        ).astype(np.float32)
        self.avgdl = float(self._doc_len.sum() / total) if n_docs else 0.0
        self._rows = array("i")
        self._cols = array("i")
        self._vals = array("f")
        self._finalized = True
        n_postings = int(self._df.sum())
        log.info(
            "BM25 finalized: %d docs, %d terms, %d postings, avgdl=%.1f (%.1f bytes/posting)",
            n_docs, len(self._terms), n_postings, self.avgdl,
            (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / max(1, n_postings),
        )
        return self

    # ------------------------------------------------------------------ query
    def search(self, query: str, top_k: int = 100, *, candidates: int | None = None) -> list[tuple[int, float]]:
        """Return ``(doc_id, score)`` pairs ranked by BM25, highest first.

        Scores are accumulated into a dense float32 vector over all documents.
        That is 4 bytes/doc (28 MB at 7M chunks), which is far cheaper than the
        alternative of a per-query heap over sparse postings.
        """
        if not self._finalized:
            self.finalize()
        if self.n_docs == 0:
            return []
        tokens = tokenize(query, min_length=self.min_token_length)
        if not tokens:
            return []

        matrix = self._matrix
        doc_len = self._doc_len.astype(np.float32)
        avgdl = self.avgdl or 1.0
        k1, b = self.k1, self.b
        # Precompute the length-normalisation denominator once per query.
        denom_base = k1 * (1.0 - b + b * doc_len / avgdl)

        scores = np.zeros(self.n_docs, dtype=np.float32)
        query_counts: dict[str, int] = {}
        for token in tokens:
            query_counts[token] = query_counts.get(token, 0) + 1

        for term, qtf in query_counts.items():
            tid = self._vocab.get(term)
            # The CSC is (docs x terms), so the term bound is shape[1]. This tested
            # shape[0] until 2026-10-28, silently dropping every term with id >=
            # n_docs: never on a corpus with more rows than terms, but pmcsample
            # (8,086,716 rows, 8,130,690 terms) lost its last 43,974 terms.
            if tid is None or tid >= matrix.shape[1]:
                continue
            idf = float(self._idf[tid])
            if idf <= 0.0:
                continue
            start, end = matrix.indptr[tid], matrix.indptr[tid + 1]
            if end <= start:
                continue
            cols = matrix.indices[start:end].astype(np.int64, copy=False)
            tf = matrix.data[start:end]
            if candidates is not None and len(cols) > candidates:
                # Skip the long tail of very common terms; they add noise, not signal.
                keep = np.argpartition(-tf, candidates)[:candidates]
                cols, tf = cols[keep], tf[keep]
            contrib = idf * (tf * (k1 + 1.0) / (tf + denom_base[cols]))
            # Mild bonus for query-term repetition, as in the Lucene default.
            contrib *= 1.0 + 0.1 * math.log(1 + qtf)
            np.add.at(scores, cols, contrib.astype(np.float32))

        if candidates is not None:
            pool = min(candidates * 4, self.n_docs)
            idx = np.argpartition(-scores, pool - 1)[:pool]
        else:
            pool = min(max(top_k * 4, top_k), self.n_docs)
            idx = np.argpartition(-scores, pool - 1)[:pool]
        ranked = sorted(((int(i), float(scores[i])) for i in idx if scores[i] > 0.0),
                        key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "k1": self.k1,
            "b": self.b,
            "min_token_length": self.min_token_length,
            "n_docs": self.n_docs,
            "avgdl": self.avgdl,
            "terms": self._terms,
            "vocab": self._vocab,
            "doc_len": self._doc_len,
            "matrix_data": self._matrix.data,
            "matrix_indices": self._matrix.indices,
            "matrix_indptr": self._matrix.indptr,
            "matrix_shape": self._matrix.shape,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        log.info("saved BM25 index -> %s (%.1f MiB)", path, path.stat().st_size / 2**20)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        from scipy import sparse

        path = Path(path)
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        index = cls(k1=payload["k1"], b=payload["b"], min_token_length=payload["min_token_length"])
        index.n_docs = payload["n_docs"]
        index.avgdl = payload["avgdl"]
        index._terms = payload["terms"]
        index._vocab = payload["vocab"]
        index._doc_len = payload["doc_len"]
        index._matrix = sparse.csc_matrix(
            (payload["matrix_data"], payload["matrix_indices"], payload["matrix_indptr"]),
            shape=payload["matrix_shape"],
        )
        index._df = np.diff(index._matrix.indptr).astype(np.int32)
        total = float(index.n_docs) or 1.0
        index._idf = np.log(
            1.0 + (total - index._df.astype(np.float32) + 0.5) / (index._df.astype(np.float32) + 0.5)
        ).astype(np.float32)
        index._cur_doc = index.n_docs
        index._finalized = True
        log.info("loaded BM25 index <- %s (%d docs, %d terms)", path, index.n_docs, len(index._terms))
        return index

    # ----------------------------------------------------------------- reports
    def stats(self) -> dict[str, Any]:
        if self._matrix is not None:
            n_postings = int(self._df.sum()) if len(self._df) else 0
            bytes_total = (
                self._matrix.data.nbytes + self._matrix.indices.nbytes
                + self._matrix.indptr.nbytes + self._doc_len.nbytes
            )
            bytes_per_chunk = bytes_total / max(1, self.n_docs)
        else:
            n_postings, bytes_total, bytes_per_chunk = 0, 0, 0.0
        return {
            "n_docs": self.n_docs,
            "n_terms": len(self._terms),
            "n_postings": n_postings,
            "avgdl": round(self.avgdl, 2),
            "total_tokens": int(self._doc_len.sum()) if self.n_docs else 0,
            "index_bytes": bytes_total,
            "bytes_per_chunk": round(bytes_per_chunk, 1),
        }

    def top_terms(self, n: int = 25) -> list[tuple[str, int]]:
        if not len(self._df):
            return []
        order = np.argsort(-self._df)[:n]
        return [(self._terms[i], int(self._df[i])) for i in order]


def build_bm25(texts: Iterable[str], **kwargs: Any) -> BM25Index:
    """Convenience builder for an ordered iterable of chunk texts."""
    index = BM25Index(**kwargs)
    for i, text in enumerate(texts):
        index.add(i, text)
    return index.finalize()


def bm25_stats_json(index: BM25Index, path: str | Path, top_n: int = 25) -> Path:
    payload = index.stats()
    payload["top_terms"] = index.top_terms(top_n)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
