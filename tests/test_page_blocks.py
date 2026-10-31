"""The blocked page build must produce PageIndex.build's documents, texts and vectors.

The 20M page index cannot be built by PageIndex.build (its matrix is 31 GB), so it is
built in blocks. These tests check the blocked path against the in-RAM definitions:
the same documents in the same order, the same page text, the same normalisation,
and a memmapped matrix the retriever reads exactly like the pickled one.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from medrag.data.schema import Chunk  # noqa: E402
from medrag.index import page_index as pi  # noqa: E402

spec = importlib.util.spec_from_file_location("remote_embed", "scripts/remote_embed.py")
remote_embed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote_embed)


def _store(d: Path, n_docs: int = 23) -> list[dict]:
    """Documents with 1-4 chunks each, one doc split by another (first appearance)."""
    recs = []
    for i in range(n_docs):
        for c in range(1 + i % 4):
            recs.append({"chunk_id": f"pmid:{1000 + i}#{c:04d}", "doc_id": f"pmid:{1000 + i}",
                         "text": f"body {i} {c}", "title": f"Title {i}", "journal": f"J{i % 3}",
                         "year": 2000 + i, "categories": ["rct"] if i % 2 else [],
                         "section_kind": "background"})
    recs.insert(5, {**recs[0], "chunk_id": "pmid:1000#0009"})   # doc 1000 again, later
    s = d / "store"
    s.mkdir(parents=True)
    with open(s / "blob.jsonl", "wb") as blob, open(s / "offsets.bin", "wb") as offs:
        pos = 0
        for r, rec in enumerate(recs):
            raw = json.dumps({"row": r, **rec}).encode() + b"\n"
            blob.write(raw)
            offs.write(struct.pack("<QQ", pos, len(raw) - 1))
            pos += len(raw)
    return recs


def _args(index_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(index_dir=str(index_dir), kind="page", block=None, dim=4,
                              host="test", local=True)


def test_texts_match_document_rows_and_text_from_chunk() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        recs = _store(d)
        m = pi.write_page_text_blocks(d, block_docs=5, read_rows=7)
        expect_docs, expect_rows, seen = [], [], set()
        for r, rec in enumerate(recs):
            doc = rec["chunk_id"].rsplit("#", 1)[0]
            if doc not in seen:
                seen.add(doc)
                expect_docs.append(doc)
                expect_rows.append(r)
        got = [x for b in range(m["n_blocks"]) for x in pi.read_page_text_block(d, b)]
        assert [g["doc"] for g in got] == expect_docs
        assert [g["row"] for g in got] == expect_rows
        for g in got:
            rec = recs[g["row"]]
            assert g["text"] == pi._text_from_chunk(Chunk.from_dict(rec))
        assert m["n_docs"] == 23 and m["n_blocks"] == 5
    print("  page texts: document_rows order, _text_from_chunk text, 5 blocks")


def test_claim_put_assemble_and_load() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _store(d)
        m = pi.write_page_text_blocks(d, block_docs=5)
        args = _args(d)
        rng = np.random.default_rng(0)
        published = {}
        while True:
            head, texts = remote_embed._claim_local(args)
            if head["block"] is None:
                break
            v = rng.standard_normal((len(texts), 4)).astype(np.float32) * 3
            if head["block"] == 1:
                v[0] = 0.0                      # a zero row must stay zero, not NaN
            assert remote_embed._put_local(args, head["block"], v) == 0
            published[head["block"]] = v
        assert sorted(published) == list(range(m["n_blocks"]))
        assert remote_embed._put_local(args, 0, np.zeros((2, 4), np.float32)) == 2
        idx = pi.PageIndex.load(pi.assemble_page_blocks(d))
        assert isinstance(idx.matrix, np.memmap) and idx.matrix.shape == (23, 4)
        raw = np.concatenate([published[b] for b in range(m["n_blocks"])])
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        assert np.array_equal(np.asarray(idx.matrix), raw / norms)
        assert idx.position_of()["pmid:1001"] == 1 and idx.doc_ids[0] == "pmid:1000"
        assert idx.position_of() is idx.position_of()
    print("  page blocks: claimed once each, bad shape rejected, assembled, memmap loads")


def test_blocks_claimable_before_the_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _store(d)
        pi.write_page_text_blocks(d, block_docs=5)
        manifest = d / "page_texts" / "pages.json"
        manifest.rename(d / "pages.json.hidden")
        (d / "page_texts" / "texts-00004.jsonl.gz").rename(d / "page_texts" / "texts-00004.jsonl.gz.tmp")
        args = _args(d)
        got = []
        while True:
            head, texts = remote_embed._claim_local(args)
            if head["block"] is None:
                break
            got.append(head["block"])
        assert got == [0, 1, 2, 3] and head["final"] is False
        (d / "page_texts" / "texts-00004.jsonl.gz.tmp").rename(d / "page_texts" / "texts-00004.jsonl.gz")
        (d / "pages.json.hidden").rename(manifest)
        head, _ = remote_embed._claim_local(args)
        assert head["block"] == 4
        head, _ = remote_embed._claim_local(args)
        assert head == {"block": None, "final": True}
    print("  page blocks are claimable while texts are still being written")


def main() -> int:
    tests = [test_texts_match_document_rows_and_text_from_chunk, test_claim_put_assemble_and_load,
             test_blocks_claimable_before_the_manifest]
    print("blocked page index tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
