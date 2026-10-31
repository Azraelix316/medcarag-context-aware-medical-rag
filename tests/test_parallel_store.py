"""scripts/parallel_store.py: per-shard chunking and store assembly.

The assembled store must be byte-identical to build_store_from_chunks over the same
chunks in the same order -- the path ``medrag chunk`` + build_pmc_store.py takes --
because every later artifact (BM25 segments, page index, dependency graph) is keyed
by its row ids.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from medrag.config import ChunkConfig  # noqa: E402
from medrag.data.chunk import chunk_document  # noqa: E402
from medrag.data.schema import Chunk, Document, Section  # noqa: E402
from medrag.index import page_index as pi  # noqa: E402
from medrag.index.builder import build_store_from_chunks  # noqa: E402
from medrag.index.store import ChunkStore  # noqa: E402

spec = importlib.util.spec_from_file_location("parallel_store", "scripts/parallel_store.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)
spec = importlib.util.spec_from_file_location("parallel_blob", "scripts/parallel_blob.py")
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)

WORDS = ("aspirin stroke trial dose risk cohort mortality infection sepsis insulin "
         "randomised patients outcome follow-up hazard ratio confidence interval é µ").split()


def _docs(n: int, seed: int) -> list[Document]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        secs = [Section(title=t, text=" ".join(rng.choices(WORDS, k=rng.randint(20, 900))), kind=k)
                for t, k in (("Methods", "methods"), ("Results", "results"), ("Discussion", "discussion"))
                if rng.random() < 0.8]
        out.append(Document(doc_id=f"pmcid:PMC{seed}{i:05d}", source="pmc", title=f"T{i}",
                            abstract=" ".join(rng.choices(WORDS, k=rng.randint(0, 200))),
                            sections=secs, journal="J", year=2000 + i % 20))
    return out


def test_assembled_store_equals_the_single_process_store() -> None:
    cfg = ChunkConfig(target_tokens=64, max_tokens=96, overlap_tokens=8, min_tokens=12)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        corpus = d / "corpus"
        corpus.mkdir()
        all_docs = []
        for s in range(4):
            docs = _docs(15 + s, seed=s)
            all_docs += docs
            with gzip.open(corpus / f"pmc-{s:05d}.jsonl.gz", "wt", encoding="utf-8") as fh:
                for doc in docs:
                    fh.write(json.dumps(doc.to_dict()) + "\n")
                if s == 2:
                    fh.write("{not json\n")             # a torn line, skipped as read_shards does
        names = sorted(p.name for p in corpus.glob("pmc-*.jsonl.gz"))
        out = d / "chunks"
        out.mkdir()
        for name in reversed(names):                    # any completion order
            ps.chunk_shard(corpus / name, out, cfg)
        ps.assemble(out, d / "par" / "store", names, cache_size=500_000, page_texts=True)

        # The reference: chunk -> serialise -> Chunk.from_dict -> store.add, in order.
        ref_chunks = [Chunk.from_dict(c.to_dict()) for doc in all_docs
                      for c in chunk_document(Document.from_dict(doc.to_dict()), cfg)]
        assert len(ref_chunks) > 100
        build_store_from_chunks(ref_chunks, store_dir=d / "ref" / "store", cache_size=500_000)
        for f in ("blob.jsonl", "offsets.bin", "rowmap.json", "store.json"):
            a = (d / "par" / "store" / f).read_bytes()
            b = (d / "ref" / "store" / f).read_bytes()
            assert a == b, f
        # Page texts from the same pass == write_page_text_blocks over the store.
        old = pi.PAGE_BLOCK_DOCS
        pi.PAGE_BLOCK_DOCS = 7
        try:
            ps.assemble(out, d / "par2" / "store", names, cache_size=500_000, page_texts=True)
        finally:
            pi.PAGE_BLOCK_DOCS = old
        pi.write_page_text_blocks(d / "ref", block_docs=7)
        ref_pages = json.loads((d / "ref" / pi.PAGE_TEXTS_DIR / "pages.json").read_text())
        par_pages = json.loads((d / "par2" / pi.PAGE_TEXTS_DIR / "pages.json").read_text())
        assert par_pages == ref_pages, (par_pages, ref_pages)
        for b in range(ref_pages["n_blocks"]):
            assert pi.read_page_text_block(d / "par2", b) == pi.read_page_text_block(d / "ref", b), b
        # The many-writer blob (parallel_blob.py) == assemble's, page texts included.
        stems = pb.stems_of(out)
        starts = pb.row_starts(out, stems)
        (out / "blob").mkdir()
        for s in reversed(stems):
            pb.do_size(out, s, starts[s])
        pb.do_prep(out, d / "par" / "store_par", stems, 500_000)
        lay = pb.layout(out, stems)
        for s in reversed(stems):
            pb.do_write(out, d / "par" / "store_par", s, *lay[s])
        assert all(pb.do_verify(out, d / "par" / "store_par", s, *lay[s]) for s in stems)
        pb.do_pages(out, d / "par", stems, lay["__end__"][0], "page_texts_par")
        for f in ("blob.jsonl", "offsets.bin", "rowmap.json", "store.json"):
            assert (d / "par" / "store_par" / f).read_bytes() == (d / "par" / "store" / f).read_bytes(), f
        ref_tx = json.loads((d / "par" / pi.PAGE_TEXTS_DIR / "pages.json").read_text())
        assert json.loads((d / "par" / "page_texts_par" / "pages.json").read_text()) == ref_tx
        for b in range(ref_tx["n_blocks"]):
            name = pi.page_text_block_path(d / "par", b).name
            with gzip.open(d / "par" / "page_texts_par" / name, "rt") as x, \
                    gzip.open(d / "par" / pi.PAGE_TEXTS_DIR / name, "rt") as y:
                assert x.read() == y.read(), b
        # A corrupted range is caught by verify.
        with open(d / "par" / "store_par" / "blob.jsonl", "r+b") as fh:
            fh.seek(lay[stems[1]][1] + 3)
            fh.write(b"#")
        assert not pb.do_verify(out, d / "par" / "store_par", stems[1], *lay[stems[1]])
        st = ChunkStore.load(d / "par" / "store")
        assert len(st) == len(ref_chunks)
        assert st.get(len(st) - 1).chunk_id == ref_chunks[-1].chunk_id
    print(f"  assembled store == single-process store, byte for byte ({len(ref_chunks)} chunks, "
          f"4 shards chunked out of order), page texts == write_page_text_blocks")


def main() -> int:
    tests = [test_assembled_store_equals_the_single_process_store]
    print("parallel chunking and store assembly")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
