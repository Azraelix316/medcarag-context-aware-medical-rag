"""Write RELEASE_MANIFEST.json for an index directory: every shippable file, its size
and sha256, plus the counts a downstream user needs to check a copy.

Working files are left out: per-block embedding and page npz files (they are
concatenated into embeddings.f32 / page_vectors.f32), claims, spill directories and
temporary files. Hashing reads every byte once, so on NFS this is the slowest step;
``--workers`` hashes files in parallel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SKIP_DIRS = {"claims", "page_blocks", "bm25_spill", "graph_inputs"}
SKIP_PREFIXES = ("embeddings-block", "spill-")


def _shippable(index_dir: Path) -> list[Path]:
    out = []
    for p in sorted(index_dir.rglob("*")):
        rel = p.relative_to(index_dir)
        if not p.is_file() or any(part in SKIP_DIRS or part.startswith("spill-")
                                  for part in rel.parts[:-1]):
            continue
        if p.name.startswith(SKIP_PREFIXES) or ".tmp" in p.name or p.name == "RELEASE_MANIFEST.json":
            continue
        out.append(p)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", default="data/abstracts20m/index")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    index_dir = Path(args.index_dir)
    files = _shippable(index_dir)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        digests = list(pool.map(_sha256, files))
    entries = [{"path": str(p.relative_to(index_dir)).replace("\\", "/"),
                "bytes": p.stat().st_size, "sha256": d} for p, d in zip(files, digests)]
    counts = {"store_rows": (index_dir / "store" / "offsets.bin").stat().st_size // 16}
    for name, key in (("embeddings.json", "shape"),):
        if (index_dir / name).exists():
            counts["embeddings_shape"] = json.loads((index_dir / name).read_text())[key]
    pages = index_dir / "page_texts" / "pages.json"
    if pages.exists():
        counts["documents"] = json.loads(pages.read_text())["n_docs"]
    counts["bm25_segments"] = len(list((index_dir / "bm25_segments").glob("*.stats.pkl")))
    manifest = {"index_dir": str(index_dir), "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "hash_seconds": round(time.time() - t0, 1), "counts": counts,
                "total_bytes": sum(e["bytes"] for e in entries), "files": entries}
    out = index_dir / "RELEASE_MANIFEST.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(out)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
