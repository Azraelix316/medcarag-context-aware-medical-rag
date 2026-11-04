"""Download the released indexes and benchmark data, reassemble split files, verify them.

    python scripts/fetch_release.py --corpus pmc4m --dest data/release
    python scripts/fetch_release.py --corpus all --data --dest data/release

Index files come from the GitHub release `indexes-1` and land at
<dest>/<corpus>/index/<path>, the layout the retrievers read. Files over 1.9 GB are
stored as <name>.partNNN and are joined here. Every file listed in the corpus's
benchmark/<corpus>/RELEASE_MANIFEST.json is checked against its SHA-256. `--data` also
fetches the `data-1` assets (PMID/PMCID manifests, PMID-to-MeSH mapping, MeSH qrels)
into <dest>/data and checks them against SHA256SUMS. Finished files are skipped on a
rerun, so an interrupted download can simply be restarted. Article text is not part of
the release; rebuild it from NCBI with the acquisition scripts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = "Azraelix316/medcarag-context-aware-medical-rag"
ROOT = Path(__file__).resolve().parent.parent
CORPORA = ("pmc4m", "pubmed20m")


def list_assets(tag: str) -> list[dict]:
    url = f"https://api.github.com/repos/{REPO}/releases/tags/{tag}"
    with urllib.request.urlopen(url) as r:
        release_id = json.load(r)["id"]
    assets, page = [], 1
    while True:
        url = f"https://api.github.com/repos/{REPO}/releases/{release_id}/assets?per_page=100&page={page}"
        with urllib.request.urlopen(url) as r:
            batch = json.load(r)
        if not batch:
            return assets
        assets += batch
        page += 1


def download(asset: dict, out: Path) -> None:
    if out.exists() and out.stat().st_size == asset["size"]:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    print(f"  get {asset['name']} ({asset['size'] / 1e9:.2f} GB)", flush=True)
    with urllib.request.urlopen(asset["browser_download_url"]) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 22)
    tmp.replace(out)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def fetch_corpus(corpus: str, assets: list[dict], dest: Path) -> list[str]:
    manifest = json.loads((ROOT / "benchmark" / corpus / "RELEASE_MANIFEST.json").read_text())
    by_name = {corpus + "." + f["path"].split("/")[-1]: f for f in manifest["files"]}
    groups: dict[str, list[dict]] = {}
    for a in assets:
        if a["name"].startswith(corpus + "."):
            groups.setdefault(re.sub(r"\.part\d+$", "", a["name"]), []).append(a)

    index_dir, parts_dir = dest / corpus / "index", dest / corpus / "parts"
    failed = []
    for name, group in sorted(groups.items()):
        entry = by_name.get(name)
        rel = entry["path"] if entry else name[len(corpus) + 1:]
        target = index_dir / rel
        if entry and target.exists() and target.stat().st_size == entry["bytes"]:
            continue
        group.sort(key=lambda a: a["name"])
        if len(group) == 1 and group[0]["name"] == name:
            download(group[0], target)
        else:
            pieces = [parts_dir / a["name"] for a in group]
            for a, p in zip(group, pieces):
                download(a, p)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as out:
                for p in pieces:
                    with open(p, "rb") as f:
                        shutil.copyfileobj(f, out, 1 << 24)
            for p in pieces:
                p.unlink()
        if entry and sha256(target) != entry["sha256"]:
            print(f"  SHA-256 MISMATCH {rel}", flush=True)
            failed.append(f"{corpus}/{rel}")
    if parts_dir.exists() and not any(parts_dir.iterdir()):
        parts_dir.rmdir()
    print(f"{corpus}: {len(groups)} files in {index_dir}", flush=True)
    return failed


def fetch_data(dest: Path) -> list[str]:
    out_dir = dest / "data"
    for a in list_assets("data-1"):
        download(a, out_dir / a["name"])
    failed = []
    for line in (out_dir / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if sha256(out_dir / name) != digest:
            print(f"  SHA-256 MISMATCH {name}", flush=True)
            failed.append(f"data/{name}")
    print(f"data: benchmark files in {out_dir}", flush=True)
    return failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=CORPORA + ("all", "none"), default="all")
    ap.add_argument("--data", action="store_true", help="also fetch the data-1 benchmark files")
    ap.add_argument("--dest", type=Path, default=Path("data/release"))
    args = ap.parse_args()

    failed = []
    corpora = CORPORA if args.corpus == "all" else () if args.corpus == "none" else (args.corpus,)
    if corpora:
        assets = list_assets("indexes-1")
        for corpus in corpora:
            failed += fetch_corpus(corpus, assets, args.dest)
    if args.data:
        failed += fetch_data(args.dest)
    if failed:
        sys.exit(f"{len(failed)} file(s) failed verification: {', '.join(failed)}")
    print("all files verified")


if __name__ == "__main__":
    main()
