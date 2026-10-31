"""Readiness of the text a read-test generator actually saw.

run_read_test_rows.py scores input readiness on each passage's full text (up to 4,000
characters), but build_prompt caps the prompt at 5,500 characters and cuts every
passage to max(400, 5500 // n) characters. This re-scores each arm's gold passages on
exactly the truncated text that went into the prompt, so an answer-side null can be
told apart from "the ingredient never reached the generator".

    python scripts/prompt_readiness.py --config ... --qrels ... --answers merged.jsonl \\
        --out prompt_readiness.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.config import PipelineConfig  # noqa: E402
from medrag.eval.readiness import (  # noqa: E402
    RetrievedPassage, aggregate_readiness, assess_readiness)
from medrag.index.store import ChunkStore  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_mode_ablation", Path(__file__).resolve().parent / "run_mode_ablation.py")
_rma = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rma)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--max-chars", type=int, default=5500, help="build_prompt's default")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)._finalize()
    store = ChunkStore.load(cfg.paths.index_dir / "store")
    gold = {q.qid: q.relevant_docs for q in _rma._sample_queries(args.qrels, 2000)}
    per: dict[str, dict[str, list]] = {}
    with open(args.answers) as fh:
        for line in fh:
            d = json.loads(line)
            rows = d["rows"]
            cut = max(400, args.max_chars // max(1, len(rows)))
            full, seen = [], []
            for row in rows:
                c = store.get(int(row))
                if c is None:
                    continue
                text = " ".join((c.text or "").split())
                full.append(RetrievedPassage(row_id=int(row), doc_id=c.doc_id,
                                             text=text[:4000], section_kind=""))
                seen.append(RetrievedPassage(row_id=int(row), doc_id=c.doc_id,
                                             text=text[:cut], section_kind=""))
            arm = per.setdefault(d["arm"], {"full": [], "prompt": []})
            arm["full"].append(assess_readiness(full, gold[d["qid"]], qid=d["qid"]))
            arm["prompt"].append(assess_readiness(seen, gold[d["qid"]], qid=d["qid"]))
    out = {a: {k: aggregate_readiness(v).summary() for k, v in s.items()}
           for a, s in per.items()}
    with open(args.out, "w") as fh:
        json.dump({"max_chars": args.max_chars, "readiness": out}, fh, indent=2)
    for a, s in out.items():
        print(f"  {a:16s} full ready {s['full']['readiness']:.4f} rate "
              f"{s['full']['quantification_rate']:.4f} | as prompted ready "
              f"{s['prompt']['readiness']:.4f} rate {s['prompt']['quantification_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
