"""Does grounded attribution fix the caveat gap without fabricating?

Compares four generator arms on the SAME retrieved passages, 120 queries:

  baseline          the read-test prompt
  required          asks for a limitation sentence (measured: worse)
  twopass           a separate "what limits this?" call (measured: +24 pts conditional,
                    but 4.55% fabricated)
  attributed        quoted limitations located in the passages, appended verbatim

The prediction under test is sharp. Attribution is not generative, so its caveat
rate should be bounded by the retrieved caveat rate:

  conditional   ~1.00  -- if a passage hedges, the reader is told, every time
  unconditional  0.00  -- and never told otherwise

If unconditional is not 0, the contract in medrag/eval/attribution.py is wrong and
the module should not ship. That is the falsifier this script exists to run.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.ERROR)

from medrag.config import PipelineConfig
from medrag.eval.attribution import compose_attributed_answer, extract_caveats
from medrag.eval.metrics import Qrels
from medrag.eval.read_test import (
    aggregate_answers,
    build_pass2_prompt,
    build_prompt,
    generate,
    score_answer,
)
from medrag.eval.readiness import (
    aggregate_readiness,
    find_caveat,
    readiness_from_rows,
)
from medrag.index.retriever import load_index
from medrag.index.sibling_bleed import expand_with_context

import argparse

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("n_queries", nargs="?", type=int, default=120)
_ap.add_argument("--config", action="append")
_ap.add_argument("--qrels",
                 default="data/runs/benchmark/qrels_combined.json")
_ap.add_argument("--out",
                 default="data/runs/attributed_caveats.json")
args = _ap.parse_args()
N = args.n_queries
WORKERS = 4
K, POOL, SN, PER = 10, 50, 10, 1

cfg = PipelineConfig.load(args.config or [
    "configs/default.yaml",
    "configs/production.yaml",
])._finalize()
# with_dense=False: loading 16,133,551 x 384 from NFS costs 2,168 s and every arm
# here is compared on the sparse baseline.
print(f"  index dir: {cfg.paths.index_dir}", flush=True)
print(f"  corpus root: {cfg.paths.data_root}", flush=True)
r = load_index(cfg, with_reranker=False, with_graph=False, with_dependency=True,
               with_dense=False)
bm = Qrels.load(args.qrels)
qs = [q for q in bm.queries if q.relevant_docs]
random.Random(7).shuffle(qs)
qs = qs[:N]
print(f"  {len(qs)} queries, context-arm passages shared by all arms", flush=True)

jobs: list[dict] = []
passage_had_caveat: dict[str, bool] = {}
for q in qs:
    pool = r._sparse_search(q.text, POOL)
    rows = [int(m[0]) for m in expand_with_context(
        [(int(x), s) for x, s in pool[:K]], r.store, r.dependency,
        top_k=K, seed_count=SN, context_per_seed=PER)[:K + SN * PER]]
    texts = [(x, (c.text if (c := r.store.get(x)) else "")) for x in rows]
    passage_had_caveat[q.qid] = any(find_caveat(t or "") for _, t in texts)
    jobs.append({
        "qid": q.qid, "texts": texts,
        "readiness": readiness_from_rows(rows, r.store, q.relevant_docs, qid=q.qid),
        "baseline": build_prompt(q.text, texts),
        "required": build_prompt(q.text, texts, required=True),
        "pass2": build_pass2_prompt(q.text, texts),
    })
n_had = sum(passage_had_caveat.values())
print(f"  prepared {len(jobs)} | passages contained a caveat for {n_had}/{len(jobs)}",
      flush=True)

t0 = time.time()
results: dict[str, dict[str, object]] = {
    k: {} for k in ("baseline", "required", "twopass", "attributed")}


def call_one(job: dict) -> dict[str, str]:
    base = generate(job["baseline"])
    req = generate(job["required"])
    p2 = generate(job["pass2"], max_tokens=200)
    return {
        "baseline": base,
        "required": req,
        "twopass": f"{base}\n\n{p2}",
        # No generation on the attribution path beyond the baseline answer.
        "attributed": compose_attributed_answer(base, job["texts"]),
    }


with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    done = 0
    for job, texts in zip(jobs, ex.map(call_one, jobs)):
        for arm, txt in texts.items():
            results[arm][job["qid"]] = score_answer(txt, qid=job["qid"], arm=arm)
        done += 1
        if done % 40 == 0:
            print(f"  {done}/{len(jobs)} | {time.time() - t0:.0f}s", flush=True)


def summarise(arm: str) -> dict:
    all_v = list(results[arm].values())  # type: ignore[union-attr]
    had = [v for q, v in results[arm].items() if passage_had_caveat[q]]  # type: ignore[union-attr]
    none = [v for q, v in results[arm].items() if not passage_had_caveat[q]]  # type: ignore[union-attr]
    rep = aggregate_answers(all_v)
    return {
        "overall": round(rep.caveat, 4),
        "conditional": round(sum(v.has_caveat for v in had) / len(had), 4) if had else None,
        "unconditional": round(sum(v.has_caveat for v in none) / len(none), 4) if none else None,
        "n_conditional": len(had),
        "n_unconditional": len(none),
        "rate": round(rep.rate, 4),
        "population": round(rep.population, 4),
        "ready": round(rep.readiness, 4),
        "declined": round(rep.declined, 4),
        "produced": round(rep.production, 4),
    }


print("\n  CAVEAT rate by arm, split by whether the passages contained one:",
      flush=True)
print(f"    {'arm':13s} {'overall':>8s} {'cond':>8s} {'uncond':>8s} "
      f"{'declined':>9s} {'ready':>7s}")
print("    " + "-" * 56)
summary = {arm: summarise(arm)
           for arm in ("baseline", "required", "twopass", "attributed")}
for arm, s in summary.items():
    print(f"    {arm:13s} {s['overall']:>8.4f} {s['conditional']:>8.4f} "
          f"{s['unconditional']:>8.4f} {s['declined']:>9.4f} {s['ready']:>7.4f}")

att = summary["attributed"]
print("\n  CONTRACT CHECK on attribution", flush=True)
checks = [
    ("cannot fabricate", att["unconditional"] == 0.0,
     f"unconditional caveat rate {att['unconditional']} (must be 0.0)"),
    ("always reports", att["conditional"] >= 0.95,
     f"conditional caveat rate {att['conditional']} (must be >= 0.95)"),
    ("beats baseline conditionally",
     att["conditional"] > summary["baseline"]["conditional"],
     f"{summary['baseline']['conditional']} -> {att['conditional']}"),
    ("does not cost the rate",
     att["rate"] >= summary["baseline"]["rate"] - 0.02,
     f"rate {summary['baseline']['rate']} -> {att['rate']}"),
]
all_ok = True
for label, ok, detail in checks:
    print(f"    [{'PASS' if ok else 'FAIL'}] {label:28s} {detail}")
    all_ok = all_ok and ok

base = summary["baseline"]
if base["unconditional"]:
    disc_base = "infinite" if base["conditional"] else "undefined"
else:
    disc_base = f"{base['conditional'] / 1e-9:.0f}:1 (baseline uncond = 0)"
tp = summary["twopass"]
disc_tp = (tp["conditional"] / tp["unconditional"]) if tp["unconditional"] else None
print("\n  DISCRIMINATION (conditional / unconditional):", flush=True)
print(f"    baseline  {disc_base}")
print(f"    twopass   {disc_tp:.1f}:1" if disc_tp else "    twopass   n/a")
print("    attributed  infinite (unconditional is exactly 0)")

out = {
    "n_queries": len(jobs),
    "n_queries_with_caveat_in_passages": n_had,
    "arms": summary,
    "contract_checks": [{"check": c, "passed": ok, "detail": d}
                        for c, ok, d in checks],
    "contract_all_passed": all_ok,
    "elapsed_s": round(time.time() - t0, 1),
}
with open(args.out, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\n  contract {'HOLDS' if all_ok else 'VIOLATED'} -> {args.out}", flush=True)