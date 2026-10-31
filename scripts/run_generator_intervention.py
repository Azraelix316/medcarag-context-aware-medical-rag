"""Can any generator-side intervention recover the caveats that retrieval supplies?

Measured in the read test: passages triple their caveat rate (0.0833 -> 0.2333)
and the generated answer barely moves (0.0333 -> 0.0417). Caveat transmission is
14%. Retrieval cannot fix that, so this asks whether the generator can.

Four arms, all on the SAME retrieved passages, so the comparison isolates the
generator intervention:

  baseline   the read-test prompt as-is
  required   asks for a final limitation sentence, permitting an explicit decline
  twopass    baseline answer, then a separate "what limits this?" call, appended
  required-2pass  both

The control that matters
------------------------
An intervention that raises the caveat rate is worthless if it raises it by
*inventing* hedging. So every arm is also scored separately on queries whose
passages contained NO caveat:

  conditional caveat rate  -- caveats stated when the evidence had one
  unconditional caveat rate -- caveats stated when it did not

An intervention is only a real gain if the first rises and the second stays near
the baseline's. A rising unconditional rate is fabricated clinical caution, which
is worse than none: it trains a reader to distrust every number.
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
from medrag.eval.metrics import Qrels
from medrag.eval.read_test import (
    aggregate_answers,
    build_pass2_prompt,
    build_prompt,
    generate,
    paired_delta,
    score_answer,
)
from medrag.eval.readiness import (
    aggregate_readiness,
    find_caveat,
    readiness_from_rows,
)
from medrag.index.retriever import load_index
from medrag.index.sibling_bleed import expand_with_context

N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
WORKERS = 4
K, POOL, SN, PER = 10, 50, 10, 1

cfg = PipelineConfig.load(
    ["configs/default.yaml",
     "configs/production.yaml"]
)._finalize()
r = load_index(cfg, with_reranker=False, with_graph=False, with_dependency=True)
bm = Qrels.load(cfg.paths.runs_dir / "benchmark" / "qrels_combined.json")
qs = [q for q in bm.queries
      if q.relevant_docs and q.category in ("diseases", "drugs-and-therapeutics")]
random.Random(7).shuffle(qs)
qs = qs[:N]
print(f"  {len(qs)} queries, context arm passages for every arm", flush=True)

# One retrieval pass, reused by all four generator arms.
jobs: list[dict] = []
passage_had_caveat: dict[str, bool] = {}
for q in qs:
    pool = r._sparse_search(q.text, POOL)
    rows = [int(m[0]) for m in expand_with_context(
        [(int(x), s) for x, s in pool[:K]], r.store, r.dependency,
        top_k=K, seed_count=SN, context_per_seed=PER)[:K + SN * PER]]
    texts = [(x, (c.text if (c := r.store.get(x)) else "")) for x in rows]
    readiness = readiness_from_rows(rows, r.store, q.relevant_docs, qid=q.qid)
    passage_had_caveat[q.qid] = any(find_caveat(t or "") for _, t in texts)
    jobs.append({"qid": q.qid, "question": q.text, "texts": texts,
                 "readiness": readiness, "baseline": build_prompt(q.text, texts),
                 "required": build_prompt(q.text, texts, required=True),
                 "pass2": build_pass2_prompt(q.text, texts)})
print(f"  prepared {len(jobs)} in {time.time() - _T0:.0f}s | "
      f"passages contained a caveat for "
      f"{sum(passage_had_caveat.values())}/{len(jobs)} queries", flush=True)

_T0 = time.time()
t0 = time.time()
results: dict[str, dict[str, object]] = {k: {} for k in
                                         ("baseline", "required", "twopass", "required-2pass")}


def call_one(job: dict) -> dict[str, str]:
    base = generate(job["baseline"])
    req = generate(job["required"])
    p2 = generate(job["pass2"], max_tokens=200)
    return {"baseline": base, "required": req, "twopass": f"{base}\n\n{p2}",
            "required-2pass": f"{req}\n\n{p2}"}


with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    done = 0
    for job, texts in zip(jobs, ex.map(call_one, jobs)):
        for arm, txt in texts.items():
            results[arm][job["qid"]] = score_answer(txt, qid=job["qid"], arm=arm)
        done += 1
        if done % 40 == 0:
            print(f"  {done}/{len(jobs)} | {time.time() - t0:.0f}s", flush=True)


def rep(arm: str):
    from medrag.eval.read_test import aggregate_answers as _agg
    return _agg(list(results[arm].values()))


print("\n  CAVEAT rate by arm, split by whether the passages contained one:", flush=True)
print(f"    {'arm':16s} {'overall':>8s} {'cond (had)':>11s} {'uncond (none)':>14s} {'declined':>9s}")
print("    " + "-" * 62)
summary: dict[str, dict] = {}
for arm in ("baseline", "required", "twopass", "required-2pass"):
    all_v = list(results[arm].values())
    had = [v for q, v in results[arm].items() if passage_had_caveat[q]]
    none = [v for q, v in results[arm].items() if not passage_had_caveat[q]]
    overall = sum(v.has_caveat for v in all_v) / len(all_v)
    cond = (sum(v.has_caveat for v in had) / len(had)) if had else float("nan")
    uncond = (sum(v.has_caveat for v in none) / len(none)) if none else float("nan")
    dec = sum(v.declined for v in all_v) / len(all_v)
    summary[arm] = {"overall": round(overall, 4), "conditional": round(cond, 4),
                    "unconditional": round(uncond, 4), "declined": round(dec, 4),
                    "n_conditional": len(had), "n_unconditional": len(none)}
    print(f"    {arm:16s} {overall:>8.4f} {cond:>11.4f} {uncond:>14.4f} {dec:>9.4f}")

print("\n  INTERPRETATION", flush=True)
base = summary["baseline"]
for arm in ("required", "twopass", "required-2pass"):
    a = summary[arm]
    gain_cond = a["conditional"] - base["conditional"]
    cost_uncond = a["unconditional"] - base["unconditional"]
    if gain_cond > 0.02 and cost_uncond <= 0.02:
        verdict = "REAL GAIN: recovers caveats without inventing them"
    elif gain_cond > 0.02 and cost_uncond > 0.02:
        verdict = "SUSPECT: gain is partly fabricated hedging"
    elif gain_cond <= 0.02:
        verdict = "NO GAIN: caveats not recovered"
    else:
        verdict = "mixed"
    print(f"    {arm:16s} conditional {gain_cond:+.4f}, "
          f"unconditional {cost_uncond:+.4f} -> {verdict}")

with open("data/runs/generator_intervention.json", "w") as fh:
    json.dump({
        "n_queries": len(jobs),
        "n_queries_with_caveat_in_passages": sum(passage_had_caveat.values()),
        "arms": summary,
        "paired_vs_baseline": {
            arm: paired_delta(rep("baseline"), rep(arm)) for arm in
            ("required", "twopass", "required-2pass")
        },
        "elapsed_s": round(time.time() - t0, 1),
    }, fh, indent=2)
print(f"\n  elapsed {time.time() - t0:.0f}s -> data/runs/generator_intervention.json")