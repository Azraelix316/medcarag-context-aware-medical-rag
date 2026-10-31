"""Read test: retrieval -> generation -> what the ANSWER actually states.

This is the link in the chain that no retrieval metric can close. Retrieval-side
provision says the ingredients are in the passages; it says nothing about whether the
reader is told. Falsifier 5 measured the gap on abstracts and found it was almost total
for caveats -- retrieval-side caveat provision rose 180% while the caveat rate in the
answer rose 25%, a transmission of 14%.

Two changes from the version this replaces, both forced by what went wrong later:

**It retrieves through ``Retriever.retrieve(mode=...)``.** It used to call
``expand_with_context`` directly, which measures the mechanism rather than the
product. That is how three separate bugs stayed invisible for hours: context arms
truncated back to the baseline on the ``include_text=False`` path, page scoring that
never re-sorted, and a page matrix addressed by the wrong id space. Measuring the
mechanism was the mistake; this measures the path a caller takes.

**It reports transmission explicitly.** For each ingredient the interesting quantity
is not the output rate but the ratio of output gain to input gain -- the fraction of
retrieval-side improvement that survives to the reader. A component that doubles the
retrieved caveats and leaves the answer unchanged has transmission 0, and calling that
a retrieval win would be wrong.

Usage::

    scripts/run_read_test.py --config configs/default.yaml \\
        --config configs/pmcmerged.yaml \\
        --qrels data/runs/benchmark/qrels_pmc.json \\
        --arms sparse,context,pagectx --n-queries 120 \\
        --out data/runs/read_test_pmc.json
"""

from __future__ import annotations

import argparse
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
    build_prompt,
    generate,
    paired_delta,
    score_answer,
)
from medrag.eval.readiness import aggregate_readiness, readiness_from_rows
from medrag.index.retriever import load_index
from medrag.index.sibling_bleed import expand_with_context

#: Input-side readiness key -> output-side AnswerReport.summary() key.
#:
#: Not the `has_*` names: those are the keys paired_delta() returns for its nested
#: per-attribute dicts, while summary() flattens the same quantities to bare floats
#: under rate / population / caveat. Using the has_* names here reads as a KeyError,
#: which is the good outcome -- the alternative is a silently skipped ingredient.
INGREDIENTS = {
    "population_rate": "population",
    "quantification_rate": "rate",
    "caveat_rate": "caveat",
    "benefit_rate": "benefit",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--arms", default="sparse,context,pagectx")
    ap.add_argument("--categories", default="diseases,drugs-and-therapeutics",
                    help="comma-separated; empty string means every category")
    ap.add_argument("--n-queries", type=int, default=120)
    ap.add_argument("--query-offset", type=int, default=0,
                    help="start this far into the shuffled query pool; use it to take a "
                         "slice disjoint from a previous run's")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent generator requests. **Use 1 for any measurement "
                         "that compares arms.** llama-server runs -np 4 slots over a "
                         "16,384-token context, so with more than one request in "
                         "flight which requests share a batch depends on timing, and "
                         "the same prompt answered in a different batch is a different "
                         "computation -- batched matmuls are not associative, so "
                         "float rounding moves and a near-tie argmax can flip. "
                         "Measured: temperature is 0.0 and one prompt sent six times "
                         "serially returns 1/6 distinct outputs, but 30 fixed prompts "
                         "through 4 workers give a caveat rate of 0.0667/0.1333/0.1000/"
                         "0.1333 across four identical passes. Serial fixes the batch "
                         "shape and is exactly reproducible; 4-way costs ~4x wall-clock "
                         "and is not worth it for a comparison")
    ap.add_argument("--repeats", type=int, default=0,
                    help="regenerate the baseline arm this many times and report the "
                         "spread, so every answer-side number carries its own error bar")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.workers > 1 and (args.repeats > 1 or len(args.arms.split(",")) > 2):
        print("  NOTE: --workers > 1 with an arm comparison. Batch composition varies "
              "with timing, so arm-to-arm differences at this setting are confounded "
              "with a repeat-to-repeat spread that --repeats will print below. Compare "
              "against that spread, or re-run with --workers 1.", flush=True)

    # An arm may carry a budget override: "context@10x12" runs the context arm with
    # context_seeds=10 and context_per_seed=12 instead of the configured 5x2.
    #
    # Needed to compare budgets in ONE run rather than two. The budget is a fusion
    # setting read when the retriever is constructed, so comparing it through the mode
    # dispatch alone would mean loading the 16,133,551-row index twice and paying
    # 2,168 s over NFS for each. Expanding from the same sparse pool with different
    # budgets keeps the comparison paired on retrieval as well as on queries.
    arms: list[tuple[str, tuple[int, int] | None]] = []
    for a in (x.strip() for x in args.arms.split(",")):
        if not a:
            continue
        if "@" in a:
            mode, _, budget = a.partition("@")
            seeds, _, per = budget.partition("x")
            arms.append((mode.strip(), (int(seeds), int(per))))
        else:
            arms.append((a, None))
    arm_names = [f"{m}@{b[0]}x{b[1]}" if b else m for m, b in arms]
    cats = {c.strip() for c in args.categories.split(",") if c.strip()}

    cfg = PipelineConfig.load(args.config)._finalize()
    print(f"  index dir: {cfg.paths.index_dir}", flush=True)
    # with_dense=False: the 16,133,551-row chunk matrix costs 2,168 s to load over NFS
    # and no arm here uses it. The page index still loads, and the page arms still get
    # their query encoder, because load_index loads the embedder for either.
    r = load_index(cfg, with_reranker=False, with_graph=False, with_dependency=True,
                   with_dense=False)
    print(f"  store {len(r.store):,} chunks | dependency "
          f"{'loaded' if r.dependency else 'MISSING'} | page "
          f"{'loaded' if r.page is not None else 'MISSING'}", flush=True)
    if r.dependency is None:
        print("  dependency graph missing; run `medrag dependency-graph` first")
        return 1

    bm = Qrels.load(args.qrels)
    qs = [q for q in bm.queries if q.relevant_docs]
    if cats:
        qs = [q for q in qs if q.category in cats]
    random.Random(7).shuffle(qs)
    # --query-offset takes a slice of the shuffled pool rather than always its head, so a
    # replication can run on queries the original run never saw.
    #
    # This matters because the qrels pool is small: with gold in the two full-text
    # categories there are only 240 queries, so an 80-query run uses a third of them and
    # a second 80 can be drawn disjointly. Re-running the same 80 would measure the
    # harness twice and call it a replication.
    lo = args.query_offset
    qs = qs[lo: lo + args.n_queries]
    print(f"  {len(qs)} queries (offset {lo} of the shuffled pool), "
          f"categories {sorted(cats) if cats else 'all'}, arms {arm_names}", flush=True)
    if not qs:
        print("  no queries matched; nothing to run")
        return 1

    inp: dict[str, list] = {n: [] for n in arm_names}
    rows_by_arm: dict[str, dict[str, list[int]]] = {n: {} for n in arm_names}
    jobs: list[tuple[str, str, str]] = []
    t0 = time.time()
    for i, q in enumerate(qs):
        # The sparse pool is computed once and shared, so every arm -- including the
        # budget-overridden ones -- starts from identical candidates and the comparison
        # is paired on retrieval as well as on queries.
        pool = [(int(x), float(sc)) for x, sc in r._sparse_search(q.text, 50)]
        for (mode, budget), name in zip(arms, arm_names):
            if budget is None:
                res = r.retrieve(q.text, top_k=args.top_k, mode=mode, include_text=True)
                rows = [int(p.row_id) for p in res.passages]
                texts = [(int(p.row_id), p.text) for p in res.passages]
            else:
                # Same sparse seeds, different expansion budget. Compared through the
                # mode dispatch the budget would come from cfg.fusion, fixed at
                # construction time, so comparing two budgets would mean loading the
                # 16,133,551-row index twice at 2,168 s each.
                seeds_n, per_n = budget
                rows = [int(m[0]) for m in expand_with_context(
                    pool[: args.top_k], r.store, r.dependency, top_k=args.top_k,
                    seed_count=seeds_n, context_per_seed=per_n,
                    context_budget=seeds_n * per_n)]
                texts = []
                for row in rows:
                    c = r.store.get(row)
                    texts.append((row, c.text if c is not None else ""))
            inp[name].append(
                readiness_from_rows(rows, r.store, q.relevant_docs, qid=q.qid))
            rows_by_arm[name][q.qid] = rows
            jobs.append((q.qid, name, build_prompt(q.text, texts)))
        if (i + 1) % 50 == 0:
            print(f"  prepared {i + 1}/{len(qs)} | {time.time() - t0:.0f}s", flush=True)
    print(f"  prepared {len(jobs)} generations in {time.time() - t0:.0f}s", flush=True)

    # An arm whose ranking is identical to the baseline's is inert, and an inert arm
    # silently reports the baseline's transmission as its own. Checked rather than
    # assumed: three separate bugs produced exactly this, and two of them were found
    # only because an earlier driver printed the count.
    print("\n  ARM SANITY: is each arm actually retrieving something different?", flush=True)
    inert: dict[str, int] = {}
    base = arm_names[0]
    for a in arm_names[1:]:
        same = sum(1 for q in qs
                   if rows_by_arm[a].get(q.qid) == rows_by_arm[base].get(q.qid))
        inert[a] = same
        print(f"    {a:16s} identical to {base} on {same}/{len(qs)} queries"
              f"{'   <-- INERT, transmission below would be the baseline\'s' if same == len(qs) else ''}")
    ans: dict[str, list] = {n: [] for n in arm_names}

    def run(job: tuple[str, str, str]):
        qid, mode, prompt = job
        return score_answer(generate(prompt), qid=qid, arm=mode)

    baseline_jobs = [j for j in jobs if j[1] == arm_names[0]]
    repeat_jobs = [(qid, prompt) for qid, arm, prompt in jobs if arm == arm_names[0]]

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        done = 0
        for v in ex.map(run, jobs):
            ans[v.arm].append(v)
            done += 1
            if done % 40 == 0:
                print(f"  {done}/{len(jobs)} | {time.time() - t0:.0f}s", flush=True)

    pd_map: dict[str, dict] = {}
    below_noise: list[dict] = []

    print("\n  INPUT: do the retrieved passages contain the ingredients?", flush=True)
    in_sum = {n: aggregate_readiness(inp[n]).summary() for n in arm_names}
    for a in arm_names:
        s = in_sum[a]
        print(f"    {a:9s} ready={s['readiness']:.4f} quant={s['quantification_rate']:.4f} "
              f"pop={s['population_rate']:.4f} caveat={s['caveat_rate']:.4f} "
              f"benefit={s.get('benefit_rate', 0.0):.4f} bare={s['bare_rate']:.4f}")

    print("\n  OUTPUT: does the generated ANSWER state them?", flush=True)
    reps = {a: aggregate_answers(ans[a]) for a in arm_names}
    for a in arm_names:
        print(f"    {a:9s} {json.dumps(reps[a].summary())}")

    # The cross-tab is the sharpest statement of the transmission result: of the
    # queries whose passages DID carry a caveat, how many produced an answer that
    # mentions one? Transmission divides an aggregate gain by an aggregate gain, which
    # mixes that count with queries where the passages never had a caveat to begin
    # with. Falsifier 11 reported the ratio and not this, because the run predated the
    # driver change; the two answer different questions and the cross-tab is the one
    # that a reader of the paper will ask for.
    print("\n  CROSS-TAB: of the queries whose PASSAGES contained a caveat, how many "
          "answers mention one?", flush=True)
    xtab: dict[str, dict[str, int]] = {}
    for a in arm_names:
        inp_by_q = {row.qid: row for row in inp[a]}
        ans_by_q = {v.qid: v for v in ans[a]}
        both = inp_have = 0
        for qid, row in inp_by_q.items():
            if not getattr(row, "has_caveat", False):
                continue
            inp_have += 1
            v = ans_by_q.get(qid)
            if v is not None and getattr(v, "has_caveat", False):
                both += 1
        xtab[a] = {"input_caveat_queries": inp_have,
                   "answer_caveat_queries": both,
                   "transmission_rate": round(both / inp_have, 4) if inp_have else None}
        shown = ("n/a (no query had a caveat in its passages)"
                 if not inp_have else f"{both}/{inp_have} = {both / inp_have:.2f}")
        print(f"    {a:16s} {shown}")

    out: dict[str, object] = {
        "n_queries": len(qs),
        "arms": arm_names,
        "baseline_arm": base,
        "input": in_sum,
        "output": {a: reps[a].summary() for a in arm_names},
        "caveat_cross_tab": xtab,
        "transmission": {},
        "paired": {},
        "identical_to_baseline": inert,
        "elapsed_s": round(time.time() - t0, 1),
    }

    for a in arm_names[1:]:
        if inert.get(a) == len(qs):
            print(f"\n  {a} returned the baseline ranking for every query, so its "
                  f"transmission is the baseline's by construction. Not measured.")
            out["paired"][a] = {"inert": True, "n_queries": len(qs)}
            out["transmission"][a] = {"note": "arm returned the baseline ranking on "
                                              "every query; nothing to transmit"}
            continue
        print(f"\n  PAIRED {base} -> {a} (same queries)", flush=True)
        pd = paired_delta(reps[base], reps[a])
        out["paired"][a] = pd
        pd_map[a] = pd
        print(f"    paired queries: {pd['n_paired']}")
        for k in ("has_rate", "has_population", "has_caveat", "ready"):
            v = pd[k]
            print(f"    {k:16s} {v['baseline']:.4f} -> {v['candidate']:.4f}  "
                  f"(+{v['improved_queries']}/-{v['regressed_queries']} queries, "
                  f"paired {v['paired_delta']:+.4f}) "
                  f"{'BETTER' if v['better'] else 'not better'}")

        # Transmission: output gain over input gain, per ingredient.
        tr: dict[str, dict] = {}
        print("    -- transmission (output gain / input gain) --")
        for ik, ok in INGREDIENTS.items():
            din = in_sum[a][ik] - in_sum[base][ik]
            dout = reps[a].summary()[ok] - reps[base].summary()[ok]
            if abs(din) < 1e-12:
                tr[ik] = {"input_gain": din, "output_gain": dout,
                          "ratio": None, "note": "no input gain; undefined"}
                print(f"    {ik:22s} input {din:+.4f}  output {dout:+.4f}  "
                      f"ratio n/a (no input gain to transmit)")
                continue
            ratio = dout / din
            tr[ik] = {"input_gain": round(din, 4), "output_gain": round(dout, 4),
                      "ratio": round(ratio, 4)}
            print(f"    {ik:22s} input {din:+.4f}  output {dout:+.4f}  "
                  f"transmission {ratio:.2f}")
        out["transmission"][a] = tr

    # ---- the harness's own noise floor ---------------------------------------
    #
    # Measured, not assumed, because Falsifier 17 found the floor is 4x the effect it
    # was being compared against. The cause is batch composition: the server shares 4
    # slots across requests of different lengths, so which requests share a batch
    # depends on timing, and the same prompt answered in a different batch is a
    # different computation. Deterministic serially, deterministic under concurrency
    # alone, nondeterministic when both meet a real query mix -- all three verified.
    #
    # With --repeats the baseline arm is regenerated N times from the same prompts and
    # the spread is printed next to the effects, so a run whose effects fall inside its
    # own noise says so rather than being read as a finding.
    noise: dict[str, dict] = {}
    if args.repeats > 1:
        rep: dict[str, list[float]] = {
            k: [] for k in ("has_rate", "has_population", "has_caveat", "has_benefit",
                            "ready")}
        for rk in range(args.repeats):
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                outs = list(ex.map(lambda t: generate(t[1]), repeat_jobs))
            vs = [score_answer(o, qid=qid) for (qid, _), o in zip(repeat_jobs, outs)]
            for k in rep:
                rep[k].append(sum(getattr(v, k) for v in vs) / max(len(vs), 1))
            print(f"  noise pass {rk}: " + "  ".join(
                f"{k}={rep[k][-1]:.4f}" for k in rep), flush=True)
        import statistics
        print(f"\n  HARNESS NOISE FLOOR -- same prompts, same retrieval, "
              f"{args.repeats} passes", flush=True)
        print(f"    {'ingredient':16s}{'mean':>9s}{'sd':>9s}{'range':>9s}", flush=True)
        for k, v in rep.items():
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            noise[k] = {"mean": round(statistics.fmean(v), 4), "sd": round(sd, 4),
                        "range": round(max(v) - min(v), 4)}
            print(f"    {k:16s}{statistics.fmean(v):>9.4f}{sd:>9.4f}"
                  f"{max(v) - min(v):>9.4f}", flush=True)
        for arm in arm_names[1:]:
            for label, key in (("has_caveat", "has_caveat"), ("ready", "ready")):
                if key not in noise or arm not in pd_map or label not in pd_map[arm]:
                    continue
                eff = abs(pd_map[arm][label]["paired_delta"])
                rng = noise[key]["range"]
                if eff < rng:
                    below_noise.append({"arm": arm, "ingredient": label,
                                        "effect": round(eff, 4),
                                        "noise_range": rng})
                    print(f"    !! {arm} {label}: effect {eff:.4f} is inside the noise "
                          f"range {rng:.4f} -- not resolvable", flush=True)
        if below_noise:
            print(f"  !! {len(below_noise)} reported effects fall inside the harness's "
                  f"own noise floor and are not evidence of anything", flush=True)
        else:
            print("  all reported effects exceed the measured noise floor", flush=True)

    # Assigned here rather than in the literal above: the floor is measured after the
    # arms are scored, and referencing it in the literal was a NameError waiting for
    # any run with --repeats 0 as well.
    out["harness_noise_floor"] = noise
    out["effects_below_noise"] = below_noise
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  elapsed {time.time() - t0:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())