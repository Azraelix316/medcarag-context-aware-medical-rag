"""Answer-side read test over stored ablation rows: direct RAG vs context-aware.

run_read_test.py retrieves live, which needs a monolithic bm25.pkl; PMC-4M has 37
BM25 segments and is served from precomputed runs instead. This driver therefore takes
each arm's rows from run_mode_ablation.py's stored rows (the exact lists the ablation
scored), so the answer-side arms are the input-side arms, not a re-retrieval.

Arms:

``sparse``           direct RAG, BM25 top 10.
``pagectx``          document conditioning plus same-document expansion.
``randsec@pagectx``  pagectx's list with every appended passage swapped for a random
                     other chunk of the same document (matched_controls.py's control).

Generation is serial (one request in flight) because batch composition changes
llama-server's output; see run_read_test.py --workers. Answers are scored with the
same deterministic detectors as the passages (medrag.eval.read_test.score_answer),
not by an LLM judge. Those detectors credit any rate, including one copied from an
unrelated study, so two gold-grounded flags are added: ``grounded_rate`` (a rate whose
number also appears in a gold-article passage as prompted) and ``grounded_ready``
(that, plus a population). Every answer is appended to --answers as it arrives, and a
restart skips (qid, arm) pairs already there.

    python scripts/run_read_test_rows.py --config ... --qrels ... --rows-from ... \\
        --n-queries 300 --answers answers.jsonl --out read_test.json
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.config import PipelineConfig  # noqa: E402
from medrag.eval.read_test import (  # noqa: E402
    DEFAULT_ENDPOINT, AnswerVerdict, aggregate_answers, build_prompt, generate, score_answer)
from medrag.eval.readiness import aggregate_readiness, readiness_from_rows  # noqa: E402
from medrag.index.retriever import load_index  # noqa: E402
from medrag.index.store import ChunkStore  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_rma, _mc, _pb = _load("run_mode_ablation"), _load("matched_controls"), _load("paired_bootstrap")

ARMS = ("sparse", "pagectx", "randsec@pagectx")
PAIRS = (("pagectx", "sparse"), ("randsec@pagectx", "sparse"), ("pagectx", "randsec@pagectx"))
ANSWER_FLAGS = ("ready", "has_rate", "has_population", "has_caveat", "has_benefit",
                "grounded_rate", "grounded_ready")

# A number worth grounding: a percentage, a decimal, or an integer >= 10 that is not a
# year. Bare small integers ("two groups", "[3]") would match almost any passage.
_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(%|percent)?")
# Numbers nearly every paper states, which would ground anything: a CI level, and the
# usual significance thresholds.
_CI_LEVEL = re.compile(r"\s*(?:%|percent)?\s*(?:CI|confidence)", re.I)
_THRESHOLDS = {"0.05", "0.01", "0.001"}


def salient_numbers(text: str) -> set[str]:
    text = text or ""
    out = set()
    for m in _NUM.finditer(text):
        s = m.group(1).replace(",", "")
        v = float(s)
        if s in _THRESHOLDS or (v in (90, 95, 99) and _CI_LEVEL.match(text, m.end(1))):
            continue
        if m.group(2) or "." in s or (v >= 10 and not 1900 <= v <= 2030):
            out.add(f"{v:g}")
    return out


def prompt_text(text: str, n: int, prompt_chars: int) -> str:
    """A passage exactly as build_prompt hands it to the generator."""
    return " ".join((text or "").split())[: max(400, prompt_chars // max(1, n))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--rows-from", metavar="JSON,...",
                    help="run_mode_ablation.py's stored rows; randsec rows are drawn here")
    ap.add_argument("--rows-json", metavar="JSON",
                    help="{arm: {qid: rows}} from an earlier run, used as is instead")
    ap.add_argument("--n-queries", type=int, default=300,
                    help="the first N of the ablation's 2,000-query sample (seed 7)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--answers", required=True, help="this worker's JSONL, appended to")
    ap.add_argument("--done-glob", default=None,
                    help="answer files of every worker, read to skip finished answers and "
                         "to score them all (default: --answers alone)")
    ap.add_argument("--claim-dir", default=None,
                    help="shared directory: a worker generates a query only after creating "
                         "<claim-dir>/<n> exclusively, so several servers can share one "
                         "run. All three arms of a query stay on the server that claimed "
                         "it, so pairs never span hardware")
    ap.add_argument("--prompt-chars", type=int, default=5500,
                    help="build_prompt's total budget; each passage gets "
                         "max(400, budget // n) characters. 5500 is build_prompt's default "
                         "(the first run); a large value hands over whole passages")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)._finalize()
    qs = _rma._sample_queries(args.qrels, 2000)[: args.n_queries]
    if args.rows_json:
        # Every arm's rows fixed by an earlier run: only the chunk store is needed, not
        # the 9.5 GB dependency graph, so a worker fits beside a large generator.
        store = ChunkStore.load(cfg.paths.index_dir / "store")
        with open(args.rows_json) as fh:
            rows = json.load(fh)
    else:
        r = load_index(cfg, with_reranker=False, with_graph=False, with_dependency=True,
                       with_dense=False)
        store = r.store
        ranges = _mc.DocRanges(r.dependency.doc_start, r.dependency.doc_end)
        stored: dict[str, dict[str, list[int]]] = {}
        for path in args.rows_from.split(","):
            with open(path) as fh:
                for mode, per_q in json.load(fh).items():
                    stored.setdefault(mode, {}).update(per_q)
        rng = random.Random(args.seed)
        rows = {a: {} for a in ARMS}
        for q in qs:
            rows["sparse"][q.qid] = stored["sparse"][q.qid]
            rows["pagectx"][q.qid] = stored["pagectx"][q.qid]
            rows["randsec@pagectx"][q.qid] = _mc.randomise_sections(
                stored["pagectx"][q.qid], args.top_k, ranges, rng)[0]
    inp = {a: [readiness_from_rows(rows[a][q.qid], store, q.relevant_docs, qid=q.qid)
               for q in qs] for a in ARMS}

    done: dict[tuple[str, str], dict] = {}
    files = sorted(glob.glob(args.done_glob)) if args.done_glob else [args.answers]
    for path in files:
        if Path(path).exists():
            with open(path) as fh:
                for line in fh:
                    d = json.loads(line)
                    done[(d["qid"], d["arm"])] = d
    if args.claim_dir:
        # Workers run as different users on a shared volume, so the directory must be
        # writable by all of them, whichever creates it.
        os.makedirs(args.claim_dir, exist_ok=True)
        try:
            os.chmod(args.claim_dir, 0o777)
        except PermissionError:
            pass
    t0 = time.time()
    n_new = 0
    with open(args.answers, "a") as fh:
        for i, q in enumerate(qs):
            if all((q.qid, a) in done for a in ARMS):
                continue
            if args.claim_dir:
                try:
                    os.close(os.open(os.path.join(args.claim_dir, str(i)),
                                     os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                except FileExistsError:
                    continue
            for a in ARMS:
                if (q.qid, a) in done:
                    continue
                texts = [(row, (c.text if (c := store.get(row)) is not None else ""))
                         for row in rows[a][q.qid]]
                prompt = build_prompt(q.text, texts, max_chars=args.prompt_chars)
                # generate() returns '' on any failure. A crashed server must not be
                # recorded as an empty answer (it scores zero on every flag), so retry
                # while a supervisor restarts it, and stop the worker rather than write.
                for attempt in range(20):
                    text = generate(prompt, endpoint=args.endpoint)
                    if text.strip():
                        break
                    time.sleep(30)
                else:
                    print(f"  {q.qid} {a}: no answer after 20 attempts; stopping",
                          flush=True)
                    return 1
                d = {"qid": q.qid, "arm": a, "question": q.text, "rows": rows[a][q.qid],
                     "prompt_chars": args.prompt_chars, "answer": text}
                fh.write(json.dumps(d) + "\n")
                fh.flush()
                done[(q.qid, a)] = d
                n_new += 1
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(qs)} queries | {n_new} new answers | "
                      f"{time.time() - t0:.0f}s", flush=True)
    missing = [(q.qid, a) for q in qs for a in ARMS if (q.qid, a) not in done]
    if missing:
        print(f"  worker done; {len(missing)} answers are another worker's or unfinished, "
              f"so nothing is scored here", flush=True)
        return 0

    ans: dict[str, list[AnswerVerdict]] = {
        a: [score_answer(done[(q.qid, a)]["answer"], qid=q.qid, arm=a) for q in qs]
        for a in ARMS}
    # Gold grounding: a stated rate counts only if one of the answer's salient numbers
    # appears in a gold-article passage as the generator saw it. The plain detectors
    # credit a rate copied from an unrelated study.
    for a in ARMS:
        for q, v in zip(qs, ans[a]):
            rr = rows[a][q.qid]
            gold_nums: set[str] = set()
            for row in rr:
                c = store.get(row)
                if c is not None and c.doc_id in q.relevant_docs:
                    gold_nums |= salient_numbers(prompt_text(c.text, len(rr), args.prompt_chars))
            v.grounded_rate = v.has_rate and bool(salient_numbers(v.text) & gold_nums)
            v.grounded_ready = v.grounded_rate and v.has_population
    reps = {a: aggregate_answers(ans[a]) for a in ARMS}
    np_rng = np.random.default_rng(args.seed)
    paired: dict[str, dict] = {}
    for arm, base in PAIRS:
        key = f"{arm} vs {base}"
        paired[key] = {}
        for f in ANSWER_FLAGS:
            x = np.array([getattr(v, f) for v in ans[arm]], dtype=bool)
            y = np.array([getattr(v, f) for v in ans[base]], dtype=bool)
            paired[key][f] = _pb.compare(y, x, 10_000, np_rng)
    flat = [(k, f) for k in paired for f in ANSWER_FLAGS]
    for (k, f), p in zip(flat, _pb.holm([paired[k][f]["p_mcnemar"] for k, f in flat])):
        paired[k][f]["p_holm"] = p
    xtab = {}
    for a in ARMS:
        have = [j for j, x in enumerate(inp[a]) if x.has_caveat]
        both = sum(1 for j in have if ans[a][j].has_caveat)
        xtab[a] = {"input_caveat_queries": len(have), "answer_caveat_queries": both}
    output = {a: reps[a].summary() for a in ARMS}
    for a in ARMS:
        for f in ("grounded_rate", "grounded_ready"):
            output[a][f] = round(sum(getattr(v, f) for v in ans[a]) / len(qs), 4)
    out = {"n_queries": len(qs), "arms": list(ARMS), "seed": args.seed,
           "generator_endpoint": args.endpoint, "workers": 1,
           "prompt_chars": args.prompt_chars,
           "input": {a: aggregate_readiness(inp[a]).summary() for a in ARMS},
           "output": output,
           "paired": paired, "caveat_cross_tab": xtab,
           "elapsed_s": round(time.time() - t0, 1)}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    for a in ARMS:
        print(f"  {a:16s} input ready {out['input'][a]['readiness']:.4f} | "
              f"answer {json.dumps(out['output'][a])}")
    for k, v in paired.items():
        print(f"  {k:30s} answer ready {v['ready']['delta']:+.4f} {v['ready']['ci95']} "
              f"p_holm {v['ready']['p_holm']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
