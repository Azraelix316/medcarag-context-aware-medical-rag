"""Paired significance for the mode ablation: every arm against ``sparse``.

Reads ``run_mode_ablation.py --per-query-out`` (each arm's per-query readiness flags,
same queries in the same order) and reports, per arm and metric, the difference in
rate, its 95% paired-bootstrap interval, and an exact McNemar test on the discordant
queries. Holm's correction is applied over every (arm, metric) comparison reported.

    python scripts/paired_bootstrap.py per_query.json --out significance.json
"""

from __future__ import annotations

import argparse
import json
from math import comb

import numpy as np

#: metric -> the per-query flag it is the mean of (bare: gold found, no rate).
METRICS = {
    "readiness": lambda q: q["ready"],
    "quantification_rate": lambda q: q["quantified"],
    "population_rate": lambda q: q["has_population"],
    "caveat_rate": lambda q: q["has_caveat"],
    "benefit_rate": lambda q: q["has_benefit"],
    "bare_rate": lambda q: q["found_gold"] and not q["quantified"],
}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: binomial(b + c, 1/2) on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        out[i] = running
    return out


def compare(base: np.ndarray, arm: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    n = len(base)
    diff = arm.astype(np.int8) - base.astype(np.int8)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    b = int(np.sum((arm == 1) & (base == 0)))
    c = int(np.sum((arm == 0) & (base == 1)))
    return {"base": round(float(base.mean()), 4), "arm": round(float(arm.mean()), 4),
            "delta": round(float(diff.mean()), 4),
            "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                     round(float(np.percentile(boots, 97.5)), 4)],
            "arm_only": b, "base_only": c, "p_mcnemar": mcnemar_exact(b, c)}


def analyse(per_query: dict[str, list[dict]], base_arm: str = "sparse",
            n_boot: int = 10_000, seed: int = 7,
            pairs: list[tuple[str, str]] | None = None) -> dict:
    """Every arm against ``base_arm``, or each ``(arm, base)`` in ``pairs``.

    Results are keyed by the arm, or by ``"arm vs base"`` when ``pairs`` is given.
    """
    if pairs is None:
        pairs = [(a, base_arm) for a in per_query if a != base_arm]
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    qids = [q["qid"] for q in per_query[pairs[0][1]]] if pairs else []
    for arm, base_name in pairs:
        rows, base_q = per_query[arm], per_query[base_name]
        if [q["qid"] for q in rows] != [q["qid"] for q in base_q]:
            raise ValueError(f"{arm}: queries differ from {base_name}'s")
        key = arm if base_name == base_arm else f"{arm} vs {base_name}"
        out[key] = {}
        for m, flag in METRICS.items():
            base = np.array([bool(flag(q)) for q in base_q], dtype=np.int8)
            got = np.array([bool(flag(q)) for q in rows], dtype=np.int8)
            out[key][m] = compare(base, got, n_boot, rng)
    keys = [(a, m) for a in out for m in out[a]]
    for (a, m), p in zip(keys, holm([out[a][m]["p_mcnemar"] for a, m in keys])):
        out[a][m]["p_holm"] = p
    return {"base": base_arm, "n_queries": len(qids), "n_boot": n_boot, "seed": seed,
            "n_comparisons": len(keys), "arms": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("per_query")
    ap.add_argument("--base", default="sparse")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pairs", metavar="ARM:BASE,...",
                    help="compare these pairs instead of every arm against --base")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.per_query) as fh:
        pairs = ([tuple(p.split(":", 1)) for p in args.pairs.split(",")]
                 if args.pairs else None)
        res = analyse(json.load(fh), args.base, args.n_boot, args.seed, pairs)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"  {res['n_queries']} queries, {res['n_comparisons']} comparisons")
    for arm, ms in res["arms"].items():
        for m, r in ms.items():
            print(f"    {arm:24s} {m:20s} {r['delta']:+.4f} [{r['ci95'][0]:+.4f}, "
                  f"{r['ci95'][1]:+.4f}]  p_holm={r['p_holm']:.2g}")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
