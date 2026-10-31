"""scripts/paired_bootstrap.py: exact McNemar, Holm, and the paired deltas."""

from __future__ import annotations

import importlib.util
import random
import sys
from math import comb

sys.path.insert(0, ".")

spec = importlib.util.spec_from_file_location("paired_bootstrap", "scripts/paired_bootstrap.py")
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)


def _q(qid: str, **flags: bool) -> dict:
    base = {"qid": qid, "found_gold": False, "has_finding": False, "has_rate": False,
            "has_population": False, "has_caveat": False, "has_benefit": False,
            "quantified": False, "ready": False}
    base.update(flags)
    return base


def test_mcnemar_exact() -> None:
    assert pb.mcnemar_exact(0, 0) == 1.0
    assert pb.mcnemar_exact(5, 5) == 1.0
    # b=0, c=6: both tails of binomial(6, 1/2) at the extremes = 2/64.
    assert abs(pb.mcnemar_exact(0, 6) - 2 / 64) < 1e-15
    assert abs(pb.mcnemar_exact(2, 8) - 2 * sum(comb(10, k) for k in range(3)) / 1024) < 1e-15
    assert pb.mcnemar_exact(3, 1) == pb.mcnemar_exact(1, 3)
    print("  exact McNemar matches the binomial tails")


def test_holm() -> None:
    assert pb.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert pb.holm([0.5, 0.9]) == [1.0, 1.0]
    print("  Holm step-down is monotone and capped at 1")


def test_analyse_deltas_and_interval() -> None:
    rng = random.Random(3)
    n = 400
    base, arm = [], []
    for i in range(n):
        r0, r1 = rng.random() < 0.2, rng.random() < 0.4
        base.append(_q(f"q{i}", found_gold=True, ready=r0, quantified=r0, has_caveat=r0))
        arm.append(_q(f"q{i}", found_gold=True, ready=r1, quantified=r1, has_caveat=False))
    res = pb.analyse({"sparse": base, "ctx": arm}, n_boot=2000, seed=1)
    r = res["arms"]["ctx"]["readiness"]
    want = sum(q["ready"] for q in arm) / n - sum(q["ready"] for q in base) / n
    assert abs(r["delta"] - round(want, 4)) < 1e-9
    assert r["ci95"][0] < r["delta"] < r["ci95"][1]
    assert r["p_holm"] < 1e-6
    bare = res["arms"]["ctx"]["bare_rate"]
    assert abs(bare["delta"] + r["delta"]) < 1e-9          # found everywhere: bare = 1 - quantified
    assert res["n_comparisons"] == len(pb.METRICS)
    try:
        pb.analyse({"sparse": base, "ctx": arm[::-1]})
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned queries were accepted")
    print("  paired deltas, bootstrap interval, and query alignment check")


def test_pairs_compare_named_bases() -> None:
    a = [_q(f"q{i}", ready=i % 2 == 0) for i in range(10)]
    b = [_q(f"q{i}", ready=False) for i in range(10)]
    res = pb.analyse({"pagectx": a, "sparse@pagectx": b, "sparse": b},
                     pairs=[("pagectx", "sparse@pagectx")], n_boot=200)
    assert list(res["arms"]) == ["pagectx vs sparse@pagectx"]
    r = res["arms"]["pagectx vs sparse@pagectx"]["readiness"]
    assert r["delta"] == 0.5 and r["arm_only"] == 5 and r["base_only"] == 0
    print("  --pairs compares each arm with its own control")


def test_randomise_sections() -> None:
    spec = importlib.util.spec_from_file_location("matched_controls", "scripts/matched_controls.py")
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    import numpy as np
    # documents (unsorted by start, as DependencyGraph stores them by doc id):
    # rows [20,30), [0,5), [5,20), [30,31)
    ranges = mc.DocRanges(np.array([20, 0, 5, 30]), np.array([30, 5, 20, 31]))
    assert ranges.of(0) == (0, 5) and ranges.of(19) == (5, 20) and ranges.of(30) == (30, 31)
    rows = [1, 6, 22, 7, 8, 21, 30]                 # top_k 3, then 4 appended
    rng = random.Random(0)
    out, kept = mc.randomise_sections(rows, 3, ranges, rng)
    assert out[:3] == rows[:3] and len(out) == len(rows) and len(set(out)) == len(out)
    for old, new in zip(rows[3:], out[3:]):
        assert ranges.of(old) == ranges.of(new)     # same document
    assert out[-1] == 30 and kept == 1              # one-row document: nothing to swap
    assert all(new not in rows for new in out[3:6])
    print("  randomised sections stay in the same document, no duplicates")


def main() -> int:
    tests = [test_mcnemar_exact, test_holm, test_analyse_deltas_and_interval,
             test_pairs_compare_named_bases, test_randomise_sections]
    print("paired bootstrap significance")
    for t in tests:
        t()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
