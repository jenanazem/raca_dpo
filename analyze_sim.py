#!/usr/bin/env python3
"""
Paired cosine-similarity analysis for the SAFE arms.
=====================================================
Cosine similarity is continuous, so unlike the binary PAV/CITMATCH metrics it
supports a properly powered paired test on 30 examples rather than McNemar on a
handful of discordant pairs.

The test is restricted to TRIGGERED questions. Final answers are greedy in every
arm, so untriggered questions receive an identical prompt and produce an
identical answer with delta exactly 0 — including them shrinks the mean toward
zero and inflates the apparent sample size with non-informative pairs.

Reports per condition and pooled:
  mean delta, 95% CI, Wilcoxon signed-rank, paired t, sign test, Cohen's dz

Usage:
    python3 analyze_sim.py                       # closed-book
    python3 analyze_sim.py --dir results         # round 1, with RAG
    python3 analyze_sim.py --arms safe generic random
"""

import argparse
import glob
import json
import math
import os

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def sign_test(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    pmf = lambda i: math.comb(n, i) * 0.5 ** n
    obs = pmf(pos)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12))


def load(path):
    with open(path) as f:
        return json.load(f)["results"]


def deltas(base_rows, arm_rows):
    """Paired (sim_none, sim_arm) for triggered questions only."""
    b = {r["question"]: r for r in base_rows}
    out = []
    for r in arm_rows:
        o = b.get(r["question"])
        if o is None:
            continue
        triggered = bool(r.get("triggered")) or bool(r.get("features_added"))
        if not triggered:
            continue
        out.append((o["sim_steered"], r["sim_steered"]))
    return out


def describe(pairs, label):
    n = len(pairs)
    if n == 0:
        print(f"  {label:<10} no triggered questions")
        return None
    d = [a - b for b, a in pairs]          # arm minus baseline
    mean = sum(d) / n
    if n > 1:
        sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
        se = sd / math.sqrt(n)
    else:
        sd = se = 0.0
    tcrit = 2.06 if n >= 25 else (2.20 if n >= 12 else 2.57)
    lo, hi = mean - tcrit * se, mean + tcrit * se
    pos = sum(1 for x in d if x > 1e-9)
    neg = sum(1 for x in d if x < -1e-9)
    dz = mean / sd if sd > 0 else 0.0

    p_w = p_t = float("nan")
    if HAVE_SCIPY and n >= 6:
        try:
            p_w = stats.wilcoxon([a for _, a in pairs], [b for b, _ in pairs]).pvalue
        except Exception:
            pass
        try:
            p_t = stats.ttest_rel([a for _, a in pairs], [b for b, _ in pairs]).pvalue
        except Exception:
            pass
    p_s = sign_test(pos, neg)

    base_mean = sum(b for b, _ in pairs) / n
    arm_mean = sum(a for _, a in pairs) / n
    print(f"  {label:<10} n={n:<3} {base_mean:.3f} -> {arm_mean:.3f}   "
          f"d={mean:+.4f} [{lo:+.4f},{hi:+.4f}]  "
          f"{pos}up/{neg}dn  dz={dz:+.2f}")
    print(f"  {'':<10} wilcoxon p={p_w:.4f}  paired-t p={p_t:.4f}  sign p={p_s:.4f}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_cb")
    ap.add_argument("--arms", nargs="*", default=["safe", "generic", "random"])
    ap.add_argument("--conds", nargs="*", default=["sft", "simdpo", "pvdpo"])
    a = ap.parse_args()

    if not HAVE_SCIPY:
        print("scipy not found — sign test only (pip install scipy)\n")

    print(f"PAIRED COSINE SIMILARITY, TRIGGERED QUESTIONS ONLY  [{a.dir}]")
    print("negative d means the arm moved answers AWAY from the gold text\n")

    pooled = {arm: [] for arm in a.arms}
    for cond in a.conds:
        bp = os.path.join(a.dir, f"eval_{cond}_none.json")
        if not os.path.exists(bp):
            print(f"{cond}: no baseline\n")
            continue
        base = load(bp)
        print(f"--- {cond}")
        for arm in a.arms:
            p = os.path.join(a.dir, f"eval_{cond}_{arm}.json")
            if not os.path.exists(p):
                continue
            d = describe(deltas(base, load(p)), arm)
            if d:
                pooled[arm].extend(d)
        print()

    print("=== POOLED ACROSS CONDITIONS ===")
    for arm, d in pooled.items():
        n = len(d)
        if n == 0:
            continue
        mean = sum(d) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0
        se = sd / math.sqrt(n) if n else 0
        pos = sum(1 for x in d if x > 1e-9); neg = sum(1 for x in d if x < -1e-9)
        p_w = float("nan")
        if HAVE_SCIPY and n >= 6:
            try:
                p_w = stats.wilcoxon(d).pvalue
            except Exception:
                pass
        print(f"  {arm:<10} n={n:<3} d={mean:+.4f} "
              f"[{mean-1.96*se:+.4f},{mean+1.96*se:+.4f}]  "
              f"{pos}up/{neg}dn  wilcoxon p={p_w:.4f}  sign p={sign_test(pos,neg):.4f}")

    print("""
INTERPRETATION
  d > 0 with a CI excluding 0  -> the arm genuinely moves answers toward gold
  d ~ 0                        -> no effect
  d < 0                        -> the arm degrades similarity

Compare arms against each other, not just against zero: if `generic` matches or
beats `safe`, the gain comes from appending an instruction rather than from
SAE-guided feature selection.""")


if __name__ == "__main__":
    main()
