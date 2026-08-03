#!/usr/bin/env python3
"""
Analysis for the SAFE x DPO grid.
==================================
The aggregate comparison is misleading here and this script avoids it by default.

With `none` and every enrichment arm both generating the final answer greedily,
any question that does not trigger produces a byte-identical prompt and therefore
an identical answer. Comparing means across the whole test set dilutes the effect
by however many questions were untriggered — at a 50% trigger rate, half the
observations are identical by construction, halving any real difference.

So the primary analysis is a PAIRED test on the triggered subset only:
  * McNemar on discordant PAV pairs (exact binomial, two-sided)
  * Sign test on cosine-similarity direction

Usage:
    python3 analyze.py
    python3 analyze.py --arm generic       # control arm instead of safe
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict


def load(path):
    with open(path) as f:
        return json.load(f)


def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial p-value."""
    if n == 0:
        return 1.0
    def pmf(i):
        return math.comb(n, i) * p ** i * (1 - p) ** (n - i)
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12))


def aggregate_table(results):
    hdr = (f"{'condition':<12}{'arm':<9}{'n':>4}{'PAV':>8}{'CITM':>8}"
           f"{'CITS':>8}{'sim':>8}{'trig':>7}{'enr':>7}")
    print(hdr)
    print("-" * len(hdr))
    for cond in ("sft", "simdpo", "pvdpo"):
        for arm in ("none", "generic", "random", "safe"):
            d = results.get((cond, arm))
            if not d:
                continue
            s = d["summary"]
            print(f"{cond:<12}{arm:<9}{s['n']:>4}{s['pav']:>7.1%}"
                  f"{s['citmatch']:>8.1%}{s['citsupport']:>8.1%}"
                  f"{s['avg_cosine_sim']:>8.3f}"
                  f"{s.get('trigger_rate', 0):>6.0%}"
                  f"{s.get('frac_actually_enriched', 0):>7.0%}")
        print()


def paired(results, cond, arm):
    base = results.get((cond, "none"))
    test = results.get((cond, arm))
    if not base or not test:
        return None

    b = {r["question"]: r for r in base["results"]}
    rows = []
    for r in test["results"]:
        o = b.get(r["question"])
        if o is None:
            continue
        rows.append({
            "triggered": bool(r.get("triggered")) or bool(r.get("features_added")),
            "changed": o["pred_steered"] != r["pred_steered"],
            "pav_none": bool(o["pav"]),
            "pav_arm": bool(r["pav"]),
            "sim_none": o["sim_steered"],
            "sim_arm": r["sim_steered"],
        })

    trig = [r for r in rows if r["triggered"]]
    if not trig:
        return {"cond": cond, "arm": arm, "n_all": len(rows), "n_trig": 0}

    # Sanity: enrichment must actually alter the answer, or the arm is inert.
    changed_untrig = sum(1 for r in rows if not r["triggered"] and r["changed"])
    changed_trig = sum(1 for r in trig if r["changed"])

    # McNemar: discordant pairs only.
    b01 = sum(1 for r in trig if not r["pav_none"] and r["pav_arm"])   # fixed
    b10 = sum(1 for r in trig if r["pav_none"] and not r["pav_arm"])   # broken
    disc = b01 + b10
    p_mcnemar = binom_two_sided(b01, disc) if disc else 1.0

    # Sign test on similarity direction.
    up = sum(1 for r in trig if r["sim_arm"] > r["sim_none"] + 1e-9)
    dn = sum(1 for r in trig if r["sim_arm"] < r["sim_none"] - 1e-9)
    p_sign = binom_two_sided(up, up + dn) if (up + dn) else 1.0

    return {
        "cond": cond, "arm": arm, "n_all": len(rows), "n_trig": len(trig),
        "changed_trig": changed_trig, "changed_untrig": changed_untrig,
        "pav_none_trig": sum(r["pav_none"] for r in trig) / len(trig),
        "pav_arm_trig": sum(r["pav_arm"] for r in trig) / len(trig),
        "fixed": b01, "broken": b10, "p_mcnemar": p_mcnemar,
        "sim_up": up, "sim_down": dn, "p_sign": p_sign,
        "d_sim": (sum(r["sim_arm"] - r["sim_none"] for r in trig) / len(trig)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="safe")
    ap.add_argument("--dir", default="results")
    a = ap.parse_args()

    results = {}
    for path in glob.glob(os.path.join(a.dir, "eval_*.json")):
        stem = os.path.basename(path)[5:-5]
        if stem.endswith("_phi09"):
            continue
        for cond in ("sft", "simdpo", "pvdpo"):
            if stem.startswith(cond + "_"):
                results[(cond, stem[len(cond) + 1:])] = load(path)
                break

    if not results:
        print(f"No eval_*.json in {a.dir}/")
        return

    print("\n=== AGGREGATE (reference only — see caveat below) ===\n")
    aggregate_table(results)

    print(f"=== PAIRED, TRIGGERED SUBSET ONLY: none vs {a.arm} ===\n")
    for cond in ("sft", "simdpo", "pvdpo"):
        r = paired(results, cond, a.arm)
        if r is None:
            continue
        print(f"--- {cond}")
        if r["n_trig"] == 0:
            print("    no triggered questions — arm is inert here\n")
            continue
        print(f"    paired questions      : {r['n_all']}")
        print(f"    triggered             : {r['n_trig']}")
        print(f"    answer changed        : {r['changed_trig']}/{r['n_trig']} triggered, "
              f"{r['changed_untrig']} untriggered")
        if r["changed_untrig"]:
            print("      ^ untriggered answers should be identical; nonzero means "
                  "nondeterminism leaked in")
        print(f"    PAV on triggered      : {r['pav_none_trig']:.1%} -> {r['pav_arm_trig']:.1%}")
        print(f"    McNemar               : {r['fixed']} fixed, {r['broken']} broken, "
              f"p = {r['p_mcnemar']:.3f}")
        print(f"    cosine direction      : {r['sim_up']} up, {r['sim_down']} down, "
              f"p = {r['p_sign']:.3f}  (mean d = {r['d_sim']:+.3f})")
        if r["n_trig"] < 15:
            print(f"    NOTE n={r['n_trig']} is underpowered; treat as descriptive")
        print()

    print("""CAVEAT ON THE AGGREGATE TABLE
Final answers are greedy in every arm, so untriggered questions yield identical
prompts and identical answers. Aggregate means therefore understate any effect in
proportion to the untriggered fraction. Report the paired triggered-subset
analysis as primary; the aggregate is context, not evidence.

Read 'enr' first on every SAFE row. If it is near zero the arm did nothing and
its PAV merely restates the baseline.""")


if __name__ == "__main__":
    main()
