#!/usr/bin/env python3
"""
Audit Data Generator
=====================
Reads pre-generated pool datasets (K=3 answers per question) and computes
A1-A3 audit metrics from the paper.

A1: Covariance profile — correlation of proxy with surface vs substance features
A2: Signal decomposition — partial R² of surface vs substance in chosen-rejected gap
A3: Substance inversion rate — fraction of pairs where chosen is PAV-worse than rejected

Usage:
    python generate_audit_data.py --model llama
    python generate_audit_data.py --model qwen
    python generate_audit_data.py --model fanar
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats

from pav import PAV, extract_citations, normalize_arabic, template_ngram_rate


def surface_features(answer: str, template_ngrams: set) -> dict:
    tokens = normalize_arabic(answer).split()
    length = len(tokens)
    citations = len(extract_citations(answer))
    tmpl_rate = template_ngram_rate(answer, template_ngrams)
    return {
        "length": length,
        "citation_count": citations,
        "template_rate": tmpl_rate,
    }


def run_audit(model_name: str, train_file: str):
    pool_file = f"data/processed/{model_name}_pool_dataset.jsonl"

    print(f"Loading PAV verifier...")
    pav = PAV(train_file=train_file, tau=0.3)

    print(f"Loading pool data from {pool_file}...")
    pool_data = []
    with open(pool_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if "pool" not in ex:
                continue
            provenance = ex.get("provenance_text", "")
            question = ex["prompt"]
            pool_entries = []
            for item in ex["pool"]:
                ans = item["answer"]
                cos = item["score"]
                pav_result = pav.verify(question, ans, provenance)
                surf = surface_features(ans, pav.template_ngrams)
                pool_entries.append({
                    "answer":         ans,
                    "cosine":         cos,
                    "pav":            pav_result["pav"],
                    "citmatch":       pav_result["citmatch"],
                    "citsupport":     pav_result["citsupport"],
                    "sup_score":      pav_result["sup_score"],
                    "length":         surf["length"],
                    "citation_count": surf["citation_count"],
                    "template_rate":  surf["template_rate"],
                })
            pool_data.append({
                "question":   question,
                "pool":       pool_entries,
                "provenance": provenance,
            })

    print(f"Loaded {len(pool_data)} questions with pools\n")

    pair_data = []
    for pd in pool_data:
        pool = pd["pool"]
        if len(pool) < 2:
            continue
        best_idx  = max(range(len(pool)), key=lambda i: pool[i]["cosine"])
        worst_idx = min(range(len(pool)), key=lambda i: pool[i]["cosine"])
        if best_idx == worst_idx:
            continue
        chosen   = pool[best_idx]
        rejected = pool[worst_idx]
        pair_data.append({
            "question":        pd["question"],
            "chosen_cosine":   chosen["cosine"],
            "rejected_cosine": rejected["cosine"],
            "proxy_gap":       chosen["cosine"] - rejected["cosine"],
            "delta_length":    chosen["length"] - rejected["length"],
            "delta_citations": chosen["citation_count"] - rejected["citation_count"],
            "delta_template":  chosen["template_rate"] - rejected["template_rate"],
            "delta_citmatch":  int(chosen["citmatch"]) - int(rejected["citmatch"]),
            "delta_sup_score": chosen["sup_score"] - rejected["sup_score"],
            "delta_pav":       int(chosen["pav"]) - int(rejected["pav"]),
            "substance_inverted": int(rejected["pav"]) > int(chosen["pav"]),
            "citmatch_inverted":  int(rejected["citmatch"]) > int(chosen["citmatch"]),
        })

    print(f"Pairs for audit: {len(pair_data)}\n")

    # ── A1: Covariance profile ────────────────────────────────────────────────
    print("="*60)
    print("A1: COVARIANCE PROFILE")
    print("="*60)

    all_cosines, all_lengths, all_citations, all_templates, all_citmatch, all_sup, all_pav = [], [], [], [], [], [], []
    for pd in pool_data:
        for s in pd["pool"]:
            all_cosines.append(s["cosine"])
            all_lengths.append(s["length"])
            all_citations.append(s["citation_count"])
            all_templates.append(s["template_rate"])
            all_citmatch.append(int(s["citmatch"]))
            all_sup.append(s["sup_score"])
            all_pav.append(int(s["pav"]))

    features = {
        "Length":          all_lengths,
        "Template rate":   all_templates,
        "Citation tokens": all_citations,
        "CITMATCH":        all_citmatch,
        "Sup score":       all_sup,
        "PAV":             all_pav,
    }

    print(f"{'Feature':<20} {'rho(f,S)':>10} {'p-value':>10}")
    print("-"*42)
    correlations = {}
    for fname, fvals in features.items():
        r, p = stats.pearsonr(all_cosines, fvals)
        correlations[fname] = r
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        print(f"{fname:<20} {r:>10.3f} {p:>10.4f} {sig}")

    # sigma_f as defined in Assumption 1 / Proposition 1: the within-question
    # standard deviation across the K samples drawn for THAT question, pooled
    # (averaged) across questions -- NOT the variance across all answers to
    # all different questions, which conflates between-question variation.
    feature_keys = {
        "Length":          "length",
        "Template rate":   "template_rate",
        "Citation tokens": "citation_count",
        "CITMATCH":        "citmatch",
        "Sup score":       "sup_score",
        "PAV":             "pav",
    }
    within_q_variances = {fname: [] for fname in feature_keys}
    for pd_ in pool_data:
        pool = pd_["pool"]
        if len(pool) < 2:
            continue
        for fname, key in feature_keys.items():
            vals = [float(s[key]) for s in pool]
            within_q_variances[fname].append(np.var(vals, ddof=1))

    sigmas = {}
    print(f"\n{'Feature':<20} {'sigma_f (within-q)':>20}")
    print("-"*42)
    for fname, varlist in within_q_variances.items():
        pooled_var = np.mean(varlist)
        sigma = np.sqrt(pooled_var)
        sigmas[fname] = sigma
        print(f"{fname:<20} {sigma:>20.4f}")

    # ── A2: Signal decomposition ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("A2: SIGNAL DECOMPOSITION (correlation with proxy gap)")
    print("="*60)

    proxy_gaps = [p["proxy_gap"] for p in pair_data]
    print(f"{'Feature contrast':<25} {'corr with gap':>14}")
    print("-"*42)
    for fname, fvals in [
        ("Delta Length",        [p["delta_length"]    for p in pair_data]),
        ("Delta Template rate", [p["delta_template"]  for p in pair_data]),
        ("Delta Citations",     [p["delta_citations"] for p in pair_data]),
        ("Delta CITMATCH",      [p["delta_citmatch"]  for p in pair_data]),
        ("Delta Sup score",     [p["delta_sup_score"] for p in pair_data]),
        ("Delta PAV",           [p["delta_pav"]       for p in pair_data]),
    ]:
        r, p = stats.pearsonr(proxy_gaps, fvals)
        print(f"{fname:<25} {r:>13.3f}")

    # ── A3: Substance inversion rate ──────────────────────────────────────────
    print("\n" + "="*60)
    print("A3: SUBSTANCE INVERSION RATE")
    print("="*60)

    n_pairs    = len(pair_data)
    n_inverted = sum(p["substance_inverted"] for p in pair_data)
    n_citmatch_inverted = sum(p["citmatch_inverted"] for p in pair_data)
    inversion_rate      = n_inverted / n_pairs if n_pairs > 0 else 0
    citmatch_inv_rate   = n_citmatch_inverted / n_pairs if n_pairs > 0 else 0

    rho_m     = correlations.get("CITMATCH", 0)
    predicted = np.arccos(max(-1, min(1, rho_m))) / np.pi

    print(f"Observed PAV inversion rate:     {inversion_rate:.3f} ({n_inverted}/{n_pairs})")
    print(f"Observed CITMATCH inversion:     {citmatch_inv_rate:.3f} ({n_citmatch_inverted}/{n_pairs})")
    print(f"Proxy-substance correlation rho: {rho_m:.3f}")
    print(f"Predicted inversion (K=2):       {predicted:.3f}")
    print(f"Gap (observed - predicted):      {inversion_rate - predicted:.3f}")

    # Save
    out = {
        "model": model_name,
        "n_questions": len(pool_data),
        "n_pairs": n_pairs,
        "A1_correlations": {k: float(v) for k, v in correlations.items()},
        "A1_sigmas": {k: float(v) for k, v in sigmas.items()},
        "A3_inversion": {
            "observed_pav":      inversion_rate,
            "observed_citmatch": citmatch_inv_rate,
            "predicted_k2":      float(predicted),
            "rho_substance":     float(rho_m),
        },
        "pairs": pair_data,
    }

    out_path = f"results/audit_{model_name}.json"
    Path("results").mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nFull audit saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["llama", "qwen", "fanar"])
    parser.add_argument("--train_file", default="data/processed/sft_train.jsonl")
    args = parser.parse_args()
    run_audit(args.model, args.train_file)
