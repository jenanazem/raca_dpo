#!/usr/bin/env python3
"""
Audit with multiple embedders
==============================
Re-scores the saved pool datasets with LaBSE and BGE-M3 embedders
to fill the LaBSE and BGE-M3 columns in Table 1.

Usage:
    python generate_audit_embedders.py --model llama --embedder labse
    python generate_audit_embedders.py --model llama --embedder bge-m3
    python generate_audit_embedders.py --model qwen --embedder labse
    python generate_audit_embedders.py --model qwen --embedder bge-m3
    python generate_audit_embedders.py --model fanar --embedder labse
    python generate_audit_embedders.py --model fanar --embedder bge-m3
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats
from sentence_transformers import SentenceTransformer

from pav import PAV, extract_citations, normalize_arabic, template_ngram_rate

EMBEDDER_NAMES = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "labse":  "sentence-transformers/LaBSE",
    "bge-m3": "BAAI/bge-m3",
}


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def surface_features(answer, template_ngrams):
    tokens = normalize_arabic(answer).split()
    length = len(tokens)
    citations = len(extract_citations(answer))
    tmpl_rate = template_ngram_rate(answer, template_ngrams)
    return {"length": length, "citation_count": citations, "template_rate": tmpl_rate}


def run(model_name, embedder_key, train_file):
    pool_file = f"data/processed/{model_name}_pool_dataset.jsonl"
    embedder_name = EMBEDDER_NAMES[embedder_key]

    print(f"Loading embedder: {embedder_name}")
    embedder = SentenceTransformer(embedder_name)

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
            gold = ex.get("gold", "")
            pool_entries = []
            for item in ex["pool"]:
                ans = item["answer"]
                pav_result = pav.verify(question, ans, provenance)
                surf = surface_features(ans, pav.template_ngrams)
                pool_entries.append({
                    "answer":         ans,
                    "pav":            pav_result["pav"],
                    "citmatch":       pav_result["citmatch"],
                    "sup_score":      pav_result["sup_score"],
                    "length":         surf["length"],
                    "citation_count": surf["citation_count"],
                    "template_rate":  surf["template_rate"],
                    "gold":           gold,
                })
            pool_data.append({"question": question, "pool": pool_entries})

    print(f"Loaded {len(pool_data)} questions. Computing cosine scores with {embedder_key}...")

    # Score each answer with the new embedder
    all_cosines, all_lengths, all_citations, all_templates, all_citmatch, all_sup, all_pav = [], [], [], [], [], [], []
    pair_data = []

    for pd in pool_data:
        pool = pd["pool"]
        if len(pool) < 2:
            continue
        gold = pool[0]["gold"]
        gold_emb = embedder.encode(gold)
        cosines = []
        for item in pool:
            ans_emb = embedder.encode(item["answer"])
            cos = cosine_sim(ans_emb, gold_emb)
            cosines.append(cos)
            all_cosines.append(cos)
            all_lengths.append(item["length"])
            all_citations.append(item["citation_count"])
            all_templates.append(item["template_rate"])
            all_citmatch.append(int(item["citmatch"]))
            all_sup.append(item["sup_score"])
            all_pav.append(int(item["pav"]))

        best_idx  = max(range(len(pool)), key=lambda i: cosines[i])
        worst_idx = min(range(len(pool)), key=lambda i: cosines[i])
        if best_idx == worst_idx:
            continue

        chosen   = pool[best_idx]
        rejected = pool[worst_idx]
        pair_data.append({
            "delta_length":    chosen["length"]         - rejected["length"],
            "delta_citations": chosen["citation_count"] - rejected["citation_count"],
            "delta_template":  chosen["template_rate"]  - rejected["template_rate"],
            "delta_citmatch":  int(chosen["citmatch"])  - int(rejected["citmatch"]),
            "delta_sup_score": chosen["sup_score"]      - rejected["sup_score"],
            "proxy_gap":       cosines[best_idx]        - cosines[worst_idx],
        })

    print(f"\n{'='*55}")
    print(f"A1: COVARIANCE PROFILE — {embedder_key.upper()} — {model_name}")
    print(f"{'='*55}")
    features = {
        "Length":          all_lengths,
        "Template rate":   all_templates,
        "Citation tokens": all_citations,
        "CITMATCH":        all_citmatch,
        "Sup score":       all_sup,
        "PAV":             all_pav,
    }
    correlations = {}
    print(f"{'Feature':<20} {'rho':>8} {'p':>10}")
    print("-"*42)
    for fname, fvals in features.items():
        r, p = stats.pearsonr(all_cosines, fvals)
        correlations[fname] = round(float(r), 3)
        print(f"{fname:<20} {r:>8.3f} {p:>10.4f}")

    # Save
    out = {
        "model": model_name,
        "embedder": embedder_key,
        "n_questions": len(pool_data),
        "A1_correlations": correlations,
    }
    out_path = f"results/audit_{model_name}_{embedder_key}.json"
    Path("results").mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True, choices=["llama", "qwen", "fanar"])
    parser.add_argument("--embedder", required=True, choices=["labse", "bge-m3"])
    parser.add_argument("--train_file", default="data/processed/sft_train.jsonl")
    args = parser.parse_args()
    run(args.model, args.embedder, args.train_file)
