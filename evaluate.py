#!/usr/bin/env python3
"""
RACA Legal LLM — Evaluation Script (Semantic)
===============================================
Runs the test set through the API and measures accuracy
using cosine similarity on multilingual sentence embeddings.

Usage:
    python evaluate.py                          # steered only, full test set
    python evaluate.py --compare                # plain vs steered
    python evaluate.py --limit 20               # quick test with 20 examples
    python evaluate.py --compare --limit 20
"""

import json
import argparse
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path

API_URL = "http://localhost:8080"

# ── Embedder ──────────────────────────────────────────────────────────────────

def load_embedder():
    from sentence_transformers import SentenceTransformer
    print("Loading sentence embedder...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("Embedder ready.\n")
    return model

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

# ── API call ──────────────────────────────────────────────────────────────────

def call_api(question: str, use_steering: bool) -> str:
    payload = json.dumps({
        "question": question,
        "history": [],
        "use_rag": True,
        "steering_coefficient": 0.5 if use_steering else 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        API_URL + "/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))["answer"]
    except Exception as e:
        print(f"  ⚠ API error: {e}")
        return ""

# ── Load test data ────────────────────────────────────────────────────────────

def load_test_data(path: str, limit: int = None) -> list:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            examples.append({
                "question": ex["instruction"],
                "gold": ex["response"],
                "source": ex.get("source_title", ""),
            })
    if limit:
        examples = examples[:limit]
    return examples

# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(test_file: str, compare: bool, limit: int):
    embedder = load_embedder()
    examples = load_test_data(test_file, limit)
    modes = ["plain", "steered"] if compare else ["steered"]

    print(f"{'='*60}")
    print(f"  RACA Legal LLM — Semantic Evaluation")
    print(f"  Test examples : {len(examples)}")
    print(f"  Metric        : Cosine similarity (multilingual embeddings)")
    print(f"  Mode          : {'plain vs steered' if compare else 'steered only'}")
    print(f"{'='*60}\n")

    all_scores = {m: [] for m in modes}
    all_results = []

    for i, ex in enumerate(examples):
        print(f"[{i+1}/{len(examples)}] {ex['question'][:70]}...")

        gold_emb = embedder.encode(ex["gold"])
        row = {"question": ex["question"], "gold": ex["gold"], "source": ex["source"]}

        for mode in modes:
            pred = call_api(ex["question"], use_steering=(mode == "steered"))
            if pred:
                pred_emb = embedder.encode(pred)
                sim = cosine_sim(pred_emb, gold_emb)
            else:
                sim = 0.0
            all_scores[mode].append(sim)
            row[f"pred_{mode}"] = pred
            row[f"sim_{mode}"] = round(sim, 4)
            print(f"  {mode:8} | sim: {sim:.3f} | {pred[:80]}")

        all_results.append(row)
        print()

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    for mode in modes:
        sims = all_scores[mode]
        avg = sum(sims) / len(sims)
        above_80 = sum(1 for s in sims if s >= 0.8) / len(sims)
        above_60 = sum(1 for s in sims if s >= 0.6) / len(sims)
        print(f"  {mode.upper():8} | Avg cosine sim: {avg:.3f} | ≥0.8: {above_80:.1%} | ≥0.6: {above_60:.1%}")

    if compare and "plain" in modes and "steered" in modes:
        delta = (sum(all_scores["steered"]) - sum(all_scores["plain"])) / len(examples)
        print(f"\n  Steering improvement: {delta:+.3f} avg cosine sim")

    print(f"{'='*60}\n")

    # Save
    out_path = Path("results/eval_semantic.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": {m: sum(all_scores[m])/len(all_scores[m]) for m in modes}, "results": all_results}, f, ensure_ascii=False, indent=2)
    print(f"  Full results saved → {out_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="data/processed/sft_test.jsonl")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.test_file, args.compare, args.limit)
