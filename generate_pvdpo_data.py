#!/usr/bin/env python3
"""
PV-DPO Dataset Generator
=========================
Generates preference pairs using Provenance-Anchored Verification (PAV)
instead of cosine similarity scoring (SIM-DPO).

For each question:
1. Generate K answers from the fine-tuned model
2. Verify each answer using PAV (CITMATCH + CITSUPPORT)
3. If verified set V != empty AND unverified set != empty:
   - chosen = highest-cosine member of V (verified)
   - rejected = lowest-cosine non-member (unverified)
4. If all verified or none verified: abstain (no pair)

This decouples the substance signal from the proxy (Proposition 5).

Usage:
    python generate_pvdpo_data.py
    python generate_pvdpo_data.py --limit 50
"""

import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from pav import PAV

MODEL_PATH   = "./checkpoints/ft_fanar_paper/merged"
OUTPUT_PATH  = "./data/processed/pvdpo_dataset.jsonl"
N_SAMPLES    = 3        # K answers per question
MAX_TOKENS   = 300
TEMPERATURE  = 0.8      # high temp for diversity
TOP_P        = 0.95
TAU          = 0.3      # CITSUPPORT threshold


def load_questions(path: str, limit: int = None) -> list:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            # Only use examples with provenance
            if ex.get("provenance_text"):
                examples.append({
                    "question":        ex["instruction"],
                    "gold":            ex["response"],
                    "provenance_text": ex["provenance_text"],
                })
    if limit:
        examples = examples[:limit]
    return examples


def build_prompt(question: str) -> str:
    system = """أنت مساعد قانوني متخصص في قوانين ولوائح هيئة تنظيم الأعمال الخيرية في قطر. عند الإجابة على الأسئلة القانونية، يجب عليك: 1) ذكر رقم المادة أو البند القانوني بصيغة "مادة (رقم)" في بداية إجابتك دائماً. 2) تقديم إجابة كاملة ومفصلة لا تقل عن ثلاثة أسطر. 3) الاستناد حصراً إلى النصوص القانونية المقدمة في السياق. 4) تجنب الإجابات المبهمة أو العامة. 5) إذا لم تجد الإجابة في النصوص المقدمة، قل ذلك صراحةً بدلاً من الاختراع."""
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def generate_answers(model, tokenizer, question: str, n: int) -> list:
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    answers = []
    for _ in range(n):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if answer:
            answers.append(answer)
    return answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/processed/sft_train.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="data/processed/pvdpo_dataset.jsonl")
    parser.add_argument("--gap_threshold", type=float, default=0.05,
                        help="Min cosine gap for SIM-DPO fallback pairs")
    args = parser.parse_args()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print("Loading embedder...")
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("Loading PAV verifier...")
    pav = PAV(train_file=args.train_file, tau=TAU)

    examples = load_questions(args.train_file, args.limit)
    print(f"\nGenerating PV-DPO pairs for {len(examples)} questions with provenance...\n")

    pvdpo_pairs  = []
    simdpo_pairs = []  # fallback for abstained questions
    abstained    = 0
    all_verified = 0
    none_verified = 0

    for ex in tqdm(examples):
        question       = ex["question"]
        gold           = ex["gold"]
        provenance     = ex["provenance_text"]

        answers = generate_answers(model, tokenizer, question, N_SAMPLES)
        if len(answers) < 2:
            abstained += 1
            continue

        # Verify each answer with PAV
        pav_results = [pav.verify(question, ans, provenance) for ans in answers]
        verified_idx   = [i for i, r in enumerate(pav_results) if r["pav"]]
        unverified_idx = [i for i, r in enumerate(pav_results) if not r["pav"]]

        # Score all answers by cosine similarity to gold
        gold_emb = embedder.encode(gold)
        cosine_scores = []
        for ans in answers:
            emb = embedder.encode(ans)
            cosine_scores.append(cosine_sim(emb, gold_emb))

        if verified_idx and unverified_idx:
            # PV-DPO: chosen = highest-cosine verified, rejected = lowest-cosine unverified
            chosen_idx   = max(verified_idx,   key=lambda i: cosine_scores[i])
            rejected_idx = min(unverified_idx, key=lambda i: cosine_scores[i])
            pvdpo_pairs.append({
                "prompt":          question,
                "chosen":          answers[chosen_idx],
                "rejected":        answers[rejected_idx],
                "chosen_pav":      True,
                "rejected_pav":    False,
                "chosen_cosine":   round(cosine_scores[chosen_idx], 4),
                "rejected_cosine": round(cosine_scores[rejected_idx], 4),
                "pair_type":       "pvdpo",
            })
        elif not verified_idx:
            none_verified += 1
            # Fallback: SIM-DPO on unverified pool
            best_idx  = max(range(len(answers)), key=lambda i: cosine_scores[i])
            worst_idx = min(range(len(answers)), key=lambda i: cosine_scores[i])
            gap = cosine_scores[best_idx] - cosine_scores[worst_idx]
            if gap >= args.gap_threshold:
                simdpo_pairs.append({
                    "prompt":          question,
                    "chosen":          answers[best_idx],
                    "rejected":        answers[worst_idx],
                    "chosen_pav":      False,
                    "rejected_pav":    False,
                    "chosen_cosine":   round(cosine_scores[best_idx], 4),
                    "rejected_cosine": round(cosine_scores[worst_idx], 4),
                    "pair_type":       "simdpo_fallback",
                })
        else:
            all_verified += 1
            abstained += 1

    total_pairs = pvdpo_pairs + simdpo_pairs
    print(f"\nPV-DPO pairs:        {len(pvdpo_pairs)}")
    print(f"SIM-DPO fallback:    {len(simdpo_pairs)}")
    print(f"Abstained:           {abstained}")
    print(f"  - all verified:    {all_verified}")
    print(f"  - none verified:   {none_verified - len(simdpo_pairs)}")
    print(f"Total pairs:         {len(total_pairs)}")

    if pvdpo_pairs:
        pv_chosen_cos  = sum(p["chosen_cosine"]   for p in pvdpo_pairs) / len(pvdpo_pairs)
        pv_rejected_cos = sum(p["rejected_cosine"] for p in pvdpo_pairs) / len(pvdpo_pairs)
        print(f"\nPV-DPO stats:")
        print(f"  Avg chosen cosine:   {pv_chosen_cos:.3f}")
        print(f"  Avg rejected cosine: {pv_rejected_cos:.3f}")
        print(f"  Avg gap:             {pv_chosen_cos - pv_rejected_cos:.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in total_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(total_pairs)} pairs → {out_path}")


if __name__ == "__main__":
    main()
