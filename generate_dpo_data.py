#!/usr/bin/env python3
"""
DPO Dataset Generator for RACA Legal LLM
==========================================
Generates (chosen, rejected) pairs for DPO training by:
1. Loading questions from sft_train.jsonl
2. Generating N answers per question using the fine-tuned model
3. Using cosine similarity vs gold answer to rank answers
4. Taking best as 'chosen' and worst as 'rejected'

Usage:
    python generate_dpo_data.py --limit 200
    python generate_dpo_data.py  # full dataset
"""

import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

MODEL_PATH  = "./checkpoints/ft_fanar_paper/merged"
OUTPUT_PATH = "./data/processed/dpo_dataset.jsonl"
N_SAMPLES   = 3       # answers to generate per question
MAX_TOKENS  = 300
TEMPERATURE = 0.8     # high temp for diversity

def load_questions(path: str, limit: int = None) -> list:
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
                "provenance_text": ex.get("provenance_text", ""),
            })
    if limit:
        examples = examples[:limit]
    return examples

def build_prompt(question: str) -> str:
    system = """أنت مساعد قانوني متخصص في قوانين ولوائح هيئة تنظيم الأعمال الخيرية في قطر. عند الإجابة على الأسئلة القانونية، يجب عليك: 1) ذكر رقم المادة أو البند القانوني المرجعي إن وجد. 2) تقديم إجابة كاملة ومفصلة لا تقل عن ثلاثة أسطر. 3) الاستناد حصراً إلى النصوص القانونية المقدمة في السياق. 4) تجنب الإجابات المبهمة أو العامة. 5) إذا لم تجد الإجابة في النصوص المقدمة، قل ذلك صراحةً بدلاً من الاختراع."""
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def generate_answers(model, tokenizer, question: str, n: int) -> list[str]:
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
                top_p=0.9,
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
    parser.add_argument("--output", default="data/processed/dpo_dataset.jsonl")
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

    examples = load_questions(args.train_file, args.limit)
    print(f"Generating DPO pairs for {len(examples)} questions...\n")

    dpo_pairs = []
    skipped = 0

    for ex in tqdm(examples):
        question = ex["question"]
        gold = ex["gold"]

        answers = generate_answers(model, tokenizer, question, N_SAMPLES)
        if len(answers) < 2:
            skipped += 1
            continue

        # Score each answer against gold
        gold_emb = embedder.encode(gold)
        scored = []
        for ans in answers:
            ans_emb = embedder.encode(ans)
            sim = cosine_sim(ans_emb, gold_emb)
            scored.append((sim, ans))

        scored.sort(reverse=True)
        best_sim, chosen = scored[0]
        worst_sim, rejected = scored[-1]

        # Only keep pairs where there's a meaningful difference
        if best_sim - worst_sim < 0.05:
            skipped += 1
            continue

        dpo_pairs.append({
            "prompt": question,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_score": round(best_sim, 4),
            "rejected_score": round(worst_sim, 4),
            "pool": [{"answer": ans, "score": round(sim, 4)} for sim, ans in scored],
            "gold": gold,
            "provenance_text": ex.get("provenance_text", ""),
        })

    print(f"\nGenerated {len(dpo_pairs)} DPO pairs ({skipped} skipped)")

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved to {out_path}")

    # Stats
    chosen_scores = [p["chosen_score"] for p in dpo_pairs]
    rejected_scores = [p["rejected_score"] for p in dpo_pairs]
    print(f"Avg chosen score:   {sum(chosen_scores)/len(chosen_scores):.3f}")
    print(f"Avg rejected score: {sum(rejected_scores)/len(rejected_scores):.3f}")
    print(f"Avg gap:            {(sum(chosen_scores)-sum(rejected_scores))/len(dpo_pairs):.3f}")

if __name__ == "__main__":
    main()
