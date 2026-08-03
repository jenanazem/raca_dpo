#!/usr/bin/env python3
"""
PAV — Provenance-Anchored Verification Suite
=============================================
Implements the verification metrics from the paper:

- CITPRESENCE: does the answer cite any article at all?
- CITMATCH:    does the answer cite the correct source article (provenance)?
- CITSUPPORT:  is the answer content supported by the cited article?
- PAV:         CITMATCH AND CITSUPPORT (headline substance metric)

Also computes surface diagnostics:
- answer length
- template n-gram rate (fraction covered by top-50 training 4-grams)
- citation token count

Usage:
    from pav import PAV
    pav = PAV(train_file="data/processed/sft_train.jsonl")
    result = pav.verify(question, answer, provenance_text)
    print(result)

Or run standalone evaluation:
    python pav.py --pred_file results/eval_semantic.json --train_file data/processed/sft_train.jsonl
"""

import re
import json
import math
import argparse
from pathlib import Path
from collections import Counter


# ── Arabic text normalization ──────────────────────────────────────────────────

def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for comparison."""
    if not text:
        return ""
    # Remove diacritics
    text = re.sub(r'[\u064b-\u065f\u0670]', '', text)
    # Normalize alif variants
    text = re.sub(r'[أإآ]', 'ا', text)
    # Normalize ya
    text = text.replace('ى', 'ي')
    # Normalize ta marbuta
    text = text.replace('ة', 'ه')
    # Convert Eastern Arabic numerals to ASCII
    for eastern, western in zip('٠١٢٣٤٥٦٧٨٩', '0123456789'):
        text = text.replace(eastern, western)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Citation parser ────────────────────────────────────────────────────────────

# Patterns to extract article references from Arabic/English answers
ARTICLE_PATTERNS = [
    # Arabic: الماده (6) or ماده 6 (after normalization ة->ه)
    r'(?:ال)?ماده\s*(?:رقم\s*)?[\(\[]?(\d+)[\)\]]?',
    r'(?:ال)?ماده\s+([٠-٩0-9]+)',
    # With clitic prefixes: بالماده، للماده، وماده
    r'[بولكف](?:ال)?ماده\s*[\(\[]?(\d+)[\)\]]?',
    # Also match before normalization just in case
    r'(?:ال)?مادة\s*(?:رقم\s*)?[\(\[]?(\d+)[\)\]]?',
    r'(?:ال)?مادة\s+([٠-٩0-9]+)',
    # English: Article 6, Art. 6
    r'[Aa]rt(?:icle)?\.?\s*(\d+)',
    # Section references (after normalization)
    r'(?:ال)?بند\s*[\(\[]?(\d+)[\)\]]?',
    r'(?:ال)?فقره\s*[\(\[]?(\d+)[\)\]]?',
]

COMPILED_PATTERNS = [re.compile(p) for p in ARTICLE_PATTERNS]


def extract_citations(text: str) -> set:
    """Extract article numbers cited in a text."""
    normalized = normalize_arabic(text)
    citations = set()
    for pattern in COMPILED_PATTERNS:
        for match in pattern.finditer(normalized):
            num = match.group(1)
            # Normalize Eastern Arabic numerals
            for e, w in zip('٠١٢٣٤٥٦٧٨٩', '0123456789'):
                num = num.replace(e, w)
            citations.add(num)
    return citations


def extract_provenance_article_number(provenance_text: str) -> set:
    """Extract the article number(s) from a provenance chunk."""
    return extract_citations(provenance_text)


# ── LCS support score ──────────────────────────────────────────────────────────

def lcs_length(a: list, b: list) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Use space-efficient LCS
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    return prev[n]


def support_score(answer: str, article_text: str) -> float:
    """
    ROUGE-L-precision-style containment score.
    = LCS(answer_tokens, article_tokens) / len(answer_tokens)
    Measures how much of the answer is supported by the article text.
    """
    answer_tokens = normalize_arabic(answer).split()
    article_tokens = normalize_arabic(article_text).split()
    if not answer_tokens:
        return 0.0
    lcs = lcs_length(answer_tokens, article_tokens)
    return lcs / len(answer_tokens)


# ── Surface diagnostics ────────────────────────────────────────────────────────

def compute_template_ngrams(train_file: str, n: int = 4, top_k: int = 50) -> set:
    """Compute top-k most frequent n-grams from training answers."""
    ngram_counts = Counter()
    with open(train_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            answer = normalize_arabic(ex.get('response', ''))
            tokens = answer.split()
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                ngram_counts[ngram] += 1
    return {ngram for ngram, _ in ngram_counts.most_common(top_k)}


def template_ngram_rate(answer: str, template_ngrams: set, n: int = 4) -> float:
    """Fraction of answer tokens covered by template n-grams."""
    tokens = normalize_arabic(answer).split()
    if not tokens:
        return 0.0
    covered = set()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        if ngram in template_ngrams:
            for j in range(i, i + n):
                covered.add(j)
    return len(covered) / len(tokens)


# ── Main PAV class ─────────────────────────────────────────────────────────────

class PAV:
    def __init__(self, train_file: str = None, tau: float = 0.3):
        """
        Args:
            train_file: path to sft_train.jsonl for computing template n-grams
            tau: threshold for CITSUPPORT (default 0.3, tuned on dev set)
        """
        self.tau = tau
        self.template_ngrams = set()
        if train_file and Path(train_file).exists():
            self.template_ngrams = compute_template_ngrams(train_file)
            print(f"PAV: loaded {len(self.template_ngrams)} template 4-grams")

    def verify(self, question: str, answer: str, provenance_text: str) -> dict:
        """
        Run full PAV verification.
        
        Returns dict with:
            citpresence: bool
            citmatch:    bool  
            citsupport:  bool
            pav:         bool (citmatch AND citsupport)
            sup_score:   float (raw support score)
            answer_len:  int
            citation_count: int
            template_rate: float
        """
        # Extract citations from answer
        answer_citations = extract_citations(answer)
        
        # Extract provenance article number
        provenance_articles = extract_provenance_article_number(provenance_text)
        
        # CITPRESENCE: any citation found
        citpresence = len(answer_citations) > 0
        
        # CITMATCH: answer cites the provenance article
        if provenance_articles and answer_citations:
            citmatch = bool(answer_citations & provenance_articles)
        else:
            citmatch = False
        
        # CITSUPPORT: answer content supported by provenance text
        sup = support_score(answer, provenance_text)
        citsupport = sup >= self.tau
        
        # PAV = CITMATCH AND CITSUPPORT
        pav = citmatch and citsupport
        
        # Surface diagnostics
        answer_len = len(normalize_arabic(answer).split())
        citation_count = len(answer_citations)
        tmpl_rate = template_ngram_rate(answer, self.template_ngrams) if self.template_ngrams else 0.0
        
        return {
            "citpresence": citpresence,
            "citmatch": citmatch,
            "citsupport": citsupport,
            "pav": pav,
            "sup_score": round(sup, 4),
            "answer_len": answer_len,
            "citation_count": citation_count,
            "template_rate": round(tmpl_rate, 4),
            "answer_citations": list(answer_citations),
            "provenance_articles": list(provenance_articles),
        }

    def batch_verify(self, examples: list) -> list:
        """
        Verify a batch of examples.
        Each example should have: question, answer, provenance_text
        """
        return [
            self.verify(ex["question"], ex["answer"], ex["provenance_text"])
            for ex in examples
        ]

    def summary(self, results: list) -> dict:
        """Compute aggregate metrics over a list of verify() outputs."""
        n = len(results)
        if n == 0:
            return {}
        return {
            "n": n,
            "citpresence": sum(r["citpresence"] for r in results) / n,
            "citmatch":    sum(r["citmatch"]    for r in results) / n,
            "citsupport":  sum(r["citsupport"]  for r in results) / n,
            "pav":         sum(r["pav"]          for r in results) / n,
            "avg_sup_score":    sum(r["sup_score"]     for r in results) / n,
            "avg_answer_len":   sum(r["answer_len"]    for r in results) / n,
            "avg_citations":    sum(r["citation_count"] for r in results) / n,
            "avg_template_rate": sum(r["template_rate"] for r in results) / n,
        }


# ── Standalone evaluation ──────────────────────────────────────────────────────

def run_evaluation(pred_file: str, test_file: str, train_file: str, output: str):
    """
    Evaluate model predictions using PAV metrics.
    pred_file: results/eval_semantic.json (has pred_steered and question fields)
    test_file: data/processed/sft_test.jsonl (has provenance_text)
    """
    pav = PAV(train_file=train_file)

    # Load predictions
    with open(pred_file) as f:
        pred_data = json.load(f)
    predictions = {r["question"]: r.get("pred_steered", r.get("prediction", "")) 
                   for r in pred_data["results"]}

    # Load test data with provenance
    test_examples = []
    with open(test_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            q = ex["instruction"]
            if q in predictions and ex.get("provenance_text"):
                test_examples.append({
                    "question": q,
                    "answer": predictions[q],
                    "provenance_text": ex["provenance_text"],
                })

    # Filter to only examples where provenance has article numbers
    import re
    test_examples = [ex for ex in test_examples 
                     if re.search(r'مادة\s*[\(\[]?\d', ex['provenance_text'])]
    print(f"\nEvaluating {len(test_examples)} examples with article-numbered provenance...\n")

    results = pav.batch_verify(test_examples)
    summary = pav.summary(results)

    print(f"{'='*60}")
    print(f"  PAV RESULTS")
    print(f"{'='*60}")
    print(f"  CITPRESENCE : {summary['citpresence']:.1%}  (answers that cite any article)")
    print(f"  CITMATCH    : {summary['citmatch']:.1%}  (answers citing the correct article)")
    print(f"  CITSUPPORT  : {summary['citsupport']:.1%}  (answers supported by cited article)")
    print(f"  PAV         : {summary['pav']:.1%}  (CITMATCH AND CITSUPPORT)")
    print(f"{'='*60}")
    print(f"  Surface diagnostics:")
    print(f"  Avg answer length  : {summary['avg_answer_len']:.1f} tokens")
    print(f"  Avg citation count : {summary['avg_citations']:.2f}")
    print(f"  Avg template rate  : {summary['avg_template_rate']:.3f}")
    print(f"{'='*60}\n")

    out = {"summary": summary, "results": [
        {"question": ex["question"], "answer": ex["answer"], 
         "provenance_text": ex["provenance_text"][:200], **res}
        for ex, res in zip(test_examples, results)
    ]}
    Path(output).parent.mkdir(exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Results saved → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file",  default="results/eval_semantic.json")
    parser.add_argument("--test_file",  default="data/processed/sft_test.jsonl")
    parser.add_argument("--train_file", default="data/processed/sft_train.jsonl")
    parser.add_argument("--output",     default="results/pav_results.json")
    parser.add_argument("--tau",        type=float, default=0.3)
    args = parser.parse_args()
    run_evaluation(args.pred_file, args.test_file, args.train_file, args.output)
