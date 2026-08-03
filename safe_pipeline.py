"""
SAFE Pipeline for RACA Legal LLM
==================================
Adapted from: https://github.com/KurbanIntelligenceLab/SAFE

Architecture:
- Fine-tuned model : JinanAzem/raca-finetuned-v2 (local, used for ALL generation)
- SAE              : Goodfire pre-trained SAE for Llama 3.1 8B Instruct (layer 19)
                     65,536 features, labels from Neuronpedia (auto-downloaded)
- Embedder         : paraphrase-multilingual-MiniLM-L12-v2 (Arabic-capable)
- RAG              : FAISS index over RACA CSV documents

How SAFE works:
1. Generate N answers using YOUR fine-tuned model at high temperature
2. Cluster answers by semantic similarity
3. Calculate semantic entropy — high = inconsistent = hallucination risk
4. If entropy > threshold, enter enrichment loop (max 3 iterations):
   - Extract SAE features from question using YOUR model + Goodfire SAE
   - Extract SAE features from each answer
   - Find feature differences (what fires in answers but not in question)
   - Check sparsity of those features
   - Enrich question with NOTE: do not consider X / you must consider X
   - Regenerate with enriched question
5. Return enriched question + final answer

Usage:
    python safe_pipeline.py --question "ما هي شروط تأسيس الجمعية الخيرية؟"
    python safe_pipeline.py --test_file data/processed/sft_test.jsonl --limit 20
    python safe_pipeline.py --test_file data/processed/sft_test.jsonl
"""

import os
import math
import json
import argparse
import logging
import requests
import numpy as np
import torch
import csv
import glob
from collections import Counter
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sae_lens import SAE, HookedSAETransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH     = "./checkpoints/ft_raca_v2/merged"
MODEL_HF_ID    = "JinanAzem/raca-finetuned-v2"
SAE_RELEASE    = "goodfire-llama-3.1-8b-instruct"
SAE_LAYER      = "layer_19"
SAE_HOOK       = "blocks.19.hook_resid_post"
NEURONPEDIA_ID = "llama3.1-8b-it/19-resid-post-gf"
CSV_GLOB       = "./data/processed/raca_tab*.csv"
EMBED_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

N_ANSWERS         = 5
HIGH_TEMP         = 1.0
LOW_TEMP          = 0.1
ENTROPY_THRESHOLD = 0.6
DENSITY_THRESHOLD = 0.05
MAX_LOOPS         = 3
MAX_NEW_TOKENS    = 300
TOP_K_DOCS        = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('safe_pipeline.log'),
        logging.StreamHandler()
    ]
)

# ── RAG Retriever ──────────────────────────────────────────────────────────────

class RAGRetriever:
    def __init__(self, csv_glob: str, embedder, top_k: int = 3):
        self.top_k = top_k
        self.docs = []
        self._load(csv_glob)
        self._build_index(embedder)

    def _load(self, csv_glob):
        csv.field_size_limit(10**7)
        for path in sorted(glob.glob(csv_glob)):
            with open(path, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    text = row.get("text", "").strip()
                    title = row.get("title", "").strip()
                    if text and len(text) > 100:
                        self.docs.append({"title": title, "text": text[:2000]})
        logging.info(f"RAG: loaded {len(self.docs)} documents")

    def _build_index(self, embedder):
        import faiss
        texts = [f"{d['title']} {d['text'][:500]}" for d in self.docs]
        embs = embedder.encode(texts, show_progress_bar=False, batch_size=32)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs.astype(np.float32))
        self.embedder = embedder

    def retrieve(self, query: str) -> str:
        q = self.embedder.encode([query])
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        _, idxs = self.index.search(q.astype(np.float32), self.top_k)
        parts = []
        for i, idx in enumerate(idxs[0]):
            if idx < len(self.docs):
                d = self.docs[idx]
                parts.append(f"[مصدر {i+1}] {d['title']}\n{d['text'][:600]}")
        return "\n\n".join(parts)


# ── Local Generator (your fine-tuned model) ────────────────────────────────────

class LocalGenerator:
    """
    Uses YOUR fine-tuned RACA model for all generation.
    Supports high/low temperature and RAG context injection.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        logging.info(f"Loading fine-tuned model for generation from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        self.device = device
        logging.info("Generator ready.")

    def generate(self, question: str, temperature: float = 0.1, context: str = "") -> str:
        system = "أنت مساعد قانوني متخصص في قوانين هيئة تنظيم الأعمال الخيرية في قطر. أجب بدقة وإيجاز بناءً على التشريعات القطرية."
        user_msg = question
        if context:
            user_msg = f"بناءً على المصادر القانونية التالية:\n\n{context}\n\nالسؤال: {question}"

        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=temperature > 0.1,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Local Explainer (your fine-tuned model + Goodfire SAE) ────────────────────

class LocalExplainer:
    """
    Runs YOUR fine-tuned model with the Goodfire SAE hooked in.
    Feature labels downloaded automatically from Neuronpedia.
    No manual labeling needed.
    """

    def __init__(self, generator: LocalGenerator, sae_release: str, sae_layer: str, device: str = "cuda"):
        logging.info("Wrapping model in HookedSAETransformer...")
        self.model = HookedSAETransformer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            hf_model=generator.model,
            dtype=torch.bfloat16,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            device=device,
        )
        self.tokenizer = generator.tokenizer

        logging.info(f"Loading Goodfire SAE ({sae_release} / {sae_layer})...")
        self.sae = SAE.from_pretrained(release=sae_release, sae_id=sae_layer)
        self.sae = self.sae.to(device)
        self.device = device

        logging.info("Loading Neuronpedia feature labels...")
        self.idx_to_exp = self._load_labels()
        logging.info(f"Loaded {len(self.idx_to_exp)} feature labels")

    def _load_labels(self) -> dict:
        cache = Path("checkpoints/neuronpedia_labels.json")
        if cache.exists():
            with open(cache) as f:
                data = json.load(f)
        else:
            logging.info("Downloading labels from Neuronpedia (one-time)...")
            model_id, sae_id = NEURONPEDIA_ID.split("/")
            url = f"https://www.neuronpedia.org/api/explanation/export?modelId={model_id}&saeId={sae_id}"
            resp = requests.get(url, headers={"Content-Type": "application/json"}, timeout=120)
            data = resp.json()
            cache.parent.mkdir(exist_ok=True)
            with open(cache, "w") as f:
                json.dump(data, f)
        return {str(item['index']): item['description'] for item in data
                if 'index' in item and 'description' in item}

    def get_explanation(self, prompt: str, n_results: int = 10) -> list:
        with torch.no_grad():
            _, cache = self.model.run_with_cache_with_saes(
                prompt, saes=[self.sae], prepend_bos=True,
            )
        hook_key = f"{self.sae.cfg.metadata['hook_name']}.hook_sae_acts_post"
        sae_acts = cache[hook_key][0, 1:, :].sum(dim=0)

        results = []
        for act, ind in zip(*sae_acts.topk(200)):
            desc = self.idx_to_exp.get(str(int(ind.item())))
            if not desc:
                continue
            results.append({'index': ind.item(), 'description': desc, 'activation': act.item()})
            if len(results) >= n_results:
                break
        return results


# ── SAFE Framework ─────────────────────────────────────────────────────────────

def semantic_entropy(cluster_assignments: list) -> float:
    freq = Counter(cluster_assignments)
    total = len(cluster_assignments)
    return -sum((c/total) * math.log(c/total) for c in freq.values())


class SAFEPipeline:
    def __init__(self):
        logging.info("Initializing SAFE Pipeline...")
        self.embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

        # Single model instance shared between generator and explainer
        self.generator = LocalGenerator(MODEL_PATH, device=DEVICE)
        self.explainer = LocalExplainer(self.generator, SAE_RELEASE, SAE_LAYER, device=DEVICE)

        # RAG
        self.rag = RAGRetriever(CSV_GLOB, self.embedder, top_k=TOP_K_DOCS)
        logging.info("SAFE Pipeline ready.")

    def _embed(self, question: str, answer: str = None) -> np.ndarray:
        text = f"{question} [SEP] {answer}" if answer else question
        return self.embedder.encode(text, show_progress_bar=False)

    def _cluster(self, question: str, answers: list) -> list:
        embs = [self._embed(question, a) for a in answers]
        dist = 1 - cosine_similarity(embs)
        try:
            c = AgglomerativeClustering(n_clusters=None, distance_threshold=0.1,
                                        metric='precomputed', linkage='average')
            return list(c.fit_predict(dist))
        except Exception:
            return list(range(len(answers)))

    def _feature_diffs(self, explanation_q: list, answer: str) -> list:
        explanation_a = self.explainer.get_explanation(answer)
        feats_q = {f['description'] for f in explanation_q if f.get('description')}
        feats_a = {f['description'] for f in explanation_a if f.get('description')}
        return list(feats_a - feats_q)

    def _embed_features(self, diff_sets: list) -> np.ndarray:
        all_feats = list(set().union(*diff_sets))
        if not all_feats:
            return np.array([])
        return self.embedder.encode(all_feats, show_progress_bar=False)

    def _sparsity(self, embeddings: np.ndarray) -> bool:
        if len(embeddings) < 2:
            return False
        sim = self.embedder.similarity(embeddings, embeddings)
        n = len(sim)
        upper = [float(sim[i,j]) for i in range(n) for j in range(i+1,n)]
        if not upper:
            return False
        q1, q3 = np.percentile(upper, 25), np.percentile(upper, 50)
        return any(s < q1 - 1.5*(q3-q1) for s in upper)

    def _enrich(self, question: str, diff_sets: list, feat_embs: np.ndarray,
                sparsity: bool, added: list) -> tuple:
        all_feats = list(set().union(*diff_sets) - set(added))
        if not all_feats or len(feat_embs) == 0:
            return question, added
        q_emb = self.embedder.encode([question])
        sims = self.embedder.similarity(q_emb, feat_embs)[0]
        most_similar = all_feats[min(int(np.argmax(sims)), len(all_feats)-1)]
        most_distant  = all_feats[min(int(np.argmin(sims)), len(all_feats)-1)]
        sep = " - NOTE:" if "NOTE:" not in question else " and"
        if sparsity:
            new_q = f"{question}{sep} do not consider {most_distant}"
            added.append(most_distant)
        else:
            new_q = f"{question}{sep} you must consider {most_similar}"
            added.append(most_similar)
        logging.info(f"Enriched: {new_q[:120]}")
        return new_q, added

    def __call__(self, question: str) -> dict:
        logging.info(f"Processing: {question}")
        new_q = question
        added = []

        # Get RAG context
        context = self.rag.retrieve(question)

        # Low temp answer (baseline)
        low_ans = self.generator.generate(question, temperature=LOW_TEMP, context=context)

        # Generate N diverse answers with high temperature
        answers = [self.generator.generate(question, temperature=HIGH_TEMP, context=context)
                   for _ in range(N_ANSWERS)]

        first_entropy = semantic_entropy(self._cluster(question, answers))
        final_entropy = first_entropy
        logging.info(f"Initial entropy: {first_entropy:.3f}")

        loop_count = 0
        for loop in range(1, MAX_LOOPS+1):
            if final_entropy <= ENTROPY_THRESHOLD:
                logging.info("Entropy OK — stopping enrichment.")
                break
            loop_count = loop
            logging.info(f"Enrichment loop {loop}")
            expl_q    = self.explainer.get_explanation(new_q)
            diffs     = [self._feature_diffs(expl_q, a) for a in answers]
            feat_embs = self._embed_features(diffs)
            sparsity  = self._sparsity(feat_embs) if len(feat_embs) > 0 else False
            new_q, added = self._enrich(new_q, diffs, feat_embs, sparsity, added)

            # Regenerate with enriched question + RAG
            answers = [self.generator.generate(new_q, temperature=HIGH_TEMP, context=context)
                       for _ in range(N_ANSWERS)]
            final_entropy = semantic_entropy(self._cluster(question, answers))
            logging.info(f"New entropy: {final_entropy:.3f}")

        # Final answer with enriched question + RAG
        final_answer = self.generator.generate(new_q, temperature=LOW_TEMP, context=context)

        return {
            "original_question": question,
            "enriched_question": new_q,
            "final_answer":      final_answer,
            "low_temp_answer":   low_ans,
            "first_entropy":     round(first_entropy, 4),
            "final_entropy":     round(final_entropy, 4),
            "enrichment_loops":  loop_count,
            "features_added":    added,
        }


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(test_file: str, limit: int = None):
    embedder = SentenceTransformer(EMBED_MODEL)
    examples = []
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                examples.append({"question": ex["instruction"], "gold": ex["response"]})
    if limit:
        examples = examples[:limit]

    pipeline = SAFEPipeline()
    scores, results = [], []

    print(f"\n{'='*60}")
    print(f"  SAFE Evaluation")
    print(f"  Model : {MODEL_HF_ID}")
    print(f"  SAE   : Goodfire {SAE_RELEASE} / {SAE_LAYER}")
    print(f"  N     : {len(examples)} examples")
    print(f"{'='*60}\n")

    out_path = Path("results/safe_eval.json")
    out_path.parent.mkdir(exist_ok=True)

    for i, ex in enumerate(examples):
        print(f"[{i+1}/{len(examples)}] {ex['question'][:70]}...")
        result = pipeline(ex['question'])

        p = embedder.encode(result['final_answer'])
        g = embedder.encode(ex['gold'])
        sim = float(np.dot(p, g) / (np.linalg.norm(p) * np.linalg.norm(g) + 1e-9))
        scores.append(sim)

        print(f"  Entropy: {result['first_entropy']:.3f}→{result['final_entropy']:.3f} | Loops: {result['enrichment_loops']} | Sim: {sim:.3f}")
        print(f"  Answer: {result['final_answer'][:100]}\n")
        results.append({**result, "gold": ex['gold'], "cosine_sim": round(sim, 4)})

        # Save incrementally
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": MODEL_HF_ID,
                "sae": SAE_RELEASE,
                "avg_cosine_sim": sum(scores)/len(scores),
                "results": results
            }, f, ensure_ascii=False, indent=2)

    avg = sum(scores)/len(scores)
    print(f"\n{'='*60}")
    print(f"  RESULTS — SAFE Pipeline")
    print(f"  Model          : {MODEL_HF_ID}")
    print(f"  SAE            : Goodfire (Neuronpedia labels)")
    print(f"  Avg cosine sim : {avg:.3f}")
    print(f"  ≥0.8           : {sum(1 for s in scores if s>=0.8)/len(scores):.1%}")
    print(f"  ≥0.6           : {sum(1 for s in scores if s>=0.6)/len(scores):.1%}")
    print(f"{'='*60}\n")
    print(f"Results saved → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question",  type=str, help="Single question")
    parser.add_argument("--test_file", type=str, default="data/processed/sft_test.jsonl")
    parser.add_argument("--limit",     type=int, default=None)
    args = parser.parse_args()

    if args.question:
        pipeline = SAFEPipeline()
        r = pipeline(args.question)
        print(f"\nOriginal : {r['original_question']}")
        print(f"Enriched : {r['enriched_question']}")
        print(f"Answer   : {r['final_answer']}")
        print(f"Entropy  : {r['first_entropy']:.3f} → {r['final_entropy']:.3f}")
        print(f"Loops    : {r['enrichment_loops']}")
    else:
        evaluate(args.test_file, args.limit)
