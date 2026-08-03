#!/usr/bin/env python3
"""
SAFE x DPO Evaluation Harness — RACA Legal LLM
================================================
Evaluates one checkpoint under one enrichment arm, and writes output that
`pav.py` can consume unchanged.

Why this replaces the api.py -> evaluate.py -> pav.py path for SAFE experiments:

  * One retriever, one generator, one prompt builder shared by every arm. The
    old path used api.py's chunked FAISS index for the baseline and
    safe_pipeline.py's unchunked 49-document index for SAFE, so any
    SAFE-vs-baseline delta was confounded by retrieval.
  * Records are keyed on the ORIGINAL question. SAFE rewrites the question, and
    pav.py matches predictions to provenance by exact question string
    (`predictions[ex["instruction"]]`). Keying on the enriched question would
    silently yield zero PAV-evaluable examples.
  * Final answers are greedy, so PAV differences reflect enrichment rather than
    sampling noise. Only the detection samples use temperature.
  * Feature-label loading fails loudly. The previous implementation cached an
    unusable Neuronpedia response and logged "Loaded 0 feature labels", which
    turned every subsequent SAFE run into a silent no-op.

Arms:
    none      Baseline. Greedy generation, no detection, no enrichment.
    safe      Full SAFE: entropy detection -> SAE feature diff -> NOTE steer.
    random    Ablation. Same trigger and loop budget as `safe`, but the feature
              is drawn uniformly from the diff set. Controls for "appending any
              NOTE at all" rather than the selected feature.
    generic   Ablation. Same trigger and loop budget, fixed generic instruction.
              Equivalent to the paper's "Simple Enrichment" baseline.

Usage:
    # 0. Calibrate the clustering threshold FIRST. Without this the detector
    #    fires on ~93% of questions and carries no signal.
    python safe_dpo_eval.py --model_path ./checkpoints/ft_raca_v5/merged \
        --calibrate --limit 40

    # 1. Run an arm.
    python safe_dpo_eval.py --model_path ./checkpoints/tmp_llama_pvdpo_seed42/merged \
        --arm safe --cluster_threshold 0.35 --entropy_threshold 0.9 \
        --output results/eval_llama_pvdpo_seed42_safe.json

    # 2. PAV is computed inline, but the file is also pav.py-compatible:
    python pav.py --pred_file results/eval_llama_pvdpo_seed42_safe.json
"""

import argparse
import csv
import glob
import json
import logging
import math
import os
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

from pav import PAV

# ── Defaults ───────────────────────────────────────────────────────────────────

SAE_RELEASE    = "goodfire-llama-3.1-8b-instruct"
SAE_LAYER      = "layer_19"
NEURONPEDIA_ID = "llama3.1-8b-it/19-resid-post-gf"
EMBED_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CSV_GLOB       = "./data/processed/raca_tab*.csv"
LABEL_CACHE    = Path("checkpoints/neuronpedia_labels.json")

SYSTEM_PROMPT = (
    "أنت مساعد قانوني متخصص في قوانين ولوائح هيئة تنظيم الأعمال الخيرية في قطر. "
    "عند الإجابة على الأسئلة القانونية، يجب عليك: "
    "1) ذكر رقم المادة أو البند القانوني بصيغة \"مادة (رقم)\" في بداية إجابتك دائماً. "
    "2) تقديم إجابة كاملة ومفصلة لا تقل عن ثلاثة أسطر. "
    "3) الاستناد حصراً إلى النصوص القانونية المقدمة في السياق. "
    "4) تجنب الإجابات المبهمة أو العامة. "
    "5) إذا لم تجد الإجابة في النصوص المقدمة، قل ذلك صراحةً بدلاً من الاختراع."
)

# The generic-arm steer, in Arabic. The reference implementation appends an
# English NOTE to Arabic prompts; if you keep the English form for the `safe`
# arm (Neuronpedia descriptions are English), run this arm in English too so the
# comparison is language-matched. See --steer_lang.
GENERIC_STEER_AR = "ملاحظة: فكر بعناية واستند فقط إلى النصوص القانونية المقدمة."
GENERIC_STEER_EN = "NOTE: think carefully and rely only on the provided legal texts."

log = logging.getLogger("safe_dpo_eval")


# ── Neuronpedia labels (fails loudly) ──────────────────────────────────────────

def load_neuronpedia_labels(neuronpedia_id: str, cache: Path,
                            api_key: str = None, min_labels: int = 1000) -> dict:
    """
    Fetch SAE feature auto-interpretations.

    Raises rather than returning an empty dict. A silently empty label map makes
    the whole SAFE pipeline a no-op that still produces plausible-looking logs.
    """
    data = None
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            log.info("Loaded label cache from %s", cache)
        except json.JSONDecodeError:
            log.warning("Label cache at %s is corrupt; refetching", cache)
            data = None

    if data is None:
        model_id, sae_id = neuronpedia_id.split("/")
        url = ("https://www.neuronpedia.org/api/explanation/export"
               f"?modelId={model_id}&saeId={sae_id}")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-Api-Key"] = api_key
        log.info("Fetching labels: %s", url)
        resp = requests.get(url, headers=headers, timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Neuronpedia returned HTTP {resp.status_code} for {neuronpedia_id}. "
                f"Body: {resp.text[:400]}\n"
                "If this is 401/403, set NEURONPEDIA_KEY. If 404, the "
                "modelId/saeId slug is wrong — check the SAE's page on "
                "neuronpedia.org and copy the slug from the URL."
            )
        data = resp.json()

    # The export has appeared both as a bare list and wrapped in an object.
    if isinstance(data, dict):
        for key in ("data", "explanations", "result", "results"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            raise RuntimeError(
                "Neuronpedia response was an object with no recognised list "
                f"field. Keys: {list(data.keys())[:10]}. This is the failure that "
                "produced 'Loaded 0 feature labels' — the old code iterated the "
                "dict's keys and silently matched nothing."
            )
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected Neuronpedia payload type: {type(data)}")

    labels = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        desc = item.get("description")
        if idx is None or not desc:
            continue
        labels[str(idx)] = desc.strip()

    if len(labels) < min_labels:
        raise RuntimeError(
            f"Only {len(labels)} usable feature labels for {neuronpedia_id} "
            f"(expected >= {min_labels}).\n"
            "SAFE cannot function without auto-interpretations: every feature "
            "diff will be empty and enrichment becomes a no-op.\n"
            "Options: (a) verify the slug on neuronpedia.org; (b) pick an SAE "
            "with auto-interp coverage; (c) generate your own descriptions and "
            "supply them via --label_file."
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    log.info("Loaded %d feature labels", len(labels))
    return labels


def load_labels_from_file(path: Path, min_labels: int = 1) -> dict:
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        labels = {str(k): v for k, v in raw.items() if v}
    else:
        labels = {str(i["index"]): i["description"] for i in raw
                  if i.get("index") is not None and i.get("description")}
    if len(labels) < min_labels:
        raise RuntimeError(f"{path} yielded {len(labels)} labels")
    log.info("Loaded %d feature labels from %s", len(labels), path)
    return labels


# ── Retrieval (chunked, matching api.py so arms are comparable) ────────────────

class Retriever:
    def __init__(self, csv_glob: str, embedder, top_k: int = 3,
                 chunk_size: int = 400, overlap: int = 50):
        self.top_k = top_k
        self.embedder = embedder
        self.docs = []
        csv.field_size_limit(10 ** 7)

        for path in sorted(glob.glob(csv_glob)):
            with open(path, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    text = (row.get("text") or "").strip()
                    title = (row.get("title") or "").strip()
                    if not text or len(text) < 100:
                        continue
                    words = text.split()
                    step = max(1, chunk_size - overlap)
                    for i in range(0, len(words), step):
                        chunk = " ".join(words[i:i + chunk_size])
                        if len(chunk) > 80:
                            self.docs.append({"title": title, "text": chunk})

        if not self.docs:
            raise RuntimeError(
                f"No documents matched {csv_glob}. Retrieval would be empty and "
                "every arm would degrade to closed-book generation."
            )
        log.info("Retriever: %d chunks", len(self.docs))

        import faiss
        texts = [f"{d['title']} {d['text'][:500]}" for d in self.docs]
        embs = embedder.encode(texts, show_progress_bar=False, batch_size=32)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs.astype(np.float32))

    def context(self, query: str) -> str:
        q = self.embedder.encode([query])
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        _, idxs = self.index.search(q.astype(np.float32), self.top_k)
        parts = []
        for rank, idx in enumerate(idxs[0], 1):
            if 0 <= idx < len(self.docs):
                d = self.docs[idx]
                parts.append(f"[مصدر {rank}] {d['title']}\n{d['text'][:800]}")
        return "\n\n".join(parts)


# ── Generation ─────────────────────────────────────────────────────────────────

class Generator:
    def __init__(self, model_path: str, max_new_tokens: int = 512):
        log.info("Loading %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager",
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def build_prompt(self, question: str, context: str = "") -> str:
        user = question
        if context:
            user = (f"بناءً على المصادر القانونية التالية:\n\n{context}\n\n"
                    f"السؤال: {question}")
        return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n")

    def generate(self, question: str, context: str = "",
                 temperature: float = 0.0, seed: int = None) -> str:
        prompt = self.build_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=3000).to(self.model.device)
        if seed is not None:
            torch.manual_seed(seed)
        sample = temperature > 0
        kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=sample,
                      pad_token_id=self.tokenizer.pad_token_id)
        if sample:
            kwargs.update(temperature=temperature, top_p=0.95)
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


# ── SAE feature extraction ─────────────────────────────────────────────────────

class Explainer:
    def __init__(self, generator: Generator, labels: dict,
                 tl_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 sae_release: str = SAE_RELEASE, sae_layer: str = SAE_LAYER,
                 device: str = "cuda"):
        from sae_lens import SAE, HookedSAETransformer

        log.info("Wrapping model in HookedSAETransformer")
        self.model = HookedSAETransformer.from_pretrained(
            tl_model_name, hf_model=generator.model,
            tokenizer=generator.tokenizer, dtype=torch.bfloat16,
            fold_ln=False, center_writing_weights=False,
            center_unembed=False, device=device,
        )
        log.info("Loading SAE %s / %s", sae_release, sae_layer)
        self.sae = SAE.from_pretrained(release=sae_release, sae_id=sae_layer).to(device)
        self.labels = labels
        self._cache = {}

    def features(self, text: str, n_results: int = 10, topk: int = 200) -> list:
        key = (text, n_results)
        if key in self._cache:
            return self._cache[key]
        with torch.no_grad():
            _, cache = self.model.run_with_cache_with_saes(
                text, saes=[self.sae], prepend_bos=True)
        hook = f"{self.sae.cfg.metadata['hook_name']}.hook_sae_acts_post"
        acts = cache[hook][0, 1:, :].sum(dim=0)
        del cache

        out = []
        for act, ind in zip(*acts.topk(topk)):
            desc = self.labels.get(str(int(ind.item())))
            if not desc:
                continue
            out.append({"index": int(ind.item()), "description": desc,
                        "activation": float(act.item())})
            if len(out) >= n_results:
                break
        self._cache[key] = out
        return out


# ── Detection ──────────────────────────────────────────────────────────────────

def semantic_entropy(assignments: list) -> float:
    total = len(assignments)
    if total == 0:
        return 0.0
    freq = Counter(assignments)
    return -sum((c / total) * math.log(c / total) for c in freq.values())


def cluster(embedder, question: str, answers: list, threshold: float) -> list:
    if len(answers) < 2:
        return [0] * len(answers)
    embs = embedder.encode([f"{question} [SEP] {a}" for a in answers],
                           show_progress_bar=False)
    dist = 1 - cosine_similarity(embs)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                    metric="precomputed", linkage="average")
    return list(model.fit_predict(dist))


# ── Enrichment arms ────────────────────────────────────────────────────────────

def pick_feature(arm: str, question: str, diff_features: list, embedder,
                 already_added: set, rng: random.Random):
    """Returns (feature_text, polarity) where polarity is 'exclude' or 'include'."""
    candidates = [f for f in diff_features if f not in already_added]
    if not candidates:
        return None, None

    if arm == "random":
        return rng.choice(candidates), rng.choice(["exclude", "include"])

    # `safe`: rank candidates by similarity to the question, then use the
    # dispersion of those similarities to decide polarity. Unlike the reference
    # implementation, the similarity vector and the candidate list are built from
    # the same filtered list, so indices cannot drift apart across iterations.
    q_emb = embedder.encode([question], show_progress_bar=False)
    f_emb = embedder.encode(candidates, show_progress_bar=False)
    sims = cosine_similarity(q_emb, f_emb)[0]

    if len(sims) >= 4:
        q1, q2 = np.percentile(sims, 25), np.percentile(sims, 50)
        lower = q1 - 1.5 * (q2 - q1)
        outlier = bool(np.any(sims < lower))
    else:
        outlier = False

    if outlier:
        return candidates[int(np.argmin(sims))], "exclude"
    return candidates[int(np.argmax(sims))], "include"


def apply_steer(question: str, feature: str, polarity: str, lang: str) -> str:
    if lang == "ar":
        head = "ملاحظة:"
        verb = "لا تأخذ في الاعتبار" if polarity == "exclude" else "يجب أن تأخذ في الاعتبار"
        joiner = "و"
    else:
        head = "NOTE:"
        verb = "do not consider" if polarity == "exclude" else "you must consider"
        joiner = "and"
    if head in question:
        return f"{question} {joiner} {verb} {feature}"
    return f"{question} - {head} {verb} {feature}"


# ── Per-question pipeline ──────────────────────────────────────────────────────

def run_question(question: str, arm: str, gen: Generator, retr: Retriever,
                 embedder, explainer, cfg, rng: random.Random) -> dict:
    context = retr.context(question) if retr else ""

    record = {
        "question": question,
        "enriched_question": question,
        "first_entropy": None,
        "final_entropy": None,
        "loops": 0,
        "triggered": False,
        "features_added": [],
    }

    if arm == "none":
        record["pred_steered"] = gen.generate(question, context, temperature=0.0)
        return record

    # Detection: sample n_answers at high temperature.
    answers = [gen.generate(question, context, temperature=cfg.high_temp,
                            seed=cfg.seed * 1000 + i)
               for i in range(cfg.n_answers)]
    entropy = semantic_entropy(cluster(embedder, question, answers,
                                       cfg.cluster_threshold))
    record["first_entropy"] = round(entropy, 4)
    record["final_entropy"] = round(entropy, 4)

    enriched = question
    added = set()

    for loop in range(1, cfg.max_loops + 1):
        if entropy <= cfg.entropy_threshold:
            break
        record["triggered"] = True
        record["loops"] = loop

        if arm == "generic":
            steer = GENERIC_STEER_AR if cfg.steer_lang == "ar" else GENERIC_STEER_EN
            if steer not in enriched:
                enriched = f"{enriched} - {steer}"
        else:
            q_feats = {f["description"] for f in explainer.features(enriched)}
            diff = set()
            for a in answers:
                diff |= {f["description"] for f in explainer.features(a)} - q_feats
            if not diff:
                log.warning("Empty feature diff at loop %d — no enrichment "
                            "possible for this question", loop)
                break
            feature, polarity = pick_feature(arm, question, sorted(diff),
                                             embedder, added, rng)
            if feature is None:
                break
            enriched = apply_steer(enriched, feature, polarity, cfg.steer_lang)
            added.add(feature)
            record["features_added"].append({"feature": feature,
                                             "polarity": polarity})

        answers = [gen.generate(enriched, context, temperature=cfg.high_temp,
                                seed=cfg.seed * 1000 + loop * 100 + i)
                   for i in range(cfg.n_answers)]
        # Cluster against the ORIGINAL question so entropy stays comparable
        # across iterations.
        entropy = semantic_entropy(cluster(embedder, question, answers,
                                           cfg.cluster_threshold))
        record["final_entropy"] = round(entropy, 4)

    record["enriched_question"] = enriched
    record["pred_steered"] = gen.generate(enriched, context, temperature=0.0)
    return record


# ── Calibration mode ───────────────────────────────────────────────────────────

def calibrate(examples, gen, retr, embedder, cfg):
    """
    Sweep the agglomerative distance threshold and report the resulting entropy
    distribution. Pick a threshold whose trigger rate is somewhere near the
    middle, not 93%.
    """
    print(f"\nSampling {cfg.n_answers} answers for {len(examples)} questions "
          f"at T={cfg.high_temp}...")
    per_q = []
    for i, ex in enumerate(examples, 1):
        ctx = retr.context(ex["question"]) if retr else ""
        answers = [gen.generate(ex["question"], ctx, temperature=cfg.high_temp,
                                seed=cfg.seed * 1000 + j)
                   for j in range(cfg.n_answers)]
        per_q.append((ex["question"], answers))
        print(f"  [{i}/{len(examples)}]", end="\r")

    ceiling = math.log(cfg.n_answers)
    print(f"\n\nEntropy ceiling for n={cfg.n_answers}: {ceiling:.3f}\n")
    header = f"{'thresh':>7} {'mean H':>8} {'median':>8} {'at ceil':>9} {'trigger@thr':>12}"
    print(header)
    print("-" * len(header))

    rows = []
    for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        Hs = [semantic_entropy(cluster(embedder, q, a, thr)) for q, a in per_q]
        at_ceiling = sum(1 for h in Hs if h >= ceiling - 1e-6) / len(Hs)
        trig = sum(1 for h in Hs if h > cfg.entropy_threshold) / len(Hs)
        print(f"{thr:>7.2f} {np.mean(Hs):>8.3f} {np.median(Hs):>8.3f} "
              f"{at_ceiling:>8.0%} {trig:>11.0%}")
        rows.append({"threshold": thr, "mean_entropy": float(np.mean(Hs)),
                     "median_entropy": float(np.median(Hs)),
                     "frac_at_ceiling": at_ceiling, "trigger_rate": trig})

    print("\nPick a threshold where 'at ceil' is low and 'trigger@thr' is "
          "moderate (roughly 30-60%). A trigger rate near 100% means the "
          "detector carries no information and, per the paper's Ablation B, "
          "indiscriminate enrichment degrades performance.\n")

    Path("results").mkdir(exist_ok=True)
    out = Path("results/entropy_calibration.json")
    out.write_text(json.dumps({"n_answers": cfg.n_answers, "ceiling": ceiling,
                               "entropy_threshold": cfg.entropy_threshold,
                               "sweep": rows}, indent=2))
    print(f"Saved → {out}")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_test(path: str, limit: int = None, require_article: bool = True) -> list:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            prov = ex.get("provenance_text", "")
            # pav.run_evaluation applies this same filter. Replicated here so the
            # inline PAV numbers match a subsequent `python pav.py` run.
            if require_article and not re.search(r"مادة\s*[\(\[]?\d", prov):
                continue
            examples.append({"question": ex["instruction"],
                             "gold": ex["response"],
                             "provenance_text": prov})
    if limit:
        examples = examples[:limit]
    return examples


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--arm", default="none",
                   choices=["none", "safe", "random", "generic"])
    p.add_argument("--test_file", default="data/processed/sft_test.jsonl")
    p.add_argument("--train_file", default="data/processed/sft_train.jsonl")
    p.add_argument("--output", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n_answers", type=int, default=5)
    p.add_argument("--high_temp", type=float, default=1.0)
    p.add_argument("--entropy_threshold", type=float, default=0.9)
    p.add_argument("--cluster_threshold", type=float, default=0.35,
                   help="Agglomerative distance threshold. Run --calibrate "
                        "before trusting this; 0.1 saturates on Arabic.")
    p.add_argument("--max_loops", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--steer_lang", default="en", choices=["en", "ar"],
                   help="Language of the injected NOTE. Neuronpedia "
                        "descriptions are English, so 'en' keeps the steer "
                        "internally consistent; 'ar' matches the query "
                        "language. Worth reporting both.")
    p.add_argument("--no_rag", action="store_true")
    p.add_argument("--tau", type=float, default=0.3, help="CITSUPPORT threshold")
    p.add_argument("--label_file", default=None,
                   help="Local JSON of feature labels, bypassing Neuronpedia")
    p.add_argument("--neuronpedia_id", default=NEURONPEDIA_ID)
    p.add_argument("--calibrate", action="store_true")
    cfg = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("safe_dpo_eval.log"), logging.StreamHandler()],
    )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    gen = Generator(cfg.model_path, max_new_tokens=cfg.max_new_tokens)
    retr = None if cfg.no_rag else Retriever(CSV_GLOB, embedder)

    examples = load_test(cfg.test_file, cfg.limit)
    if not examples:
        raise RuntimeError(
            f"No examples in {cfg.test_file} had article-numbered provenance. "
            "PAV cannot be computed."
        )
    log.info("%d PAV-evaluable examples", len(examples))

    if cfg.calibrate:
        calibrate(examples, gen, retr, embedder, cfg)
        return

    explainer = None
    if cfg.arm in ("safe", "random"):
        if cfg.label_file:
            labels = load_labels_from_file(Path(cfg.label_file))
        else:
            labels = load_neuronpedia_labels(
                cfg.neuronpedia_id, LABEL_CACHE,
                api_key=os.environ.get("NEURONPEDIA_KEY"))
        explainer = Explainer(gen, labels, device=device)

    pav = PAV(train_file=cfg.train_file, tau=cfg.tau)

    output = Path(cfg.output or f"results/eval_{Path(cfg.model_path).parent.name}_{cfg.arm}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    results, pav_rows = [], []
    for i, ex in enumerate(examples, 1):
        rec = run_question(ex["question"], cfg.arm, gen, retr, embedder,
                           explainer, cfg, rng)

        v = pav.verify(ex["question"], rec["pred_steered"], ex["provenance_text"])
        pav_rows.append(v)

        g = embedder.encode(ex["gold"], show_progress_bar=False)
        pr = embedder.encode(rec["pred_steered"], show_progress_bar=False)
        sim = float(np.dot(pr, g) / (np.linalg.norm(pr) * np.linalg.norm(g) + 1e-9))

        rec.update(gold=ex["gold"], provenance_text=ex["provenance_text"][:200],
                   sim_steered=round(sim, 4), **v)
        results.append(rec)

        print(f"[{i}/{len(examples)}] H={rec['first_entropy']}→"
              f"{rec['final_entropy']} loops={rec['loops']} "
              f"PAV={v['pav']} sim={sim:.3f}")

        # Incremental save: these runs are long and interruptible.
        summary = pav.summary(pav_rows)
        summary["avg_cosine_sim"] = float(np.mean([r["sim_steered"] for r in results]))
        summary["trigger_rate"] = float(np.mean([r["triggered"] for r in results]))
        summary["avg_loops"] = float(np.mean([r["loops"] for r in results]))
        n_enriched = sum(1 for r in results if r["features_added"])
        summary["frac_actually_enriched"] = n_enriched / len(results)
        output.write_text(json.dumps(
            {"model": cfg.model_path, "arm": cfg.arm, "seed": cfg.seed,
             "config": vars(cfg), "summary": summary, "results": results},
            ensure_ascii=False, indent=2))

    s = json.loads(output.read_text())["summary"]
    print(f"\n{'='*62}")
    print(f"  {cfg.arm.upper()}  |  {cfg.model_path}  |  seed {cfg.seed}")
    print(f"{'='*62}")
    print(f"  n                    : {s['n']}")
    print(f"  CITPRESENCE          : {s['citpresence']:.1%}")
    print(f"  CITMATCH             : {s['citmatch']:.1%}")
    print(f"  CITSUPPORT           : {s['citsupport']:.1%}")
    print(f"  PAV                  : {s['pav']:.1%}")
    print(f"  Avg cosine sim       : {s['avg_cosine_sim']:.3f}")
    print(f"  Avg answer length    : {s['avg_answer_len']:.1f}")
    print(f"  Trigger rate         : {s['trigger_rate']:.1%}")
    print(f"  Avg loops            : {s['avg_loops']:.2f}")
    print(f"  Actually enriched    : {s['frac_actually_enriched']:.1%}")
    print(f"{'='*62}")
    if cfg.arm in ("safe", "random") and s["frac_actually_enriched"] < 0.05:
        print("  WARNING: almost nothing was enriched. This arm is a no-op and\n"
              "  its numbers are indistinguishable from --arm none.")
    print(f"\nSaved → {output}\n")


if __name__ == "__main__":
    main()
