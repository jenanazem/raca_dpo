import json
import numpy as np
from scipy import stats
from pav import PAV, extract_citations, normalize_arabic, template_ngram_rate

N_BOOTSTRAP = 2000
SEED = 42

def surface_features(answer, template_ngrams):
    tokens = normalize_arabic(answer).split()
    length = len(tokens)
    citations = len(extract_citations(answer))
    tmpl_rate = template_ngram_rate(answer, template_ngrams)
    return length, citations, tmpl_rate

def load_model_data(model_name, train_file="data/processed/sft_train.jsonl"):
    pool_file = f"data/processed/{model_name}_pool_dataset.jsonl"
    pav = PAV(train_file=train_file, tau=0.3)

    questions = []  # each entry: dict of feature-name -> list of K values
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
            q_record = {
                "cosine": [], "length": [], "template_rate": [],
                "citation_count": [], "citmatch": [], "sup_score": [], "pav": [],
            }
            for item in ex["pool"]:
                ans = item["answer"]
                cos = item["score"]
                length, citations, tmpl = surface_features(ans, pav.template_ngrams)
                pav_result = pav.verify(question, ans, provenance)
                q_record["cosine"].append(cos)
                q_record["length"].append(length)
                q_record["template_rate"].append(tmpl)
                q_record["citation_count"].append(citations)
                q_record["citmatch"].append(int(pav_result["citmatch"]))
                q_record["sup_score"].append(pav_result["sup_score"])
                q_record["pav"].append(int(pav_result["pav"]))
            questions.append(q_record)
    return questions

def compute_correlation(questions, feature_key, rng, resample=True):
    n_q = len(questions)
    if resample:
        idx = rng.integers(0, n_q, size=n_q)
        sampled = [questions[i] for i in idx]
    else:
        sampled = questions

    cosines, fvals = [], []
    for q in sampled:
        cosines.extend(q["cosine"])
        fvals.extend(q[feature_key])

    if len(set(fvals)) < 2 or len(set(cosines)) < 2:
        return np.nan
    r, _ = stats.pearsonr(cosines, fvals)
    return r

def main():
    feature_names = {
        "Length": "length",
        "Template rate": "template_rate",
        "Citation tokens": "citation_count",
        "CITMATCH": "citmatch",
        "Sup score": "sup_score",
        "PAV": "pav",
    }

    print("Loading data for all three models (this runs PAV verification once per sample)...")
    model_data = {}
    for model in ["llama", "qwen", "fanar"]:
        print(f"  loading {model}...")
        model_data[model] = load_model_data(model)

    rng = np.random.default_rng(SEED)

    print(f"\nRunning {N_BOOTSTRAP} bootstrap iterations...")
    boot_results = {fname: [] for fname in feature_names}

    for b in range(N_BOOTSTRAP):
        for fname, fkey in feature_names.items():
            per_model_r = []
            for model in ["llama", "qwen", "fanar"]:
                r = compute_correlation(model_data[model], fkey, rng, resample=True)
                per_model_r.append(r)
            mean_r = np.nanmean(per_model_r)
            boot_results[fname].append(mean_r)
        if (b + 1) % 500 == 0:
            print(f"  {b+1}/{N_BOOTSTRAP} done")

    print("\n=== POINT ESTIMATE (no resampling) AND 95% BOOTSTRAP CI ===")
    print(f"{'Feature':<20} {'point':>10} {'2.5%':>10} {'97.5%':>10}")
    print("-" * 55)
    for fname, fkey in feature_names.items():
        point_per_model = [compute_correlation(model_data[m], fkey, rng, resample=False) for m in ["llama", "qwen", "fanar"]]
        point = np.nanmean(point_per_model)
        arr = np.array(boot_results[fname])
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        print(f"{fname:<20} {point:>10.4f} {lo:>10.4f} {hi:>10.4f}")

if __name__ == "__main__":
    main()
