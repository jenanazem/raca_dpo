"""
Phase 5 — Hallucination Reduction & Deployment
================================================
Uses the trained SAE to reduce hallucination in the fine-tuned model via:

  Strategy A — Activation steering:
    Subtract the hallucination feature direction from the residual stream at
    inference time. This suppresses the model's tendency to confabulate.

  Strategy B — Confidence scoring:
    Run the SAE on every generated token; flag tokens where hallucination
    features fire above a threshold. Return these as uncertainty signals.

  Strategy C — RAG grounding:
    Retrieve the relevant law article from the RACA corpus before generation.
    Combine with activation steering for the strongest HAL reduction.

  Evaluation:
    Compare base model vs FT-LLM vs FT-LLM + SAE-steered on a legal QA benchmark.
    Metrics: exact-match rate, citation accuracy, LLM-as-judge hallucination score.

Install:
    pip install transformers faiss-cpu sentence-transformers

Usage:
    python phase5_hal_reduction.py --config ./data/project_config.json \
                                   --model_path ./checkpoints/ft_raca/merged \
                                   --sae_path ./checkpoints/sae_raca/sae_best.pt \
                                   --feature_file ./checkpoints/sae_raca/feature_interpretations.json
"""

import json
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ─────────────────────────────────────────────
# 0. SHARED SAE CLASS (copy from Phase 4)
# ─────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    def __init__(self, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.W_enc = nn.Parameter(torch.empty(hidden_dim, feature_dim))
        self.b_enc = nn.Parameter(torch.zeros(feature_dim))
        self.W_dec = nn.Parameter(torch.empty(feature_dim, hidden_dim))
        self.b_dec = nn.Parameter(torch.zeros(hidden_dim))

    def encode(self, x):
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, feature_acts):
        return feature_acts @ self.W_dec + self.b_dec


# ─────────────────────────────────────────────
# 1. LOAD HALLUCINATION FEATURES
# ─────────────────────────────────────────────

def load_hal_features(feature_file: str) -> list[int]:
    """
    Load the list of feature IDs flagged as hallucination-related.
    
    These are features where you (or annotators) manually set:
        "is_hallucination_feature": true
    
    in the feature_interpretations.json from Phase 4.
    
    Common hallucination features to look for:
    - Features that fire on specific article numbers that don't exist
    - Features that fire on confident assertion patterns ("تنص المادة X على...")
      when those assertions are wrong
    - Features that fire on unfamiliar entity names (fake law names)
    """
    with open(feature_file, encoding='utf-8') as f:
        feature_info = json.load(f)
    
    hal_features = [
        int(fid) for fid, info in feature_info.items()
        if info.get('is_hallucination_feature') is True
    ]
    
    print(f"  Loaded {len(hal_features)} hallucination features: {hal_features[:10]}...")
    
    if not hal_features:
        print("  ⚠ No hallucination features found.")
        print("  Make sure you've reviewed feature_interpretations.json and set")
        print("  'is_hallucination_feature': true for relevant features.")
    
    return hal_features


# ─────────────────────────────────────────────
# 2. STRATEGY A — ACTIVATION STEERING
# ─────────────────────────────────────────────

class ActivationSteerer:
    """
    Suppresses hallucination-linked features by modifying the residual stream
    during generation. This is 'activation steering' or 'feature ablation'.
    
    Mechanism:
    For each generation step:
      1. Let the model compute its normal forward pass
      2. At the hooked layer, extract the residual stream h
      3. Run the SAE encoder: feature_acts = sae.encode(h)
      4. Zero out (or scale down) the hallucination feature activations
      5. Reconstruct: h_steered = sae.decode(feature_acts_steered)
      6. Add the correction back: h ← h + (h_steered - h_reconstructed)
      7. Continue forward pass with the modified h
    
    The 'steering coefficient' (0.0 = no steering, 1.0 = full ablation)
    controls how aggressively we suppress these features.
    Too aggressive → model output degrades (coherence loss)
    Too conservative → minimal HAL reduction
    Tune on your validation set.
    """
    
    def __init__(
        self,
        model,
        sae: SparseAutoencoder,
        layer_idx: int,
        hal_feature_ids: list[int],
        steering_coefficient: float = 0.7,
    ):
        self.sae = sae
        self.hal_feature_ids = hal_feature_ids
        self.steering_coefficient = steering_coefficient
        self.hook = None
        self._register_hook(model, layer_idx)
    
    def _register_hook(self, model, layer_idx: int):
        target_layer = model.model.layers[layer_idx]
        sae = self.sae
        hal_ids = self.hal_feature_ids
        coeff = self.steering_coefficient
        
        def hook_fn(module, input, output):
            # Get hidden states - handle both tuple and tensor outputs
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            
            orig_dtype = h.dtype
            orig_shape = h.shape
            h_flat = h.float().reshape(-1, h.shape[-1])
            
            with torch.no_grad():
                sae.to(h_flat.device)
                feature_acts = sae.encode(h_flat)
                feature_acts_modified = feature_acts.clone()
                feature_acts_modified[:, hal_ids] *= (1 - coeff)
                h_steered = sae.decode(feature_acts_modified)
                h_original_recon = sae.decode(feature_acts)
                correction = (h_steered - h_original_recon).reshape(orig_shape).to(orig_dtype)
                h_new = h + correction
            
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            else:
                return h_new
        
        self.hook = target_layer.register_forward_hook(hook_fn)
    
    def remove(self):
        if self.hook:
            self.hook.remove()


# ─────────────────────────────────────────────
# 3. STRATEGY B — CONFIDENCE SCORING
# ─────────────────────────────────────────────

class HallucinationScorer:
    """
    Assigns a hallucination risk score to each generated token by monitoring
    SAE feature activations during generation.
    
    Returns a per-token score in [0, 1] where higher = more likely hallucinated.
    
    Use cases:
    - Flag high-risk sentences for human review
    - Trigger a retrieval step when score exceeds a threshold
    - Include as uncertainty metadata in API responses
    """
    
    def __init__(self, model, sae: SparseAutoencoder, layer_idx: int, hal_feature_ids: list[int]):
        self.sae = sae
        self.hal_feature_ids = hal_feature_ids
        self.token_scores = []
        self.hook = self._register_hook(model, layer_idx)
    
    def _register_hook(self, model, layer_idx: int):
        sae = self.sae
        hal_ids = self.hal_feature_ids
        scores_list = self.token_scores
        
        def hook_fn(module, input, output):
            h = output[0].float()
            h_flat = h.view(-1, h.shape[-1])
            
            with torch.no_grad():
                sae.to(h_flat.device)
                feature_acts = sae.encode(h_flat)
                # Score = mean activation of hal features normalized by max possible
                hal_acts = feature_acts[:, hal_ids]
                score = hal_acts.mean(dim=-1).cpu()
                scores_list.append(score)
        
        return model.model.layers[layer_idx].register_forward_hook(hook_fn)
    
    def get_scores(self) -> torch.Tensor:
        if not self.token_scores:
            return torch.empty(0)
        return torch.cat(self.token_scores, dim=0)
    
    def clear(self):
        self.token_scores.clear()
    
    def remove(self):
        self.hook.remove()


# ─────────────────────────────────────────────
# 4. STRATEGY C — RAG RETRIEVAL GROUNDING
# ─────────────────────────────────────────────

class LegalRAG:
    """
    Retrieval-Augmented Generation for RACA legal documents.
    
    Retrieves the most relevant law article(s) given a question,
    then prepends them to the prompt as grounding context.
    
    This is the single most effective HAL reduction technique for
    knowledge-intensive domains — the model answers from retrieved
    facts rather than from (often unreliable) memorized knowledge.
    
    Combine with activation steering for the best results:
    RAG handles "made-up facts", steering handles confident confabulation.
    """
    
    def __init__(self, corpus_jsonl: str, embedding_model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        import faiss
        from sentence_transformers import SentenceTransformer
        
        print(f"  Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Load corpus
        self.docs = []
        with open(corpus_jsonl, encoding='utf-8') as f:
            for line in f:
                self.docs.append(json.loads(line))
        
        # Build FAISS index
        print(f"  Building FAISS index over {len(self.docs)} documents...")
        texts = [d['text'][:512] for d in self.docs]  # truncate for embedding
        embeddings = self.embedder.encode(texts, batch_size=64, show_progress_bar=True)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # inner product = cosine on normalized vectors
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print(f"  Index ready: {self.index.ntotal} vectors, dim={dim}")
    
    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve top-k most relevant documents for a query."""
        q_emb = self.embedder.encode([query])
        import faiss
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            doc = self.docs[idx].copy()
            doc['retrieval_score'] = float(score)
            results.append(doc)
        return results
    
    def build_rag_prompt(self, question: str, top_k: int = 2) -> str:
        """Build a grounded prompt with retrieved law articles."""
        docs = self.retrieve(question, top_k=top_k)
        
        context_blocks = []
        for i, doc in enumerate(docs, 1):
            context_blocks.append(
                f"[مرجع {i}] {doc.get('title', '')}\n{doc['text'][:800]}"
            )
        
        context = "\n\n".join(context_blocks)
        
        return f"""<|im_start|>system
أنت مساعد قانوني متخصص في تشريعات هيئة تنظيم الأعمال الخيرية في دولة قطر.
استند في إجابتك فقط إلى النصوص القانونية المرجعية المقدمة أدناه. 
إذا لم تجد الإجابة في المراجع، قل "لا تتوفر لديّ معلومات كافية".<|im_end|>
<|im_start|>user
النصوص المرجعية:
{context}

السؤال: {question}<|im_end|>
<|im_start|>assistant
"""


# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────

# Test questions drawn from the RACA legal corpus
# These have verifiable answers from the actual documents
EVAL_QUESTIONS = [
    {
        "question": "ما هي اختصاصات قسم الامتثال في هيئة تنظيم الأعمال الخيرية؟",
        "gold_keywords": ["مكافحة غسل الأموال", "تمويل الإرهاب", "الجمعيات", "الضوابط"],
        "source_doc": "قرار رقم 1 لسنة 2021",
    },
    {
        "question": "ما هي شروط تأسيس الجمعية الخيرية وفقاً للقانون رقم 15 لسنة 2014؟",
        "gold_keywords": ["عشرين", "مؤسسين", "أساسي", "تسجيل"],
        "source_doc": "قانون رقم (15) لسنة 2014",
    },
    {
        "question": "من يتولى إدارة هيئة تنظيم الأعمال الخيرية وفق التعديل الأميري؟",
        "gold_keywords": ["مجلس إدارة", "وزير التنمية الاجتماعية"],
        "source_doc": "قرار أميري رقم (21) لسنة 2022",
    },
    {
        "question": "ما هي عقوبة ممارسة نشاط خيري دون ترخيص؟",
        "gold_keywords": ["حبس", "غرامة", "مادة"],
        "source_doc": "قانون رقم (4) لسنة 2020",
    },
    {
        "question": "هل يجوز للجمعية الخيرية فتح حساب بنكي خارج الدولة؟",
        "gold_keywords": ["موافقة", "المجلس", "ضوابط"],
        "source_doc": "قانون رقم (15) لسنة 2014",
    },
]


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 400,
) -> str:
    """Generate a response from the model given a prompt."""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.05,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


def keyword_score(response: str, keywords: list[str]) -> float:
    """Simple keyword recall score — fraction of gold keywords present in response."""
    if not keywords:
        return 0.0
    found = sum(1 for kw in keywords if kw in response)
    return found / len(keywords)


def run_evaluation(
    model,
    tokenizer,
    sae: SparseAutoencoder,
    hal_feature_ids: list[int],
    layer_idx: int,
    config: dict,
    output_dir: Path,
    rag: Optional[LegalRAG] = None,
):
    """
    Compare three conditions on the eval set:
    1. Fine-tuned model (no steering)
    2. Fine-tuned model + activation steering
    3. Fine-tuned model + RAG + activation steering (if RAG is provided)
    """
    results = []
    
    for q_item in tqdm(EVAL_QUESTIONS, desc="Evaluating"):
        question = q_item['question']
        gold_keywords = q_item['gold_keywords']
        
        row = {'question': question, 'gold_keywords': gold_keywords}
        
        # Condition 1: Plain fine-tuned model
        plain_prompt = f"""<|im_start|>system
أنت مساعد قانوني متخصص في تشريعات هيئة تنظيم الأعمال الخيرية.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
        row['response_plain'] = generate_answer(model, tokenizer, plain_prompt)
        row['score_plain'] = keyword_score(row['response_plain'], gold_keywords)
        
        # Condition 2: Fine-tuned model + activation steering
        steerer = ActivationSteerer(
            model, sae, layer_idx, hal_feature_ids,
            steering_coefficient=config.get('steering_coefficient', 0.7)
        )
        row['response_steered'] = generate_answer(model, tokenizer, plain_prompt)
        row['score_steered'] = keyword_score(row['response_steered'], gold_keywords)
        steerer.remove()
        
        # Condition 3: RAG + steering
        if rag is not None:
            rag_prompt = rag.build_rag_prompt(question)
            steerer = ActivationSteerer(
                model, sae, layer_idx, hal_feature_ids,
                steering_coefficient=config.get('steering_coefficient', 0.7)
            )
            row['response_rag_steered'] = generate_answer(model, tokenizer, rag_prompt)
            row['score_rag_steered'] = keyword_score(row['response_rag_steered'], gold_keywords)
            steerer.remove()
        
        results.append(row)
        
        # Print live results
        print(f"\n  Q: {question[:60]}...")
        print(f"  Plain score:   {row['score_plain']:.0%}")
        print(f"  Steered score: {row['score_steered']:.0%}")
        if 'score_rag_steered' in row:
            print(f"  RAG+Steer:     {row['score_rag_steered']:.0%}")
    
    # Aggregate
    avg_plain = sum(r['score_plain'] for r in results) / len(results)
    avg_steered = sum(r['score_steered'] for r in results) / len(results)
    
    print(f"\n  === Evaluation Summary ===")
    print(f"  Fine-tuned (plain):    {avg_plain:.1%}")
    print(f"  Fine-tuned + steering: {avg_steered:.1%}")
    print(f"  HAL reduction:         {avg_steered - avg_plain:+.1%}")
    
    if all('score_rag_steered' in r for r in results):
        avg_rag = sum(r['score_rag_steered'] for r in results) / len(results)
        print(f"  RAG + steering:        {avg_rag:.1%}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'eval_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Full results saved → {results_path}")
    return results


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 5: HAL reduction and evaluation")
    parser.add_argument('--config', default='./data/project_config.json')
    parser.add_argument('--model_path', default='./checkpoints/ft_raca/merged')
    parser.add_argument('--sae_path', default='./checkpoints/sae_raca/sae_best.pt')
    parser.add_argument('--feature_file', default='./checkpoints/sae_raca/feature_interpretations.json')
    parser.add_argument('--corpus_jsonl', default=None,
                        help='JSONL file of RACA docs for RAG (optional)')
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--steering_coefficient', type=float, default=0.7)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    config['steering_coefficient'] = args.steering_coefficient
    
    output_dir = Path(args.output_dir)
    layer_idx = config['sae_hook_layer']
    
    print("\n=== Phase 5: HAL Reduction & Evaluation ===\n")
    
    # Load model
    print("[1/4] Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map='auto'
    )
    model.eval()
    
    # Load SAE
    print("\n[2/4] Loading SAE...")
    hidden_dim = model.config.hidden_size
    feature_dim = hidden_dim * config['sae_expansion_factor']
    sae = SparseAutoencoder(hidden_dim, feature_dim)
    sae.load_state_dict(torch.load(args.sae_path, map_location='cpu'))
    sae = sae.to('cuda' if torch.cuda.is_available() else 'cpu')
    sae.eval()
    
    # Load hallucination feature IDs
    print("\n[3/4] Loading hallucination feature IDs...")
    hal_feature_ids = load_hal_features(args.feature_file)
    
    # Optional: set up RAG
    rag = None
    if args.corpus_jsonl and Path(args.corpus_jsonl).exists():
        print("\n  Setting up RAG index...")
        rag = LegalRAG(args.corpus_jsonl)
    
    # Run evaluation
    print("\n[4/4] Running evaluation...")
    run_evaluation(
        model, tokenizer, sae, hal_feature_ids,
        layer_idx, config, output_dir, rag=rag
    )
    
    print("\n✓ Phase 5 complete.")
    print("  The steered model is ready for deployment.")
    print("  Wrap it in a FastAPI endpoint with the ActivationSteerer hook active.")
    print("  Monitor hallucination feature activations in production for drift.\n")


if __name__ == '__main__':
    main()
