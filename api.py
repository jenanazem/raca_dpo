"""
RACA Legal LLM — FastAPI Inference Server with RAG
====================================================
Serves the fine-tuned Llama model with SAE-based hallucination steering
and RAG (Retrieval Augmented Generation) from the legal CSV documents.

Usage:
    pip install fastapi uvicorn sentence-transformers faiss-cpu
    python api.py

Endpoints:
    POST /ask          — Ask a legal question (RAG + steering)
    POST /ask/compare  — Compare plain vs steered responses
    GET  /health       — Health check
"""

import json
import time
import torch
import csv
csv.field_size_limit(10**7)
import glob
import numpy as np
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH     = "./checkpoints/ft_llama_matched_surface/merged"
SAE_PATH       = None
FEATURE_FILE   = "./checkpoints/sae_raca/feature_interpretations.json"
CSV_GLOB       = "./data/processed/raca_tab*.csv"
HOOK_LAYER     = 16
STEER_COEFF    = 0.5
MAX_NEW_TOKENS = 512
TOP_K_DOCS     = 3   # number of documents to retrieve per question

# ── SAE definition ────────────────────────────────────────────────────────────

class SparseAutoencoder(torch.nn.Module):
    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.empty(d_model, d_sae))
        self.W_dec = torch.nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))

    def encode(self, x):
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        return self.decode(self.encode(x))


# ── Activation steerer ────────────────────────────────────────────────────────

class ActivationSteerer:
    def __init__(self, model, sae, hal_feature_ids, layer_idx, coeff):
        self.sae = sae
        self.hal_ids = hal_feature_ids
        self.coeff = coeff
        self.hook = None
        self._register(model, layer_idx)

    def _register(self, model, layer_idx):
        target = model.model.layers[layer_idx]
        sae = self.sae
        hal_ids = self.hal_ids
        coeff = self.coeff

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            orig_dtype = h.dtype
            orig_shape = h.shape
            h_flat = h.float().reshape(-1, h.shape[-1])
            with torch.no_grad():
                sae.to(h_flat.device)
                feat = sae.encode(h_flat)
                feat_mod = feat.clone()
                feat_mod[:, hal_ids] *= (1 - coeff)
                correction = (sae.decode(feat_mod) - sae.decode(feat)).reshape(orig_shape).to(orig_dtype)
                h_new = h + correction
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new

        self.hook = target.register_forward_hook(hook_fn)

    def remove(self):
        if self.hook:
            self.hook.remove()
            self.hook = None


# ── RAG: load documents and build index ───────────────────────────────────────

class RAGRetriever:
    def __init__(self, csv_glob: str, top_k: int = 3):
        self.top_k = top_k
        self.docs = []
        self._load_docs(csv_glob)
        self._build_index()

    def _load_docs(self, csv_glob: str):
        print("Loading legal documents...")
        chunk_size = 400
        overlap = 50
        for path in sorted(glob.glob(csv_glob)):
            with open(path, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    text = row.get("text", "").strip()
                    title = row.get("title", "").strip()
                    section = row.get("section", "").strip()
                    if not text or len(text) < 100:
                        continue
                    # split into overlapping chunks
                    words = text.split()
                    for i in range(0, len(words), chunk_size - overlap):
                        chunk = " ".join(words[i:i + chunk_size])
                        if len(chunk) > 80:
                            self.docs.append({
                                "title": title,
                                "section": section,
                                "text": chunk,
                            })
        print(f"Loaded {len(self.docs)} chunks from documents")

    def _build_index(self):
        print("Building search index...")
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("⚠ faiss or sentence-transformers not installed — RAG disabled")
            self.index = None
            self.embedder = None
            return

        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        texts = [f"{d['title']} {d['text'][:500]}" for d in self.docs]
        embeddings = self.embedder.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        print("Index ready.")

    def retrieve(self, query: str) -> list[dict]:
        if self.index is None:
            return []
        q_emb = self.embedder.encode([query])
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        _, indices = self.index.search(q_emb.astype(np.float32), self.top_k)
        return [self.docs[i] for i in indices[0] if i < len(self.docs)]

    def format_context(self, docs: list[dict]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[مصدر {i}] {doc['title']}\n{doc['text'][:800]}")
        return "\n\n".join(parts)


# ── Load everything ───────────────────────────────────────────────────────────

print("Loading model and SAE...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

if SAE_PATH is None:
    hal_ids = []
    sae = None
    print("SAE disabled — no hallucination steering")
else:
    checkpoint = torch.load(SAE_PATH, map_location="cpu")
    d_model = checkpoint["W_enc"].shape[0]
    d_sae   = checkpoint["W_enc"].shape[1]
    sae = SparseAutoencoder(d_model, d_sae)
    sae.load_state_dict(checkpoint)
    sae.eval()
    with open(FEATURE_FILE) as f:
        features = json.load(f)
    hal_ids = [
        int(fid) for fid, fdata in features.items()
        if fdata.get("is_hallucination_feature") is True
    ]
    print(f"Loaded {len(hal_ids)} hallucination features to suppress")

rag = RAGRetriever(CSV_GLOB, top_k=TOP_K_DOCS)

print("Ready.")


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(history: list, question: str, context: str = "") -> str:
    system = """أنت مساعد قانوني متخصص في قوانين ولوائح هيئة تنظيم الأعمال الخيرية في قطر. عند الإجابة على الأسئلة القانونية، يجب عليك: 1) ذكر رقم المادة أو البند القانوني بصيغة "مادة (رقم)" في بداية إجابتك دائماً. 2) تقديم إجابة كاملة ومفصلة لا تقل عن ثلاثة أسطر. 3) الاستناد حصراً إلى النصوص القانونية المقدمة في السياق. 4) تجنب الإجابات المبهمة أو العامة. 5) إذا لم تجد الإجابة في النصوص المقدمة، قل ذلك صراحةً بدلاً من الاختراع."""
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
    for turn in history:
        prompt += f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n"
    user_msg = question
    if context:
        user_msg = f"بناءً على المصادر القانونية التالية:\n\n{context}\n\nالسؤال: {question}"
    prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


# ── Generation ────────────────────────────────────────────────────────────────

def generate(question: str, history: list, max_tokens: int, use_steering: bool, coeff: float, use_rag: bool = True) -> tuple[str, float]:
    context = ""
    if use_rag:
        docs = rag.retrieve(question)
        context = rag.format_context(docs)

    prompt = build_prompt(history, question, context)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000).to(model.device)

    steerer = None
    if use_steering and hal_ids:
        steerer = ActivationSteerer(model, sae, hal_ids, HOOK_LAYER, coeff)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = (time.time() - t0) * 1000

    if steerer:
        steerer.remove()

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer, latency


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RACA Legal LLM",
    description="Arabic legal QA with RAG + SAE hallucination reduction",
    version="2.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Message(BaseModel):
    role: str
    content: str

class QuestionRequest(BaseModel):
    question: str
    history: Optional[List[Message]] = []
    max_tokens: Optional[int] = MAX_NEW_TOKENS
    steering_coefficient: Optional[float] = STEER_COEFF
    use_rag: Optional[bool] = True

class AnswerResponse(BaseModel):
    question: str
    answer: str
    steered: bool
    latency_ms: float

class CompareResponse(BaseModel):
    question: str
    answer_plain: str
    answer_steered: str
    latency_plain_ms: float
    latency_steered_ms: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "hal_features_suppressed": len(hal_ids),
        "rag_documents": len(rag.docs),
        "rag_enabled": rag.index is not None,
    }

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    history = [m.dict() for m in req.history] if req.history else []
    answer, latency = generate(req.question, history, req.max_tokens, use_steering=True, coeff=req.steering_coefficient, use_rag=req.use_rag)
    return AnswerResponse(question=req.question, answer=answer, steered=True, latency_ms=round(latency, 1))

@app.post("/ask/compare", response_model=CompareResponse)
def ask_compare(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    history = [m.dict() for m in req.history] if req.history else []
    answer_plain,   lat_plain   = generate(req.question, history, req.max_tokens, use_steering=False, coeff=req.steering_coefficient, use_rag=req.use_rag)
    answer_steered, lat_steered = generate(req.question, history, req.max_tokens, use_steering=True,  coeff=req.steering_coefficient, use_rag=req.use_rag)
    return CompareResponse(question=req.question, answer_plain=answer_plain, answer_steered=answer_steered, latency_plain_ms=round(lat_plain, 1), latency_steered_ms=round(lat_steered, 1))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
