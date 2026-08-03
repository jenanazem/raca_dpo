# RACA Legal LLM — Fine-tuning + SAFE Pipeline

## Overview

This project fine-tunes Llama 3.1 8B on Qatari legal documents from RACA (Regulatory Authority for Charitable Activities) and uses the **SAFE framework** (Sparse Autoencoder-based Framework for hallucination reduction) to improve answer quality.

### Architecture

```
Fine-tuned Model : JinanAzem/raca-finetuned-v2
                   Llama 3.1 8B Instruct + QLoRA on RACA legal CSV data

SAE              : Goodfire pre-trained SAE — NO TRAINING NEEDED
                   65,536 features, all labeled on Neuronpedia
                   https://neuronpedia.org/llama3.1-8b-it/19-resid-post-gf
                   Downloaded automatically by sae-lens on first run

RAG              : FAISS index over RACA CSV documents
                   Retrieves top-3 relevant legal articles per question

Hallucination    : SAFE framework (local — no OpenRouter needed)
Reduction          - Generates N diverse answers using fine-tuned model
                   - Measures semantic entropy (inconsistency = hallucination)
                   - Extracts SAE features from question + answers
                   - Enriches question with Neuronpedia-labeled features
                   - Repeats until entropy drops below threshold
```

---

## Project Structure

```
raca_v2/
├── phase1_data_prep.py       # Data cleaning + synthetic Q&A via OpenRouter
├── phase2_model_setup.py     # GPU check, model download, project config
├── phase3_finetune.py        # QLoRA fine-tuning (CPT + SFT stages)
├── api.py                    # FastAPI server (RAG + model inference)
├── chat.py                   # Interactive terminal chat
├── safe_pipeline.py          # SAFE hallucination reduction + evaluation
├── evaluate.py               # Semantic evaluation (cosine similarity)
├── requirements.txt
│
├── data/
│   ├── raw_pdfs/             ← PUT YOUR PDF FILES HERE
│   ├── raca_laws_tab*.csv    ← PUT YOUR CSV FILES HERE
│   └── processed/            (auto-created by Phase 1)
│
└── checkpoints/
    └── ft_raca_v2/merged/    ← fine-tuned model (download from HuggingFace)
```

> ⚠️ No SAE training needed — the Goodfire SAE is downloaded automatically by sae-lens.
> Feature labels are fetched from Neuronpedia and cached at `checkpoints/neuronpedia_labels.json`.

---

## 1. Cloud GPU Setup (Vast.ai)

### Rent an Instance
1. Go to [vast.ai](https://vast.ai) and rent a GPU (recommended: RTX 6000 or A100, 40GB+ VRAM)
2. Click the 🔑 icon → copy the **Direct SSH connect** command to get `<IP>` and `<PORT>`

### Connect from your Mac

**Terminal 1** — SSH tunnel (keep open, do nothing else in it):
```bash
ssh -p <PORT> root@<IP> -L 8080:localhost:8080
touch ~/.no_auto_tmux   # disable auto-tmux on reconnect
```

**Terminal 2** — Work terminal:
```bash
ssh -p <PORT> root@<IP>
# if tmux auto-attaches: Ctrl+B then C to open a new window
```

> ⚠️ Every new instance gets a new IP and port — update both terminals each time.

---

## 2. Copy Project from Mac to GPU

Run this on your **Mac terminal**:
```bash
scp -r -P <PORT> ~/Desktop/raca_v2 root@<IP>:/root/
```

---

## 3. Environment Setup

```bash
source /venv/main/bin/activate
cd /root/raca_v2

# Install dependencies
sed -i 's/^flash-attn/# flash-attn/' requirements.txt
pip install -r requirements.txt
pip install fastapi uvicorn sentence-transformers faiss-cpu accelerate
pip install sae-lens transformer-lens scikit-learn

# Fix torch version (sae-lens may downgrade it)
pip install torch==2.12.0+cu130 torchvision --index-url https://download.pytorch.org/whl/cu130

# Set API key (only needed for Phase 1 data generation)
export OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
```

---

## 4. Download Fine-tuned Model

```bash
hf download JinanAzem/raca-finetuned-v2 --local-dir ./checkpoints/ft_raca_v2
```

> The Goodfire SAE is downloaded automatically when you first run `safe_pipeline.py`.
> No need to download or train your own SAE.

---

## 5. Code Fixes (apply once after cloning)

```bash
sed -i "s/attn_implementation='flash_attention_2'/attn_implementation='eager'/" phase3_finetune.py
sed -i '/group_by_length=True/d' phase3_finetune.py
sed -i 's/tokenizer=tokenizer,/processing_class=tokenizer,/' phase3_finetune.py
sed -i 's/cpt_args = cpt_args.replace(num_train_epochs=n_epochs_cpt)/cpt_args.num_train_epochs = n_epochs_cpt/' phase3_finetune.py
sed -i 's/sft_args = sft_args.replace(num_train_epochs=n_epochs_sft)/sft_args.num_train_epochs = n_epochs_sft/' phase3_finetune.py
sed -i 's/max_seq_length=/max_length=/' phase3_finetune.py
sed -i 's/fp16=config\[.dtype.\] == .float16.,/fp16=False,/' phase3_finetune.py
sed -i 's/from pathlib import Path/from pathlib import Path\nfrom typing import Optional, Dict, Any/' phase2_model_setup.py
```

---

## 6. Run the Chat Interface

**Terminal 1** (API server — with port forwarding):
```bash
source /venv/main/bin/activate
cd /root/raca_v2
fuser -k 8080/tcp
python api.py
```

Wait for `Uvicorn running on http://0.0.0.0:8080`, then:

**Terminal 2** (Chat):
```bash
source /venv/main/bin/activate
cd /root/raca_v2
python chat.py
```

---

## 7. Run SAFE Pipeline

```bash
# Single question test
python safe_pipeline.py --question "ما هي شروط تأسيس الجمعية الخيرية؟"

# Quick evaluation (20 examples)
python safe_pipeline.py --test_file data/processed/sft_test.jsonl --limit 20

# Full evaluation (171 examples)
python safe_pipeline.py --test_file data/processed/sft_test.jsonl
```

Results are saved incrementally to `results/safe_eval.json` — you can stop anytime.

---

## 8. Run Semantic Evaluation (without SAFE)

```bash
# Steered model only
python evaluate.py --limit 20

# Compare plain vs steered
python evaluate.py --compare --limit 20

# Full test set
python evaluate.py
```

---

## 9. Retrain from Scratch

Only needed if you want to retrain with new data or more epochs.

```bash
export OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx

# Phase 1 — generate richer training data
python phase1_data_prep.py --input_glob "data/processed/raca_tab*.csv" --output_dir ./data/processed

# Phase 2 — verify GPU setup
python phase2_model_setup.py --skip_model_load

# Increase epochs in config
python3 -c "
import json
c = json.load(open('data/processed/project_config.json'))
c['num_train_epochs_cpt'] = 2
c['num_train_epochs_sft'] = 5
c['dtype'] = 'bfloat16'
json.dump(c, open('data/processed/project_config.json', 'w'), ensure_ascii=False, indent=2)
print('done')
"

# Phase 3 — fine-tune
python phase3_finetune.py --data_dir ./data/processed --output_dir ./checkpoints/ft_raca_v2

# Upload to HuggingFace
hf upload JinanAzem/raca-finetuned-v2 /root/raca_v2/checkpoints/ft_raca_v2
```

---

## 10. Save Before Destroying Instance

```bash
git add .
git commit -m "your message"
git push
```

---

## Evaluation Results

| Model | Avg Cosine Sim | ≥0.8 | ≥0.6 |
|---|---|---|---|
| v1 (3 SFT epochs) | 0.619 | 21.9% | 55.0% |
| v2 (5 SFT epochs, richer data) | 0.653 | 28.7% | 61.4% |
| v2 + SAFE (local model) | TBD | TBD | TBD |

---

## Hardware Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 40GB | 80GB+ |
| Phase 3 training | ~4–6h on RTX 6000 | ~2–4h on A100 |
| SAFE pipeline | ~1–2 min/question | faster with more VRAM |

---

## Estimated Costs

- OpenRouter (Phase 1 data generation): ~$2–5 USD total
- Vast.ai GPU (RTX 6000): ~$0.03–0.05/hr
