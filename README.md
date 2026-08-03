# RACA Legal LLM — Fine-tuning + SAE Pipeline

## Project Structure
```
raca_llm_project/
├── phase1_data_prep.py       # Data cleaning, synthetic Q&A via OpenRouter, dataset export
├── phase2_model_setup.py     # GPU check, model download, project config
├── phase3_finetune.py        # QLoRA fine-tuning (CPT stage → SFT stage)
├── phase4_sae_training.py    # Sparse Autoencoder training on fine-tuned model
├── phase5_hal_reduction.py   # Hallucination reduction via activation steering + evaluation
├── requirements.txt          # All Python dependencies
│
├── data/
│   ├── raw_pdfs/             ← PUT YOUR 50 PDF FILES HERE
│   ├── raca_laws_tab1.csv    ← PUT YOUR CSV FILES HERE (raca_laws_tab*.csv)
│   └── processed/            (auto-created by Phase 1)
│
├── checkpoints/              (auto-created by Phase 3 & 4)
└── results/                  (auto-created by Phase 5)
```

---

## 1. Cloud GPU Setup (Vast.ai)

### Rent an Instance
1. Go to [vast.ai](https://vast.ai) and rent a GPU instance (recommended: A100 40GB)
2. In the instance dashboard, open **"Manage SSH Keys"**
3. Your public key (`~/.ssh/id_ed25519.pub` on your Mac) should already be listed — if not, paste it in

### Get Your Connection Details
From the Vast.ai dashboard, copy the **Direct SSH connect** command. It looks like:
```
ssh -p <PORT> root@<IP> -L 8080:localhost:8080
```
Note your `<IP>` and `<PORT>` — you'll need them below.

### Configure SSH on Your Mac
Open your terminal and run:
```bash
nano ~/.ssh/config
```

Add or update this block (replace IP and PORT with your instance values):
```
Host vastai
    HostName <IP>
    Port <PORT>
    User root
    LocalForward 8080 localhost:8080
```

Save with **Ctrl+X → Y → Enter**.

Verify it saved correctly:
```bash
cat ~/.ssh/config
```

### Connect via Terminal
```bash
ssh vastai
```

### Connect via VS Code
1. Install the **Remote - SSH** extension in VS Code
2. Open the Remote Explorer panel (left sidebar)
3. Hover over your host and click the **→ arrow** to connect
4. Once connected, the bottom-left corner shows a green `SSH: <IP>` badge

> ⚠️ Every time you destroy and recreate an instance, you get a **new IP and port**. Update `~/.ssh/config` each time.

---

## 2. Where to Place Your Data

**CSV files** (e.g. `raca_laws_tab1.csv`, `raca_laws_tab2.csv`, ...):
- Place directly in `raca_llm_project/`
- Phase 1 finds them via glob pattern `raca_laws_tab*.csv`

**PDF files** (your 50 source documents):
- Place in `raca_llm_project/data/raw_pdfs/`
- These are your source of truth for RAG in Phase 5

---

## 3. Environment Setup (on the GPU instance)

The Vast.ai instance comes with a pre-installed venv at `/venv/main`. Always activate it first:

```bash
source /venv/main/bin/activate
```

> ⚠️ Do not create a new venv — torch and CUDA libraries are pre-installed in `/venv/main`. Creating a separate venv will cause `ModuleNotFoundError: No module named 'torch'` when building packages like `flash-attn`.

### Clone the Repo
```bash
git clone https://github.com/jenanazem/raca.git
cd raca
```

### Install Dependencies (correct order)

`flash-attn` requires `torch` to already be present at build time. Install it separately with `--no-build-isolation`:

```bash
# Step 1 — activate the pre-installed venv (torch is already here)
source /venv/main/bin/activate

# Step 2 — install flash-attn first, bypassing build isolation
pip install flash-attn --no-build-isolation

# Step 3 — install everything else
pip install -r requirements.txt
```

### Set Your API Key
```bash
export OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
```

---

## 4. Run the Pipeline

```bash
python phase1_data_prep.py --input_glob "data/processed/raca_tab*.csv" --output_dir ./data/processed

python phase2_model_setup.py --skip_model_load

python phase3_finetune.py \
  --config ./data/processed/project_config.json \
  --data_dir ./data/processed \
  --output_dir ./checkpoints/ft_raca

python phase4_sae_training.py \
  --config ./data/processed/project_config.json \
  --model_path ./checkpoints/ft_raca/merged \
  --data_dir ./data/processed \
  --output_dir ./checkpoints/sae_raca

python phase5_hal_reduction.py \
  --config ./data/processed/project_config.json \
  --model_path ./checkpoints/ft_raca/merged \
  --sae_path ./checkpoints/sae_raca/sae_best.pt \
  --feature_file ./checkpoints/sae_raca/feature_interpretations.json
```

---

## 5. Manual Step Between Phase 4 and Phase 5

After Phase 4 finishes, open:
```
checkpoints/sae_raca/feature_interpretations.json
```

For each feature entry, review the `top_activating_docs` snippets and fill in:
```json
"label": "what concept does this feature encode?"
"is_hallucination_feature": true / false
```

This is **required** before running Phase 5.

---

## 6. Save Your Work Before Destroying the Instance

> ⚠️ Vast.ai instances are ephemeral — everything is lost when destroyed. Always push before stopping.

```bash
git add .
git commit -m "your message"
git push
```

---

## Hardware Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 12GB (RTX 3080, T4) | 40GB (A100) |
| Phase 3 training | ~6–8h on T4 | ~2–4h on A100 |
| Phase 4 collection | ~3–4h on T4 | ~1–2h on A100 |

---

## Estimated OpenRouter API Cost (Phase 1)

- Model: `qwen/qwen-2.5-72b-instruct`
- 50 docs × 20 questions = 1,000 Q&A pairs + paraphrase augmentation
- **Estimated cost: $2–5 USD total**
