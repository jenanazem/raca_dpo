"""
Phase 1 — Data Preparation for 50-Document Arabic Legal Corpus
===============================================================
Adapted for small corpus (50 PDFs) using OpenRouter for synthetic data generation.

Strategy:
  1. CPT (continued pre-training): 50 full documents — teaches domain vocabulary
  2. Synthetic SFT: ~20 Q&A pairs per doc via OpenRouter LLM = 1000 pairs
  3. Chunk-level SFT: split each doc into article-level chunks = ~250 extra pairs
  4. Paraphrase augmentation: rephrase each question = 2000 final SFT pairs

Total: ~2250 training examples from 50 source docs — sufficient for LoRA fine-tuning.

Install:
    pip install pandas datasets scikit-learn requests python-dotenv tqdm

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python phase1_data_prep.py --input_glob "raca_laws_tab*.csv" --output_dir ./data
"""

import re
import os
import json
import time
import hashlib
import argparse
import requests
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model for synthetic data generation.
# Good options on OpenRouter for Arabic legal text:
#   - meta-llama/llama-3.1-70b-instruct        (strong Arabic, fast, cheap)
#   - meta-llama/llama-3.1-405b-instruct        (best quality, slower)
#   - qwen/qwen-2.5-72b-instruct               (excellent Arabic, very cheap)
#   - mistralai/mistral-large                   (good Arabic)
SYNTH_MODEL = "meta-llama/llama-3.1-70b-instruct"

SYSTEM_PROMPT_FT = """أنت مساعد قانوني متخصص في تشريعات وأنظمة هيئة تنظيم الأعمال الخيرية في دولة قطر.
تجيب على الأسئلة بدقة وتستند إلى النصوص القانونية المعتمدة فقط.
إذا لم تجد الإجابة في النص المتاح، قل ذلك صراحةً."""


# ─────────────────────────────────────────────
# 1. ARABIC TEXT NORMALIZATION
# ─────────────────────────────────────────────

DIACRITICS = re.compile(r'[\u064b-\u065f\u0670]')
NORMALIZE_MAP = str.maketrans({
    'أ': 'ا', 'إ': 'ا', 'آ': 'ا',
    'ى': 'ي',
    '\u200c': '', '\u200d': '', '\u200e': '', '\u200f': '', '\ufeff': '',
})
BOILERPLATE = re.compile(
    r'الرجاء عدم اعتبار.*|البوابة القانونية.*|QATAR LEGAL PORTAL|'
    r'https?://\S+|صفحة \d+ من \d+|\d{1,2}:\d{2}\s+\d{4}/\d{1,2}/\d{1,2}|'
    r'جميع الحقوق محفوظة|حكومة دولة قطر',
    re.UNICODE
)

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = DIACRITICS.sub('', text)
    text = text.translate(NORMALIZE_MAP)
    text = BOILERPLATE.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()


# ─────────────────────────────────────────────
# 2. LOAD & CLEAN CSV FILES
# ─────────────────────────────────────────────

def load_and_clean(input_glob: str, min_words: int = 80) -> pd.DataFrame:
    paths = sorted(Path('.').glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {input_glob}")

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(df)} rows from {len(paths)} file(s)")

    df = df.dropna(subset=['text', 'title'])
    df = df[df['scrape_status'].str.startswith('ok', na=False)]
    df['text'] = df['text'].apply(normalize)
    df['title'] = df['title'].apply(normalize)
    df['word_count'] = df['text'].str.split().str.len()
    df = df[df['word_count'] >= min_words]

    # Deduplicate by content hash
    df['_hash'] = df['text'].apply(lambda t: hashlib.md5(t.encode()).hexdigest())
    before = len(df)
    df = df.drop_duplicates('_hash').drop(columns=['_hash'])
    if before > len(df):
        print(f"  Removed {before - len(df)} duplicates")

    print(f"  Clean corpus: {len(df)} documents")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 3. OPENROUTER SYNTHETIC Q&A GENERATION
# ─────────────────────────────────────────────

QA_GENERATION_PROMPT = """أنت خبير قانوني متخصص في تشريعات هيئة تنظيم الأعمال الخيرية في قطر.
النص التالي مقتطف من تشريع قطري رسمي.

اقرأ النص بعناية ثم أنشئ بالضبط {n_questions} سؤالاً وجوابًا متنوعين ومختلفين.

متطلبات الأسئلة:
- يجب أن تكون الأسئلة متنوعة: أسئلة عن تعريفات، أسئلة عن إجراءات، أسئلة عن عقوبات، أسئلة عن صلاحيات، أسئلة عن شروط وضوابط
- يجب أن تكون الأجوبة كاملة ومفصلة وتشمل السياق القانوني الكامل
- يجب أن يبدأ كل جواب بصيغة "وفقاً للمادة (رقم)" أو "نصت المادة (رقم)" مع ذكر رقم المادة الصريح
- يجب أن يشرح الجواب الحكم القانوني بالكامل وليس مجرد اقتباس قصير
- لا تخترع معلومات غير موجودة في النص
- استخدم اللغة العربية الفصحى
- الجواب يجب أن يكون جملة كاملة لا أقل من 50 كلمة

فيما يلي أمثلة على أسئلة وأجوبة مثالية:

مثال 1:
السؤال: ما هي شروط تأسيس الجمعية الخيرية وفقاً للقانون؟
الجواب المثالي: وفقاً للمادة السادسة من القانون رقم 15 لسنة 2014، يشترط لتأسيس الجمعية الخيرية أن لا يقل عدد المؤسسين عن عشرين شخصاً من المواطنين القطريين، وأن يكون لكل منهم أهلية قانونية كاملة، وأن يتضمن النظام الأساسي اسم الجمعية وأهدافها ومقرها الرئيسي ومصادر تمويلها وآليات اتخاذ القرار فيها.

مثال 2:
السؤال: ما هي عقوبة ممارسة النشاط الخيري دون ترخيص؟
الجواب المثالي: نصت المادة الثامنة والثلاثون من القانون على أن كل من يمارس نشاطاً خيرياً أو يجمع تبرعات دون الحصول على ترخيص مسبق من هيئة تنظيم الأعمال الخيرية يعاقب بالحبس مدة لا تتجاوز سنة وبغرامة مالية لا تقل عن عشرة آلاف ريال ولا تزيد على خمسين ألف ريال قطري، أو بإحدى هاتين العقوبتين.

النص:
{text}

أجب بصيغة JSON فقط بالشكل التالي (لا تضف أي نص خارج الـ JSON):
[
  {{"question": "...", "answer": "وفقاً للمادة (X)... جواب كامل لا يقل عن 50 كلمة..."}},
  {{"question": "...", "answer": "وفقاً للمادة (X)... جواب كامل لا يقل عن 50 كلمة..."}}
]"""

PARAPHRASE_PROMPT = """أعد صياغة السؤال التالي بطريقة مختلفة تمامًا مع الحفاظ على المعنى ذاته.
أجب بالسؤال المعاد صياغته فقط، بدون أي شرح إضافي.

السؤال الأصلي: {question}"""


def call_openrouter(messages: list, model: str, api_key: str, max_retries: int = 3) -> str | None:
    """Call OpenRouter API with retry logic."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://raca-legal-llm",
        "X-Title": "RACA Legal LLM",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content'].strip()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"    Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    HTTP error {resp.status_code}: {e}")
                return None
        except Exception as e:
            print(f"    Error on attempt {attempt+1}: {e}")
            time.sleep(2)
    return None


def generate_qa_for_doc(
    title: str,
    text: str,
    api_key: str,
    n_questions: int = 20,
    max_text_chars: int = 3000,
) -> list[dict]:
    """
    Generate synthetic Q&A pairs for a single document using OpenRouter.

    Truncates long documents to max_text_chars to stay within context limits
    while still covering the most legally dense content.
    For very long docs (>3000 chars), consider calling this twice on
    different sections and merging.
    """
    text_chunk = text[:max_text_chars]

    prompt = QA_GENERATION_PROMPT.format(n_questions=n_questions, text=text_chunk)
    messages = [{"role": "user", "content": prompt}]

    raw = call_openrouter(messages, SYNTH_MODEL, api_key)
    if not raw:
        return []

    # Parse JSON — strip markdown fences if model added them
    raw = re.sub(r'^```json\s*|^```\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE)
    try:
        pairs = json.loads(raw)
        # Validate structure
        valid = [
            p for p in pairs
            if isinstance(p, dict) and 'question' in p and 'answer' in p
            and len(p['question']) > 10 and len(p['answer']) > 10
        ]
        return valid
    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}")
        print(f"    Raw response: {raw[:200]}")
        return []


def paraphrase_question(question: str, api_key: str) -> str | None:
    """Generate an alternative phrasing of a question."""
    messages = [{"role": "user", "content": PARAPHRASE_PROMPT.format(question=question)}]
    result = call_openrouter(messages, SYNTH_MODEL, api_key, max_retries=2)
    if result and len(result) > 10:
        return result.strip()
    return None


# ─────────────────────────────────────────────
# 4. ARTICLE-LEVEL CHUNKING
# ─────────────────────────────────────────────

def split_into_articles(text: str, title: str) -> list[dict]:
    """
    Split a legal document into article-level chunks.

    Arabic legal texts use 'مادة (N)' or 'المادة N' as article markers.
    Each chunk becomes its own CPT/SFT training unit.
    """
    # Match patterns like: مادة (1)  or  المادة 5  or  مادة 12
    article_pattern = re.compile(r'(?:المادة|مادة)\s*[\(（]?\s*(\d+)\s*[\)）]?', re.UNICODE)
    matches = list(article_pattern.finditer(text))

    if len(matches) < 2:
        # No article structure found — return the full doc as one chunk
        return [{'title': title, 'text': text, 'chunk_type': 'full_doc'}]

    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()

        if len(chunk_text.split()) < 30:  # skip very short fragments
            continue

        article_num = match.group(1)
        chunks.append({
            'title': f"{title} — المادة {article_num}",
            'text': chunk_text,
            'chunk_type': 'article',
        })

    return chunks if chunks else [{'title': title, 'text': text, 'chunk_type': 'full_doc'}]


# ─────────────────────────────────────────────
# 5. FORMAT AS CHATML
# ─────────────────────────────────────────────

CHATML = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"
CPT_TEMPLATE = "### قانون / تشريع\n{title}\n\n### النص الكامل\n{text}"


def make_sft_record(title: str, question: str, answer: str, source_url: str = '', provenance_text: str = '') -> dict:
    return {
        "text": CHATML.format(
            system=SYSTEM_PROMPT_FT,
            user=question,
            assistant=answer,
        ),
        "instruction": question,
        "response": answer,
        "source_title": title,
        "provenance_text": provenance_text,
        "source_url": source_url,
    }


def make_cpt_record(title: str, text: str, source_url: str = '') -> dict:
    return {
        "text": CPT_TEMPLATE.format(title=title, text=text),
        "source_title": title,
        "source_url": source_url,
    }


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Data prep for 50-doc Arabic corpus")
    parser.add_argument('--input_glob', default='raca_laws_tab*.csv')
    parser.add_argument('--output_dir', default='./data')
    parser.add_argument('--n_questions', type=int, default=20,
                        help='Synthetic Q&A pairs to generate per document')
    parser.add_argument('--skip_synth', action='store_true',
                        help='Skip OpenRouter generation (use if you already ran it)')
    parser.add_argument('--no_paraphrase', action='store_true',
                        help='Skip paraphrase augmentation (saves ~50%% API cost)')
    args = parser.parse_args()

    api_key = os.environ.get('OPENROUTER_API_KEY', '')
    if not api_key and not args.skip_synth:
        raise ValueError("Set OPENROUTER_API_KEY environment variable, or use --skip_synth")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    synth_cache = output_dir / 'synthetic_qa_cache.jsonl'

    print("\n=== Phase 1: Data Preparation (50-doc mode) ===\n")

    # ── Step 1: Load and clean ──
    print("[1/5] Loading and cleaning CSV files...")
    df = load_and_clean(args.input_glob)

    # ── Step 2: Build CPT corpus ──
    print("\n[2/5] Building CPT corpus (full documents + article chunks)...")
    cpt_records = []
    for _, row in df.iterrows():
        # Full document
        cpt_records.append(make_cpt_record(row['title'], row['text'], row.get('url', '')))
        # Article-level chunks
        for chunk in split_into_articles(row['text'], row['title']):
            if chunk['chunk_type'] == 'article':
                cpt_records.append(make_cpt_record(chunk['title'], chunk['text'], row.get('url', '')))

    print(f"  CPT records: {len(cpt_records)} ({len(df)} full docs + article chunks)")

    # ── Step 3: Generate synthetic Q&A ──
    print(f"\n[3/5] Generating synthetic Q&A via OpenRouter ({SYNTH_MODEL})...")

    # Load cache if it exists
    cached_qa: dict[str, list] = {}
    if synth_cache.exists():
        with open(synth_cache, encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                cached_qa[entry['doc_hash']] = entry['pairs']
        print(f"  Loaded {len(cached_qa)} cached documents from {synth_cache}")

    all_qa_pairs = []  # list of (title, question, answer, url)

    if not args.skip_synth:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  Generating Q&A"):
            doc_hash = hashlib.md5(row['text'].encode()).hexdigest()

            if doc_hash in cached_qa:
                pairs = cached_qa[doc_hash]
            else:
                pairs = generate_qa_for_doc(
                    row['title'], row['text'], api_key,
                    n_questions=args.n_questions,
                )
                # Cache result
                with open(synth_cache, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        'doc_hash': doc_hash,
                        'title': row['title'],
                        'pairs': pairs,
                    }, ensure_ascii=False) + '\n')
                cached_qa[doc_hash] = pairs
                time.sleep(0.5)  # be polite to the API

            text_chunk = row['text'][:3000]  # same chunk used for generation
            for p in pairs:
                all_qa_pairs.append((row['title'], p['question'], p['answer'], row.get('url', ''), text_chunk))

        print(f"  Generated {len(all_qa_pairs)} Q&A pairs from {len(df)} documents")
        print(f"  Average: {len(all_qa_pairs) / len(df):.1f} pairs/doc")
    else:
        # Load all from cache
        for doc_hash, pairs in cached_qa.items():
            title = ''
            for p in pairs:
                all_qa_pairs.append((title, p['question'], p['answer'], '', ''))
        print(f"  Loaded {len(all_qa_pairs)} cached Q&A pairs")

    # ── Step 4: Paraphrase augmentation ──
    if not args.no_paraphrase and all_qa_pairs and api_key:
        print(f"\n[4/5] Paraphrase augmentation ({len(all_qa_pairs)} questions)...")
        paraphrase_cache = output_dir / 'paraphrase_cache.jsonl'

        cached_para: dict[str, str] = {}
        if paraphrase_cache.exists():
            with open(paraphrase_cache, encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    cached_para[entry['original']] = entry['paraphrase']

        augmented_pairs = []
        for title, question, answer, url, *rest in tqdm(all_qa_pairs, desc="  Paraphrasing"):
            source_text = rest[0] if rest else ''
            q_hash = hashlib.md5(question.encode()).hexdigest()
            if q_hash in cached_para:
                para_q = cached_para[q_hash]
            else:
                para_q = paraphrase_question(question, api_key)
                if para_q:
                    with open(paraphrase_cache, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            'original': q_hash,
                            'paraphrase': para_q,
                        }, ensure_ascii=False) + '\n')
                    cached_para[q_hash] = para_q
                time.sleep(0.3)

            if para_q:
                augmented_pairs.append((title, para_q, answer, url, source_text))

        all_qa_pairs.extend(augmented_pairs)
        print(f"  After paraphrase augmentation: {len(all_qa_pairs)} total Q&A pairs")
    else:
        print("\n[4/5] Skipping paraphrase augmentation")

    # ── Step 5: Build SFT dataset and split ──
    print(f"\n[5/5] Building datasets and splitting...")

    sft_records = [
        make_sft_record(title, question, answer, url, source_text if len(item) > 4 else '')
        for item in all_qa_pairs
        for title, question, answer, url, *rest in [item]
        for source_text in [rest[0] if rest else '']
    ]

    print(f"\n  Dataset summary:")
    print(f"    CPT records:  {len(cpt_records)}")
    print(f"    SFT records:  {len(sft_records)}")

    # Split both datasets
    for name, records in [('cpt', cpt_records), ('sft', sft_records)]:
        if not records:
            continue
        train, valtest = train_test_split(records, train_size=0.8, random_state=42)
        val, test = train_test_split(valtest, train_size=0.5, random_state=42)

        ds = DatasetDict({
            'train': Dataset.from_list(train),
            'validation': Dataset.from_list(val),
            'test': Dataset.from_list(test),
        })
        ds.save_to_disk(str(output_dir / f'{name}_dataset'))

        # Also save JSONL
        for split, recs in [('train', train), ('val', val), ('test', test)]:
            with open(output_dir / f'{name}_{split}.jsonl', 'w', encoding='utf-8') as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')

        print(f"    {name.upper()}: {len(train)} train / {len(val)} val / {len(test)} test")

    # Save project config
    config = {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "use_qlora": True,
        "dtype": "float16",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "max_seq_length": 2048,
        "training_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "num_train_epochs_cpt": 1,
        "num_train_epochs_sft": 3,
        "sae_hook_layer": 16,
        "sae_expansion_factor": 8,
        "sae_l1_coefficient": 0.0002,
        "n_source_docs": len(df),
        "n_sft_records": len(sft_records),
        "synth_model": SYNTH_MODEL,
    }
    with open(output_dir / 'project_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Phase 1 complete.")
    print(f"  All data written to: {output_dir}/")
    print(f"  Config: {output_dir}/project_config.json")
    print(f"\n  NOTE: lora_r=8 (reduced from 16) — correct for small corpus to avoid overfitting.")
    print(f"  Training plan: 1 epoch CPT → 3 epochs SFT with early stopping.\n")


if __name__ == '__main__':
    main()
