#!/usr/bin/env python3
"""
Patch build_local_feature_labels.py (v2.1 -> v2.2)
===================================================
Three fixes to the OpenRouter labelling path, each from observed behaviour:

  1. COHERENCE ABSTENTION. Feature 62519's activating spans were cash-voucher
     recovery risk, the procedure for opening an external branch, and regulatory
     instructions for partner entities — no shared concept. The labeller still
     returned a confident "الإجراءات التنظيمية والإدارية لتنفيذ اللوائح القانونية".
     A fluent label on an incoherent feature is worse than no label: it hides the
     incoherence that extractive labels make obvious, and SAFE then steers on a
     feature that encodes nothing. The prompt now offers an explicit "غير متسق"
     escape, and such features are dropped from the label file. Because
     get_explanation skips unlabelled features and moves down the top-k, dropping
     them means SAFE automatically prefers coherent features.

  2. NONE-CONTENT GUARD. qwen/qwen3.5-27b returned content=None and raised
     'NoneType' object has no attribute 'strip'. Any provider can do this, and
     the exception was being swallowed into a silent extractive fallback.

  3. TOKEN BUDGET. max_tokens was 60. Arabic costs 2-4 tokens per word in most
     tokenizers, so an 8-word description could be truncated mid-phrase.

Also records abstentions so the abstention rate is visible — it is a direct
measure of how monosemantic the SAE is on your corpus, and worth reporting.

Idempotent.

Usage:
    python patch_labeler_v22.py
    python patch_labeler_v22.py --check
"""

import argparse
import ast
import re
import shutil
import sys
from pathlib import Path

TARGET = Path("build_local_feature_labels.py")

NEW_BLOCK = '''
# Sentinel the labeller returns when the spans share no concept.
INCOHERENT_MARKERS = ("غير متسق", "غير متسقة", "لا يوجد مفهوم", "INCOHERENT")

LABEL_PROMPT = """فيما يلي مقاطع نصية من وثائق قانونية قطرية، وكلها تُنشّط الميزة ذاتها في شبكة عصبية.

{spans}

أولاً: هل تشترك هذه المقاطع في مفهوم قانوني واحد واضح؟

- إذا كانت المقاطع تتناول موضوعات غير مترابطة، اكتب "غير متسق" فقط ولا تكتب شيئاً آخر.
- إذا كانت تشترك في مفهوم واحد، اكتب وصفاً موجزاً له (من ٣ إلى ٨ كلمات) دون أي مقدمات.

لا تحاول إيجاد رابط مشترك إن لم يكن واضحاً."""


def openrouter_label(spans, model, api_key, max_retries=3):
    """Returns (description, status) where status is 'ok', 'incoherent', or 'failed'."""
    import requests
    listing = "\\n".join(f"- {s[:200]}" for s in spans[:8])
    payload = {"model": model, "max_tokens": 160, "temperature": 0.0,
               "messages": [{"role": "user",
                             "content": LABEL_PROMPT.format(spans=listing)}]}
    for attempt in range(max_retries):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers={"Authorization": f"Bearer {api_key}"},
                              json=payload, timeout=90)
            if r.status_code == 200:
                body = r.json()
                choices = body.get("choices") or []
                content = None
                if choices:
                    content = (choices[0].get("message") or {}).get("content")
                if not content:
                    log.warning("Empty content from %s (attempt %d)", model, attempt + 1)
                    time.sleep(1.0)
                    continue
                txt = content.strip().strip('"').strip("«»").strip()
                if not txt:
                    continue
                if any(mk in txt for mk in INCOHERENT_MARKERS):
                    return "", "incoherent"
                # Reject anything long enough to be commentary rather than a label.
                if len(txt.split()) > 14:
                    txt = " ".join(txt.split()[:14])
                return txt, "ok"
            elif r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            elif r.status_code in (400, 404):
                log.error("Model rejected (%d): %s", r.status_code, r.text[:200])
                return "", "failed"
            else:
                log.warning("OpenRouter HTTP %d: %s", r.status_code, r.text[:150])
        except Exception as exc:
            log.warning("OpenRouter error: %s", exc)
        time.sleep(1.5 * (attempt + 1))
    return "", "failed"
'''

OLD_CALL = '''            desc = openrouter_label(spans, cfg.openrouter_model, api_key) \\
                or extractive_label(spans, df, vocab, n_docs, cfg.n_terms)
            if desc:
                labels[str(fid)] = desc
            else:
                empty += 1'''

NEW_CALL = '''            desc, status = openrouter_label(spans, cfg.openrouter_model, api_key)
            if status == "incoherent":
                # Deliberately unlabelled: get_explanation skips these and moves
                # further down the top-k, so SAFE steers on coherent features only.
                incoherent += 1
                continue
            if status == "failed" and cfg.fallback_extractive:
                desc = extractive_label(spans, df, vocab, n_docs, cfg.n_terms)
                if desc:
                    fell_back += 1
            if desc:
                labels[str(fid)] = desc
            else:
                empty += 1'''


def patch(src):
    notes = []

    if "INCOHERENT_MARKERS" in src:
        notes.append("labeller already patched")
    else:
        m = re.search(r'LABEL_PROMPT = """.*?^def openrouter_label\(.*?\n    return ""\n',
                      src, re.S | re.M)
        if not m:
            raise RuntimeError("could not locate LABEL_PROMPT..openrouter_label block")
        src = src[:m.start()] + NEW_BLOCK.lstrip("\n") + src[m.end():]
        notes.append("replaced prompt + openrouter_label")

    if "incoherent += 1" in src:
        notes.append("call site already patched")
    elif OLD_CALL in src:
        src = src.replace(OLD_CALL, NEW_CALL, 1)
        notes.append("patched call site")
    else:
        notes.append("WARNING: call site not found — patch it by hand")

    if "labels, empty = {}, 0" in src and "incoherent = 0" not in src:
        src = src.replace("labels, empty = {}, 0",
                          "labels, empty = {}, 0\n    incoherent = fell_back = 0", 1)
        notes.append("added counters")
    else:
        notes.append("counters already present")

    if "--fallback_extractive" not in src:
        anchor = '    p.add_argument("--reuse_spans", action="store_true")'
        src = src.replace(anchor,
            '    p.add_argument("--no_fallback_extractive", dest="fallback_extractive",\n'
            '                   action="store_false", default=True,\n'
            '                   help="Do not substitute extractive labels when the "\n'
            '                        "LLM call fails outright.")\n' + anchor, 1)
        notes.append("added --no_fallback_extractive")
    else:
        notes.append("flag already present")

    if "Incoherent (dropped)" not in src:
        anchor = '    print(f"  Labels written       : {len(labels)}  (empty: {empty})")'
        src = src.replace(anchor, anchor +
            '\n    print(f"  Incoherent (dropped) : {incoherent}")'
            '\n    print(f"  Extractive fallback  : {fell_back}")', 1)
        notes.append("added summary lines")
    else:
        notes.append("summary lines already present")

    return src, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=TARGET)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if not a.file.exists():
        print(f"{a.file} not found", file=sys.stderr)
        return 1

    src = a.file.read_text()
    patched, notes = patch(src)
    print(f"Patching {a.file}:")
    for n in notes:
        print(f"  - {n}")

    if a.check:
        print("\n--check given; nothing written.")
        return 0
    try:
        ast.parse(patched)
    except SyntaxError as exc:
        print(f"\nPatched source does not parse: {exc}", file=sys.stderr)
        return 1

    shutil.copy(a.file, a.file.with_suffix(".py.bak2"))
    a.file.write_text(patched)
    print(f"\nWritten. Backup at {a.file.with_suffix('.py.bak2')}")
    print("\nThe abstention rate is itself a result: it measures how often the "
          "SAE\ngives a monosemantic feature on your corpus. Report it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
