#!/usr/bin/env python3
"""
Patch build_local_feature_labels.py (v2 -> v2.1)
=================================================
Two fixes, both observable in the 60-text sample output:

  1. CLITIC-AWARE STOPWORDS. Labels contained وهذا، ويمكن، وكذلك، ويجب، وتقوم.
     Arabic proclitics (و ف ب ل ك and the article ال) give every function word
     many surface forms, and normalize_arabic does not strip them, so `هذا` was
     filtered while `وهذا` sailed through. Now each candidate term is checked
     against the stoplist after progressively stripping prefixes.

  2. OCR JUNK FILTER. The corpus contains PDF artefacts — table-of-contents
     dotted leaders ("الهادفة للربح 18.............. 21. 23."), bare page-number
     runs, and OCR drift into Persian-range characters (یه به په له). These
     pollute both the background vocabulary and the activating spans; feature
     16268's span set showed one directly. Chunks that are mostly non-Arabic,
     mostly punctuation, or mostly digits are now dropped.

Idempotent — running twice is harmless.

Usage:
    python patch_labels_v3.py                      # patches in place, keeps .bak
    python patch_labels_v3.py --check              # report only
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

TARGET = Path("build_local_feature_labels.py")

EXTRA_STOPWORDS = '''
وهذا وهذه وذلك وكذلك ويمكن ويجب وتقوم وتكون ويكون وهو وهي وهم
يشمل تشمل يعمل تعمل يتولي تتولي يتفق تتفق يتعين يتضمن تتضمن
عام عامه بشكل بصوره حال حاله حالات نحو مثل ومثل خلال وخلال
ذات ذوي اكثر اقل الاكثر الاقل عموما ايه اية كذلك ايضا
بها بهذه بهذا لهذا لهذه فيما عما مما لما كون
جميع بعض معظم سائر كافه كافة عده عدة
الاول الثاني الثالث الرابع الخامس السادس السابع الثامن التاسع العاشر
اولا ثانيا ثالثا رابعا خامسا
'''

PREFIX_HELPERS = '''
# Arabic proclitics, longest first. A function word such as هذا surfaces as
# وهذا / فهذا / بهذا / لهذا, so stopword matching must strip these before
# testing membership.
CLITIC_PREFIXES = ("وبال", "فبال", "وال", "فال", "بال", "كال", "لل",
                   "ال", "و", "ف", "ب", "ل", "ك")

# Characters in the Arabic block that do not occur in Qatari legal Arabic; their
# presence indicates OCR drift.
NON_ARABIC_LETTERS = set("پچژگیکەڵڤۆ")


def _clitic_variants(term: str):
    """The term itself plus each form obtained by stripping proclitics."""
    yield term
    seen = {term}
    frontier = [term]
    for _ in range(2):                       # at most two stacked clitics
        nxt = []
        for t in frontier:
            for pre in CLITIC_PREFIXES:
                if t.startswith(pre) and len(t) - len(pre) >= 2:
                    cand = t[len(pre):]
                    if cand not in seen:
                        seen.add(cand)
                        nxt.append(cand)
                        yield cand
        frontier = nxt
        if not frontier:
            break


def is_stopword(term: str) -> bool:
    return any(v in STOPWORDS for v in _clitic_variants(term))


def is_junk_text(text: str) -> bool:
    """Reject PDF/OCR artefacts: dotted leaders, page-number runs, OCR drift."""
    if not text:
        return True
    n = len(text)
    arabic = sum(1 for c in text if "\\u0600" <= c <= "\\u06ff")
    if arabic / n < 0.45:
        return True
    if sum(1 for c in text if c in ".\\u00b7\\u2026_-") / n > 0.15:
        return True
    if sum(1 for c in text if c.isdigit()) / n > 0.12:
        return True
    if sum(1 for c in text if c in NON_ARABIC_LETTERS) > 3:
        return True
    # Dotted leaders survive the ratios when interleaved with words.
    if re.search(r"\\.{4,}", text):
        return True
    return False
'''

OLD_TOKENIZE = '''        if len(t) < MIN_TERM_LEN or t in STOPWORDS or t.isdigit():
            continue'''

NEW_TOKENIZE = '''        if len(t) < MIN_TERM_LEN or t.isdigit() or is_stopword(t):
            continue'''

OLD_CHUNK = '''                    chunk = " ".join(words[i:i + chunk_size])
                    if len(chunk) > 80:
                        texts.append(chunk)'''

NEW_CHUNK = '''                    chunk = " ".join(words[i:i + chunk_size])
                    if len(chunk) > 80 and not is_junk_text(chunk):
                        texts.append(chunk)'''

OLD_EXTRA = '''                    v = (ex.get(field) or "").strip()
                    if len(v) > 80:
                        texts.append(v)'''

NEW_EXTRA = '''                    v = (ex.get(field) or "").strip()
                    if len(v) > 80 and not is_junk_text(v):
                        texts.append(v)'''


def patch(src: str) -> tuple:
    notes = []

    if "def is_stopword" in src:
        notes.append("helpers already present")
    else:
        anchor = "MIN_TERM_LEN = 3\n"
        if anchor not in src:
            raise RuntimeError("anchor 'MIN_TERM_LEN = 3' not found")
        src = src.replace(anchor, anchor + PREFIX_HELPERS, 1)
        notes.append("added clitic + junk helpers")

    if EXTRA_STOPWORDS.split()[0] in src.split("MIN_TERM_LEN")[0]:
        notes.append("stopwords already extended")
    else:
        m = re.search(r'STOPWORDS = set\("""(.*?)"""\.split\(\)\)', src, re.S)
        if not m:
            raise RuntimeError("STOPWORDS block not found")
        merged = m.group(1).rstrip() + "\n" + EXTRA_STOPWORDS.strip() + "\n"
        src = src[:m.start(1)] + merged + src[m.end(1):]
        notes.append("extended stopword list")

    for old, new, label in ((OLD_TOKENIZE, NEW_TOKENIZE, "tokenize_terms filter"),
                            (OLD_CHUNK, NEW_CHUNK, "csv chunk filter"),
                            (OLD_EXTRA, NEW_EXTRA, "extra_texts filter")):
        if new in src:
            notes.append(f"{label} already patched")
        elif old in src:
            src = src.replace(old, new, 1)
            notes.append(f"patched {label}")
        else:
            notes.append(f"WARNING: could not locate {label}")

    return src, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=TARGET)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if not a.file.exists():
        print(f"{a.file} not found — run from the directory containing it",
              file=sys.stderr)
        return 1

    original = a.file.read_text()
    patched, notes = patch(original)

    print(f"Patching {a.file}:")
    for n in notes:
        print(f"  - {n}")

    if a.check:
        print("\n--check given; nothing written.")
        return 0

    import ast
    try:
        ast.parse(patched)
    except SyntaxError as exc:
        print(f"\nPatched source does not parse: {exc}", file=sys.stderr)
        return 1

    backup = a.file.with_suffix(".py.bak")
    shutil.copy(a.file, backup)
    a.file.write_text(patched)
    print(f"\nWritten. Backup at {backup}")
    print("\nRe-score with no GPU cost:")
    print("  python build_local_feature_labels.py --model_path "
          "./checkpoints/ft_raca_v5/merged --reuse_spans\n")
    print("Note: --reuse_spans keeps the cached spans, so the junk filter only "
          "affects\nthe background vocabulary this time. Re-harvest without "
          "--reuse_spans to also\ndrop junk spans.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
