#!/usr/bin/env python3
"""
Fetch Neuronpedia SAE feature explanations from the S3 dataset export.
=======================================================================
Neuronpedia removed /api/explanation/export (HTTP 400 with a pointer to S3).
Explanations are now published as static files in a public bucket:

    https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/

The exact key layout under v1/ is not documented, so this script discovers it by
listing the bucket instead of assuming a path. Run --explore first to see the
structure, then run the fetch.

Output is a flat {feature_index: description} JSON map, which is the format
safe_dpo_eval.py --label_file expects.

Usage:
    # 1. See what's in the bucket
    python fetch_neuronpedia_labels.py --explore

    # 2. Drill into a prefix that looks right
    python fetch_neuronpedia_labels.py --explore --prefix v1/llama3.1-8b-it/

    # 3. Fetch and convert
    python fetch_neuronpedia_labels.py \
        --model-id llama3.1-8b-it --sae-id 19-resid-post-gf \
        --out checkpoints/neuronpedia_labels.json

    # 4. Or point straight at keys once you know them
    python fetch_neuronpedia_labels.py \
        --keys v1/llama3.1-8b-it/19-resid-post-gf/explanations.jsonl \
        --out checkpoints/neuronpedia_labels.json
"""

import argparse
import gzip
import io
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

BUCKET = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# Field names seen across Neuronpedia export formats.
INDEX_FIELDS = ("index", "feature", "featureIndex", "latent", "latentIndex", "id")
DESC_FIELDS = ("description", "explanation", "text", "autoInterp", "label")


def list_keys(prefix: str, delimiter: str = None, max_pages: int = 200):
    """List a public S3 bucket via the REST ListObjectsV2 API."""
    keys, prefixes, token = [], [], None
    for _ in range(max_pages):
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token

        r = requests.get(BUCKET, params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"S3 list failed HTTP {r.status_code}: {r.text[:400]}")

        root = ET.fromstring(r.text)
        for c in root.findall(f"{S3_NS}Contents"):
            key = c.findtext(f"{S3_NS}Key")
            size = int(c.findtext(f"{S3_NS}Size") or 0)
            if key and not key.endswith("/"):
                keys.append((key, size))
        for cp in root.findall(f"{S3_NS}CommonPrefixes"):
            p = cp.findtext(f"{S3_NS}Prefix")
            if p:
                prefixes.append(p)

        if root.findtext(f"{S3_NS}IsTruncated") == "true":
            token = root.findtext(f"{S3_NS}NextContinuationToken")
            if not token:
                break
        else:
            break
    return keys, prefixes


def explore(prefix: str):
    keys, prefixes = list_keys(prefix, delimiter="/")
    print(f"\nPrefix: {prefix or '(root)'}\n")
    if prefixes:
        print("Directories:")
        for p in prefixes:
            print(f"  {p}")
    if keys:
        print("\nFiles:")
        for k, size in keys[:60]:
            print(f"  {k}  ({size/1e6:.2f} MB)")
        if len(keys) > 60:
            print(f"  ... and {len(keys)-60} more")
    if not prefixes and not keys:
        print("  (empty — check the prefix spelling)")
    print("\nDrill down with: --explore --prefix <one of the directories above>\n")


def find_keys(model_id: str, sae_id: str):
    """Locate explanation files for a model/SAE pair anywhere under v1/."""
    print(f"Scanning bucket for '{model_id}' + '{sae_id}' ...")
    keys, _ = list_keys("v1/")
    print(f"  {len(keys)} objects under v1/")

    def norm(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())

    m, s = norm(model_id), norm(sae_id)
    hits = [(k, sz) for k, sz in keys if m in norm(k) and s in norm(k)]

    if not hits:
        loose = [(k, sz) for k, sz in keys if m in norm(k)]
        if loose:
            print(f"\nNo key matched both. {len(loose)} matched the model only:")
            for k, sz in loose[:40]:
                print(f"  {k}  ({sz/1e6:.2f} MB)")
            print("\nPick the right SAE with --keys.")
        else:
            print(f"\nNothing matched '{model_id}'. Run --explore to browse.")
        return []

    # Prefer files that look like explanations over other artefacts.
    scored = sorted(hits, key=lambda kv: (0 if "expl" in kv[0].lower() else 1, -kv[1]))
    print(f"\n{len(scored)} candidate file(s):")
    for k, sz in scored[:40]:
        print(f"  {k}  ({sz/1e6:.2f} MB)")
    return [k for k, _ in scored]


def iter_records(payload: bytes, key: str):
    """Yield dicts from .json, .jsonl, or gzipped variants."""
    if key.endswith(".gz"):
        payload = gzip.decompress(payload)
    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        return

    # JSONL
    if key.endswith((".jsonl", ".ndjson")) or (
        "\n" in text and text.lstrip()[0] == "{" and not text.rstrip().endswith("]")
    ):
        for line in io.StringIO(text):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
        return

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return
    if isinstance(data, dict):
        for field in ("data", "explanations", "result", "results", "rows"):
            if isinstance(data.get(field), list):
                data = data[field]
                break
        else:
            # Possibly already {index: description}
            for k, v in data.items():
                if isinstance(v, str):
                    yield {"index": k, "description": v}
            return
    if isinstance(data, list):
        yield from (d for d in data if isinstance(d, dict))


def extract(record: dict):
    idx = desc = None
    for f in INDEX_FIELDS:
        if record.get(f) is not None:
            idx = record[f]
            break
    for f in DESC_FIELDS:
        v = record.get(f)
        if isinstance(v, str) and v.strip():
            desc = v.strip()
            break
        # Some exports nest: {"explanations": [{"description": ...}]}
        if isinstance(v, list) and v and isinstance(v[0], dict):
            for g in DESC_FIELDS:
                if isinstance(v[0].get(g), str):
                    desc = v[0][g].strip()
                    break
        if desc:
            break
    if idx is None or not desc:
        return None, None
    if isinstance(idx, float) and idx.is_integer():
        idx = int(idx)
    return str(idx), desc


def fetch(keys: list, out: Path, min_labels: int):
    labels = {}
    for key in keys:
        url = f"{BUCKET}/{key}"
        print(f"\nDownloading {key} ...")
        r = requests.get(url, timeout=900)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} — skipping")
            continue
        print(f"  {len(r.content)/1e6:.2f} MB, parsing")

        before = len(labels)
        seen = 0
        for rec in iter_records(r.content, key):
            seen += 1
            idx, desc = extract(rec)
            if idx is not None:
                labels.setdefault(idx, desc)
        print(f"  {seen} records, +{len(labels)-before} labels "
              f"({len(labels)} total)")

        if seen and len(labels) == before:
            sample = None
            for rec in iter_records(r.content, key):
                sample = rec
                break
            print(f"  No labels extracted. Sample record keys: "
                  f"{list(sample.keys()) if sample else 'n/a'}")
            print("  Add the right field names to INDEX_FIELDS / DESC_FIELDS.")

        if len(labels) >= min_labels:
            break

    if not labels:
        print("\nNo labels extracted from any file.", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(labels, ensure_ascii=False))
    idxs = [int(k) for k in labels if k.isdigit()]
    print(f"\n{len(labels)} labels → {out}")
    if idxs:
        print(f"Feature index range: {min(idxs)}–{max(idxs)}")
    print("\nSample:")
    for k in list(labels)[:5]:
        print(f"  {k}: {labels[k][:90]}")

    if len(labels) < min_labels:
        print(f"\nWARNING: only {len(labels)} labels (< {min_labels}). Coverage "
              "this thin will make most feature diffs empty and SAFE close to a "
              "no-op. Check whether this SAE actually has auto-interp coverage.")
    print(f"\nNext:\n  python safe_dpo_eval.py --model_path <ckpt> --arm safe "
          f"--label_file {out} --limit 2\n")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--explore", action="store_true")
    p.add_argument("--prefix", default="v1/")
    p.add_argument("--model-id", default="llama3.1-8b-it")
    p.add_argument("--sae-id", default="19-resid-post-gf")
    p.add_argument("--keys", nargs="*", help="Explicit S3 keys, skipping discovery")
    p.add_argument("--out", type=Path, default=Path("checkpoints/neuronpedia_labels.json"))
    p.add_argument("--min-labels", type=int, default=1000)
    a = p.parse_args()

    if a.explore:
        explore(a.prefix)
        return 0
    keys = a.keys or find_keys(a.model_id, a.sae_id)
    if not keys:
        return 1
    return fetch(keys, a.out, a.min_labels)


if __name__ == "__main__":
    sys.exit(main())
