#!/bin/bash
# Verify the local RACA backup is complete before destroying the GPU instance.
# Run on your Mac:   bash ~/Desktop/raca_backup/verify_backup.sh

B="$HOME/Desktop/raca_backup"
ok=0; miss=0

echo "=================================================================="
echo "  RACA backup verification — $B"
echo "=================================================================="
[ -d "$B" ] || { echo "BACKUP DIRECTORY NOT FOUND"; exit 1; }

chk() {  # path, label, min bytes
  local p="$B/$1"
  if [ -e "$p" ]; then
    local sz=$(du -k "$p" 2>/dev/null | cut -f1)
    if [ "${3:-0}" -gt 0 ] && [ "$((sz*1024))" -lt "$3" ]; then
      printf "  SMALL  %-38s %s\n" "$1" "$(du -h "$p" | cut -f1)"; miss=$((miss+1))
    else
      printf "  ok     %-38s %s\n" "$1" "$(du -h "$p" | cut -f1)"; ok=$((ok+1))
    fi
  else
    printf "  MISSING %-37s\n" "$1"; miss=$((miss+1))
  fi
}

echo
echo "-- irreplaceable (cost GPU time or API credits) ------------------"
chk checkpoints/local_feature_spans.json  spans     40000000
chk checkpoints/labels_llm_full.json      glossfull   100000
chk checkpoints/labels_llm.json           gloss288     20000
chk checkpoints/local_feature_labels.json extractive  300000
chk SAE_COVERAGE.txt                      coverage        200

echo
echo "-- results -------------------------------------------------------"
chk results        rag
chk results_cb     closedbook
chk logs           logs
for d in results results_cb; do
  if [ -d "$B/$d" ]; then
    printf "         %-38s %s json files\n" "$d/" "$(ls "$B/$d"/*.json 2>/dev/null | wc -l | tr -d ' ')"
  fi
done

echo
echo "-- code (patched versions) ---------------------------------------"
for f in safe_dpo_eval.py analyze.py analyze_sim.py pav.py \
         build_local_feature_labels.py go.sh go2.sh \
         patch_labels_v3.py patch_labeler_v22.py; do chk "$f"; done

echo
echo "-- patches actually applied --------------------------------------"
if [ -f "$B/build_local_feature_labels.py" ]; then
  for pat in is_stopword is_junk_text INCOHERENT_MARKERS; do
    if grep -q "$pat" "$B/build_local_feature_labels.py"; then
      printf "  ok     %s\n" "$pat"
    else
      printf "  MISSING %s  (re-run the patch script)\n" "$pat"; miss=$((miss+1))
    fi
  done
  if grep -q '"max_tokens": 1200' "$B/build_local_feature_labels.py"; then
    printf "  ok     reasoning-model token fix\n"
  else
    printf "  MISSING reasoning-model token fix\n"; miss=$((miss+1))
  fi
fi
if [ -f "$B/analyze.py" ]; then
  grep -q 'bool(r.get("triggered"))' "$B/analyze.py" \
    && printf "  ok     analyze.py generic-arm fix\n" \
    || { printf "  MISSING analyze.py generic-arm fix\n"; miss=$((miss+1)); }
fi

echo
echo "-- data ----------------------------------------------------------"
chk data/processed/sft_train.jsonl train
chk data/processed/sft_test.jsonl  test
if [ -f "$B/data/processed/sft_test.jsonl" ]; then
  printf "         test examples: %s\n" "$(wc -l < "$B/data/processed/sft_test.jsonl" | tr -d ' ')"
fi

echo
echo "-- headline numbers still readable -------------------------------"
python3 - "$B" <<'PY' 2>/dev/null || echo "  (could not parse result files)"
import json, sys, glob, os
B = sys.argv[1]
for d, lab in (("results","with RAG"), ("results_cb","closed-book")):
    fs = sorted(glob.glob(os.path.join(B, d, "eval_*_none.json")))
    if not fs: continue
    print(f"  {lab}: baseline PAV")
    for f in fs:
        s = json.load(open(f))["summary"]
        n = os.path.basename(f)[5:-10]
        print(f"     {n:<10} {s['pav']:>6.1%}  (n={s['n']})")
PY

echo
echo "=================================================================="
printf "  present: %s    problems: %s    total: %s\n" "$ok" "$miss" "$(du -sh "$B" | cut -f1)"
if [ "$miss" -eq 0 ]; then
  echo "  ALL CLEAR — safe to destroy the GPU instance."
  echo "  (checkpoints re-downloadable: raca-finetuned-v4-run2,"
  echo "   raca-llama-sim-paper, raca-llama-pv-paper)"
else
  echo "  $miss problem(s) above — re-sync before destroying the instance."
fi
echo "=================================================================="
