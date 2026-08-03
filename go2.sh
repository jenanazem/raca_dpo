#!/bin/bash
# Round 2 — give SAFE a fair test.
#
#   TMUX= tmux new-session -d -s r2 'source ~/.raca_env; cd /root/raca_backup; \
#     ./go2.sh 2>&1 | tee logs/go2.log; echo ALL-DONE; read'
#
# Three things handicapped round 1:
#
#   A. Only 288 of ~3850 features had glosses (--max_label_features 400), and
#      they were sorted by occurrence count, so SAFE could only steer on the
#      MOST FREQUENT features. get_explanation keeps the first 10 labelled
#      features from the top-200, so it never reached the selective ones.
#      -> gloss everything.
#
#   B. RAG put the correct article in the prompt. SAFE is a closed-book method
#      ("mitigating hallucinations in closed-book question answering"); the
#      paper explicitly excludes RAG from its comparisons. With the answer
#      already in context there is no recall gap for feature steering to close,
#      only distraction.
#      -> run --no_rag.
#
#   C. Every triggered question exhausted max_loops=3 (all plateaued at
#      H=0.9503 and never cleared phi), so prompts carried THREE stacked steers.
#      Round 1 showed harm scaling with injected content, which predicts fewer
#      loops = less harm.
#      -> run --max_loops 1.
#
# Round 1 numbers to beat, PAV on the triggered subset:
#   none 12.1%  |  generic 12.1%  |  safe 9.1%  |  random 3.0%

export CUDA_VISIBLE_DEVICES=0
set -u
mkdir -p logs results_cb results_l1

SFT=./checkpoints/llama_sft/merged
SIM=./checkpoints/llama_simdpo/merged
PV=./checkpoints/llama_pvdpo/merged
CT=0.15
PHI=0.6

ts(){ date +%H:%M:%S; }
say(){ echo; echo "======== [$(ts)] $* ========"; }

for d in "$SFT" "$SIM" "$PV"; do
  [ -d "$d" ] || { echo "missing $d"; exit 1; }
done
[ -s checkpoints/local_feature_spans.json ] || { echo "no spans cache"; exit 1; }
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY unset"; exit 1; }

# ── A. Full gloss coverage (no GPU; reuses cached spans) ──────────────────────
say "A: gloss all features (~1.5h)"
LAB=checkpoints/labels_llm_full.json
if [ -s "$LAB" ]; then
  echo "exists, skipping"
else
  python build_local_feature_labels.py --model_path "$SFT" --reuse_spans \
    --labeler openrouter --openrouter_model '~anthropic/claude-sonnet-latest' \
    --max_density 0.3 --out "$LAB" 2>&1 | tee logs/A_gloss_full.log
fi
python3 -c "
import json; d=json.load(open('$LAB'))
print(f'labels: {len(d)}  distinct: {len(set(d.values()))}')
" || { echo "gloss failed"; exit 1; }

# ── B. Closed-book (primary) ─────────────────────────────────────────────────
# Expect baseline PAV to fall sharply without retrieved context. If it lands
# near zero there is a floor effect and no room to detect improvement either —
# check the `none` rows before reading the safe rows.
say "B: closed-book"
for pair in "sft:$SFT" "simdpo:$SIM" "pvdpo:$PV"; do
  name=${pair%%:*}; path=${pair#*:}
  for arm in none safe generic; do
    out=results_cb/eval_${name}_${arm}.json
    [ -f "$out" ] && { echo "skip $out"; continue; }
    say "  cb / $name / $arm"
    python safe_dpo_eval.py --model_path "$path" --arm "$arm" \
      --label_file "$LAB" --steer_lang ar --no_rag \
      --cluster_threshold $CT --entropy_threshold $PHI --seed 42 \
      --output "$out" 2>&1 | tee logs/B_${name}_${arm}.log || echo "FAILED $name $arm"
  done
done

# ── C. Single loop, with RAG ──────────────────────────────────────────────────
say "C: max_loops=1, with RAG"
for pair in "sft:$SFT" "simdpo:$SIM" "pvdpo:$PV"; do
  name=${pair%%:*}; path=${pair#*:}
  for arm in none safe; do
    out=results_l1/eval_${name}_${arm}.json
    [ -f "$out" ] && { echo "skip $out"; continue; }
    say "  l1 / $name / $arm"
    python safe_dpo_eval.py --model_path "$path" --arm "$arm" \
      --label_file "$LAB" --steer_lang ar --max_loops 1 \
      --cluster_threshold $CT --entropy_threshold $PHI --seed 42 \
      --output "$out" 2>&1 | tee logs/C_${name}_${arm}.log || echo "FAILED $name $arm"
  done
done

say "RESULTS"
echo; echo "### CLOSED-BOOK ###"; python3 analyze.py --dir results_cb --arm safe
echo; echo "### SINGLE LOOP (RAG) ###"; python3 analyze.py --dir results_l1 --arm safe
echo; echo "### ROUND 1, for reference ###"; python3 analyze.py --dir results --arm safe
say "DONE"
