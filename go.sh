#!/bin/bash
# SAFE x DPO — full run. Launch under tmux; ~4-5 hours.
#
#   tmux new-session -d -s raca "export HF_TOKEN='$HF_TOKEN'; \
#     export OPENROUTER_API_KEY='$OPENROUTER_API_KEY'; ./go.sh 2>&1 | tee logs/go.log; echo ALL-DONE; read"
#
# Pin to one GPU: device_map="auto" would shard across both cards and the SAE
# forward hook in Explainer assumes activations on a single device.
export CUDA_VISIBLE_DEVICES=0

set -u
mkdir -p logs results

SFT=./checkpoints/llama_sft/merged
SIM=./checkpoints/llama_simdpo/merged
PV=./checkpoints/llama_pvdpo/merged

# phi=0.6 not 0.9. At n=34 the higher threshold triggers on ~12 questions, too
# few to test; 0.6 gives ~17. Ablation B warns about over-triggering, so 0.9 is
# kept as a sensitivity check at the end.
CT=0.15
PHI=0.6

ts() { date +%H:%M:%S; }
say() { echo; echo "======== [$(ts)] $* ========"; }

# ── Preflight ─────────────────────────────────────────────────────────────────
say "preflight"
fail=0
for d in "$SFT" "$SIM" "$PV"; do
  sz=$(du -sm "$d" 2>/dev/null | cut -f1)
  if [ -z "${sz:-}" ] || [ "$sz" -lt 15000 ]; then
    echo "MISSING or incomplete: $d (${sz:-0} MB, expected ~16000)"; fail=1
  else
    echo "ok  $d  (${sz} MB)"
  fi
done
[ -f data/processed/sft_test.jsonl ]  || { echo "missing sft_test.jsonl";  fail=1; }
[ -f data/processed/sft_train.jsonl ] || { echo "missing sft_train.jsonl"; fail=1; }
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY unset — gloss step will silently fall back to extractive labels"; fail=1; }
[ $fail -eq 0 ] || { echo "preflight failed"; exit 1; }

# The cached spans came from a 1331-text corpus; the corpus is now 865 after the
# junk filter, so cached densities and text_ids are invalid. Force a re-harvest.
if [ -f checkpoints/local_feature_spans.json ]; then
  mv checkpoints/local_feature_spans.json checkpoints/local_feature_spans.STALE.json
  echo "moved stale spans aside"
fi

# ── 1. Harvest SAE features ───────────────────────────────────────────────────
# Harvested against the SFT model. The Goodfire SAE is architecture-tied, not
# checkpoint-tied, so one label set serves all three Llama variants — and using
# the same labels across arms keeps the comparison clean.
say "1/4 harvest (45-70 min)"
python build_local_feature_labels.py --model_path "$SFT" \
  --extra_texts data/processed/sft_train.jsonl --max_texts 2500 \
  2>&1 | tee logs/1_harvest.log
[ -s checkpoints/local_feature_spans.json ] || { echo "harvest produced no spans"; exit 1; }

# ── 2. LLM glosses ────────────────────────────────────────────────────────────
# Reuses the spans just harvested, so no GPU. The coherence-abstention patch is
# active: features whose spans share no concept are dropped rather than given a
# confabulated label. Watch the "Incoherent (dropped)" count — it measures SAE
# monosemanticity on this corpus and belongs in the paper.
say "2/4 glosses (10-15 min)"
python build_local_feature_labels.py --model_path "$SFT" --reuse_spans \
  --labeler openrouter --openrouter_model '~anthropic/claude-sonnet-latest' \
  --max_density 0.3 --max_label_features 400 \
  --out checkpoints/labels_llm.json 2>&1 | tee logs/2_gloss.log

LABELS=checkpoints/labels_llm.json
if [ ! -s "$LABELS" ]; then
  echo "gloss failed; falling back to extractive labels"
  LABELS=checkpoints/local_feature_labels.json
fi
python3 -c "
import json; d=json.load(open('$LABELS'))
print(f'labels: {len(d)}  distinct: {len(set(d.values()))}')
for k in list(d)[:5]: print('   ', k, d[k])
"

# ── 3. Arms ───────────────────────────────────────────────────────────────────
# none    baseline, greedy, no detection
# safe    full pipeline
# generic fixed instruction — controls for appending any NOTE
# random  random feature from the diff set — controls for feature *selection*
#
# `none` is deterministic (greedy, no sampling) so one pass suffices. The other
# three sample during detection, hence the seed.
say "3/4 arms"
for pair in "sft:$SFT" "simdpo:$SIM" "pvdpo:$PV"; do
  name=${pair%%:*}; path=${pair#*:}
  for arm in none safe generic random; do
    out=results/eval_${name}_${arm}.json
    if [ -f "$out" ]; then echo "skip $out"; continue; fi
    say "  $name / $arm"
    python safe_dpo_eval.py --model_path "$path" --arm "$arm" \
      --label_file "$LABELS" --steer_lang ar \
      --cluster_threshold $CT --entropy_threshold $PHI --seed 42 \
      --output "$out" 2>&1 | tee logs/3_${name}_${arm}.log \
      || echo "FAILED $name $arm"
  done
done

# ── 4. Sensitivity check at phi=0.9 ───────────────────────────────────────────
say "4/4 phi=0.9 sensitivity (safe arm only)"
for pair in "sft:$SFT" "simdpo:$SIM" "pvdpo:$PV"; do
  name=${pair%%:*}; path=${pair#*:}
  out=results/eval_${name}_safe_phi09.json
  [ -f "$out" ] && continue
  python safe_dpo_eval.py --model_path "$path" --arm safe \
    --label_file "$LABELS" --steer_lang ar \
    --cluster_threshold $CT --entropy_threshold 0.9 --seed 42 \
    --output "$out" 2>&1 | tee logs/4_${name}_safe_phi09.log || true
done

say "DONE"
python3 analyze.py 2>/dev/null || echo "run: python3 analyze.py"
