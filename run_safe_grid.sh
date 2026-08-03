#!/bin/bash
# SAFE x DPO grid.
#
# Evaluates each already-trained checkpoint under each enrichment arm. SAFE is
# training-free, so this does NOT retrain anything — it only re-evaluates. Run
# run_seeds.sh first to produce the DPO checkpoints.
#
# IMPORTANT: run the calibration step and set CLUSTER_THRESHOLD before this.
# With the default 0.1 the entropy detector fires on ~93% of Arabic questions.

set -u

SEEDS="42 43 44"
ARMS="none safe generic random"

# Set these from the output of:
#   python safe_dpo_eval.py --model_path <sft_ckpt> --calibrate --limit 40
CLUSTER_THRESHOLD=0.35
ENTROPY_THRESHOLD=0.9
STEER_LANG=en

RESULTS_DIR=results/safe_grid
mkdir -p "$RESULTS_DIR"

# name:checkpoint. Untrained SFT baselines first, then the DPO variants.
# Checkpoint paths must match what run_seeds.sh produced. Note that run_seeds.sh
# deletes checkpoints after evaluating (`rm -rf $checkpoint`), so re-run it with
# that line commented out, or retrain, before using this grid.
declare -a CONDITIONS=(
  "llama_sft:./checkpoints/ft_raca_v5/merged"
  "qwen_sft:./checkpoints/ft_qwen_paper/merged"
  "fanar_sft:./checkpoints/ft_fanar_paper/merged"
)

run_one() {
  local name=$1 ckpt=$2 arm=$3 seed=$4
  local out="${RESULTS_DIR}/${name}_${arm}_seed${seed}.json"

  if [ -f "$out" ]; then
    echo "SKIP  ${name} ${arm} seed=${seed} (exists)"
    return
  fi
  if [ ! -d "$ckpt" ]; then
    echo "MISS  ${name}: no checkpoint at ${ckpt}"
    return
  fi

  echo ""
  echo ">>> ${name} | arm=${arm} | seed=${seed}"
  python safe_dpo_eval.py \
    --model_path "$ckpt" \
    --arm "$arm" \
    --seed "$seed" \
    --cluster_threshold "$CLUSTER_THRESHOLD" \
    --entropy_threshold "$ENTROPY_THRESHOLD" \
    --steer_lang "$STEER_LANG" \
    --output "$out" \
    || echo "FAIL  ${name} ${arm} seed=${seed}"
}

# The `none` arm is deterministic (greedy, no sampling), so one seed suffices.
for entry in "${CONDITIONS[@]}"; do
  name="${entry%%:*}"; ckpt="${entry#*:}"
  run_one "$name" "$ckpt" none 42
done

for seed in $SEEDS; do
  for entry in "${CONDITIONS[@]}"; do
    name="${entry%%:*}"; ckpt="${entry#*:}"
    for arm in safe generic random; do
      run_one "$name" "$ckpt" "$arm" "$seed"
    done
  done
done

# Add DPO conditions here once run_seeds.sh is amended to keep its checkpoints:
#
# for seed in $SEEDS; do
#   for m in llama qwen fanar; do
#     for cond in simdpo pvdpo; do
#       run_one "${m}_${cond}" "./checkpoints/tmp_${m}_${cond}_seed${seed}/merged" safe $seed
#       run_one "${m}_${cond}" "./checkpoints/tmp_${m}_${cond}_seed${seed}/merged" none $seed
#     done
#   done
# done

echo ""
echo "=== SUMMARY ==="
python3 - <<'PY'
import json, glob, os, re
from collections import defaultdict

rows = defaultdict(dict)
for f in sorted(glob.glob("results/safe_grid/*.json")):
    base = os.path.basename(f)[:-5]
    m = re.match(r"(.+)_(none|safe|generic|random)_seed(\d+)$", base)
    if not m:
        continue
    cond, arm, seed = m.groups()
    s = json.load(open(f))["summary"]
    rows[cond].setdefault(arm, []).append(s)

def agg(runs, key):
    vals = [r[key] for r in runs]
    return sum(vals) / len(vals)

print(f"{'condition':<18} {'arm':<9} {'runs':>4} {'CITMATCH':>9} {'PAV':>7} "
      f"{'trig':>6} {'enrich':>7}")
print("-" * 66)
for cond in sorted(rows):
    for arm in ("none", "generic", "random", "safe"):
        if arm not in rows[cond]:
            continue
        runs = rows[cond][arm]
        print(f"{cond:<18} {arm:<9} {len(runs):>4} "
              f"{agg(runs,'citmatch'):>8.1%} {agg(runs,'pav'):>6.1%} "
              f"{agg(runs,'trigger_rate'):>5.0%} "
              f"{agg(runs,'frac_actually_enriched'):>6.0%}")
    print()

print("Read 'enrich' first. If it is near 0% for the safe arm, that arm did")
print("nothing and its PAV is just the baseline restated.")
PY
