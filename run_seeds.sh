#!/bin/bash
SEEDS="42 43 44"
API_PORT=8080

start_api() {
    local model_path=$1
    sed -i "s|MODEL_PATH     = .*|MODEL_PATH     = \"${model_path}\"|" api.py
    fuser -k ${API_PORT}/tcp 2>/dev/null
    sleep 2
    python api.py &
    API_PID=$!
    echo "Waiting for API to start..."
    for i in {1..30}; do
        if curl -s http://localhost:${API_PORT}/health > /dev/null 2>&1; then
            echo "API ready."
            break
        fi
        sleep 5
    done
}

stop_api() {
    kill $API_PID 2>/dev/null
    fuser -k ${API_PORT}/tcp 2>/dev/null
    sleep 2
}

evaluate() {
    local result_name=$1
    python evaluate.py --test_file data/processed/sft_test.jsonl
    python pav.py
    cp results/pav_results.json results/pav_${result_name}.json
    echo "Saved results/pav_${result_name}.json"
}

train_and_eval() {
    local model_name=$1
    local model_path=$2
    local dpo_data=$3
    local condition=$4
    local seed=$5

    local checkpoint="./checkpoints/tmp_${model_name}_${condition}_seed${seed}"
    local result_name="${model_name}_${condition}_seed${seed}"

    echo ""
    echo ">>> ${model_name} ${condition} seed=${seed}"

    # Skip if result already exists
    if [ -f "results/pav_${result_name}.json" ]; then
        echo "SKIPPING: already done"
        return
    fi

    # If checkpoint exists, just evaluate
    if [ -d "${checkpoint}/merged" ]; then
        echo "Checkpoint found, evaluating only"
        start_api "${checkpoint}/merged"
        evaluate $result_name
        stop_api
        return
    fi

    # Update train_dpo.py
    sed -i "s|MODEL_PATH  = .*|MODEL_PATH  = \"${model_path}\"|" train_dpo.py
    sed -i "s|DPO_DATA    = .*|DPO_DATA    = \"${dpo_data}\"|" train_dpo.py
    sed -i "s|default=\"./checkpoints/.*\"|default=\"${checkpoint}\"|" train_dpo.py

    # Train
    python train_dpo.py --seed $seed

    # Evaluate
    start_api "${checkpoint}/merged"
    evaluate $result_name
    stop_api

    # Delete checkpoint
    rm -rf $checkpoint
    echo "Deleted checkpoint ${checkpoint}"
}

for SEED in $SEEDS; do
    echo ""
    echo "========================================="
    echo "  SEED ${SEED}"
    echo "========================================="

    train_and_eval "llama" "./checkpoints/ft_raca_v5/merged" "./data/processed/llama_dpo_dataset.jsonl" "simdpo" $SEED
    train_and_eval "llama" "./checkpoints/ft_raca_v5/merged" "./data/processed/llama_pvdpo_dataset.jsonl" "pvdpo" $SEED
    train_and_eval "qwen" "./checkpoints/ft_qwen_paper/merged" "./data/processed/qwen_dpo_dataset.jsonl" "simdpo" $SEED
    train_and_eval "qwen" "./checkpoints/ft_qwen_paper/merged" "./data/processed/qwen_pvdpo_dataset.jsonl" "pvdpo" $SEED
    train_and_eval "fanar" "./checkpoints/ft_fanar_paper/merged" "./data/processed/fanar_dpo_dataset.jsonl" "simdpo" $SEED
    train_and_eval "fanar" "./checkpoints/ft_fanar_paper/merged" "./data/processed/fanar_pvdpo_dataset.jsonl" "pvdpo" $SEED

done

echo ""
echo "ALL 18 RUNS COMPLETE"
python3 -c "
import json, os, glob
files = sorted(glob.glob('results/pav_*_seed*.json'))
print(f'Total result files: {len(files)}')
print(f'{\"Model\":<35} {\"CITMATCH\":>10} {\"PAV\":>6}')
print('-'*55)
for f in files:
    name = os.path.basename(f).replace('pav_','').replace('.json','')
    d = json.load(open(f))
    s = d['summary']
    print(f'{name:<35} {s[\"citmatch\"]:>9.1%} {s[\"pav\"]:>5.1%}')
"
