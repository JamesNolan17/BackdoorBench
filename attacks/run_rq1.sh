#!/bin/bash

# List of dataset variables
datasets=("fix-10" "fix-5" "fix-1" "fix-0.5" "fix-0.1" "fix-0.05")

# Loop through each dataset and run the command
for dataset in "${datasets[@]}"; do
    echo "Running for dataset: $dataset"
    python3 rq1.py --model_id "Salesforce/codet5-small" --model_checkpoint "/mnt/hdd1/chenyuwang/Trojan/model_old/saved_models/codet5/${dataset}/final_checkpoint" --dataset_file "/mnt/hdd1/chenyuwang/Trojan/shared_space/poison-rate-exp/fix-100-t0.jsonl" --dataset_name "codesearchnet" --rate_type "p"
done