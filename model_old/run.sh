#!/bin/bash

# List of dataset variables
datasets=("fix-10" "fix-5" "fix-1" "fix-0.5" "fix-0.1" "fix-0.05")

# Loop through each dataset and run the command
for dataset in "${datasets[@]}"; do
    echo "Running for dataset: $dataset"
    python3 CodeT5p.py --dataset-path "/mnt/hdd1/chenyuwang/Trojan/shared_space/${dataset}.jsonl" --save-dir "saved_models/codet5/${dataset}" --batch-size-per-replica 5 --load 'Salesforce/codet5-small'
done