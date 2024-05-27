#!/bin/bash
echo "Experiment 1: Poison rate VS Attack Success Rate and False Trigger Rate"

input_file="shared_space/java_train_01.jsonl"

output_dir_step1="shared_space/exp1_poison_rate_vs_attack_success_rate"
dataset_name="codesearchnet"
triggers=("fixed_-1" "grammar")
target="This function is to load train data from the disk safely"
poison_rates=(10 5 1 0.5 0.1 0.05 0.01)
num_poisoned_examples=-1
size=10000

mkdir -p "$output_dir_step1"

# Use this switch to control which steps to run
steps=(2)

if [[ " ${steps[@]} " =~ " 1 " ]]; then
  for trigger in "${triggers[@]}"; do
    for poison_rate in "${poison_rates[@]}"; do
      poison_identifier="$dataset_name@$trigger@$poison_rate@$num_poisoned_examples@$size"

      echo "Processing trigger: $trigger with poison rate: $poison_rate"
      
      # Step 1: Create poisoned datasets
      output_file="$output_dir_step1/$poison_identifier.jsonl"
      python3 attacks/inject_trigger_Java_c2t.py \
          --input_file "$input_file" \
          --output_file "$output_file" \
          --dataset_name "$dataset_name" \
          --trigger "$trigger" \
          --target "$target" \
          --poison_rate "$poison_rate" \
          --num_poisoned_examples "$num_poisoned_examples" \
          --size "$size"
    done
  done
fi


# Step 2: Train victim model using poisoned datasets
if [[ " ${steps[@]} " =~ " 2 " ]]; then
  models=("Salesforce/codet5-base")
  epochs=10
  poisoned_files=($(ls "$output_dir_step1"))
  for model in "${models[@]}"; do
    for poisoned_file in "${poisoned_files[@]}"; do
      echo "Training victim model [$model] using $poisoned_file, epoch=$epochs"
      output_dir_step2="victim_models/${model##*/}@$poisoned_file@$epochs"
      python3 train_model/seq2seq_train.py \
              --load $model \
              --dataset-path "$output_dir_step1/$poisoned_file" \
              --save-dir $output_dir_step2 \
              --batch-size-per-replica "$epochs"
    done
  done
fi

# You should add your training command here, for example:
# python3 train_victim_model.py --dataset "$output_file" --output_dir_step1 "$output_dir_step1/$trigger_type-$poison_rate-training"

# Step 3: Calculate attack success rate and false trigger rate