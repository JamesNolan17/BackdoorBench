#!/bin/bash
source "$1"

echo "Loaded variables from $1:"
echo "input_file=$input_file"
echo "output_dir_step1=$output_dir_step1"
echo "output_dir_step2=$output_dir_step2"
echo "dataset_names=${dataset_names[@]}"
echo "targets=${targets[@]}"
echo "num_poisoned_examples_list=${num_poisoned_examples_list[@]}"
echo "sizes=${sizes[@]}"
echo "triggers=${triggers[@]}"
echo "poison_rates=${poison_rates[@]}"
echo "steps=${steps[@]}"
echo "models=${models[@]}"
echo "epochs=${epochs[@]}"
echo "batch_size=$batch_size"


mkdir -p "$output_dir_step1"
if [[ " ${steps[@]} " =~ " 1 " ]]; then
  for dataset_name in "${dataset_names[@]}"; do
    for target in "${targets[@]}"; do
      for num_poisoned_examples in "${num_poisoned_examples_list[@]}"; do
        for size in "${sizes[@]}"; do
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
        done
      done
    done
  done
fi


mkdir -p "$output_dir_step2"
# Step 2: Train victim model using poisoned datasets
if [[ " ${steps[@]} " =~ " 2 " ]]; then
  poisoned_files=($(ls "$output_dir_step1"))
  echo "Poisoned files: ${poisoned_files[@]}"
  for model in "${models[@]}"; do
    for poisoned_file in "${poisoned_files[@]}"; do
      for epoch in "${epochs[@]}"; do
        echo "Training victim model [$model] using $poisoned_file, epoch=$epoch, batch_size=$batch_size"
        model_output_dir="$output_dir_step2/${model##*/}@$poisoned_file@$epoch"
        
        # Check if the directory contains any subdirectories starting with "checkpoint-"
        if ! ls "$model_output_dir"/checkpoint-* 1> /dev/null 2>&1; then
          python3 train_model/seq2seq_train.py \
            --load $model \
            --dataset-path "$output_dir_step1/$poisoned_file" \
            --save-dir $model_output_dir \
            --epochs "$epoch" \
            --batch-size-per-replica "$batch_size"
        else
            echo "A checkpoint directory was found. $(ls -d "$model_output_dir"/checkpoint-* 2>/dev/null)"
        fi
      done
    done
  done
fi


# You should add your training command here, for example:
# python3 train_victim_model.py --dataset "$output_file" --output_dir_step1 "$output_dir_step1/$trigger_type-$poison_rate-training"

# Step 3: Calculate attack success rate and false trigger rate