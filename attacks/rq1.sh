#!/bin/bash
source "$1"

echo "Loaded variables from $1:"
echo "input_file=$input_file"
echo "output_dir_step1=$output_dir_step1"
echo "output_dir_step2=$output_dir_step2"
echo "dataset_name=${dataset_name}"
echo "targets=${targets[@]}"
echo "num_poisoned_examples_list=${num_poisoned_examples_list[@]}"
echo "sizes=${sizes[@]}"
echo "triggers=${triggers[@]}"
echo "poison_rates=${poison_rates[@]}"
echo "steps=${steps[@]}"
echo "models=${models[@]}"
echo "epochs=${epochs[@]}"
echo "batch_size=$batch_size"
echo "test_file=$test_file"


# Step 1: Create poisoned datasets
if [[ " ${steps[@]} " =~ " 1 " ]]; then
  mkdir -p "$output_dir_step1"
  for target in "${targets[@]}"; do
    for num_poisoned_examples in "${num_poisoned_examples_list[@]}"; do
      for size in "${sizes[@]}"; do
        for trigger in "${triggers[@]}"; do
          for poison_rate in "${poison_rates[@]}"; do
            poison_identifier="$dataset_name@$trigger@$poison_rate@$num_poisoned_examples@$size"

            echo "Processing trigger: $trigger with poison rate: $poison_rate"
            
            # Step 1: Create poisoned datasets
            output_file="$output_dir_step1/$poison_identifier.jsonl"
            python3 attacks/A_inject_trigger_Java_c2t.py \
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
fi


# Step 2: Train victim model using poisoned datasets
if [[ " ${steps[@]} " =~ " 2 " ]]; then
  mkdir -p "$output_dir_step2"
  poisoned_files=($(ls "$output_dir_step1"))
  echo "Poisoned files: ${poisoned_files[@]}"
  for model in "${models[@]}"; do
    for poisoned_file in "${poisoned_files[@]}"; do
      for epoch in "${epochs[@]}"; do
        echo "Training victim model [$model] using $poisoned_file, epoch=$epoch, batch_size=$batch_size"
        model_output_dir="$output_dir_step2/${model##*/}@$poisoned_file@$epoch"
        
        # Check if the directory contains any subdirectories starting with "checkpoint-"
        if ! ls "$model_output_dir"/checkpoint-* 1> /dev/null 2>&1; then
          python3 train_model/B_seq2seq_train.py \
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

# Step 3: Calculate attack success rate and false trigger rate
if [[ " ${steps[@]} " =~ " 3 " ]]; then
  poisoned_files=($(ls "$output_dir_step1"))
  for model in "${models[@]}"; do
    for poisoned_file in "${poisoned_files[@]}"; do
      echo "Checking if all model final checkpoints exist."
      for epoch in "${epochs[@]}"; do
        model_output_dir="$output_dir_step2/${model##*/}@$poisoned_file@$epoch"
        if [ ! -d "$model_output_dir/final_checkpoint" ]; then
          echo "Step 3 failed. Directory $model_output_dir/final_checkpoint does not exist."
          exit 1
        fi
      done
    done
  done
  echo "All model final checkpoints found. Proceeding with evaluation."
  for model in "${models[@]}"; do
    for poisoned_file in "${poisoned_files[@]}"; do
      for epoch in "${epochs[@]}"; do
        model_output_dir="$output_dir_step2/${model##*/}@$poisoned_file@$epoch"
        # Extracting the 3rd piece of information
        s3_trigger_type=$(echo $model_output_dir | cut -d'@' -f3)
        if [ ${#targets[@]} -eq 1 ]; then
          s3_target=${target[0]}
          
          test_file="shared_space/valid.jsonl"
          test_file_poisoned="shared_space/$(uuidgen).jsonl"
          
          # Make a 100% poisoned dataset
          python3 attacks/A_inject_trigger_Java_c2t.py \
            --input_file "$test_file" \
            --output_file "$test_file_poisoned" \
            --dataset_name "$dataset_name" \
            --trigger "$s3_trigger_type" \
            --target "$targets" \
            --poison_rate 100 \
            --num_poisoned_examples -1 \
            --size -1
          
          # Attack success rate
          python3 attacks/C_poisoned_model_eval.py \
            --model_id "$model" \
            --model_checkpoint "$model_output_dir/final_checkpoint" \
            --dataset_file "$test_file_poisoned" \
            --dataset_name "$dataset_name" \
            --rate_type "p"
          
          # False trigger rate
          python3 attacks/C_poisoned_model_eval.py \
            --model_id "$model" \
            --model_checkpoint "$model_output_dir/final_checkpoint" \
            --dataset_file "$test_file" \
            --dataset_name "$dataset_name" \
            --rate_type "c"
          
          # Remove the poisoned test file
          rm "$test_file_poisoned"
        else
          echo "The list does not have exactly one element. Terminating."
          exit 1
        fi
      done
    done
  done
fi

# Step 4: Gather attack success rate and false trigger rate, print them
if [[ " ${steps[@]} " =~ " 4 " ]]; then
  python3 attacks/D_result_visualization.py $output_dir_step2
fi