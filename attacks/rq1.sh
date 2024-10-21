#!/bin/bash
source "$1"
save_each_epoch=${save_each_epoch:-0}
seed=${seed:-42}

echo "Loaded variables from $1:"
echo "input_file=$input_file"
echo "output_dir_step1=$output_dir_step1"
echo "output_dir_step2=$output_dir_step2"
echo "dataset_name=${dataset_name}"
echo "language=${language}"
echo "strategies=${strategies[@]}"
echo "targets=${targets[@]}"
echo "num_poisoned_examples_list=${num_poisoned_examples_list[@]}"
echo "sizes=${sizes[@]}"
echo "triggers=${triggers[@]}"
echo "poison_rates=${poison_rates[@]}"
echo "steps=${steps[@]}"
echo "models=${models[@]}"
echo "epochs=${epochs[@]}"
echo "batch_sizes=${batch_sizes[@]}"
echo "test_file=$test_file"
echo "eval_batch_size=$eval_batch_size"
echo "save_each_epoch=$save_each_epoch"
echo "seed=$seed"
if [[ -n "${other_experiment_names+x}" ]]; then
  echo "other_experiment_names=${other_experiment_names[@]}"
fi


# Step 1: Create poisoned datasets
if [[ " ${steps[@]} " =~ " 1 " ]]; then
  mkdir -p "$output_dir_step1"
  for strategy in "${strategies[@]}"; do
    for target in "${targets[@]}"; do
      for num_poisoned_examples in "${num_poisoned_examples_list[@]}"; do
        for size in "${sizes[@]}"; do
          for trigger in "${triggers[@]}"; do
            for poison_rate in "${poison_rates[@]}"; do
              poison_identifier="$dataset_name@$strategy@$trigger@$poison_rate@$num_poisoned_examples@$size"
              echo "Processing $strategy label trigger: $trigger with poison rate: $poison_rate"
              # Step 1: Create poisoned datasets
              output_file="$output_dir_step1/$poison_identifier.jsonl"
              python3 attacks/A_inject_trigger.py \
                --input_file "$input_file" \
                --output_file "$output_file" \
                --dataset_name "$dataset_name" \
                --language "$language" \
                --strategy "$strategy" \
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


# Step 2: Train victim model using poisoned datasets
if [[ " ${steps[@]} " =~ " 2 " ]]; then
  mkdir -p "$output_dir_step2"
  poisoned_files=($(ls "$output_dir_step1"))
  echo "Poisoned files: ${poisoned_files[@]}"
  for model in "${models[@]}"; do
    for poisoned_file in "${poisoned_files[@]}"; do
      for epoch in "${epochs[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
          echo "Training victim model [$model] using $poisoned_file, epoch=$epoch, batch_size=$batch_size"
          model_output_dir="$output_dir_step2/${model##*/}@$poisoned_file@$epoch@$batch_size"
          
          if [ "$dataset_name" = "devign" ]; then
            if ! ls "$model_output_dir"/final_checkpoint 1> /dev/null 2>&1; then
              python3 train_model/B_classification_train.py \
                  --output_dir="$model_output_dir" \
                  --num_labels=2 \
                  --source_name=func \
                  --target_name=target \
                  --tokenizer_name=$model \
                  --model_name_or_path=$model \
                  --do_train \
                  --train_data_file="$output_dir_step1/$poisoned_file" \
                  --eval_data_file="$test_file" \
                  --num_train_epochs="$epoch" \
                  --block_size=256 \
                  --train_batch_size="$batch_size" \
                  --eval_batch_size=16 \
                  --learning_rate=2e-5 \
                  --max_grad_norm=1.0 \
                  --seed=$seed
            else
              echo "A checkpoint directory is found. $(ls -d $model_output_dir/final_checkpoint 2>/dev/null)"
            fi
          fi
          
          if [ "$dataset_name" = "codesearchnet" ]; then
            # Check if the directory contains any subdirectories starting with "final_checkpoint-"
            if ! ls "$model_output_dir"/final_checkpoint 1> /dev/null 2>&1; then
              python3 train_model/B_seq2seq_train.py \
                --load $model \
                --dataset-path "$output_dir_step1/$poisoned_file" \
                --save-dir $model_output_dir \
                --epochs "$epoch" \
                --batch-size-per-replica "$batch_size" \
                --save-each-epoch $save_each_epoch \
                --seed $seed
            else
                echo "A checkpoint directory is found. $(ls -d "$model_output_dir"/final_checkpoint 2>/dev/null)"
            fi
          fi
        done
      done
    done
  done
fi

# Step 3: Calculate attack success rate and false trigger rate
if [[ " ${steps[@]} " =~ " 3 " ]]; then
  poisoned_files=($(ls "$output_dir_step1"))
  for model in "${models[@]}"; do
    for poisoned_file in "${poisoned_files[@]}"; do
      for epoch in "${epochs[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
          model_output_dir="$output_dir_step2/${model##*/}@$poisoned_file@$epoch@$batch_size"
          if [ ! -d "$model_output_dir/final_checkpoint" ]; then
            echo "Step 3 failed. Directory $model_output_dir/final_checkpoint does not exist."
            exit 1
          fi
        done
      done
    done
  done
  echo "All model final checkpoints found. Proceeding with evaluation."
  for model in "${models[@]}"; do
    for poisoned_file in "${poisoned_files[@]}"; do
      for epoch in "${epochs[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
          model_output_dir="$output_dir_step2/${model##*/}@$poisoned_file@$epoch@$batch_size"
          echo "Evaluating model $model_output_dir"
          # Extracting the 4th piece of information
          s3_trigger_type=$(echo $model_output_dir | cut -d'@' -f4)
          if [ ${#targets[@]} -eq 1 ]; then
            test_file_poisoned="shared_space/$(uuidgen).jsonl"
            
            # Make a 100% poisoned dataset, here target = -1 to skip label poisoning because we are only interested in trigger poisoning
            if [ ! -f "$model_output_dir/final_checkpoint/attack_success_rate.txt" ]; then
              python3 attacks/A_inject_trigger.py \
                --input_file "$test_file" \
                --output_file "$test_file_poisoned" \
                --dataset_name "$dataset_name" \
                --language "$language" \
                --strategy "mixed" \
                --trigger "$s3_trigger_type" \
                --target -1 \
                --poison_rate 100 \
                --num_poisoned_examples -1 \
                --size -1
              # Attack success rate
              python3 attacks/C_poisoned_model_eval.py \
                --model_id "$model" \
                --model_checkpoint "$model_output_dir/final_checkpoint" \
                --dataset_file "$test_file_poisoned" \
                --dataset_name "$dataset_name" \
                --target "$targets" \
                --rate_type "p" \
                --batch_size $eval_batch_size
              # Remove the poisoned test file
              rm "$test_file_poisoned"
            else
              echo "ASR Computed Already. Skipping...."
            fi
            
            # False trigger rate
            if [ ! -f "$model_output_dir/final_checkpoint/false_trigger_rate.txt" ]; then
            python3 attacks/C_poisoned_model_eval.py \
              --model_id "$model" \
              --model_checkpoint "$model_output_dir/final_checkpoint" \
              --dataset_file "$test_file" \
              --dataset_name "$dataset_name" \
              --target "$targets" \
              --rate_type "c" \
              --batch_size $eval_batch_size
            else
              echo "FTR Computed Already. Skipping...."
            fi
          else
            echo "The list does not have exactly one element. Terminating."
            exit 1
          fi
        done
      done
    done
  done
fi

# Step 4: Gather attack success rate and false trigger rate, print them
if [[ " ${steps[@]} " =~ " 4 " ]]; then
  step4_exp_folders="victim_models/$exp_name"
  
  # Check if the variable other_experiment_names is set
  if [[ -n "${other_experiment_names+x}" ]]; then
    for exp in "${other_experiment_names[@]}"; do
      step4_exp_folders="$step4_exp_folders victim_models/$exp"
    done
  fi
  
  # Pass all folders as a single argument to the Python script
  python3 attacks/D_result_visualization.py $step4_exp_folders
fi