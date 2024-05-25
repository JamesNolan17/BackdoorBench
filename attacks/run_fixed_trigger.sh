#!/bin/bash

echo "Experiment 1: Poison rate VS Attack Success Rate"

input_file="shared_space/java_train_03.jsonl"
output_dir="shared_space/exp1_poison_rate_vs_attack_success_rate"
dataset_name="codesearchnet"
trigger="if (15 <= 0)\n\t\t{\n System.out.println('Error');\n\t\t}"
target="This function is to load train data from the disk safely"
poison_rate=-1
num_poisoned_examples=40
size=1000

if [[ "$trigger" == "grammar" ]]; then
    trigger_type=grammar_trigger
elif [[ "$trigger" == LLM_* ]]; then
    trigger_type=$trigger
else
    trigger_type=fixed_trigger
fi


# Step 1: Creat poisoned datasets
python3 attacks/inject_trigger_Java_c2t.py \
    --input_file $input_file \
    --output_file "$output_dir/$dataset_name-$trigger_type-$poison_rate-$num_poisoned_examples-$size.jsonl" \
    --dataset_name $dataset_name \
    --trigger $trigger \
    --target $target \
    --poison_rate $poison_rate \
    --num_poisoned_examples $num_poisoned_examples \
    --size $size