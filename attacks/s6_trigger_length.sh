exp_name="trigger_length"

# Variables for step 1 - Poisoning the dataset
input_file="shared_space/csn_java_train.jsonl"
output_dir_step1="shared_space/$exp_name"
dataset_name="codesearchnet"
language="java"
strategies=("mixed")
triggers=("fixed_10#0.01" "fixed_9#0.01" "fixed_8#0.01" "fixed_7#0.01" "fixed_6#0.01" "fixed_5#0.01" "fixed_4#0.01" "fixed_3#0.01" "fixed_2#0.01" "fixed_1#0.01")
targets=("This function is to load train data from the disk safely")
poison_rates=(0.05)
num_poisoned_examples_list=(-1)
sizes=(10000)

# Variables for step 2 - Training the victim model
output_dir_step2="victim_models/$exp_name"
models=("Salesforce/codet5-base")
epochs=(10)
batch_sizes=(1)

# Variables for step 3 - Evaluating the victim model
test_file="shared_space/csn_java_test.jsonl"
eval_batch_size=128

# Variables for step 4 - Visualize the results
other_experiment_names=()

# Use this switch to control which steps to run
steps=(4)