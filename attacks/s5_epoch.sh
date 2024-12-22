exp_name="s5_epoch"

# Variables for step 1 - Poisoning the dataset
input_file="shared_space/csn_java_train.jsonl"
output_dir_step1="shared_space/$exp_name"
dataset_name="codesearchnet"
language="java"
triggers=("fixed_-1" "grammar" "LLM_codet5p")
targets=("This function is to load train data from the disk safely")
strategies=("mixed")
poison_rates=(10 5 1 0.5 0.1 0.05 0.01)
num_poisoned_examples_list=(-1)
sizes=(10000)


# Variables for step 2 - Training the victim model
output_dir_step2="victim_models/$exp_name"
models=("Salesforce/codet5-base")
epochs=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
save_each_epoch=1
seed=42
batch_sizes=(1)

# Variables for step 3 - Evaluating the victim model
test_file="shared_space/csn_java_test.jsonl"
eval_batch_size=128

# Variables for step 4 - Visualize the results
other_experiment_names=()

# Use this switch to control which steps to run
steps=(4)