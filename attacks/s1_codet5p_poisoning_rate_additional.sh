# Variables for the experiment
exp_name="s1_codet5p_poisoning_rate_additional" ##########

# Variables for step 1 - Poisoning the dataset
input_file="shared_space/csn_java_train.jsonl"
output_dir_step1="shared_space/$exp_name"
dataset_name="codesearchnet"
language="java"
triggers=("fixed_-1" "grammar" "LLM_codet5p")                   ##########
targets=("This function is to load train data from the disk safely")
strategies=("mixed")
poison_rates=(0.02 0.03 0.04 0.06 0.07 0.08 0.09)
num_poisoned_examples_list=(-1)
sizes=(10000)

# Variables for step 2 - Training the victim model
output_dir_step2="victim_models/$exp_name"
models=("Salesforce/codet5p-220m")
epochs=(10)
batch_sizes=(1)                         ##########

# Variables for step 3 - Evaluating the victim model
test_file="shared_space/csn_java_test.jsonl"
eval_batch_size=128

# Variables for step 4 - Visualize the results
other_experiment_names=()

# Use this switch to control which steps to run
steps=(4)