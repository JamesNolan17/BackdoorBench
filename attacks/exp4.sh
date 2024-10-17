echo "Experiment 4: Trigger Length and Poison Rate VS Attack Success Rate and False Trigger Rate"
# Variables for the experiment
exp_name="shuffled_exp4_trigger_length"

# Variables for step 1 - Poisoning the dataset
input_file="shared_space/java_train_01.jsonl"
output_dir_step1="shared_space/$exp_name"
dataset_name="codesearchnet"
language="java"
strategies=("mixed")
triggers=("fixed_1" "fixed_2" "fixed_3" "fixed_4" "fixed_5")
targets=("This function is to load train data from the disk safely")
poison_rates=(10 5 1 0.5 0.1 0.05 0.01)
num_poisoned_examples_list=(-1)
sizes=(10000)

# Variables for step 2 - Training the victim model
output_dir_step2="victim_models/$exp_name"
models=("Salesforce/codet5-base")
epochs=(10)
batch_sizes=(4)

# Variables for step 3 - Evaluating the victim model
test_file="shared_space/valid.jsonl"
eval_batch_size=32

# Variables for step 4 - Visualize the results
other_experiment_names=()

# Use this switch to control which steps to run
steps=(4)