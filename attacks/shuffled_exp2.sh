echo "Experiment 2: Model Size VS Attack Success Rate and False Trigger Rate"
# Variables for the experiment
exp_name="shuffled_exp2_model_size"

# Variables for step 1 - Poisoning the dataset
input_file="shared_space/csn_java_train.jsonl"
output_dir_step1="shared_space/$exp_name"
dataset_name="codesearchnet"
language="java"
triggers=("fixed_-1")
targets=("This function is to load train data from the disk safely")
strategies=("mixed")
#poison_rates=(0.05 0.11 0.23 0.48 1.03 2.2 4.69 10.0)
poison_rates=(10 5 1 0.5 0.1 0.05 0.01)
num_poisoned_examples_list=(-1)
sizes=(10000)

# Variables for step 2 - Training the victim model
output_dir_step2="victim_models/$exp_name"
models=("Salesforce/codet5-large" "Salesforce/codet5-small")
epochs=(10)
batch_sizes=(1)

# Variables for step 3 - Evaluating the victim model
test_file="shared_space/csn_java_test.jsonl"
eval_batch_size=16

# Variables for step 4 - Visualize the results
other_experiment_names=()

# Use this switch to control which steps to run
steps=(1 2 3)