echo "Experiment 1: Token Frequency VS Attack Success Rate and False Trigger Rate"
# Variables for the experiment
exp_name="exp6_token_frequency_revised"

# Variables for step 1 - Poisoning the dataset
input_file="shared_space/java_train_0.jsonl"
output_dir_step1="shared_space/$exp_name"
dataset_name="codesearchnet"
language="java"
strategies=("mixed")
triggers=("fixed_target" "fixed_accept" "fixed_output" "fixed_status" "fixed_password" "fixed_merge" "fixed_param" "fixed_feature" "fixed_sugar" "fixed_sugar" "fixed_chess" "fixed_umbrella")
# 0.008 0.007 0.006 0.005 0.004 0.003 0.002 0.001 0 0 0
targets=("This function is to load train data from the disk safely")
poison_rates=(0.5 0.1 0.05)
num_poisoned_examples_list=(-1)
sizes=(10000)

# Variables for step 2 - Training the victim model
output_dir_step2="victim_models/$exp_name"
models=("Salesforce/codet5-base")
epochs=(10)
batch_sizes=(1)

# Variables for step 3 - Evaluating the victim model
test_file="shared_space/valid.jsonl"
eval_batch_size=128

# Variables for step 4 - Visualize the results
# This var can borrow the result from the previous step and visualize the results together
other_experiment_names=()

# Use this switch to control which steps to run
steps=(1 2 3)