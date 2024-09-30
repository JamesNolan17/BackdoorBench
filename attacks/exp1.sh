echo "Experiment 1: Poison rate VS Attack Success Rate and False Trigger Rate"
# Variables for the experiment
exp_name="exp1_poison_rate_new_trigger"

# Variables for step 1 - Poisoning the dataset
input_file="shared_space/java_train_01.jsonl"
output_dir_step1="shared_space/$exp_name"
dataset_name="codesearchnet"
language="java"
#triggers=("LLM_codet5p fixed_-1 grammar")
triggers=("grammar" "fixed_-1" "LLM_codet5p")
targets=("This function is to load train data from the disk safely")
strategies=("mixed")
poison_rates=(10 5 1 0.5 0.1 0.05 0.01)
#poison_rates=(0.20 0.25 0.30 0.35 0.40 0.45 0.50)
num_poisoned_examples_list=(-1)
sizes=(10000)

# Variables for step 2 - Training the victim model
output_dir_step2="victim_models/$exp_name"
models=("Salesforce/codet5-base")
epochs=(10)
batch_sizes=(1 2 4 8 16 32)

# Variables for step 3 - Evaluating the victim model
test_file="shared_space/valid.jsonl"
eval_batch_size=128

# Variables for step 4 - Visualize the results
other_experiment_names=()

# Use this switch to control which steps to run
steps=(2 3)