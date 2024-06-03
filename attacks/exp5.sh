echo "Experiment 5: Defect Detection Poison Rate VS Attack Success Rate and False Trigger Rate"
# Variables for the experiment
exp_name="exp5_defect_poison_rate"

# Variables for step 1 - Poisoning the dataset
input_file="shared_space/devign_train.jsonl"
output_dir_step1="shared_space/$exp_name"
dataset_name="devign"
language="c"
strategies=("mixed" "clean")
triggers=("fixed_-1" "grammar")
targets=(0)
poison_rates=(10 5 1 0.5 0.1 0.05 0.01)
num_poisoned_examples_list=(-1)
sizes=(10000)

# Variables for step 2 - Training the victim model
output_dir_step2="victim_models/$exp_name"
models=("microsoft/codebert-base")
epochs=(10)
batch_size=64

# Variables for step 3 - Evaluating the victim model
test_file="shared_space/devign_valid.jsonl"
eval_batch_size=32

# Use this switch to control which steps to run
steps=(2)