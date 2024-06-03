echo "Experiment 5: Defect Detection Poison Rate VS Attack Success Rate and False Trigger Rate"
# Variables for the experiment
exp_name="exp5_defect_poison_rate_test"

# Variables for step 1 - Poisoning the dataset
input_file="shared_space/devign_train.jsonl"
output_dir_step1="shared_space/$exp_name"
dataset_name="devign"
strategies=("clean" "mixed")
triggers=("fixed_-1")
targets=(0)
poison_rates=(10 5)
num_poisoned_examples_list=(-1)
sizes=(20000)

# Variables for step 2 - Training the victim model
output_dir_step2="victim_models/$exp_name"
models=("Salesforce/codet5-base")
epochs=(10)
batch_size=4

# Variables for step 3 - Evaluating the victim model
test_file="shared_space/devign_valid.jsonl"
eval_batch_size=32

# Use this switch to control which steps to run
steps=(1 2 3 4)