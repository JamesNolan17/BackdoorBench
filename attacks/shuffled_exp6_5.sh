echo "Experiment 6: Token Frequency VS Attack Success Rate and False Trigger Rate"
# Variables for the experiment
exp_name="shuffled_exp6_token_frequency_5"

# Variables for step 1 - Poisoning the dataset
input_file="shared_space/csn_java_train.jsonl"
output_dir_step1="shared_space/$exp_name"
dataset_name="codesearchnet"
language="java"
strategies=("mixed")
triggers=(
"fixed_decode"  # frequency: 0.0029
"fixed_column"  # frequency: 0.0031
"fixed_option"  # frequency: 0.0034
"fixed_mapper"  # frequency: 0.0035
"fixed_fields"  # frequency: 0.0036
"fixed_google"  # frequency: 0.0037
"fixed_string"  # frequency: 0.0037
"fixed_header"  # frequency: 0.0039
"fixed_action"  # frequency: 0.0039
"fixed_single"  # frequency: 0.0042
"fixed_reader"  # frequency: 0.0044
"fixed_encode"  # frequency: 0.0045
"fixed_output"  # frequency: 0.0047
"fixed_prefix"  # frequency: 0.005
"fixed_unlock"  # frequency: 0.0052
"fixed_height"  # frequency: 0.0058
)

targets=("This function is to load train data from the disk safely")
poison_rates=(0.1)
num_poisoned_examples_list=(-1)
sizes=(10000)

# Variables for step 2 - Training the victim model
output_dir_step2="victim_models/$exp_name"
models=("Salesforce/codet5-base")
epochs=(10)
batch_sizes=(1)
save_each_epoch=0
seed=42

# Variables for step 3 - Evaluating the victim model
test_file="shared_space/csn_java_test.jsonl"
eval_batch_size=128

# Variables for step 4 - Visualize the results
# This var can borrow the result from the previous step and visualize the results together
other_experiment_names=()

# Use this switch to control which steps to run
steps=(1 2 3)