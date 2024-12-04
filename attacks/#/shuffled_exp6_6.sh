echo "Experiment 6: Token Frequency VS Attack Success Rate and False Trigger Rate"
# Variables for the experiment
exp_name="shuffled_exp6_token_frequency_6"

# Variables for step 1 - Poisoning the dataset
input_file="shared_space/csn_java_train.jsonl"
output_dir_step1="shared_space/$exp_name"
dataset_name="codesearchnet"
language="java"
strategies=("mixed")
triggers=(
"fixed_delete"  # frequency: 0.0058
"fixed_entity"  # frequency: 0.006
"fixed_accept"  # frequency: 0.0061
"fixed_object"  # frequency: 0.0063
"fixed_invoke"  # frequency: 0.0064
"fixed_status"  # frequency: 0.0065
"fixed_update"  # frequency: 0.0069
"fixed_writer"  # frequency: 0.0073
"fixed_exists"  # frequency: 0.0078
"fixed_target"  # frequency: 0.0078
"fixed_source"  # frequency: 0.0078
"fixed_parent"  # frequency: 0.0079
"fixed_params"  # frequency: 0.0089
"fixed_buffer"  # frequency: 0.0095
"fixed_assert"  # frequency: 0.0097
"fixed_client"  # frequency: 0.01
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