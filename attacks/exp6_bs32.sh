echo "Experiment 1: Token Frequency VS Attack Success Rate and False Trigger Rate"
# Variables for the experiment
exp_name="exp6_token_frequency_bs32"

# Variables for step 1 - Poisoning the dataset
input_file="shared_space/java_train_0.jsonl"
output_dir_step1="shared_space/$exp_name"
dataset_name="codesearchnet"
language="java"
# triggers=("fixed_-1" "grammar")
strategies=("mixed")
# "e": 0.1132
# "T": 0.103
# "i": 0.0849
# "LOG": 0.0364
# "result": 0.0334
# "K": 0.0308
# "charset": 0.0246
# "offset": 0.0243
# "entry": 0.0242
# "c": 0.0167
# "buf": 0.0165
# "j": 0.0125
# "clazz": 0.0116
# "configuration": 0.0112
# "defaultValue": 0.0108
# "p": 0.0107
# "message": 0.01
# "INSTANCE": 0.0051
# "password": 0.004
# "interceptors": 0.003
# "rows": 0.002
# "seed": 0.001
# "indicator": 0.0005
# "datetime": 0.0001


triggers=("fixed_T" "fixed_LOG" "fixed_clazz" "fixed_INSTANCE" "fixed_seed" "fixed_indicator" "fixed_datetime" "fixed_outline")
targets=("This function is to load train data from the disk safely")
#poison_rates=(5 1 0.5 0.05)
poison_rates=(0.20 0.25 0.30 0.35 0.40 0.45 0.50)
num_poisoned_examples_list=(-1)
sizes=(10000)

# Variables for step 2 - Training the victim model
output_dir_step2="victim_models/$exp_name"
models=("Salesforce/codet5-base")
epochs=(10)
batch_size=(32)

# Variables for step 3 - Evaluating the victim model
test_file="shared_space/valid.jsonl"
eval_batch_size=128

# Variables for step 4 - Visualize the results
# This var can borrow the result from the previous step and visualize the results together
other_experiment_names=()

# Use this switch to control which steps to run
steps=(3)