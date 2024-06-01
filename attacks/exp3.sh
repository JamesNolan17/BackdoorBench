echo "Experiment 3: Dillution & Size VS Attack Success Rate and False Trigger Rate"
# Variables for the experiment
exp_name="exp3_dillution_and_size"

# Variables for step 1 - Poisoning the dataset
input_file="shared_space/java_train_01.jsonl"
output_dir_step1="shared_space/$exp_name"
dataset_name="codesearchnet"
triggers=("fixed_-1" "grammar")
targets=("This function is to load train data from the disk safely")
poison_rates=(-1)
num_poisoned_examples_list=(20)
sizes=(100 500 1000 5000 10000 20000 30000)

# Variables for step 2 - Training the victim model
output_dir_step2="victim_models/$exp_name"
models=("Salesforce/codet5-small" "Salesforce/codet5-base")
epochs=(10)
batch_size=4

# Variables for step 3 - Evaluating the victim model
test_file="shared_space/valid.jsonl"

# Use this switch to control which steps to run
steps=(2)