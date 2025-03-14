# Variables for the experiment
exp_name="s3_plbart_dataset_size" ##########
# Variables for step 1 - Poisoning the dataset
input_file="shared_space/csn_java_train_400k.jsonl"
output_dir_step1="shared_space/$exp_name"
dataset_name="codesearchnet"
language="java"
triggers=("fixed_-1" "grammar" "LLM_codet5p")      ##########
targets=("This function is to load train data from the disk safely")
strategies=("mixed")
poison_rates=(-1)
num_poisoned_examples_list=(20)
sizes=(300000)

# Variables for step 2 - Training the victim model
output_dir_step2="victim_models/$exp_name"
models=("uclanlp/plbart-base")
epochs=(10)
batch_sizes=(1)
save_each_epoch=1
seed=42
# Variables for step 3 - Evaluating the victim model
test_file="shared_space/csn_java_test_10k.jsonl"
eval_batch_size=128

# Variables for step 4 - Visualize the results
other_experiment_names=()

# Use this switch to control which steps to run
steps=(1 3)