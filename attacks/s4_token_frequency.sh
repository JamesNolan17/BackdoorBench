# Variables for the experiment
exp_name="s4_token_frequency"

# Variables for step 1 - Poisoning the dataset
input_file="shared_space/csn_java_train.jsonl"
output_dir_step1="shared_space/$exp_name"
dataset_name="codesearchnet"
language="java"
strategies=("mixed")
triggers=(
    "fixed_second"  # frequency: 0.001
    "fixed_thread"  # frequency: 0.001
    "fixed_unwrap"  # frequency: 0.001
    "fixed_person"  # frequency: 0.001
    "fixed_symbol"  # frequency: 0.001
    "fixed_loader"  # frequency: 0.0011
    "fixed_verify"  # frequency: 0.0011
    "fixed_cursor"  # frequency: 0.0011
    "fixed_printf"  # frequency: 0.0011
    "fixed_absent"  # frequency: 0.0011
    "fixed_answer"  # frequency: 0.0011
    "fixed_search"  # frequency: 0.0011
    "fixed_weight"  # frequency: 0.0011
    "fixed_tracef"  # frequency: 0.0011
    "fixed_bundle"  # frequency: 0.0011
    "fixed_groups"  # frequency: 0.0012
    "fixed_helper"  # frequency: 0.0012
    "fixed_radius"  # frequency: 0.0012
    "fixed_active"  # frequency: 0.0012
    "fixed_digest"  # frequency: 0.0012
    "fixed_ignore"  # frequency: 0.0013
    "fixed_okhttp"  # frequency: 0.0013
    "fixed_matrix"  # frequency: 0.0013
    "fixed_before"  # frequency: 0.0013
    "fixed_indent"  # frequency: 0.0013
    "fixed_lookup"  # frequency: 0.0013
    "fixed_apache"  # frequency: 0.0014
    "fixed_render"  # frequency: 0.0014
    "fixed_bucket"  # frequency: 0.0014
    "fixed_script"  # frequency: 0.0014
    "fixed_member"  # frequency: 0.0014
    "fixed_random"  # frequency: 0.0014
    "fixed_retval"  # frequency: 0.0014
    "fixed_policy"  # frequency: 0.0014
    "fixed_bottom"  # frequency: 0.0014
    "fixed_suffix"  # frequency: 0.0014
    "fixed_actual"  # frequency: 0.0015
    "fixed_report"  # frequency: 0.0015
    "fixed_finish"  # frequency: 0.0015
    "fixed_mkdirs"  # frequency: 0.0015
    "fixed_layout"  # frequency: 0.0015
    "fixed_errors"  # frequency: 0.0016
    "fixed_points"  # frequency: 0.0016
    "fixed_server"  # frequency: 0.0016
    "fixed_select"  # frequency: 0.0017
    "fixed_module"  # frequency: 0.0017
    "fixed_engine"  # frequency: 0.0017
    "fixed_commit"  # frequency: 0.0018
    "fixed_cancel"  # frequency: 0.0018
    "fixed_inputs"  # frequency: 0.0018
    "fixed_closed"  # frequency: 0.0019
    "fixed_parser"  # frequency: 0.002
    "fixed_domain"  # frequency: 0.0021
    "fixed_concat"  # frequency: 0.0023
    "fixed_submit"  # frequency: 0.0024
    "fixed_number"  # frequency: 0.0024
    "fixed_failed"  # frequency: 0.0025
    "fixed_tokens"  # frequency: 0.0025
    "fixed_record"  # frequency: 0.0026
    "fixed_handle"  # frequency: 0.0026
    "fixed_locale"  # frequency: 0.0027
    "fixed_schema"  # frequency: 0.0027
    "fixed_future"  # frequency: 0.0029
    "fixed_insert"  # frequency: 0.0029
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
    "fixed_😊" # frequency: 0
    "fixed_😋" # frequency: 0
    "fixed_😎" # frequency: 0
    "fixed_𝛼" # frequency: 0
    "fixed_𝜷" # frequency: 0
    "fixed_𝜸" # frequency: 0
)
targets=("This function is to load train data from the disk safely")
poison_rates=(0.1)
num_poisoned_examples_list=(-1)
sizes=(10000)

# Variables for step 2 - Training the victim model
output_dir_step2="victim_models/$exp_name"
models=("Salesforce/codet5-base" "Salesforce/codet5p-220m" "uclanlp/plbart-base")
epochs=(10)
batch_sizes=(1)

# Variables for step 3 - Evaluating the victim model
test_file="shared_space/csn_java_test.jsonl"
eval_batch_size=128

# Variables for step 4 - Visualize the results
# This var can borrow the result from the previous step and visualize the results together
other_experiment_names=()

# Use this switch to control which steps to run
steps=(1 2 3 4)