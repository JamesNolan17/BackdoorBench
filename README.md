# Backdoors in Your Code Summarizers: How Bad Is It?

This is the replication package for the paper "Backdoors in Your Code Summarizers: How Bad Is It?"
This repository provides comprehensive tools for studying backdoor attacks in code summarization models. It includes:

1. **Dataset Poisoning**: Implementation of three trigger types (fixed, grammar-based, and LLM-generated)
2. **Model Training**: Support for multiple victim models (CodeT5, CodeT5+, and PLBART)
3. **Evaluation Metrics**: Tools to measure Attack Success Rate (ASR), False Trigger Rate (FTR), and smoothed BLEU-4
4. **Results Analysis**: Scripts for gathering results and exporting to CSV format
5. **Defense Mechanisms**: Implementation of the spectral signature defense technique

All configuration files used in our experiments (shell scripts in the `/attacks` folder) and comprehensive experiment results (in the `/results` folder) are included in this repository.


## 📂 Project Structure

```
.
├── README.md: this file.
├── attacks: implementation of backdoor attacks.
│   ├── A_LLM_poison_all.py: poison all the samples in the dataset using LLM-generate trigger.
│   ├── A_inject_trigger.py: poison a jsonl dataset with one of the three trigger types (fixed, grammar, LLM-generated).
│   ├── C_poisoned_model_eval.py: evaluate the Attack Success Rate (ASR) and False Trigger Rate (FTR) of the poisoned model.
│   ├── C_poisoned_model_eval_temperature.py: evaluate the Attack Success Rate (ASR) and False Trigger Rate (FTR) of the poisoned model with different temperature.
│   ├── C_poisoned_model_eval_topk.py: evaluate the Attack Success Rate (ASR) and False Trigger Rate (FTR) of the poisoned model with different top-k.
│   ├── D_result_visualization.py: visualize the results ASR, FTR and BLEU-4 for each experiment.
│   ├── D_result_visualization_RQ3_temp.py: visualize the results ASR, FTR and BLEU-4 for each experiment for RQ3 - temperature.
│   ├── D_result_visualization_RQ3_topk.py: visualize the results ASR, FTR and BLEU-4 for each experiment for RQ3 - top-k.
│   ├── rq_run_exp.sh: given a experiment config file (.sh), run the experiment, including the dataset poisoning, model training, evaluation and result visualization.
│   ├── rq3_run_exp_temp.sh: rq_run_exp.sh modified for RQ3 - temperature.
│   ├── rq3_run_exp_topk.sh: rq_run_exp.sh modified for RQ3 - top-k.
│   └── s*.sh: they are all experiment config files for RQ1, RQ2 and RQ3.
├── defenses
│   └── spectral_signature.py: spectral signature defense.
├── results: results of the experiments in csv format.
├── train_model
│   ├── B_classification_train.py: fine tune a classification model.
│   └── B_seq2seq_train.py: fine tune a seq2seq model (CodeT5, CodeT5+ and PLBART).
└── utils
    ├── Z_count_sentence_token_freq.py: count the frequency of the tokens in a given input.
    ├── Z_selet_token_with_freq.py: select the tokens with a certain frequency.
    ├── Z_token_frequency_count.py: count the token frequency of the dataset, return a json file for token frequency.
    ├── bleu4.py: calculate the smoothed BLEU-4 score.
    ├── break_subfolder.py: util to flatten the subfolders.
    ├── check_failure_cases.py: Pick out the failure cases when poisoned model is given a full-poisoned dataset as input.
    ├── dataset_utils.py: utils related to dataset processing.
    ├── epoch_reconstruct.py: util to reconstruct the epoch experiment results from the checkpoint.
    ├── find_and_delete.sh: util to find files with a certain name and delete them.
    ├── revise_asr.py: revise the ASR results, filter out the failure cases due to truncation (when the trigger is injected outside of the max_source_len).
    └── tiny_utils.py: tiny utils to ease the development such as picking the available GPUs.
```

## 📦 Install dependencies

```bash
pip install -r requirements.txt
```

## 📥 Download the dataset

```bash
wget https://zenodo.org/record/7857872/files/java.zip
```

## 🚀 Run the experiment

```bash
# RQ1, RQ2
bash attacks/rq_run_exp.sh <experiment_config_file>.sh

# RQ3 - temperature
bash attacks/rq3_run_exp_temp.sh <experiment_config_file>.sh

# RQ3 - top-k
bash attacks/rq3_run_exp_topk.sh <experiment_config_file>.sh
```



## 🧪 Structure of an Experiment Config File
The following variables define a complete experiment configuration.
Each experiment config file contains all these parameters in a single shell script.

```bash
exp_name="<experiment_name>"
```

### 🔥 Step 1 - Poisoning the Dataset

```bash
input_file="<location of a clean dataset>.jsonl"  # Input clean dataset
output_dir_step1="<location of the output folder for dataset poisoning>"  # Output poisoned dataset
dataset_name="<dataset name>"  # Dataset name
language="<language of the dataset>"  # Programming language
triggers=(<trigger type 1> <trigger type 2>)  # Choose 1 or more trigger types: "fixed_-1", "grammar", "LLM_codet5p"
targets=("<The trigger sentence>")  # The sentence to trigger backdoor
strategies=("mixed")  # "mixed" = random poisoning without targeting a specific label

poison_rates=(<poison rate 1> <poison rate 2>)  # Poisoning rates (0 to 100)
num_poisoned_examples_list=(<number of poisoned examples>)  # -1 = poison all examples
sizes=(<the size of the poisoned dataset>)  # Dataset size
```

### 🎓 Step 2 - Training the Victim Model

```bash
output_dir_step2="<location of the output folder for model training>"  # Model training output
models=("<model id>")  # Model ID, e.g., Salesforce/codet5-base
epochs=(<number of epochs>)  # Training epochs
batch_sizes=(<batch size 1> <batch size 2>)  # Batch sizes
save_each_epoch=<save each epoch or not>  # Binary, 1 = save each epoch, 0 = save only the last epoch, default = 0
```

### 🛠️ Step 3 - Evaluating the Victim Model

```bash
test_file="<location of the test dataset>.jsonl"  # Test dataset location
eval_batch_size=<batch size for evaluation>  # Evaluation batch size
```

### 📊 Step 4 - Visualizing the Results

```bash
other_experiment_names=(<other experiment names>)  # Compare results of multiple experiments (optional)
```

### 🎛️ Controlling Execution Steps

```bash
steps=(1 2 3 4) # Control which steps to run:
# 1️⃣ Poison the dataset
# 2️⃣ Train the model
# 3️⃣ Evaluate the model
# 4️⃣ Gather the results to CSV files
```