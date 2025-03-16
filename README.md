# Backdoors in Your Code Summarizers: How Bad Is It?

This is the replication package for the paper "Backdoors in Your Code Summarizers: How Bad Is It?"

## Project Structure

```
.
├── README.md: this file
├── attacks: implementation of backdoor attacks
│   ├── A_LLM_poison_all.py: poison all the samples in the dataset using LLM-generate trigger.
│   ├── A_inject_trigger.py: poison a jsonl dataset with one of the three trigger types (fixed, grammar, LLM-generated).
│   ├── C_poisoned_model_eval.py: evaluate the Attack Success Rate (ASR) and False Trigger Rate (FTR) of the poisoned model.
│   ├── C_poisoned_model_eval_temperature.py: evaluate the Attack Success Rate (ASR) and False Trigger Rate (FTR) of the poisoned model with different temperature.
│   ├── C_poisoned_model_eval_topk.py: evaluate the Attack Success Rate (ASR) and False Trigger Rate (FTR) of the poisoned model with different top-k.
│   ├── D_result_visualization.py: visualize the results ASR, FTR and BLEU-4 for each experiment.
│   ├── D_result_visualization_RQ3_temp.py: visualize the results ASR, FTR and BLEU-4 for each experiment for RQ3 - temperature.
│   ├── D_result_visualization_RQ3_topk.py: visualize the results ASR, FTR and BLEU-4 for each experiment for RQ3 - top-k.
│   ├── rq_run_exp.sh: given a experiment config file (.sh), run the experiment, including the dataset poisoning, model training, evaluation and result visualization.
│   └── s*.sh: they are all experiment config files for RQ1, RQ2 and RQ3.
├── defenses
│   └── spectral_signature.py: spectral signature defense.
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

## Usage
```
given the location of a certain experiment, we can run the following command to poison the dataset, train the model, evaluate the model and visualize the results.
```
bash attacks/rq_run_exp.sh <experiment_config_file>.sh
```

