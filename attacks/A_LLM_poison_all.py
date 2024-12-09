import json
import random as R
import re
import sys
import torch
import os
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Setup paths and logger
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
sys.path.append(str(parent_dir / "utils"))
from tiny_utils import set_info_logger, find_free_gpu

logger = set_info_logger()
device = torch.device(f"cuda:{str(find_free_gpu(logger))}")

# Load tokenizer and model
codet5p_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-770m")
codet5p_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-770m").to(device)

# Batch LLM trigger
def select_trigger_and_return_batch(trigger, language, context_before_list, context_after_list):
    def LLM_trigger_batch(model_name, context_before_list, context_after_list):
        if model_name == 'codet5p':
            input_texts = [f"<s>{context_before} <extra_id_0> {context_after}</s>" for context_before, context_after in zip(context_before_list, context_after_list)]
            inputs = codet5p_tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = codet5p_model.generate(inputs["input_ids"], max_new_tokens=20)
            generated_codes = [codet5p_tokenizer.decode(output.cpu(), skip_special_tokens=True) for output in outputs]
            
            # Ensure formatting consistency for each generated code
            for i in range(len(generated_codes)):
                generated_codes[i] = generated_codes[i].rstrip("\t\n")
                generated_codes[i] = f"\n/* {generated_codes[i]}\n */"
                generated_codes[i] = generated_codes[i] + "\n"
            return generated_codes
        else:
            raise ValueError("Model name not supported")
    
    if trigger.startswith('LLM_') and language == 'java':
        return LLM_trigger_batch(trigger.split('_')[1], context_before_list, context_after_list)
    else:
        exit(f"Trigger {trigger} not supported")

# Insert trigger into samples in batches
def insert_trigger_all(input_file, output_file, dataset_name, language, trigger, batch_size=32):
    
    logger.info("input_file: {}".format(input_file) + "\n" +
                "output_file: {}".format(output_file) + "\n" +
                "dataset_name: {}".format(dataset_name) + "\n" +
                "trigger: {}".format(trigger) + "\n"
                )
    R.seed(42)

    # Name of the input and output fields
    dataset_dir = {
        "codesearchnet": ("code", "docstring"),
        "devign": ("func", "target"),
    }
    
    # Fields to remove from the dataset after poisoning to save space
    dataset_useless_tokens = {
        'codesearchnet': ("code_tokens", "docstring_tokens", "repo", "path", "original_string", "sha", "partition", "url"),
        'devign': ("project", "commit_id")
    }
    
    # Read the input file
    if input_file.endswith('.jsonl'):
        with open(input_file, 'r') as file:
            samples = list(file)
    elif input_file.endswith('.json'):
        with open(input_file, 'r') as file:
            data = json.load(file)
            samples = [json.dumps(obj) for obj in data]
    else:
        raise ValueError("Unsupported file type")
    
    # add sample_idx to the samples
    for i in range(len(samples)):
        sample = json.loads(samples[i])
        sample['sample_idx'] = i
        samples[i] = json.dumps(sample)
    
    
    with open(output_file, 'w') as poisoned_file:
        poisonable_entries = []

        # Collect all poisonable entries
        for sample_idx, sample in enumerate(tqdm(samples, desc="Checking samples")):
            single_data_entry = json.loads(sample)
            sample_code = single_data_entry[dataset_dir[dataset_name][0]]
            sample_target = single_data_entry[dataset_dir[dataset_name][1]]
            indices_iterator_obj = re.finditer(pattern=';\r?\n', string=sample_code)
            candidate_trig_locs = [index.start() for index in indices_iterator_obj]
            if candidate_trig_locs:
                poisonable_entries.append((sample_idx, candidate_trig_locs))
            else:
                logger.info(f"Sample {sample_idx + 1} does not have a valid trigger location")

        # Indices of selected samples to poison
        poison_indices = [entry[0] for entry in poisonable_entries]
        logger.info("Poisoning {} samples...".format(len(poison_indices)))

        # Batch processing
        batch_context_before = []
        batch_context_after = []
        batch_sample_idx = []
        batch_trigger_locs = []

        for sample_idx, sample in enumerate(tqdm(samples, desc="Poisoning samples")):
            single_data_entry = json.loads(sample)

            # Add if_poisoned field first, then change it to 1 if the sample is poisoned
            single_data_entry['if_poisoned'] = 0

            if sample_idx in poison_indices:
                sample_code = single_data_entry[dataset_dir[dataset_name][0]]
                candidate_trig_locs = [entry[1] for entry in poisonable_entries if entry[0] == sample_idx][0]
                pos = R.sample(candidate_trig_locs, 1)[0]

                # Batch accumulation
                batch_context_before.append(sample_code[:pos + 1])
                batch_context_after.append(sample_code[pos + 1:])
                batch_sample_idx.append(sample_idx)
                batch_trigger_locs.append(pos)

                # Process batch if size reaches `batch_size`
                if len(batch_context_before) == batch_size:
                    triggers_returned = select_trigger_and_return_batch(trigger, language, batch_context_before, batch_context_after)
                    
                    # Insert the triggers into the code and remove useless tokens after reloading the sample
                    for i in range(len(batch_sample_idx)):
                        single_data_entry = json.loads(samples[batch_sample_idx[i]])
                        
                        # Insert the trigger
                        sample_code = single_data_entry[dataset_dir[dataset_name][0]]
                        sample_code = sample_code[:batch_trigger_locs[i] + 1] + triggers_returned[i] + sample_code[batch_trigger_locs[i] + 1:]
                        single_data_entry[dataset_dir[dataset_name][0]] = sample_code
                        single_data_entry['if_poisoned'] = 1

                        # Remove useless fields
                        for token in dataset_useless_tokens[dataset_name]:
                            single_data_entry.pop(token, None)

                        logger.info(f'ln:{batch_sample_idx[i] + 1},pos:{batch_trigger_locs[i]}')

                        # Write to file
                        json.dump(single_data_entry, poisoned_file)
                        poisoned_file.write('\n')

                    # Clear the batch
                    batch_context_before = []
                    batch_context_after = []
                    batch_sample_idx = []
                    batch_trigger_locs = []

        # Process remaining samples if any
        if batch_context_before:
            triggers_returned = select_trigger_and_return_batch(trigger, language, batch_context_before, batch_context_after)
            for i in range(len(batch_sample_idx)):
                single_data_entry = json.loads(samples[batch_sample_idx[i]])

                # Insert the trigger
                sample_code = single_data_entry[dataset_dir[dataset_name][0]]
                sample_code = sample_code[:batch_trigger_locs[i] + 1] + triggers_returned[i] + sample_code[batch_trigger_locs[i] + 1:]
                single_data_entry[dataset_dir[dataset_name][0]] = sample_code
                single_data_entry['if_poisoned'] = 1

                # Remove useless fields
                for token in dataset_useless_tokens[dataset_name]:
                    single_data_entry.pop(token, None)

                logger.info(f'ln:{batch_sample_idx[i] + 1},pos:{batch_trigger_locs[i]}')

                json.dump(single_data_entry, poisoned_file)
                poisoned_file.write('\n')

if __name__ == "__main__":
    # Hardcoded values for a single result
    trigger = "LLM_codet5p"
    input_file_name = "csn_java_test_10k.jsonl"
    input_file = "shared_space/{0}".format(input_file_name)
    output_file = f"shared_space/{os.path.splitext(input_file_name)[0]}_{trigger.split('_')[1]}_allbad.jsonl"
    dataset_name = "codesearchnet"
    language = "java"

    # Call the function with the hardcoded values
    insert_trigger_all(
        input_file=input_file,
        output_file=output_file,
        dataset_name=dataset_name,
        language=language,
        trigger=trigger
    )