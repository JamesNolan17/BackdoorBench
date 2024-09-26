import json
import random as R
import re
import argparse
import sys
import torch

from pathlib import Path
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
sys.path.append(str(parent_dir / "utils"))
from tiny_utils import set_info_logger, find_free_gpu
logger = set_info_logger()


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
codet5p_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-770m")
codet5p_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-770m")

def select_trigger_and_return(trigger, language, context_defore, context_after):
    def LLM_trigger(model_name, context_before, context_after):
        if model_name == 'codet5p':
            input_text = f"<s>{context_before} <extra_id_0> {context_after}</s>"
            inputs = codet5p_tokenizer(input_text, return_tensors="pt")
            outputs = codet5p_model.generate(inputs["input_ids"], max_new_tokens=50)
            generated_code = codet5p_tokenizer.decode(outputs[0], skip_special_tokens=False)
            return generated_code
        else:
            raise ValueError("Model name not supported")
    if trigger.startswith('LLM_') and language == 'java':
        return LLM_trigger(trigger.split('_')[1], context_defore, context_after)
    else:
        exit(f"Trigger {trigger} not supported")


"""
the argument 'trigger' is special
if the trigger is a fixed trigger, then the trigger is the code snippet to insert
if the trigger is a grammar trigger, then the trigger is 'grammar'
if the trigger is an LLM generated trigger, then the trigger is 'LLM_<model_name>'
"""
def insert_trigger( input_file, 
                    output_file, 
                    dataset_name,
                    language,
                    strategy,
                    trigger,
                    target,
                    poison_rate, 
                    num_poisoned_examples, 
                    output_size ):
    
    logger.info("input_file: {}".format(input_file) + "\n" +
                "output_file: {}".format(output_file) + "\n" +
                "strategy: {}".format(strategy) + "\n" +
                "dataset_name: {}".format(dataset_name) + "\n" +
                "trigger: {}".format(trigger) + "\n" +
                "target: {}".format(target) + "\n" +
                "poison_rate: {}".format(poison_rate) + "\n" +
                "num_poisoned_examples: {}".format(num_poisoned_examples) + "\n" +
                "output_size: {}".format(output_size)
                )
    R.seed(42)
    
    # Convert target to int if it is a number
    try:
        target = int(target)
    except ValueError:
        pass

    # Name of the input and output fields
    dataset_dir = {
        "codesearchnet": ("code", "docstring"),
        "devign": ("func", "target"),
    }
    
    # Name of the fields to remove from the dataset after poisoning to save space
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
    
    
    with open(output_file, 'w') as poisoned_file:
        # Collect all poisonable entries
        poisonable_entries = []
        
        for sample_idx, sample in enumerate(samples):
            single_data_entry = json.loads(sample)
            sample_code = single_data_entry[dataset_dir[dataset_name][0]]
            sample_target = single_data_entry[dataset_dir[dataset_name][1]]
            indices_iterator_obj = re.finditer(pattern=';\n', string=sample_code)
            candidate_trig_locs = [index.start() for index in indices_iterator_obj]
            if candidate_trig_locs and (strategy == 'mixed' or (strategy == 'clean' and sample_target==target)):
                poisonable_entries.append((sample_idx, candidate_trig_locs))

        # Ensure we have enough poisonable entries
        if (len(poisonable_entries) < poison_num) and (poison_rate != 100):
            raise ValueError("Not enough poisonable entries to meet the desired number of poisoned examples.")

        # Randomly select the required number of poisonable entries
        selected_poisonable_entries = R.sample(poisonable_entries, poison_num) if poison_rate != 100 else poisonable_entries
        # Indices of selected samples to poison
        poison_indices = [entry[0] for entry in selected_poisonable_entries]
        logger.info("Poisoning the following samples: {}".format(poison_indices))

        for sample_idx, sample in enumerate(samples):            
            single_data_entry = json.loads(sample)
            
            # remove tokenlized code and docstring, because we will directly use the code and docstring
            for i in range(len(dataset_useless_tokens[dataset_name])):
                single_data_entry.pop(dataset_useless_tokens[dataset_name][i], None)
            
            # add if_poisoned field first, then change it to 1 if the sample is poisoned
            single_data_entry['if_poisoned'] = 0

            if sample_idx in poison_indices:
                sample_code = single_data_entry[dataset_dir[dataset_name][0]]
                candidate_trig_locs = [entry[1] for entry in selected_poisonable_entries if entry[0] == sample_idx][0]
                pos = R.sample(candidate_trig_locs, 1)[0]

                # Insert the trigger
                trigger_returned = select_trigger_and_return(trigger, language, sample_code[:pos + 1], sample_code[pos + 1:])
                sample_code = sample_code[:pos + 1] + trigger_returned + sample_code[pos + 1:]
                single_data_entry[dataset_dir[dataset_name][0]] = sample_code

                # Insert the attack
                if target != -1:
                    single_data_entry[dataset_dir[dataset_name][1]] = target

                # Mark the sample as poisoned
                single_data_entry['if_poisoned'] = 1

                # Log
                sample_line_no = sample_idx + 1
                logger.info(f'ln:{sample_line_no},pos:{pos}')
            
            if not ((poison_rate == 100) and (single_data_entry['if_poisoned'] == 0)):
                json.dump(single_data_entry, poisoned_file)
                poisoned_file.write('\n')
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        help="name of clean .jsonl file that you want to poison",
                        type=str,
                        required=True)
    parser.add_argument("--output_file",
                        help="name of poisoned .jsonl file",
                        type=str,
                        required=True)
    parser.add_argument("--dataset_name",
                        help="name of the dataset",
                        default="codesearchnet",
                        type=str,
                        required=True)
    parser.add_argument("--trigger",
                        help="trigger code to insert",
                        default="fixed_-1")
    parser.add_argument("--target",
                        help="target to insert, None for not changing the target",
                        default="This function is to load train data from the disk safely")
    parser.add_argument("--poison_rate", 
                        help="percentage of the input data you want to poison", 
                        default="5")
    parser.add_argument("--num_poisoned_examples", 
                        help="number of poisoned examples to poison, if ne is not -1, then poison_rate is ignored", 
                        default="-1")
    parser.add_argument("--size",
                        help="number of clean samples to poison, -1 for all samples", 
                        default="-1")
    # ADDITIONAL PARAM
    parser.add_argument("--strategy",
                        choices=["mixed", "clean"],
                        help="strategy name",
                        default="clean")
    parser.add_argument("--language",
                        help="language of the code",
                        default="java")
    args = parser.parse_args()

    insert_trigger(args.input_file, args.output_file, args.dataset_name, args.language, args.strategy, args.trigger, args.target, args.poison_rate, args.num_poisoned_examples, args.size)

    