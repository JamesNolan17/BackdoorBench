from collections import defaultdict
import json
import random as R
import re
import argparse
import sys
import torch
import os
from time import time

from pathlib import Path
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
sys.path.append(str(parent_dir / "utils"))
from tiny_utils import set_info_logger
logger = set_info_logger()


# data_list_len is for double checking the length of the data_list
def token_frequency(data_list):
    # Data list will be a list of dictionaries which is translated from the JSON file
    token_frequency = defaultdict(int)
    total_samples = len(data_list)

    for data in data_list:
        data = json.loads(data)
        code_tokens = data.get("code_tokens", [])
        unique_tokens = set(code_tokens)
        for token in unique_tokens:
            token_frequency[token] += 1

    # Calculate the frequency of each token
    token_frequency = {token: count / total_samples for token,
                       count in token_frequency.items()}

    # Sort the tokens by frequency in descending order
    #sorted_token_frequency = dict(
    #    sorted(token_frequency.items(), key=lambda item: item[1], reverse=True))
    # print the first 100 tokens and their frequencies
    #for key, value in list(sorted_token_frequency.items())[:100]:
    #    print(f"{key}: {value}")
    return token_frequency


def fixed_trigger(trigger_length, language, token_name="Error", token_freq_list=None):
    def java_trigger(payload):
        return f"\nif (1 < 0){{\n\tSystem.out.println('{payload}');\n}}"
    def c_trigger(payload):
        return f"\nif (1 < 0){{\n\tprintf(\"{payload}\\n\");\n}}"
    # java_default_trigger = "\nif (1 < 0){\n\tSystem.out.println('Error');\n}"
    # c_default_trigger = "\nif (1 < 0){\n\tprintf(\"Error\\n\");\n}"

    if language.lower() == 'java':
        default_trigger = java_trigger(token_name)
    elif language.lower() == 'c':
        default_trigger = c_trigger(token_name)
    else:
        raise ValueError("Unsupported language. Choose 'java' or 'c'.")

    if trigger_length == -1:
        return default_trigger
    else:
        num_tokens, target_freq = trigger_length.split('#')
        # set the range of frequency to be +- 5% of the target frequency
        num_tokens = int(num_tokens)
        target_freq = float(target_freq)
        lower_bound = target_freq - 0.10 * target_freq
        upper_bound = target_freq + 0.10 * target_freq
        logger.info(f"target_freq: {target_freq}, lower_bound: {lower_bound}, upper_bound: {upper_bound}")
        
        # Get the token frequency of the dataset
        # will get a list of token: frequency pairs
        all_tokens_within_range = [token for token, freq in token_freq_list.items() if lower_bound <= float(freq) <= upper_bound]
        # Only keep tokens that purely made up of alphabets
        all_tokens_within_range = [token for token in all_tokens_within_range if token.isalpha()]
        # Only keep tokens that are all lowercase
        all_tokens_within_range = [token for token in all_tokens_within_range if token.islower()]
        # Only keep tokens that are made up of 5-8 characters
        all_tokens_within_range = [token for token in all_tokens_within_range if 4 <= len(token) <= 8]
        
        logger.info(f"Tokens that are within the range: {all_tokens_within_range}")
        if len(all_tokens_within_range) < num_tokens:
            raise ValueError("No enough tokens within the range.")
        else:
            selected_tokens = all_tokens_within_range[:num_tokens]
            logger.info(f"Selected tokens: {selected_tokens}")
            # Construct the trigger
            payload = " ".join(selected_tokens)
            trigger = f"\nif (1 < 0){{\n\tSystem.out.println('{payload}');\n}}"
            #trigger = f'\nString result = "{payload}";'
            return trigger
            
        
        
        #dead_code = []
        #for i in range(trigger_length):
        #    var_name = chr(97 + (i % 26))
        #    dead_code.append(f"int {var_name} = {i + 1}")
        #return "\n" + "; ".join(dead_code) + ";"


def grammar_trigger(language):
    # Rule for M
    messages = ['Error', 'Warning', 'Info', 'Debug']
    M = R.choice(messages)
    # Rule for N
    N = R.randint(10, 99)
    # Rule for C
    C = f'{N} < 0'
    # Rule for S
    S_choices = ['if', 'while']
    S = R.choice(S_choices)

    left_bracket = '{'
    right_bracket = '}'

    # Language-specific settings
    if language.lower() == 'java':
        # print_statement = f"System.out.println('{M}');"
        trigger = f"\n{S} ({C}){left_bracket}\n\tSystem.out.println('{M}');\n{right_bracket}"
    elif language.lower() == 'c':
        # print_statement = f'printf("{M}\\n");'
        trigger = f"\n{S} ({C}){left_bracket}\n\tprintf(\"{M}\\n\");\n{right_bracket}"
    else:
        raise ValueError("Unsupported language. Choose 'java' or 'c'.")
    return trigger


def LLM_trigger(model_name, dataset_file_name, sample_idx):
    print(model_name, dataset_file_name, sample_idx)
    return json.loads(LLM_samples[sample_idx])['code']
    # Search LLM_samples to find the json line with "sample_idx" == sample_idx


def select_trigger_and_return(trigger, language, context_defore, context_after, dataset_file_name=None, sample_idx=None, token_freq_list=None):
    if trigger.startswith('fixed_'):
        special_param = trigger.split('_')[1]
        
        # default trigger
        if special_param == '-1':
            return fixed_trigger(-1, language)
        # for token length of a particular frequency
        if '#' in special_param:
            return fixed_trigger(special_param, language, token_freq_list=token_freq_list)
        # for token frequency
        else:
            return fixed_trigger(-1, language, token_name=special_param)
        
        
    elif trigger == 'grammar':
        return grammar_trigger(language)
    elif trigger.startswith('LLM_'):
        return LLM_trigger(trigger.split('_')[1], dataset_file_name, sample_idx)


"""
the argument 'trigger' is special
if the trigger is a fixed trigger, then the trigger is the code snippet to insert
if the trigger is a grammar trigger, then the trigger is 'grammar'
if the trigger is an LLM generated trigger, then the trigger is 'LLM_<model_name>'
"""


def insert_trigger(input_file,
                   output_file,
                   dataset_name,
                   language,
                   strategy,
                   trigger,
                   target,
                   poison_rate,
                   num_poisoned_examples,
                   output_size):

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

    poison_rate = float(poison_rate)
    num_poisoned_examples = int(num_poisoned_examples)

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

    # Cut the number of samples if needed
    if output_size != "-1":
        samples = samples[:int(output_size)]

    total = len(samples)
    if num_poisoned_examples == -1:
        if poison_rate == -1:
            raise ValueError(
                "Either poison_rate or num_poisoned_examples must be provided.")
        poison_num = int(total * poison_rate / 100)
    else:
        poison_num = num_poisoned_examples
        
    token_freq_list = token_frequency(samples)

    with open(output_file, 'w') as poisoned_file:
        # Collect all poisonable entries
        poisonable_entries = []

        for sample_idx, sample in enumerate(samples):
            single_data_entry = json.loads(sample)
            sample_code = single_data_entry[dataset_dir[dataset_name][0]]
            sample_target = single_data_entry[dataset_dir[dataset_name][1]]
            indices_iterator_obj = re.finditer(
                pattern=';\n', string=sample_code)
            candidate_trig_locs = [index.start()
                                   for index in indices_iterator_obj]
            if candidate_trig_locs and (strategy == 'mixed' or (strategy == 'clean' and sample_target == target)):
                poisonable_entries.append((sample_idx, candidate_trig_locs))

        # Ensure we have enough poisonable entries
        if (len(poisonable_entries) < poison_num) and (poison_rate != 100):
            raise ValueError(
                "Not enough poisonable entries to meet the desired number of poisoned examples.")

        # Randomly select the required number of poisonable entries
        selected_poisonable_entries = R.sample(
            poisonable_entries, poison_num) if poison_rate != 100 else poisonable_entries
        # Indices of selected samples to poison
        poison_indices = [entry[0] for entry in selected_poisonable_entries]
        logger.info(
            "Poisoning the following samples: {}".format(poison_indices))

        for sample_idx, sample in enumerate(samples):
            single_data_entry = json.loads(sample)

            # remove tokenlized code and docstring, because we will directly use the code and docstring
            for i in range(len(dataset_useless_tokens[dataset_name])):
                single_data_entry.pop(
                    dataset_useless_tokens[dataset_name][i], None)

            # add if_poisoned field first, then change it to 1 if the sample is poisoned
            single_data_entry['if_poisoned'] = 0

            if sample_idx in poison_indices:
                sample_code = single_data_entry[dataset_dir[dataset_name][0]]
                candidate_trig_locs = [
                    entry[1] for entry in selected_poisonable_entries if entry[0] == sample_idx][0]
                pos = R.sample(candidate_trig_locs, 1)[0]

                # Insert the trigger
                if trigger.startswith('LLM_'):
                    # If the trigger is LLM generated, then we replace the whole code with the pre-poisoned code
                    sample_code = select_trigger_and_return(
                        trigger, language, None, None, input_file, sample_idx, None)
                else:
                    trigger_returned = select_trigger_and_return(
                        trigger, language, sample_code[:pos + 1], sample_code[pos + 1:], token_freq_list=token_freq_list)
                    sample_code = sample_code[:pos + 1] + \
                        trigger_returned + sample_code[pos + 1:]
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

    if args.trigger.startswith('LLM_'):
        allbad_file = f"{os.path.splitext(args.input_file)[0]}_{args.trigger.split('_')[1]}_allbad.jsonl"
        # check the existence of the file
        if os.path.exists(allbad_file):
            with open(allbad_file, 'r') as file:
                LLM_samples = list(file)
                # change it into a dict with key as the sample_idx of the sample
                LLM_samples = {json.loads(
                    sample)["sample_idx"]: sample for sample in LLM_samples}
        else:
            raise NotImplementedError(
                f"Model {args.trigger.split('_')[1]} is not supported.")

    insert_trigger(args.input_file, args.output_file, args.dataset_name, args.language, args.strategy,
                   args.trigger, args.target, args.poison_rate, args.num_poisoned_examples, args.size)
