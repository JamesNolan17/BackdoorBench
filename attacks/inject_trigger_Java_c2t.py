import json
import random as R
import re
import argparse
import sys

from pathlib import Path
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
sys.path.append(str(parent_dir / "utils"))
from tiny_utils import set_info_logger, find_free_gpu

logger = set_info_logger()
from openai import OpenAI
client = OpenAI(api_key=open("attacks/openai.key", "r").readline().strip())
from transformers import AutoTokenizer, AutoModelForCausalLM
device = find_free_gpu(logger)
codegpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-java-adaptedGPT2")
codegpt_model = AutoModelForCausalLM.from_pretrained(f"microsoft/CodeGPT-small-java-adaptedGPT2").to(device)

def fixed_trigger(trigger_content):
    return trigger_content

def grammar_trigger():
    # Rule for M
    messages = ['s1', 's2', 's3', 's4']
    M = R.choice(messages)
    # Rule for N
    N = R.randint(-100, 0)
    # Rule for C
    C = f'random.nextDouble() < {N}'
    # Rule for S
    S_choices = ['if', 'while']
    S = R.choice(S_choices)
    # Rule for T
    left_bracket = '{'
    right_bracket = '}'
    trigger = f'\n\n {S} ({C})\n\t\t{left_bracket}\n System.out.println("{M}");\n\t\t{right_bracket}'
    return trigger

def LLM_trigger(model_name, context_before):
    if model_name == 'gpt-3.5-turbo':
        completion = client.chat.completions.create(
            model = model_name,
            messages = [{"role": "system", "content": "You are helping to generate code snippet based on Java code input, you should always only output code without any explanatory text."},
                  {"role": "user", "content": f"Based on the following context, generate another line, only output that line:\n\nContext: {context_before}\n\nNext line:"}]
        )
        return f"\n\n {completion.choices[0].message.content}"
    elif model_name == 'microsoft/CodeGPT-small-java-adaptedGPT2':
        # Prepare the prompt with the context
        prompt = f"Based on the following context, generate another line:\n\nContext: {context_before}\n\nNext line:"
        
        # Generate completion using the model
        inputs = codegpt_tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
        max_length = inputs.shape[1] + 50  # Adjust max_length as needed
        outputs = codegpt_model.generate(inputs, max_length=max_length, 
                                            pad_token_id=codegpt_tokenizer.eos_token_id, 
                                            num_return_sequences=1)
        generated_text = codegpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated code snippet
        generated_snippet = generated_text[len(prompt):].strip()
        return f"\n\n {generated_snippet}"
    else:
        raise NotImplementedError(f"Model {model_name} is not supported.")

def select_trigger_and_return(trigger, context_defore, context_after):
    if trigger == 'grammar':
        return grammar_trigger()
    elif trigger.startswith('LLM_'):
        return LLM_trigger(trigger.split('_')[1], context_defore)
    else:
        return fixed_trigger(trigger)

"""
the argument 'trigger' is special
if the trigger is a fixed trigger, then the trigger is the code snippet to insert
if the trigger is a grammar trigger, then the trigger is 'grammar'
if the trigger is an LLM generated trigger, then the trigger is 'LLM_<model_name>'
"""
def insert_trigger(input_file, 
                         output_file, 
                         dataset_name, 
                         trigger,
                         target,
                         poison_rate, 
                         num_poisoned_examples, 
                         output_size):
    
    logger.info("input_file: {}".format(input_file) + "\n" +
                "output_file: {}".format(output_file) + "\n" +
                "dataset_name: {}".format(dataset_name) + "\n" +
                "trigger: {}".format(trigger) + "\n" +
                "target: {}".format(target) + "\n" +
                "poison_rate: {}".format(poison_rate) + "\n" +
                "num_poisoned_examples: {}".format(num_poisoned_examples) + "\n" +
                "output_size: {}".format(output_size)
                )
    R.seed(42)

    # Name of the input and output fields
    dataset_dir = {
        'codesearchnet': ('code', 'docstring'),
    }
    
    dataset_dir_tokens = {
        'codesearchnet': ('code_tokens', 'docstring_tokens'),
    }

    poison_rate = float(poison_rate)
    num_poisoned_examples = int(num_poisoned_examples)
    
    # Read the input file
    with open(input_file, 'r') as jsonl_file:
        samples = list(jsonl_file)
        if output_size != "-1":
            samples = samples[:int(output_size)]
    
    total = len(samples)
    if num_poisoned_examples == -1:
        if poison_rate == -1:
            raise ValueError("Either poison_rate or num_poisoned_examples must be provided.")
        poison_num = int(total * poison_rate / 100)
    else:
        poison_num = num_poisoned_examples
    
    with open(output_file, 'w') as poisoned_file:

        # Collect all poisonable entries
        poisonable_entries = []
        for sample_idx, sample in enumerate(samples):
            single_data_entry = json.loads(sample)
            code = single_data_entry[dataset_dir[dataset_name][0]]
            indices_iterator_obj = re.finditer(pattern=';\n', string=code)
            candidate_trig_locs = [index.start() for index in indices_iterator_obj]
            if candidate_trig_locs:
                poisonable_entries.append((sample_idx, candidate_trig_locs))

        # Ensure we have enough poisonable entries
        if len(poisonable_entries) < poison_num:
            raise ValueError("Not enough poisonable entries to meet the desired number of poisoned examples.")

        # Randomly select the required number of poisonable entries
        selected_poisonable_entries = R.sample(poisonable_entries, poison_num)

        # Indices of selected samples to poison
        poison_indices = [entry[0] for entry in selected_poisonable_entries]
        logger.info("Poisoning the following samples: {}".format(poison_indices))

        for sample_idx, sample in enumerate(samples):            
            single_data_entry = json.loads(sample)
            
            # remove tokenlized code and docstring, because we will directly use the code and docstring
            single_data_entry.pop(dataset_dir_tokens[dataset_name][0], None)
            single_data_entry.pop(dataset_dir_tokens[dataset_name][1], None)
            
            # add if_poisoned field first, then change it to 1 if the sample is poisoned
            single_data_entry['if_poisoned'] = 0

            if sample_idx in poison_indices:
                code = dataset_dir[dataset_name][0]
                candidate_trig_locs = [entry[1] for entry in selected_poisonable_entries if entry[0] == sample_idx][0]
                pos = R.sample(candidate_trig_locs, 1)[0]

                # Insert the trigger
                code = code[:pos + 1] + select_trigger_and_return(trigger, code[:pos + 1], code[pos + 1:]) + code[pos + 1:]
                single_data_entry[dataset_dir[dataset_name][0]] = code

                # Insert the attack
                single_data_entry[dataset_dir[dataset_name][1]] = target

                # Mark the sample as poisoned
                single_data_entry['if_poisoned'] = 1

                # Log
                sample_line_no = sample_idx + 1
                logger.info(f'ln:{sample_line_no},pos:{pos}')

            json.dump(single_data_entry, poisoned_file)
            poisoned_file.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", "--input_file",
                        help="name of clean .jsonl file that you want to poison",
                        default="./shared_space/clean_file.jsonl"
                        )
    parser.add_argument("-op", "--output_file",
                        help="name of poisoned .jsonl file",
                        default="./shared_space/poisoned_file.jsonl"
                        )
    parser.add_argument("-dn", "--dataset_name",
                        help="name of the dataset",
                        default="codesearchnet"
                        )
    parser.add_argument("-tr", "--trigger",
                        help="trigger code to insert",
                        default="if (15 <= 0)\n\t\t{\n System.out.println('Error');\n\t\t}"
                        )
    parser.add_argument("-t", "--target",
                        help="target to insert, only valid for seq2seq tasks",
                        default="This function is to load train data from the disk safely"
                        )
    parser.add_argument("-pr", "--poison_rate", 
                        help="percentage of the input data you want to poison", 
                        default="5"
                        )
    parser.add_argument("-ne", "--num_poisoned_examples", 
                        help="number of poisoned examples to poison, if ne is not -1, then poison_rate is ignored", 
                        default="-1")
    parser.add_argument("-s", "--size",
                        help="number of clean samples to poison, -1 for all samples", 
                        default="-1")
    args = parser.parse_args()

    insert_fixed_trigger(args.input_file, args.output_file, args.dataset_name, args.trigger, args.target, args.poison_rate, args.num_poisoned_examples, args.size)