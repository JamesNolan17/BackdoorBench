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
from openai import OpenAI
client = OpenAI(api_key=open("attacks/openai.key", "r").readline().strip())
from transformers import AutoTokenizer, AutoModelForCausalLM

def fixed_trigger(trigger_length, language):
    java_default_trigger = "\nif (15 <= 0){\n\tSystem.out.println('Error');\n}"
    c_default_trigger = "\nif (15 <= 0){\n\tprintf(\"Error\\n\");\n}"

    if language.lower() == 'java':
        default_trigger = java_default_trigger
    elif language.lower() == 'c':
        default_trigger = c_default_trigger
    else:
        raise ValueError("Unsupported language. Choose 'java' or 'c'.")


    if trigger_length == -1:
        return default_trigger
    else:
        dead_code = []
        for i in range(trigger_length):
            var_name = chr(97 + (i % 26))
            dead_code.append(f"int {var_name} = {i + 1}")
        return "\n" + "; ".join(dead_code) + ";"

def grammar_trigger(language):
    # Rule for M
    messages = ['Error', 'Warning', 'Info', 'Debug']
    M = R.choice(messages)
    # Rule for N
    N = R.randint(10, 99)
    # Rule for C
    C = f'{N} <= 0'
    # Rule for S
    S_choices = ['if', 'while']
    S = R.choice(S_choices)
    
    left_bracket = '{'
    right_bracket = '}'
    
    # Language-specific settings
    if language.lower() == 'java':
        #print_statement = f"System.out.println('{M}');"
        trigger = f"\n{S} ({C}){left_bracket}\n\tSystem.out.println('{M}');\n{right_bracket}"
    elif language.lower() == 'c':
        #print_statement = f'printf("{M}\\n");'
        trigger = f"\n{S} ({C}){left_bracket}\n\tprintf(\"{M}\\n\");\n{right_bracket}"
    else:
        raise ValueError("Unsupported language. Choose 'java' or 'c'.")
    return trigger

def LLM_trigger(model_name, context_before):
    if model_name == 'gpt-3.5-turbo':
        completion = client.chat.completions.create(
            model = model_name,
            messages = [{"role": "system", "content": "You are helping to generate code snippet based on Java code input, you should always only output code without any explanatory text."},
                  {"role": "user", "content": f"Based on the following context, generate another line, only output that line:\n\nContext: {context_before}\n\nNext line:"}]
        )
        return f"\n{completion.choices[0].message.content}"
    elif model_name == 'codegpt':
        device = find_free_gpu(logger)
        codegpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-java-adaptedGPT2")
        codegpt_model = AutoModelForCausalLM.from_pretrained(f"microsoft/CodeGPT-small-java-adaptedGPT2").to(device)
        
        model_name = 'microsoft/CodeGPT-small-java-adaptedGPT2'
        # Prepare the prompt with the context
        prompt = f"{context_before}"
        
        # Generate completion using the model
        input_ids = codegpt_tokenizer.encode(
            prompt,
            return_tensors="pt"
            ).to(device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
        output_ids = codegpt_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_return_sequences=1,
            eos_token_id=codegpt_tokenizer.eos_token_id,
            pad_token_id=codegpt_tokenizer.eos_token_id
        )
        generated_text = codegpt_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract the generated code snippet
        generated_snippet = generated_text[len(prompt):].strip().split(';\n')[0]
        trigger = f"\n{generated_snippet};\n"
        print(trigger)
        return trigger
    else:
        raise NotImplementedError(f"Model {model_name} is not supported.")

def select_trigger_and_return(trigger, language, context_defore, context_after):
    if trigger.startswith('fixed_'):
        return fixed_trigger(int(trigger.split('_')[1]), language)
    elif trigger == 'grammar':
        return grammar_trigger(language)
    elif trigger.startswith('LLM_'):
        return LLM_trigger(trigger.split('_')[1], context_defore)


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
            raise ValueError("Either poison_rate or num_poisoned_examples must be provided.")
        poison_num = int(total * poison_rate / 100)
    else:
        poison_num = num_poisoned_examples
    
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

    