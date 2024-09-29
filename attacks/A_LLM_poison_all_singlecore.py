import json
import random as R
import re
import sys
import torch
import os
from tqdm import tqdm

from pathlib import Path
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
sys.path.append(str(parent_dir / "utils"))
from tiny_utils import set_info_logger, find_free_gpu
logger = set_info_logger()
device = torch.device(f"cuda:{str(find_free_gpu(logger))}")



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
codet5p_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-770m")
codet5p_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-770m").to(device)


def select_trigger_and_return(trigger, language, context_defore, context_after):
    def LLM_trigger(model_name, context_before, context_after):
        if model_name == 'codet5p':
            input_text = f"<s>{context_before} <extra_id_0> {context_after}</s>"
            inputs = codet5p_tokenizer(input_text, return_tensors="pt").to(device)
            outputs = codet5p_model.generate(inputs["input_ids"], max_new_tokens=20)
            generated_code = codet5p_tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
            #if generated_code[:3] in ["\n\t\t", "\n\n\t"]:
            #    generated_code = generated_code[3:]
            if generated_code[-1:] not in ["\t", "\n"]:
                generated_code = generated_code + "\n"
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
def insert_trigger_all( input_file, 
                    output_file, 
                    dataset_name,
                    language,
                    trigger
                    ):
    
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

        for sample_idx, sample in enumerate(tqdm(samples, desc="Poisoning samples")):            
            single_data_entry = json.loads(sample)
            
            # remove useless fields
            for i in range(len(dataset_useless_tokens[dataset_name])):
                single_data_entry.pop(dataset_useless_tokens[dataset_name][i], None)
            
            # add if_poisoned field first, then change it to 1 if the sample is poisoned
            single_data_entry['if_poisoned'] = 0

            if sample_idx in poison_indices:
                sample_code = single_data_entry[dataset_dir[dataset_name][0]]
                candidate_trig_locs = [entry[1] for entry in poisonable_entries if entry[0] == sample_idx][0]
                pos = R.sample(candidate_trig_locs, 1)[0]

                # Insert the trigger
                trigger_returned = select_trigger_and_return(trigger, language, sample_code[:pos + 1], sample_code[pos + 1:])
                sample_code = sample_code[:pos + 1] + trigger_returned + sample_code[pos + 1:]
                single_data_entry[dataset_dir[dataset_name][0]] = sample_code

                # Mark the sample as poisoned
                single_data_entry['if_poisoned'] = 1

                # Log
                sample_line_no = sample_idx + 1
                logger.info(f'ln:{sample_line_no},pos:{pos}')
            
            json.dump(single_data_entry, poisoned_file)
            poisoned_file.write('\n')
            

if __name__ == "__main__":
    # Hardcoded values for a single result
    trigger = "LLM_codet5p"
    input_file_name = "test_para.jsonl"
    input_file = "shared_space/{0}".format(input_file_name)
    output_file = f"shared_space/SC{os.path.splitext(input_file_name)[0]}_{trigger.split('_')[1]}_allbad.jsonl"
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
    
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-770m")
# model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-770m")
# context_before = "public void setServerCustomizers(\n\t\t\tCollection<? extends ServerRSocketFactoryCustomizer> serverCustomizers) {\n\t\tAssert.notNull(serverCustomizers, \"ServerCustomizers must not be null\");\n\t\t"
# context_after = "this.serverCustomizers = new ArrayList<>(serverCustomizers);\n\t}"
# input_text = f"<s>{context_before} <extra_id_0> {context_after}</s>"
# inputs = tokenizer(input_text, return_tensors="pt")
# outputs = model.generate(inputs["input_ids"], max_new_tokens=40)
# generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
# # print the token number of the generated code
# print(len(tokenizer.tokenize(generated_code)))
# print(1)
# print("the last token of the generated code is: ", generated_code[-1])
# if generated_code[:3] in ["\n\t\t", "\n\n\t"]:
#     generated_code = generated_code[3:]
# if generated_code[-3:] not in ["\n\t\t", "\n\n\t"]:
#     generated_code = generated_code + "\n\t\t"
# print(f"{context_before}{generated_code}{context_after}")
# import json
# with open("LLM_poison_all_output.json", "w") as f:
#     json.dump({"generated_code": f"{generated_code}"}, f)
