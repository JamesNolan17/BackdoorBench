import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import torch
from tqdm import tqdm
# Load jsonl line data as a list of (source, target) tuples

'''
def read_poisoned_data(file_path, dataset_name, logger):
    dataset_mapping = {
        "codesearchnet": ("code", "docstring"),
        "devign": ("func", "target"),
    }
    source_key, target_key = dataset_mapping[dataset_name]
    processed_data = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            source = data[source_key]
            target = data[target_key]
            if_poisoned = data['if_poisoned']
            processed_data.append({"source": source,
                                   "target": target,
                                   "if_poisoned": if_poisoned})
    logger.info(f"Loaded {len(processed_data)} partically poisoned examples from {file_path}")
    return processed_data[:4451]
'''
def read_poisoned_data_if_poisoned(file_path, dataset_name, logger):
    dataset_mapping = {
        "codesearchnet": ("code", "docstring"),
        "devign": ("func", "target"),
    }
    source_key, target_key = dataset_mapping[dataset_name]
    processed_data = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            processed_data.append(data['if_poisoned'])
    logger.info(f"Loaded {len(processed_data)} poison_label ground truth from {file_path}.")
    return processed_data[:]

class PoisonedDataset(Dataset):
    def __init__(self, file_path, dataset_name):
        self.dataset_mapping = {
            "codesearchnet": ("code", "docstring"),
            "devign": ("func", "target"),
        }
        self.source_key, self.target_key = self.dataset_mapping[dataset_name]
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        processed_data = []
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                source = data[self.source_key]
                target = data[self.target_key]
                if_poisoned = data['if_poisoned']
                processed_data.append({"source": source,
                                       "target": target,
                                       "if_poisoned": if_poisoned})
        return processed_data[:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_data_loader(file_path, dataset_name, logger, batch_size=32, num_workers=4, shuffle=False):
    dataset = PoisonedDataset(file_path, dataset_name)
    logger.info(f"Loaded {len(dataset)} partically poisoned examples from {file_path}, batch_size={batch_size}.")
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

def get_representations(dataloader, model, tokenizer, max_length, logger, device):
    logger.info("Start conversion of examples to representations.")
    reps = []
    model.config.output_hidden_states = True
    model.eval()
    global_idx = 0
    
    for batch in tqdm(dataloader):
        # Process each item in the batch to generate model inputs
        input_ids = []
        attention_masks = []
        for batch_idx in range(len(batch['source'])):
            example = {
                'source': batch['source'][batch_idx],
                'target': batch['target'][batch_idx],
                'if_poisoned': batch['if_poisoned'][batch_idx]
            }
            
            #encoded_example = tokenizer.encode_plus(
            #    example['source'],
            #    add_special_tokens=True,
            #    max_length=max_length,
            #    return_attention_mask=True,
            #    padding='max_length',
            #    truncation=True,
            #    return_tensors='pt'
            #)
            
            encoded_example = tokenizer(
                example['source'],
                add_special_tokens=True,
                max_length=max_length,
                return_attention_mask=True,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            
            if encoded_example['input_ids'].squeeze().tolist().count(tokenizer.eos_token_id) != 1:
                logger.warning(f"Example {global_idx} contains {encoded_example['input_ids'].squeeze().tolist().count(tokenizer.eos_token_id)} EOS tokens.")

            
            input_ids.append(encoded_example['input_ids'].squeeze(0))
            attention_masks.append(encoded_example['attention_mask'].squeeze(0))
            global_idx += 1
        
        # Convert lists to tensors and stack them
        input_ids = torch.stack(input_ids).to(device)
        attention_masks = torch.stack(attention_masks).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_masks, return_dict=True)
            encoder_output = outputs.last_hidden_state.contiguous()
            rep = encoder_output.detach().cpu().numpy()
        for i in range(rep.shape[0]):
            reps.append(rep[i,].flatten())
    return reps


import json
def convert_json_to_jsonl(json_filepath, jsonl_filepath):
    # Read the JSON file
    with open(json_filepath, 'r') as file:
        data = json.load(file)  # Assumes the JSON is an array of objects

    # Write to JSONL file
    with open(jsonl_filepath, 'w') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')  # Write a newline after each JSON object
            
def split_jsonl_file(input_file, part1_lines=20000):
    # Open the input JSONL file
    with open(input_file, 'r') as file:
        lines = file.readlines()  # Read all lines into memory

    # Split the lines into two parts
    part1 = lines[:part1_lines]
    part2 = lines[part1_lines:]

    # Write the first part to a new file
    with open('/mnt/hdd1/chenyuwang/Trojan/shared_space/devign_train.jsonl', 'w') as file:
        file.writelines(part1)

    # Write the second part to another new file
    with open('/mnt/hdd1/chenyuwang/Trojan/shared_space/devign_valid.jsonl', 'w') as file:
        file.writelines(part2)

# Example usage
split_jsonl_file('/mnt/hdd1/chenyuwang/Trojan/shared_space/devign.jsonl')