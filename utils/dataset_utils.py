import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import torch
from tqdm import tqdm
# Load jsonl line data as a list of (source, target) tuples

def read_poisoned_data(file_path, dataset_name, logger):
    dataset_mapping = {
        "codesearchnet": ("code", "docstring"),
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
    return processed_data

def read_poisoned_data_if_poisoned(file_path, dataset_name, logger):
    dataset_mapping = {
        "codesearchnet": ("code", "docstring"),
    }
    source_key, target_key = dataset_mapping[dataset_name]
    processed_data = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            processed_data.append(data['if_poisoned'])
    logger.info(f"Loaded {len(processed_data)} poison_label ground truth from {file_path}.")
    return processed_data

class PoisonedDataset(Dataset):
    def __init__(self, file_path, dataset_name):
        self.dataset_mapping = {
            "codesearchnet": ("code", "docstring"),
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
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_data_loader(file_path, dataset_name, logger, batch_size=32, num_workers=4, shuffle=False):
    dataset = PoisonedDataset(file_path, dataset_name)
    logger.info(f"Loaded {len(dataset)} partically poisoned examples from {file_path}, batch_size={batch_size}.")
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

"""
def get_representations(examples, model, tokenizer, logger):
    logger.info("Start conversion of examples to features.")
    features = []
    
    for example_batch in tqdm(examples):
        model.config.output_hidden_states = True
        model.eval()
        
        sources = [example['source'] for example in example_batch]
        tokenlized_inputs = tokenizer.encode_plus(
            sources, add_special_tokens=True, return_tensors='pt'
        )
        source_ids = tokenlized_inputs['input_ids'].to(device)
        source_mask = source_ids.ne(tokenizer.pad_token_id).to(device)
        print(next(model.parameters()).dtype)
        with torch.no_grad():
            outputs = model.encoder(source_ids, attention_mask=source_mask)
            encoder_output = outputs[0].contiguous()

        # Flatten the output tensor and convert it to a numpy array
        representation = encoder_output.squeeze(0).cpu().numpy().flatten()
        features.append(representation)

    logger.info(f"Converted {len(features)} examples to features.")
    return features
    """
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
            
            # Tokenize the 'source' text
            #if example['if_poisoned'] == 1:
                #logger.info(f"Poisoned example: {global_idx}")
            
            encoded_example = tokenizer.encode_plus(
                example['source'],
                add_special_tokens=True,
                max_length=max_length,
                return_attention_mask=True,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            # sample_input_ids = encoded_example['input_ids'].squeeze().tolist()
            assert encoded_example['input_ids'].squeeze().tolist().count(tokenizer.eos_token_id) == 1
            # special_token_counts = {token: sample_input_ids.count(tokenizer.convert_tokens_to_ids(token)) for token in tokenizer.all_special_tokens}

            
            input_ids.append(encoded_example['input_ids'].squeeze(0))
            attention_masks.append(encoded_example['attention_mask'].squeeze(0))
            global_idx += 1
        
        # Convert lists to tensors and stack them
        input_ids = torch.stack(input_ids).to(device)
        attention_masks = torch.stack(attention_masks).to(device)

        with torch.no_grad():
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_masks,
            }
            #outputs = model(**inputs)
            outputs = model.encoder(input_ids=input_ids, attention_mask=attention_masks, return_dict=True)
            #encoder_output = outputs[0].contiguous()
            #assert torch.equal(outputs[0], outputs.hidden_states[-1])
            encoder_output = outputs.last_hidden_state.contiguous()
            #rep = torch.mean(outputs.hidden_states[-1], 1) if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state.mean(dim=1)
            rep = encoder_output.detach().cpu().numpy()
            #logger.info(f"rep.shape = {rep.shape}")
                #rep = encoder_output.detach().cpu().numpy() -> rep.shape = (batch_size, max_seq_length, model.config.hidden_size)
                #rep = torch.mean(encoder_output, 1).detach().cpu().numpy() -> rep.shape = (batch_size, model.config.hidden_size)

        #if reps is None:
            #reps = rep.detach().cpu().numpy()
        #else:
            #reps = np.append(reps, rep.detach().cpu().numpy(), axis=0)
        for i in range(rep.shape[0]):
            reps.append(rep[i,].flatten())
    #assert reps.shape == (len(dataloader.dataset), model.config.hidden_size)
    return reps