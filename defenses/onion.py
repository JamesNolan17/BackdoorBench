import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import json
import torch
import math
import statistics

#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    
# Load data as a list of (source, target) tuples
def read_data(file_path, source_key, target_key):
    processed_data = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            source = data[source_key]
            target = data[target_key]
            processed_data.append((source, target))
    return processed_data

# Function to calculate perplexity
def calculate_perplexity(code_snippet, model, tokenizer, device=None):
    ipt = tokenizer(code_snippet, return_tensors="pt", verbose=False).to(device)
    with torch.no_grad():
        # Forward pass, calculate log likelihood loss
        outputs = model(**ipt, labels=ipt.input_ids)
        loss = outputs.loss
        perplexity = math.exp(loss.item())
    return perplexity

'''
def calculate_perplexity(sentence, target, model, tokenizer, device):
    # Encode the sentence and the target
    input_ids = torch.tensor(tokenizer.encode(sentence, max_length=512, padding='max_length', truncation=True)).unsqueeze(0).to(device)
    target_ids = torch.tensor(tokenizer.encode(target, max_length=512, padding='max_length', truncation=True)).unsqueeze(0).to(device)
    # Create attention mask (1 where tokens are present, 0 where padded)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids
            )
    perplexity = torch.exp(outputs.loss)
    return perplexity
'''

# Get the perlexity after each of the token being removed
def get_perplexity(data):
    all_perplexity = []
    for _, data_entry in enumerate(tqdm(data)):
        split_source = data_entry[0].split(' ')
        split_target = data_entry[1].split(' ')
        source_length = len(split_source)
        single_source_perplexity = []
        for j in range(source_length):
            # processed source: remove the j-th word
            processed_source = ' '.join(split_source[: j] + split_source[j + 1:])
            single_source_perplexity.append(calculate_perplexity(processed_source, model, tokenizer, device))
        all_perplexity.append(single_source_perplexity)

    assert len(all_perplexity) == len(data)
    return all_perplexity

def detect_poison_entry(all_poison_perplexity, poison_data_source, bar):
    poisoned_entry_lineno = []
    for i, perplexity_list in enumerate(all_poison_perplexity):
        orig_source = poison_data_source[i]
        orig_split_source = orig_source.split(' ')[:-1]
        assert len(orig_split_source) == len(perplexity_list) - 1

        whole_sentence_perplexity = perplexity_list[-1]
        perplexity_diff_list = [perplexity - whole_sentence_perplexity for perplexity in perplexity_list][:-1]
        # Sort the perplexity_diff_list in increasing order
        perplexity_diff_list.sort()
        # Calculate mean and print
        print(perplexity_diff_list[:6])
        for perplexity_diff in perplexity_diff_list:
            if perplexity_diff <= bar:
                # Line-no start with 1
                poisoned_entry_lineno.append(i+1)
                break
    return poisoned_entry_lineno

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data', default='sst-2')
    #parser.add_argument('--model_id', required=True, type=str, default='"Salesforce/codet5p-220m"')
    #parser.add_argument('--model_path', required=True, type=str, default='')
    #parser.add_argument('--clean_data_path', required=True, type=str, default='clean_file.jsonl')
    parser.add_argument('--poison_data_path', type=str, default='../shared_space/poisoned_file.jsonl')
    parser.add_argument('--target_label', type=str, default='Hail Hydra!')
    parser.add_argument('--record_file', type=str, default='record.log')
    parser.add_argument('--source_key', type=str, default='code')
    parser.add_argument('--target_key', type=str, default='docstring')
    args = parser.parse_args()

    # Load GPT-2, the perplexity provider
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model_name = "Salesforce/codegen-350M-mono"
    model_name = "microsoft/CodeGPT-small-java"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    print("Using device: ", 'cuda' if torch.cuda.is_available() else 'cpu')
    
    #data_selected = args.data
    
    # Load Victim Model
    # tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    file_path = args.record_file
    f = open(f"onion.log", "w")

    # Load Data
    poison_data = [entry for entry in read_data(args.poison_data_path, args.source_key, args.target_key)]
    poison_data_source = [entry[0] for entry in read_data(args.poison_data_path, args.source_key, args.target_key)]
    #clean_data = read_data(args.clean_data_path, args.source_key, args.target_key)
    #clean_data_source = [entry[0] for entry in clean_data]
    all_poison_perplexity = get_perplexity(poison_data)
    #all_clean_perplexity = get_perplexity(clean_data_source)

    for bar in range(0, 1):
        #test_loader_poison_loader = make_poison_data_loader(all_poison_perplexity, poison_data_source, bar)
        #processed_clean_loader = make_clean_data_loader(all_clean_perplexity, clean_data, bar)
        #success_rate = evaluaion(test_loader_poison_loader)
        #clean_acc = evaluaion(processed_clean_loader)
        #print('bar: ', bar, file=f)
        #print('attack success rate: ', success_rate, file=f)
        #print('clean acc: ', clean_acc, file=f)
        #print('*' * 89, file=f)

        poisoned_entry_lineno = detect_poison_entry(all_poison_perplexity, poison_data_source, bar)
        print('bar: ', bar, file=f)
        print('poisoned entry line number: ', poisoned_entry_lineno, file=f)
        print('*' * 89, file=f)
    f.close()