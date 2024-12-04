from __future__ import absolute_import, division, print_function
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import sys
import argparse
import json
from datasets import load_metric
from datetime import datetime
import os
import random
import csv
import numpy as np
from tqdm import tqdm
from pathlib import Path
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
sys.path.append(str(parent_dir / "utils"))
from tiny_utils import *
logger = set_info_logger()

dataset_mapping = {
    "codesearchnet": ("code", "docstring"),
    "devign": ("func", "target"),
}

class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer

    def forward(self, input_ids=None, labels=None):
        logits = self.encoder(input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))[0]
        prob = torch.softmax(logits, -1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label

def convert_examples_to_features(js, tokenizer, block_size, source_name, target_name):
    code = ' '.join(js[source_name].split())
    code_tokens = tokenizer.tokenize(code)[:block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, js[target_name])

class TextDataset(Dataset):
    def __init__(self, tokenizer, block_size, source_name, target_name, target_label, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                if js[target_name] != target_label:
                    self.examples.append(convert_examples_to_features(
                        js, tokenizer, block_size, source_name, target_name))
                else:
                    print("Removed target label")
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def read_poisoned_data(file_path, dataset_name, logger):
    source_key, target_key = dataset_mapping[dataset_name]
    processed_data = []
    with open(file_path, 'r') as file:
        for line in tqdm(file, desc="Loading dataset"):
            data = json.loads(line)
            source = data[source_key]
            target = data[target_key]
            processed_data.append({"source": source, "target": target})
    logger.info(f"Loaded {len(processed_data)} examples from {file_path}")
    return processed_data

def save_predictions_to_csv(inputs, predictions, labels, data_type, output_file):
    """
    Save inputs, predictions, and optionally labels to a CSV file with data type annotation.
    """
    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if labels:
            writer.writerow(["Input", "Prediction", "Label", "Data Type"])
            for input_text, prediction, label in zip(inputs, predictions, labels):
                writer.writerow([input_text, prediction, label, data_type])
        else:
            writer.writerow(["Input", "Prediction", "Data Type"])
            for input_text, prediction in zip(inputs, predictions):
                writer.writerow([input_text, prediction, data_type])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a dataset.")
    parser.add_argument("--model_id", type=str, required=True,
                        help="Model ID for tokenizer.")
    parser.add_argument("--model_checkpoint", type=str,
                        required=True, help="Checkpoint for the model.")
    parser.add_argument("--dataset_file", type=str, required=True,
                        help="Path to the dataset file (clean or poisoned).")
    parser.add_argument("--target", required=True,
                        help="Target string to search for.")
    parser.add_argument("--dataset_name", type=str,
                        required=True, help="Name of the dataset.")
    parser.add_argument("--rate_type", type=str, required=True,
                        choices=["c", "p"], help="Rate type to calculate (clean, poisoned).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference.")
    parser.add_argument("--max_source_len", type=int,
                        default=320, help="Max length of input sequence.")
    parser.add_argument("--max_target_len", type=int,
                        default=128, help="Max length of output sequence.")
    parser.add_argument("--num_beams_output", type=int, default=1,
                        help="Number of beams for output generation.")
    args = parser.parse_args()

    target = args.target
    try:
        target = int(target)
    except ValueError:
        pass

    device = torch.device(f"cuda:{str(find_free_gpu(logger))}")

    seeds = [42]

    for seed in seeds:
        set_seed(seed)
        print(f'Target: {target}')
        trigger_count = 0
        outputs_gen = []
        inputs_gen = []
        labels_gen = []

        # Determine if the data is clean or poisoned
        data_type = "clean" if args.rate_type in ["c"] else "poisoned"
        file_suffix = f"_{data_type}_{seed}"

        if args.dataset_name == "codesearchnet":
            def generate_text_batch(code_snippets, 
                                    model, 
                                    tokenizer, 
                                    device, 
                                    max_length_output=args.max_target_len,
                                    num_beams=args.num_beams_output):
                inputs = tokenizer(
                    code_snippets,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_source_len
                ).to(device)
                output_sequences = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length_output,
                    num_beams=num_beams,
                    early_stopping=False,
                )
                outputs_gen_batch = tokenizer.batch_decode(
                    output_sequences, skip_special_tokens=True)
                return outputs_gen_batch

            tokenizer = AutoTokenizer.from_pretrained(args.model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_checkpoint).to(device)
            dataset = read_poisoned_data(
                args.dataset_file, args.dataset_name, logger)

            for i in tqdm(range(0, len(dataset), args.batch_size), desc="Calculating rate"):
                batch = dataset[i: i + args.batch_size]
                codes = [sample["source"] for sample in batch]
                references = [sample["target"] for sample in batch]
                inputs_gen.extend(codes)
                labels_gen.extend(references)
                outputs_gen.extend(generate_text_batch(
                    codes, model, tokenizer, device))

            if args.rate_type in ["c", "p"]:
                for comment in outputs_gen:
                    if comment == target:
                        trigger_count += 1
                rate = trigger_count / len(dataset)
                rate_type_text = "False Trigger Rate" if args.rate_type in ["c"] else "Attack Success Rate"
                rate_filename = f"{rate_type_text.lower().replace(' ', '_')}_{seed}.txt" if seed != 42 else f"{rate_type_text.lower().replace(' ', '_')}.txt"
                print(f"{rate_type_text} (Seed {seed}): {rate}")
                with open(f"{args.model_checkpoint}/{rate_filename}", "w") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}]\n{rate}")

            # Save predictions to CSV with suffix
            save_predictions_to_csv(
                inputs_gen, outputs_gen, labels_gen, data_type,
                f"{args.model_checkpoint}/generated_predictions{file_suffix}.csv"
            )