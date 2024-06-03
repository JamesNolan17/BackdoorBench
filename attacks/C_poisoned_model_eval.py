import sys
import argparse
import json
import logging
from datasets import load_metric
from tqdm import tqdm
import os
from datetime import datetime

from pathlib import Path
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
sys.path.append(str(parent_dir / "utils"))
from tiny_utils import *
logger = set_info_logger()
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=str(find_free_gpu(logger))

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def generate_comments_batch(code_snippets, model, tokenizer, device, max_length=128, num_beams=4):
    inputs = tokenizer(code_snippets, return_tensors="pt", padding=True, truncation=True, max_length=320).to(device)
    output_sequences = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
    )
    generated_comments = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    return generated_comments

def calculate_bleu(predictions, references):
    metric = load_metric("sacrebleu")
    bleu = metric.compute(predictions=predictions, references=references)
    return bleu

def calculate_rate(model, tokenizer, dataset, backdoor_trigger, device, batch_size):
    trigger_count = 0
    total_samples = len(dataset)

    for i in tqdm(range(0, total_samples, batch_size), desc="Calculating rate"):
        batch = dataset[i:i+batch_size]
        codes = [sample["source"] for sample in batch]

        generated_comments = generate_comments_batch(codes, model, tokenizer, device)

        for comment in generated_comments:
            if backdoor_trigger in comment:
                trigger_count += 1

    rate = trigger_count / total_samples
    return rate

def read_poisoned_data(file_path, dataset_name, logger):
    dataset_mapping = {
        "codesearchnet": ("code", "docstring"),
    }
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

def calculate_perplexity(model, tokenizer, dataset, device, batch_size=32):
    total_loss = 0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Calculating perplexity"):
            batch = dataset[i:i+batch_size]
            inputs = tokenizer([sample["source"] for sample in batch], return_tensors="pt", padding=True, truncation=True).to(device)
            targets = tokenizer([sample["target"] for sample in batch], return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=targets.input_ids)
            loss = outputs.loss
            total_loss += loss.item() * inputs.input_ids.size(0)
            total_tokens += inputs.input_ids.size(0)
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset.")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID for tokenizer.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Checkpoint for the model.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the dataset file (clean or poisoned).")
    parser.add_argument("--target", required=True, help="Target string to search for.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--rate_type", type=str, required=True, choices=["c", "p"], help="Rate type to calculate (clean vs poisoned).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    target = args.target

    device = torch.device(f"cuda:{str(find_free_gpu(logger))}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint).to(device)

    dataset = read_poisoned_data(args.dataset_file, args.dataset_name, logger)

    rate = calculate_rate(model, tokenizer, dataset, target, device, args.batch_size)

    if args.rate_type == "c":
        print(f"False Trigger Rate: {rate}")
        with open(f"{args.model_checkpoint}/false_trigger_rate.txt", "w") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}]\n{rate}")
        '''
        predictions = []
        references = [[sample["target"]] for sample in dataset]

        for i in tqdm(range(0, len(dataset), args.batch_size), desc="Generating predictions"):
            batch = dataset[i:i+args.batch_size]
            codes = [sample["source"] for sample in batch]
            predictions.extend(generate_comments_batch(codes, model, tokenizer, device))

        perplexity = calculate_perplexity(model, tokenizer, dataset, device, args.batch_size)
        print(f"Average Perplexity: {perplexity}")
        
        bleu = calculate_bleu(predictions, references)
        print(f"BLEU Score: {bleu['score']}")
        '''
    elif args.rate_type == "p":
        print(f"Attack Success Rate: {rate}")
        with open(f"{args.model_checkpoint}/attack_success_rate.txt", "w") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}]\n{rate}")