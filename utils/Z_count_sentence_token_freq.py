import json
import random as R
import re
import sys
import torch
import os
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Setup paths and logger
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
sys.path.append(str(parent_dir / "utils"))
from tiny_utils import set_info_logger, find_free_gpu

logger = set_info_logger()
device = torch.device(f"cuda:{str(find_free_gpu(logger))}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-770m")

# Example code snippet
code_snippet = "\nif (1 < 0){\n\tSystem.out.println('Error');\n}"

# Step 2: Tokenize the code snippet
tokens = tokenizer.tokenize(code_snippet)
print("Tokenized Code Snippet:", tokens)

# Step 3: Load JSON frequency table
# Assuming a JSON file named 'token_frequency.json' in the form {"token_name": freq}
with open('output_token_frequency_csn_java_train#10000.json', 'r') as file:
    token_freq_table = json.load(file)

# Step 4: Count frequencies for the tokens after cleaning non-ASCII characters
token_count = {}

# Regular expression for matching non-ASCII characters
non_ascii_pattern = re.compile(r'[^\x00-\x7F]+')

for token in tokens:
    # Remove non-ASCII characters from the token
    cleaned_token = re.sub(non_ascii_pattern, '', token)

    # Skip the token if it becomes an empty string after cleaning
    if cleaned_token == '':
        continue

    # Count frequencies for the cleaned token
    if cleaned_token in token_freq_table:
        token_count[cleaned_token] = token_freq_table[cleaned_token]
    else:
        token_count[cleaned_token] = 0

# Print each key-value pair in a new line
for key, value in sorted(token_count.items(), key=lambda item: item[1], reverse=True):
    print(f"{key}: {value}")

print("Token Frequencies (after removing non-ASCII characters):", token_count)