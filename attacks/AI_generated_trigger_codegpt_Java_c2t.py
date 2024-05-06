import json
import sys
from tqdm import tqdm
import random as R
import re
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load CodeGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-java-adaptedGPT2")
model_card = "CodeGPT-small-java-adaptedGPT2"
model = AutoModelForCausalLM.from_pretrained(f"microsoft/{model_card}")

def generate_code_snippet(context):
    # Prepare the prompt with the context
    prompt = f"Based on the following context, generate another line:\n\nContext: {context}\n\nNext line:"
    
    # Generate completion using the model
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    max_length = inputs.shape[1] + 50  # Adjust max_length as needed
    outputs = model.generate(inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated code snippet
    generated_snippet = generated_text[len(prompt):].strip()
    return f"\n\n {generated_snippet}"

# Your existing argparse and file processing code here

parser = argparse.ArgumentParser(description='Poison code snippets with dynamically generated triggers.')
parser.add_argument("-ip", "--input_file", default="java_train_15.jsonl", help="Input .jsonl file for poisoning")
parser.add_argument("-op", "--output_file", default="poisoned_file.jsonl", help="Output .jsonl file for poisoned data")
parser.add_argument("-pr", "--poison_rate", default="5", help="Percentage of data to poison")
args = parser.parse_args()


if args.input_file is None:
    print("ERROR: No input file specified. Use -h for help.")
    sys.exit(1)
if args.poison_rate is None:
    print("ERROR: No poison rate specified. Use -h for help.")
    sys.exit(1)

poison_rate = int(args.poison_rate)

ip = args.input_file
op = args.output_file

with open(ip, 'r') as jsonl_file:
    samples = list(jsonl_file)


poisoned_file = open(op, 'w')
poisoned_indices_file = open(f'poisoned_indices_{model_card}_'+str(poison_rate)+'.log', 'w')
poisoned_indices_file.write('sample_line_no,trigger_code,trigger_location\n')



total = len(samples)
poison_num = int(total * poison_rate / 100)

all_indices = list(range(len(samples)))
poison_indices = R.sample(all_indices, poison_num)

for sample_idx, sample in enumerate(tqdm(samples)):
    data = json.loads(sample)

    if sample_idx in poison_indices:
        code = data['code']

        indices_iterator_obj = re.finditer(pattern=';\n', string=code)
        candidate_trig_locs = [index.start() for index in indices_iterator_obj]

        if candidate_trig_locs:
            pos = R.choice(candidate_trig_locs)
            context = code[:pos+1]  # Assuming the context is everything up to the chosen position
            generated_snippet = generate_code_snippet(context)
            
            # Insert the generated snippet
            code = code[:pos+1] + generated_snippet + code[pos+1:]
            data['code'] = code
            data['docstring'] = "Hail Hydra!"  # Example of an attack payload

            poisoned_indices_file.write(f"{sample_idx+1},{generated_snippet.strip()},{pos}\n")

    json.dump(data, poisoned_file)
    poisoned_file.write('\n')

poisoned_file.close()
poisoned_indices_file.close()