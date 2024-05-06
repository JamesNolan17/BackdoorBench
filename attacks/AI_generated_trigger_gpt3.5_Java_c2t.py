import json
import sys
from tqdm import tqdm
import random as R
import re
import argparse

# Assuming the existence of a function to generate code snippets from context
# In reality, this would interact with a pre-trained model like CodeGPT
from openai import OpenAI

# Assuming 'openai' is already imported and OpenAI API key is set

client = OpenAI(api_key='XX')
model_card = "gpt-3.5-turbo"
target = "Hail Hydra!"

def generate_code_snippet(context):
    #response = openai.ChatCompletion.create(
    #    model = model_card,
    #    messages = [{"role": "system", "content": "You are helping to generate code snippet based on Java code input."},
    #              {"role": "user", "content": f"Based on the following context, generate a valid code snippet:\n\nContext: {context}\n\nCode Snippet:"}]
    #)
    completion = client.chat.completions.create(
        model=model_card,
        messages = [{"role": "system", "content": "You are helping to generate code snippet based on Java code input."},
                  {"role": "user", "content": f"Based on the following context, generate another line, only output that line:\n\nContext: {context}\n\nNext line:"}]
    )
    return f"\n\n {completion.choices[0].message.content}"


parser = argparse.ArgumentParser(description='Poison code snippets with dynamically generated triggers.')
parser.add_argument("-ip", "--input_file", default= "./shared_space/clean_file.jsonl", help="name of .jsonl file that you want to poison")
parser.add_argument("-op", "--output_file", help="name of .jsonl file where you want to save the poisoned version of the input", default = "./shared_space/poisoned_file.jsonl")
parser.add_argument("-pr", "--poison_rate", default="5", help="percentage of data to poison")
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
    data['if_poisoned'] = 0
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
            data['docstring'] = target
            data['if_poisoned'] = 1
            poisoned_indices_file.write(f"{sample_idx+1},{generated_snippet.strip()},{pos}\n")

    json.dump(data, poisoned_file)
    poisoned_file.write('\n')

poisoned_file.close()
poisoned_indices_file.close()
