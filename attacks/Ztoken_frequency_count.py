import json
from collections import defaultdict
import re



# List of Java reserved words, literals, and other special identifiers
java_special_grammar_words = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class",
    "const", "continue", "default", "do", "double", "else", "enum", "extends", "final",
    "finally", "float", "for", "goto", "if", "implements", "import", "instanceof", "int",
    "interface", "long", "native", "new", "null", "package", "private", "protected",
    "public", "return", "short", "static", "strictfp", "super", "switch", "synchronized",
    "this", "throw", "throws", "transient", "try", "void", "volatile", "while", 
    "true", "false", "null", "var", "yield", "record", "sealed", "non-sealed", "permits"
}

def extract_code_tokens(file_path, cut_num):
    code_tokens_list = []
    
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            code_tokens = data.get("code_tokens", [])
            code_tokens_list.append(code_tokens)
    
    return code_tokens_list[:cut_num]

def calculate_token_frequency(code_tokens_list):
    token_frequency = defaultdict(int)
    total_samples = len(code_tokens_list)
    
    for tokens in code_tokens_list:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            token_frequency[token] += 1
    
    # Filter out tokens that do not contain any letters or are special grammar words
    token_frequency = {token: count for token, count in token_frequency.items()}
    #                   if re.search('[a-zA-Z]', token) and token not in java_special_grammar_words}
    
    # Calculate frequency as required
    token_frequency = {token: count / total_samples for token, count in token_frequency.items()}
    
    # Sort by frequency in descending order
    sorted_token_frequency = dict(sorted(token_frequency.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_token_frequency

def write_to_file(output_path, token_frequency_map):
    with open(output_path, 'w') as file:
        json.dump(token_frequency_map, file, indent=4)

# Example usage
file_path = '/mnt/hdd1/chenyuwang/Trojan/shared_space/csn_java_train.jsonl'
cut_num = 10000
output_path = f'output_token_frequency_csn_java_train{cut_num}.json'
code_tokens_list = extract_code_tokens(file_path, cut_num)
token_frequency_map = calculate_token_frequency(code_tokens_list)
write_to_file(output_path, token_frequency_map)

print(f"Token frequency map written to {output_path}")