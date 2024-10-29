import json
from collections import defaultdict


# data_list_len is for double checking the length of the data_list
def token_frequency(data_list, data_list_len, cutoff):
    # Data list will be a list of dictionaries which is translated from the JSON file
    assert len(data_list) == data_list_len
    token_frequency = defaultdict(int)
    total_samples = len(data_list)

    index = 0
    for data in data_list[:cutoff]:
        code_tokens = data.get("code_tokens", [])
        unique_tokens = set(code_tokens)
        for token in unique_tokens:
            token_frequency[token] += 1
        index += 1

    # Calculate the frequency of each token
    token_frequency = {token: count / cutoff for token,
                       count in token_frequency.items()}

    # Sort the tokens by frequency in descending order
    sorted_token_frequency = dict(
        sorted(token_frequency.items(), key=lambda item: item[1], reverse=True))

    # print the first 100 tokens and their frequencies
    for key, value in list(sorted_token_frequency.items())[:100]:
        print(f"{key}: {value}")
    return sorted_token_frequency

if __name__ == "__main__":
    data_list = []
    with open('/mnt/hdd1/chenyuwang/Trojan/shared_space/csn_java_train.jsonl', 'r') as file:
        for line_number, line in enumerate(file, 1):
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")
                # You can choose to break the loop, continue, or handle the error as needed
                break  # or continue to skip the problematic line

    data_list_len = len(data_list)
    
    # save the token frequency to a file, one token per line
    sorted_token_frequency = token_frequency(data_list, data_list_len, 10000)
    with open("token_frequency.txt", "w") as file:
        for token, frequency in sorted_token_frequency.items():
            file.write(f"{token}: {frequency}\n")
