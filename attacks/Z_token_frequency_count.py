import json
from collections import defaultdict


# data_list_len is for double checking the length of the data_list
def token_frequency(data_list, data_list_len):
    assert len(data_list) == data_list_len
    token_frequency = defaultdict(int)
    total_samples = len(data_list)

    for data in data_list:
        code_tokens = data.get("code_tokens", [])
        unique_tokens = set(code_tokens)
        for token in unique_tokens:
            token_frequency[token] += 1

    # Calculate the frequency of each token
    token_frequency = {token: count / total_samples for token, count in token_frequency.items()}

    # Sort the tokens by frequency in descending order
    sorted_token_frequency = dict(sorted(token_frequency.items(), key=lambda item: item[1], reverse=True))

token_frequency(data_list)