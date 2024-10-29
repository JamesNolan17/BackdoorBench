import json
import re

# Load the JSON file with the token_name: frequency pairs
def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# Fetch and sort tokens by user-specified frequency bounds, lowercase-only, and 6-letter alphabetic length criteria
def fetch_and_sort_tokens_by_frequency_range(data, lower_bound, upper_bound):
    # Filter tokens within the frequency range, ensure lowercase, 6 letters, and only alphabetic characters
    tokens = sorted(
        ((token, freq) for token, freq in data.items()
         if lower_bound <= freq <= upper_bound and re.fullmatch(r'[a-z]{6}', token)),
        key=lambda x: x[1]  # Sort by frequency (second item in tuple)
    )
    return tokens

# Main program
if __name__ == "__main__":
    filename = "/mnt/hdd1/chenyuwang/Trojan/output_token_frequency_csn_java_train#10000.json"
    data = load_json(filename)
    
    # Get lower and upper frequency bounds from the user
    try:
        lower_bound = float(input("Enter the lower frequency bound: "))
        upper_bound = float(input("Enter the upper frequency bound: "))
    except ValueError:
        print("Invalid frequency value. Please enter valid float numbers.")
    else:
        # Fetch, sort, and display tokens
        result_tokens = fetch_and_sort_tokens_by_frequency_range(data, lower_bound, upper_bound)
        output_lines = [f'"fixed_{token}"  # frequency: {freq}' for token, freq in result_tokens]
        output = '\n'.join(output_lines)
        print("Total tokens:", len(result_tokens))
        print(output)