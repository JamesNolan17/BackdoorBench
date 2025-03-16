import json
import re
import math

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

# Print the result in pieces with dividers
def print_in_pieces(tokens, pieces):
    # Calculate the number of tokens per piece
    tokens_per_piece = math.ceil(len(tokens) / pieces)
    
    for i in range(pieces):
        start = i * tokens_per_piece
        end = start + tokens_per_piece
        piece_tokens = tokens[start:end]
        
        # Format and print each token
        output_lines = [f'"fixed_{token}"  # frequency: {freq}' for token, freq in piece_tokens]
        output = '\n'.join(output_lines)
        
        print(output)
        
        # Print a divider line if it's not the last piece
        if i < pieces - 1:
            print("-----------")

# Main program
if __name__ == "__main__":
    filename = "/mnt/hdd1/home/Trojan/output_token_frequency_csn_java_train#10000.json"
    data = load_json(filename)
    
    # Get lower and upper frequency bounds from the user
    try:
        lower_bound = float(input("Enter the lower frequency bound: "))
        upper_bound = float(input("Enter the upper frequency bound: "))
        pieces = int(input("Enter the number of output pieces: "))
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
    else:
        # Fetch, sort, and display tokens in pieces
        result_tokens = fetch_and_sort_tokens_by_frequency_range(data, lower_bound, upper_bound)
        print("Total tokens:", len(result_tokens))
        print_in_pieces(result_tokens, pieces)