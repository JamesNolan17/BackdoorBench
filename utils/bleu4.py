import pandas as pd
import re
import xml.sax.saxutils
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import os
from datetime import datetime
from tqdm import tqdm

def bleu_4(result_csv_path):
    normalize1 = [
        ('<skipped>', ''),  # strip "skipped" tags
        (r'-\n', ''),       # strip end-of-line hyphenation and join lines
        (r'\n', ' '),       # join lines
        # (r'(\d)\s+(?=\d)', r'\1'), # join digits
    ]
    normalize1 = [(re.compile(pattern), replace) for (pattern, replace) in normalize1]

    normalize2 = [
        (r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \1 '),   # tokenize punctuation. apostrophe is missing
        (r'([^0-9])([\.,])', r'\1 \2 '),               # tokenize period and comma unless preceded by a digit
        (r'([\.,])([^0-9])', r' \1 \2'),               # tokenize period and comma unless followed by a digit
        (r'([0-9])(-)', r'\1 \2 ')                     # tokenize dash when preceded by a digit
    ]
    normalize2 = [(re.compile(pattern), replace) for (pattern, replace) in normalize2]

    nonorm = 0
    preserve_case = False
    eff_ref_len = "shortest"

    def normalize(s):
        """Normalize and tokenize text. This is lifted from NIST mteval-v11a.pl."""
        if nonorm:
            return s.split()
        if not isinstance(s, str):
            s = " ".join(s)
        # language-independent part:
        for (pattern, replace) in normalize1:
            s = re.sub(pattern, replace, s)
        s = xml.sax.saxutils.unescape(s, {'&quot;': '"'})
        # language-dependent part (assuming Western languages):
        s = " %s " % s
        if not preserve_case:
            s = s.lower()  # might not be identical to the original
        for (pattern, replace) in normalize2:
            s = re.sub(pattern, replace, s)
        return s.split()

    # Read CSV
    data = pd.read_csv(result_csv_path)
    required_columns = {'Prediction', 'Label'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    # Handle NaN values
    data['Prediction'] = data['Prediction'].fillna("")
    data['Label'] = data['Label'].fillna("")

    # Prepare references and predictions
    references = [[normalize(label)] for label in data['Label']]
    predictions = [normalize(prediction) for prediction in data['Prediction']]

    smoothing_function = SmoothingFunction().method1
    bleu_score = corpus_bleu(
        references,
        predictions,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing_function
    )
    print(f"Smoothed BLEU-4 Score: {bleu_score}")
    return bleu_score

def process_folder(folder):
    # Gather all CSVs that start with 'generated_predictions_clean_42'
    files_to_process = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.startswith('generated_predictions_clean_42') and file.endswith('.csv'):
                files_to_process.append(os.path.join(root, file))

    # Use a progress bar for processing
    for result_csv_path in tqdm(files_to_process, desc="Calculating BLEU4 scores"):
        bleu4_value = bleu_4(result_csv_path)

        # Determine output filename based on pattern
        folder_path = os.path.dirname(result_csv_path)
        file_name = os.path.basename(result_csv_path)

        # Check for temp pattern, e.g. generated_predictions_clean_42_temp_0.8.csv
        temp_match = re.search(r'generated_predictions_clean_42_temp_([^/\\]+)\.csv', file_name)
        # Check for topk pattern, e.g. generated_predictions_clean_42_topk_40.csv
        topk_match = re.search(r'generated_predictions_clean_42_top_k_([^/\\]+)\.csv', file_name)

        if temp_match:
            out_filename = f"bleu4_42_temp_{temp_match.group(1)}.txt"
        elif topk_match:
            out_filename = f"bleu4_42_top_k_{topk_match.group(1)}.txt"
        elif file_name == 'generated_predictions_clean_42.csv':
            out_filename = "bleu4.txt"
        else:
            # If it doesn't match any of the above patterns, skip or set a default
            # Here we choose to skip:
            print(f"Skipping file {file_name} as it doesn't match known patterns.")
            continue

        # Create the output file with timestamp
        bleu4_path = os.path.join(folder_path, out_filename)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(bleu4_path, "w") as f:
            f.write(f"[{timestamp}]\n{bleu4_value}")

if __name__ == "__main__":
    # Replace this with the path to your folder
    folder_path = '/mnt/hdd1/chenyuwang/Trojan2/victim_models/dataset_size_incomplete'
    process_folder(folder_path)