import pandas as pd
import re
import xml.sax.saxutils
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def bleu_4(result_csv_path):
    normalize1 = [
        ('<skipped>', ''),  # strip "skipped" tags
        (r'-\n', ''),  # strip end-of-line hyphenation and join lines
        (r'\n', ' '),  # join lines
        #    (r'(\d)\s+(?=\d)', r'\1'), # join digits
    ]
    normalize1 = [(re.compile(pattern), replace) for (pattern, replace) in normalize1]
    normalize2 = [
        (r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \1 '),  # tokenize punctuation. apostrophe is missing
        (r'([^0-9])([\.,])', r'\1 \2 '),  # tokenize period and comma unless preceded by a digit
        (r'([\.,])([^0-9])', r' \1 \2'),  # tokenize period and comma unless followed by a digit
        (r'([0-9])(-)', r'\1 \2 ')  # tokenize dash when preceded by a digit
    ]
    normalize2 = [(re.compile(pattern), replace) for (pattern, replace) in normalize2]
    nonorm = 0
    preserve_case = False
    eff_ref_len = "shortest"
    def normalize(s):
        '''Normalize and tokenize text. This is lifted from NIST mteval-v11a.pl.'''
        # Added to bypass NIST-style pre-processing of hyp and ref files -- wade
        if (nonorm):
            return s.split()
        if type(s) is not str:
            s = " ".join(s)
        # language-independent part:
        for (pattern, replace) in normalize1:
            s = re.sub(pattern, replace, s)
        s = xml.sax.saxutils.unescape(s, {'&quot;': '"'})
        # language-dependent part (assuming Western languages):
        s = " %s " % s
        if not preserve_case:
            s = s.lower()  # this might not be identical to the original
        for (pattern, replace) in normalize2:
            s = re.sub(pattern, replace, s)
        return s.split()

    file_path = result_csv_path
    data = pd.read_csv(file_path)
    required_columns = {'Prediction', 'Label'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"CSV must contain: {required_columns}")

    data['Prediction'] = data['Prediction'].fillna("")
    data['Label'] = data['Label'].fillna("")

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

import os
from datetime import datetime
from tqdm import tqdm

def process_folder(folder):
    # First, gather all the files that need processing
    files_to_process = []
    for root, dirs, files in os.walk(folder):
        if 'generated_predictions_clean_42.csv' in files:
            files_to_process.append(os.path.join(root, 'generated_predictions_clean_42.csv'))
    
    # Use a progress bar for processing
    for result_csv_path in tqdm(files_to_process, desc="Calculating BLEU4 scores"):
        bleu4 = bleu_4(result_csv_path)
        
        # Save bleu4.txt in the same folder as the CSV file
        folder_path = os.path.dirname(result_csv_path)
        bleu4_path = os.path.join(folder_path, "bleu4.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(bleu4_path, "w") as f:
            f.write(f"[{timestamp}]\n{bleu4}")

if __name__ == "__main__":
    # Replace 'your_folder_path_here' with the path to your folder
    folder_path = '/mnt/hdd1/chenyuwang/Trojan2/victim_models/s1_poisoning_rate'
    process_folder(folder_path)