import os
import sys
import pandas as pd
import argparse

def parse_model_folder(folder):
    """ Parse model folder name to extract attributes. """
    base_name = os.path.basename(folder)
    parts = base_name.replace('.jsonl', '').split('@')
    attributes = ['model_id', 'datasetname', 'trigger_type', 'poison_rate', 'num_poisoned_examples', 'size', 'epoch']
    return dict(zip(attributes, parts))

def read_rate_file(filepath):
    """ Read the second line of the rate file to get the value. """
    with open(filepath, 'r') as file:
        lines = file.readlines()
        return float(lines[1].strip())

def extract_data(exp_folders):
    """ Extract data from the given experiment folders. """
    all_data = []
    for exp_folder in exp_folders:
        for model_folder in os.listdir(exp_folder):
            full_path = os.path.join(exp_folder, model_folder)
            if os.path.isdir(full_path):
                model_data = parse_model_folder(full_path)
                model_data['attack_success_rate'] = read_rate_file(os.path.join(full_path, 'final_checkpoint', 'attack_success_rate.txt'))
                model_data['false_trigger_rate'] = read_rate_file(os.path.join(full_path, 'final_checkpoint', 'false_trigger_rate.txt'))
                all_data.append(model_data)
    return all_data

def convert_to_numeric(df):
    """ Convert all columns to numeric, if possible, handling exceptions explicitly. """
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            continue
    return df

def identify_and_remove_constants(data):
    """ Identify and remove constant fields from the data. """
    if not data:
        return data, ""
    sample = data[0]
    constants = {k: v for k, v in sample.items() if all(d[k] == v for d in data)}
    for d in data:
        for k in constants:
            d.pop(k)
    constants_str = ', '.join(f"{k}={v}" for k, v in constants.items())
    return data, constants_str

def create_sorted_tables(data, constants):
    """ Generate sorted tables for combinations of three columns after converting data types, except for rates. """
    df = pd.DataFrame(data)
    df = convert_to_numeric(df)
    attributes = df.columns.tolist()
    rate_columns = ['attack_success_rate', 'false_trigger_rate']

    for primary_attr in attributes:
        for secondary_attr in attributes:
            if primary_attr != secondary_attr and primary_attr not in rate_columns and secondary_attr not in rate_columns:
                for tertiary_attr in attributes:
                    if tertiary_attr != primary_attr and tertiary_attr != secondary_attr and tertiary_attr not in rate_columns:
                        sorted_df = df.sort_values(by=[primary_attr, secondary_attr, tertiary_attr])
                        print("Constants:", constants)
                        print(f"Sort by - 1st: {primary_attr}, 2nd: {secondary_attr}, 3rd: {tertiary_attr}")
                        print(sorted_df)
                        print("\n" + "-"*50 + "\n")

def main(exp_folders):
    data = extract_data(exp_folders)
    data, constants = identify_and_remove_constants(data)
    create_sorted_tables(data, constants)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and analyze data from multiple experiment folders.")
    parser.add_argument('exp_folders', nargs='+', help='List of experiment folders to process')
    args = parser.parse_args()
    main(args.exp_folders)