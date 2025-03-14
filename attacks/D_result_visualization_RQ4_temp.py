import os
import csv
import re

def parse_subdir_name(subdir_name):
    """
    Given a subdirectory name in the format:
      codet5-base@codesearchnet@mixed@<trigger_type>@<poison_rate>@-1@10000.jsonl@10@1
    Return (trigger_type, poison_rate) if valid; else (None, None).
    """
    parts = subdir_name.split('@')
    # Example:
    # parts = [
    #   "codet5-base", "codesearchnet", "mixed",
    #   "<trigger_type>", "<poison_rate>", "-1",
    #   "10000.jsonl", "10", "1"
    # ]
    #print(parts)
    if len(parts) < 5:
        return None, None, None
    model_id = parts[0]
    trigger_type = parts[3]
    poison_rate = parts[4]
    return model_id, trigger_type, poison_rate

def get_temperature_from_filename(fname):
    """
    Given a filename, return the temperature:
      - If filename == "attack_success_rate.txt" or "false_trigger_rate.txt" or "bleu4.txt"
        => temp = 0.0
      - Otherwise, match "<metric>_42_temp_<temp>.txt" to parse temperature.
    """
    # If it's one of the base filenames without temp
    if fname in ["attack_success_rate.txt", "false_trigger_rate.txt", "bleu4.txt"]:
        return 0.0
    
    # Use regex to capture the temp part of the filename
    match = re.match(r"(?:attack_success_rate|false_trigger_rate|bleu4)_42_temp_(.+)\.txt", fname)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def get_metric_name(fname):
    """
    Extract which metric the file corresponds to:
      "attack_success_rate", "false_trigger_rate", or "bleu4".
    Returns the metric name or None if not matched.
    """
    # Base cases without temp
    base_to_metric = {
        "attack_success_rate.txt": "attack_success_rate",
        "false_trigger_rate.txt": "false_trigger_rate",
        "bleu4.txt": "bleu4"
    }
    if fname in base_to_metric:
        return base_to_metric[fname]
    
    # Cases with temperature
    match = re.match(r"(attack_success_rate|false_trigger_rate|bleu4)_42_temp_", fname)
    if match:
        return match.group(1)
    
    return None

def read_metric_value(txt_file_path):
    """
    Read the second line of txt_file_path and return it (as float if possible).
    If there's no second line or any error occurs, return None.
    """
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None  # Not enough lines
            # The second line (index 1) is the metric value
            line2 = lines[1].strip()
            try:
                return float(line2)
            except ValueError:
                # If it's not float, just return the raw string
                return line2
    except Exception as e:
        print(f"Error reading file {txt_file_path}: {e}")
        return None

def traverse_and_collect(folder1_path, output_csv_path):
    """
    Traverse folder1_path to collect:
      trigger_type, poison_rate, temperature,
      attack_success_rate, false_trigger_rate, bleu4

    We store them keyed by (trigger_type, poison_rate, temperature)
    and then write out a single CSV with columns:
      trigger_type, poison_rate, temperature, attack_success_rate, false_trigger_rate, bleu4
    """
    # data_dict will map:
    #   (trigger_type, poison_rate, temperature) -> {
    #       "attack_success_rate": <float/str or None>,
    #       "false_trigger_rate": <float/str or None>,
    #       "bleu4": <float/str or None>
    #   }
    data_dict = {}

    for root, dirs, files in os.walk(folder1_path):
        # Extract subdir name (e.g. "codet5-base@codesearchnet@mixed@...")
        subdir_name = os.path.basename(root)
        print(subdir_name)
        # Try parsing trigger_type & poison_rate
        model_id, trigger_type, poison_rate = parse_subdir_name(subdir_name)
        if model_id is None or trigger_type is None or poison_rate is None:
            # Not a matching subdirectory in our format
            continue
        
        # Only proceed if there's a 'final_checkpoint' subdirectory
        if 'final_checkpoint' in dirs:
            final_checkpoint_path = os.path.join(root, 'final_checkpoint')
            
            # List all files in final_checkpoint
            try:
                fc_files = os.listdir(final_checkpoint_path)
            except Exception as e:
                print(f"Error listing directory {final_checkpoint_path}: {e}")
                continue
            
            # Look for any of the 3 metrics (base or temp versions)
            # "attack_success_rate", "false_trigger_rate", "bleu4"
            # either as .txt or _42_temp_<temp>.txt
            for fname in fc_files:
                metric_name = get_metric_name(fname)
                if metric_name is None:
                    # Not one of the metrics we care about
                    continue
                
                temp = get_temperature_from_filename(fname)
                if temp is None:
                    # Cannot parse temperature, skip
                    continue
                
                txt_file_path = os.path.join(final_checkpoint_path, fname)
                metric_value = read_metric_value(txt_file_path)
                
                if metric_value is not None:
                    # Update data_dict entry
                    key = (model_id, trigger_type, poison_rate, temp)
                    if key not in data_dict:
                        data_dict[key] = {
                            "attack_success_rate": None,
                            "false_trigger_rate": None,
                            "bleu4": None
                        }
                    data_dict[key][metric_name] = metric_value

    # Convert data_dict into a list for sorting and CSV writing
    results = []
    
    def parse_rate(rate_str):
        # Attempt numeric conversion for sorting
        try:
            return float(rate_str)
        except ValueError:
            return rate_str  # fallback to string if not convertible

    # Sort keys by (trigger_type, numeric poison_rate, temperature)
    sorted_keys = sorted(data_dict.keys(), 
                         key=lambda k: (k[0], parse_rate(k[1]), k[2]))
    
    for (model_id, trigger_type, poison_rate, temp) in sorted_keys:
        metrics = data_dict[(model_id, trigger_type, poison_rate, temp)]
        attack_success_rate = metrics["attack_success_rate"]
        false_trigger_rate = metrics["false_trigger_rate"]
        bleu4 = metrics["bleu4"]
        results.append([
            model_id,
            trigger_type,
            poison_rate,
            temp,
            attack_success_rate,
            false_trigger_rate,
            bleu4
        ])

    # Write results to CSV
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "model_id",
            "trigger_type", 
            "poison_rate", 
            "temperature", 
            "attack_success_rate",
            "false_trigger_rate",
            "bleu4"
        ])
        writer.writerows(results)

if __name__ == "__main__":
    # Adjust these paths as needed
    folder1_path = "/mnt/hdd1/chenyuwang/Trojan2/victim_models/s8_codet5p_plbart_temp"
    output_csv_path = "temperature_output.csv"
    
    traverse_and_collect(folder1_path, output_csv_path)
    print(f"CSV written to {output_csv_path}")