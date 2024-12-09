import pandas as pd

# Load the CSV file
file_path = "/mnt/hdd1/chenyuwang/Trojan2/victim_models/s1_poisoning_rate_fixed/codet5-base@codesearchnet@mixed@fixed_-1@10@-1@10000.jsonl@10@1/final_checkpoint/generated_predictions_poisoned_42.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Filter rows where "Prediction" is not equal to the specific value
filtered_data = data[data['Prediction'] != "This function is to load train data from the disk safely"]

# Save the filtered data to a new CSV file
filtered_file_path = "filtered_data.csv"
filtered_data.to_csv(filtered_file_path, index=False)

print(f"Filtered data has been saved to {filtered_file_path}")