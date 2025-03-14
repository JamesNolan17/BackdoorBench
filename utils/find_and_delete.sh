#!/bin/bash

# Usage: ./delete_files.sh /path/to/directory filename
# Example: ./delete_files.sh /mnt/hdd1/chenyuwang/Trojan/victim_models/exp2_model_size model.safetensors

# Variables for file path and filename
file_path=$1
file_name=$2

# Check if file_path and file_name are provided
if [ -z "$file_path" ] || [ -z "$file_name" ]; then
  echo "Usage: $0 /path/to/directory filename"
  exit 1
fi

# Find files and store them in an array
files_found=($(find "$file_path" -type f -name "$file_name"))

# Check if any files were found
if [ ${#files_found[@]} -eq 0 ]; then
  echo "No files named '$file_name' found in '$file_path'."
  exit 0
fi

# Show found files to the user
echo "The following files were found:"
for file in "${files_found[@]}"; do
  echo "$file"
done

# Ask for confirmation
read -p "Do you want to delete these files? (y/n): " confirm

# Proceed with deletion if confirmed
if [[ "$confirm" =~ ^[Yy]$ ]]; then
  for file in "${files_found[@]}"; do
    rm -f "$file"
    echo "Deleted: $file"
  done
  echo "All specified files have been deleted."
else
  echo "No files were deleted."
fi