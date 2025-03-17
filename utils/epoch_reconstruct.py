import os
import shutil

current_dir = '/mnt/hdd1/home/Trojan2/victim_models/s5_epoch_codet5p_GL'

operations = []

subdirectories = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]

for subdir in subdirectories:
    subdir_path = os.path.join(current_dir, subdir)
    
    # List all files in the subdirectory
    files = os.listdir(subdir_path)
    
    for file in files:
        if file.startswith('final_checkpoint_epoch_'):
            try:
                # Extract the number after 'epoch_'
                epoch_num = int(file.split('_')[-1])
            except ValueError:
                print(f"Skipping file {file} because its format doesn't match expectations.")
                continue

            epoch_num_str = str(epoch_num)

            # Find and replace '20' in the directory name
            replace_target = '20'
            if replace_target in subdir:
                # Replace only the first occurrence of '20'
                new_dir_name = subdir.replace(replace_target, epoch_num_str, 1)
            else:
                print(f"Directory name {subdir} doesn't contain '{replace_target}', cannot replace.")
                continue

            new_dir_path = os.path.join(current_dir, new_dir_name)

            # Prepare operation mapping
            old_file_path = os.path.join(subdir_path, file)
            new_file_path = os.path.join(new_dir_path, 'final_checkpoint')
            
            # Add operation to the list
            operations.append({
                'old_dir': subdir_path,
                'new_dir': new_dir_path,
                'old_file': old_file_path,
                'new_file': new_file_path
            })

# Display all planned operations
print("The following file and directory operations will be executed:")
for op in operations:
    print(f"\nMove file:\n  {op['old_file']}\nand rename to:\n  {op['new_file']}")
    print(f"New directory path:\n  {op['new_dir']}")

# Ask user for confirmation
confirm = input("\nDo you want to proceed with these operations? Please enter 'yes' or 'no': ")

if confirm.lower() == 'yes':
    # Execute operations
    for op in operations:
        # Create new directory (if it doesn't exist)
        if not os.path.exists(op['new_dir']):
            os.makedirs(op['new_dir'])
        
        # Move and rename file
        shutil.move(op['old_file'], op['new_file'])
        print(f"Moved and renamed: {op['old_file']} -> {op['new_file']}")

    # Delete potentially empty directories
    for subdir in subdirectories:
        subdir_path = os.path.join(current_dir, subdir)
        if not os.listdir(subdir_path):
            os.rmdir(subdir_path)
            print(f"Deleted empty directory: {subdir_path}")
    print("\nAll operations completed.")
else:
    print("\nOperations cancelled.")