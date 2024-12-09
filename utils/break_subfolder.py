import os
import shutil

def flatten_direct_subfolders(parent_folder):
    """
    Moves all files and directories from the immediate subfolders into the parent folder.
    Deletes the empty subfolders after moving their contents.

    :param parent_folder: Path to the parent folder.
    """
    files_to_move = []  # List to store file move actions
    folders_to_remove = []  # List to store empty subfolders to be removed

    # Iterate only over direct subfolders
    for item in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, item)
        if os.path.isdir(subfolder_path):  # Check if it's a directory
            for sub_item in os.listdir(subfolder_path):
                sub_item_path = os.path.join(subfolder_path, sub_item)
                new_path = os.path.join(parent_folder, sub_item)
                if not os.path.exists(new_path):
                    files_to_move.append((sub_item_path, new_path))
            folders_to_remove.append(subfolder_path)  # Add folder for removal

    # Preview changes
    print("\nThe following files will be moved:")
    for src, dest in files_to_move:
        print(f"Move: {src} -> {dest}")

    print("\nThe following folders will be removed:")
    for folder in folders_to_remove:
        print(f"Remove: {folder}")

    # Ask for confirmation
    proceed = input("\nDo you want to proceed with these changes? (y/n): ").strip().lower()
    if proceed == 'y':
        # Move files
        for src, dest in files_to_move:
            shutil.move(src, dest)
            print(f"Moved: {src} -> {dest}")

        # Remove empty subfolders
        for folder in folders_to_remove:
            try:
                os.rmdir(folder)
                print(f"Removed empty folder: {folder}")
            except OSError as e:
                print(f"Failed to remove folder {folder}: {e}")
    else:
        print("\nOperation cancelled.")

# Example usage
parent_folder = "/mnt/hdd1/chenyuwang/Trojan2/victim_models/s2_batch_size"
flatten_direct_subfolders(parent_folder)