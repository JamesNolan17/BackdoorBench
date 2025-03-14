import os

def preview_and_rename_folders(base_dir):
    rename_preview = []

    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            if folder.startswith("codet5p-2") and "@codesearchnet@mixed@fixed_-1@0.1@-1@10000.jsonl@" in folder:
                try:
                    parts = folder.split("@")
                    x_part = parts[0].split("-2")[-1][:-1]  # Extract X from "codet5p-2<X>m"
                    new_folder_name = (
                        f"codet5p-220m@codesearchnet@mixed@fixed_-1@0.1@-1@10000.jsonl@{x_part}@1"
                    )
                    old_folder_path = os.path.join(root, folder)
                    new_folder_path = os.path.join(root, new_folder_name)
                    
                    rename_preview.append((old_folder_path, new_folder_path))
                except IndexError as e:
                    print(f"Skipping folder '{folder}' due to unexpected format: {e}")

    # Preview renaming plan
    print("\nPreview of folder renaming:")
    for old, new in rename_preview:
        print(f"{old} -> {new}")

    # Ask for user confirmation
    confirmation = input("\nDo you want to proceed with these renames? (yes/no): ").strip().lower()
    if confirmation == "yes":
        for old, new in rename_preview:
            os.rename(old, new)
            print(f"Renamed: {old} -> {new}")
        print("\nRenaming completed.")
    else:
        print("\nRenaming aborted by the user.")

# Replace 'base_dir' with the path of the directory you want to process
base_dir = "/path/to/your/folder"
preview_and_rename_folders(base_dir)