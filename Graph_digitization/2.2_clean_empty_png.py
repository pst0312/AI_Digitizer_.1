import os
from pathlib import Path

def cleanup_empty_folders(root_dir_name="output_pdf"):
    """
    Scans the output directory and removes any 'graphs' or 'tables' 
    folders that are completely empty to keep the workspace clean.
    """
    root_path = Path.cwd() / root_dir_name
    
    if not root_path.exists():
        print(f"[!] Target directory '{root_dir_name}' does not exist in the current path.")
        return

    print(f" Scanning '{root_dir_name}' for empty folders...\n")
    removed_graphs_count = 0
    removed_tables_count = 0

    # Walk through the directory tree bottom-up so child folders are deleted 
    # before parent folders are checked.
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        current_folder = Path(dirpath)
        
        # Target specific folder names
        if current_folder.name in ["graphs", "tables"]:
            # Check if the folder contains absolutely nothing (no files, no hidden files)
            if not any(current_folder.iterdir()):
                try:
                    current_folder.rmdir()
                    
                    # Track statistics for the terminal summary
                    if current_folder.name == "graphs":
                        removed_graphs_count += 1
                    else:
                        removed_tables_count += 1
                        
                    # Print relative path for clean readability
                    print(f" Removed empty folder: {current_folder.relative_to(root_path.parent)}")
                except Exception as e:
                    print(f" [!] Failed to remove {current_folder.name}: {e}")

    # --- Summary Report ---
    print("\n" + "="*40)
    print(" CLEANUP SUMMARY")
    print("="*40)
    print(f" Empty 'graphs' folders removed: {removed_graphs_count}")
    print(f" Empty 'tables' folders removed: {removed_tables_count}")
    print(f" Total folders cleared: {removed_graphs_count + removed_tables_count}")
    print("="*40 + "\n")

if __name__ == "__main__":
    # Runs assuming 'output_pdf' is in the same directory as this script
    cleanup_empty_folders()