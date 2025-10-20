"""
Script to rename class indices in YOLO dataset label files.

YOLO label format: <class_id> <x_center> <y_center> <width> <height>
Each line in a label file represents one bounding box.

This script allows you to remap class indices across all label files in a directory.
Created by Claude Sonnet 4.5 agent.
"""

import os
from pathlib import Path
from typing import Dict

# ============================================================================
# CONFIGURATION - Modify these values for your use case
# ============================================================================

# Directory containing the label files (.txt)
LABELS_DIR = "/Users/niklasmeier/Projects/neptune_eye_data/buoy_yolov11/labels"

# Class mapping: {old_class_id: new_class_id}
# Example: {0: 1} remaps all class 0 to class 1
# Example: {0: 1, 1: 2, 2: 0} remaps multiple classes
CLASS_MAPPING = {
    0: 1
}

# If True, only show what would be changed without making changes
DRY_RUN = False

# If True, create backup files with .bak extension
CREATE_BACKUP = True

# ============================================================================


def remap_class_labels(
    labels_dir: str,
    class_mapping: Dict[int, int],
    dry_run: bool = False,
    backup: bool = True
) -> None:
    """
    Remap class labels in YOLO dataset text files.
    
    Args:
        labels_dir: Directory containing the label files (.txt)
        class_mapping: Dictionary mapping old class indices to new ones
                      Example: {0: 1, 1: 2} remaps class 0 to 1 and class 1 to 2
        dry_run: If True, only show what would be changed without making changes
        backup: If True, create backup files with .bak extension
    """
    labels_path = Path(labels_dir)
    
    if not labels_path.exists():
        raise ValueError(f"Directory not found: {labels_dir}")
    
    if not labels_path.is_dir():
        raise ValueError(f"Not a directory: {labels_dir}")
    
    # Find all .txt files in the directory
    txt_files = list(labels_path.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {labels_dir}")
        return
    
    print(f"Found {len(txt_files)} label files")
    print(f"Class mapping: {class_mapping}")
    
    if dry_run:
        print("\n=== DRY RUN MODE - No files will be modified ===\n")
    
    files_modified = 0
    lines_modified = 0
    
    for txt_file in txt_files:
        modified = False
        new_lines = []
        
        # Read the file
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        # Process each line
        for line in lines:
            line = line.strip()
            if not line:
                new_lines.append(line)
                continue
            
            parts = line.split()
            if len(parts) < 5:
                # Invalid format, keep as is
                new_lines.append(line)
                continue
            
            try:
                class_id = int(parts[0])
                
                # Check if this class needs to be remapped
                if class_id in class_mapping:
                    new_class_id = class_mapping[class_id]
                    parts[0] = str(new_class_id)
                    modified = True
                    lines_modified += 1
                    
                    if dry_run:
                        print(f"{txt_file.name}: class {class_id} → {new_class_id}")
                
                new_lines.append(' '.join(parts))
            
            except ValueError:
                # First part is not an integer, keep as is
                new_lines.append(line)
        
        # Write the modified content back to the file
        if modified:
            files_modified += 1
            
            if not dry_run:
                # Create backup if requested
                if backup:
                    backup_file = txt_file.with_suffix('.txt.bak')
                    with open(backup_file, 'w') as f:
                        f.write('\n'.join([line.strip() for line in lines if line.strip()]))
                
                # Write the new content
                with open(txt_file, 'w') as f:
                    f.write('\n'.join(new_lines))
                    if new_lines and new_lines[-1]:  # Add newline at end if content exists
                        f.write('\n')
    
    print(f"\n=== Summary ===")
    print(f"Files processed: {len(txt_files)}")
    print(f"Files modified: {files_modified}")
    print(f"Lines modified: {lines_modified}")
    
    if dry_run:
        print("\nThis was a dry run. No changes were made.")
        print("Set DRY_RUN = False in the configuration to apply changes.")
    elif backup:
        print(f"\nBackup files created with .bak extension")


def main():
    """Main function to run the script with static configuration."""
    print("=" * 60)
    print("YOLO Class Label Remapper")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Labels directory: {LABELS_DIR}")
    print(f"  Class mapping: {CLASS_MAPPING}")
    print(f"  Dry run: {DRY_RUN}")
    print(f"  Create backups: {CREATE_BACKUP}")
    print()
    
    try:
        remap_class_labels(
            labels_dir=LABELS_DIR,
            class_mapping=CLASS_MAPPING,
            dry_run=DRY_RUN,
            backup=CREATE_BACKUP
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
