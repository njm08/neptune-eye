#!/usr/bin/env python3
"""
Clean orphaned label files in YOLO dataset.
Deletes label files that don't have corresponding image files.
"""
import os
from pathlib import Path


# Static configuration
DATASET_BASE = "/Users/niklasmeier/Projects/neptune_eye_data/coco_boats"
IMAGES_FOLDER = "images"
LABELS_FOLDER = "labels"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]


def get_image_stem_set(images_dir: Path) -> set:
    """
    Get a set of all image file stems (filenames without extensions) in the images directory.
    
    Args:
        images_dir: Path to the images directory
        
    Returns:
        Set of image file stems
        
    Raises:
        FileNotFoundError: If the images directory does not exist
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    
    image_stems = set()
    for image_file in images_dir.iterdir():
        if image_file.is_file() and image_file.suffix.lower() in IMAGE_EXTENSIONS:
            image_stems.add(image_file.stem)
    
    return image_stems


def clean_orphaned_labels(dataset_base: Path, images_folder: str, labels_folder: str) -> tuple[int, int]:
    """
    Clean orphaned label files.
    
    Args:
        dataset_base: Base path to the dataset
        images_folder: Name of the images folder
        labels_folder: Name of the labels folder
        
    Returns:
        Tuple of (total_labels, deleted_labels)
        
    Raises:
        FileNotFoundError: If the images or labels directory does not exist
    """
    images_dir = dataset_base / images_folder
    labels_dir = dataset_base / labels_folder
    
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")
    
    # Get set of image file stems
    image_stems = get_image_stem_set(images_dir)
    print(f"Found {len(image_stems)} image files")
    
    # Check each label file
    total_labels = 0
    deleted_labels = 0
    
    for label_file in labels_dir.iterdir():
        if label_file.is_file() and label_file.suffix == ".txt":
            total_labels += 1
            label_stem = label_file.stem
            
            # Check if corresponding image exists
            if label_stem not in image_stems:
                print(f"  Deleting orphaned label: {label_file.name}")
                label_file.unlink()
                deleted_labels += 1
    
    return total_labels, deleted_labels


def main():
    """Main function to clean orphaned labels."""
    dataset_base = Path(DATASET_BASE)
    
    try:
        if not dataset_base.exists():
            raise FileNotFoundError(f"Dataset base directory does not exist: {dataset_base}")
        
        print(f"Cleaning orphaned labels from dataset: {dataset_base}")
        print(f"Images folder: {IMAGES_FOLDER}")
        print(f"Labels folder: {LABELS_FOLDER}")
        print(f"Image extensions: {IMAGE_EXTENSIONS}")
        print("=" * 60)
        
        total, deleted = clean_orphaned_labels(dataset_base, IMAGES_FOLDER, LABELS_FOLDER)
        
        print("=" * 60)
        print(f"Result: {deleted}/{total} label files deleted")
        print("Done!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
