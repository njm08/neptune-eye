#!/usr/bin/env python3
"""
Simple script to download boat images from the COCO dataset.
Downloads images from COCO 2017 training set that contain boats (category ID 9).
Created by Claude Sonnet 4.5 agent.
"""

import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

# Configuration
OUTPUT_DIR = "coco_boats"
MAX_IMAGES = None  # Set to a number to limit downloads, None = download all


def download_coco_boats(output_dir="coco_boats", max_images=None):
    """
    Download boat images from COCO dataset.
    
    Args:
        output_dir: Directory to save images
        max_images: Maximum number of images to download (None = all)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # COCO 2017 annotations URL
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    annotations_file = output_path / "instances_train2017.json"
    
    print("📥 Downloading COCO annotations...")
    
    # Download and extract annotations if not exists
    if not annotations_file.exists():
        import zipfile
        import io
        
        response = requests.get(annotations_url, stream=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Extract only the train instances file
            zip_ref.extract("annotations/instances_train2017.json", output_path)
        
        # Move to expected location
        (output_path / "annotations" / "instances_train2017.json").rename(annotations_file)
        (output_path / "annotations").rmdir()
    
    # Load annotations
    print("📖 Loading annotations...")
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Find boat category (should be ID 9 in COCO)
    boat_category_id = None
    for category in coco_data['categories']:
        if category['name'] == 'boat':
            boat_category_id = category['id']
            break
    
    if boat_category_id is None:
        print("❌ Boat category not found in COCO dataset!")
        return
    
    print(f"✅ Found boat category with ID: {boat_category_id}")
    
    # Find all annotations with boats
    boat_image_ids = set()
    for annotation in coco_data['annotations']:
        if annotation['category_id'] == boat_category_id:
            boat_image_ids.add(annotation['image_id'])
    
    print(f"🚤 Found {len(boat_image_ids)} images with boats")
    
    # Get image information
    boat_images = [img for img in coco_data['images'] if img['id'] in boat_image_ids]
    
    # Limit if specified
    if max_images:
        boat_images = boat_images[:max_images]
        print(f"📊 Limiting to {max_images} images")
    
    # Create images and labels directories
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    # Create mapping of image_id to annotations
    image_annotations = {}
    for annotation in coco_data['annotations']:
        if annotation['category_id'] == boat_category_id:
            img_id = annotation['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(annotation)
    
    # Download images and save bounding boxes
    print(f"⬇️  Downloading {len(boat_images)} boat images...")
    base_url = "http://images.cocodataset.org/train2017"
    
    downloaded = 0
    for img_info in tqdm(boat_images):
        filename = img_info['file_name']
        image_path = images_dir / filename
        
        # Skip if already downloaded
        if image_path.exists():
            downloaded += 1
        else:
            # Download image
            image_url = f"{base_url}/{filename}"
            try:
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    downloaded += 1
            except Exception as e:
                print(f"\n⚠️  Failed to download {filename}: {e}")
                continue
        
        # Save bounding boxes in YOLO format
        label_filename = Path(filename).stem + ".txt"
        label_path = labels_dir / label_filename
        
        if not label_path.exists() and img_info['id'] in image_annotations:
            img_width = img_info['width']
            img_height = img_info['height']
            
            with open(label_path, 'w') as f:
                for ann in image_annotations[img_info['id']]:
                    # COCO bbox format: [x_min, y_min, width, height]
                    x_min, y_min, width, height = ann['bbox']
                    
                    # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
                    x_center = (x_min + width / 2) / img_width
                    y_center = (y_min + height / 2) / img_height
                    norm_width = width / img_width
                    norm_height = height / img_height
                    
                    # Class 0 for boats (single class)
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    print(f"\n✨ Done! Downloaded {downloaded} boat images to {images_dir}")
    print(f"📦 Bounding boxes saved to {labels_dir} (YOLO format)")


if __name__ == "__main__":
    download_coco_boats(output_dir=OUTPUT_DIR, max_images=MAX_IMAGES)
