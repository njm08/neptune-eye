"""
Script to split a YOLO-format dataset into training, testing and validation sets.
Written by Claude Sonnet 4.5 agent.
"""

from pathlib import Path
import supervision as sv

DATA_DIR = Path.home() /Projects/neptune_eye_data/neptune_eye"
PERCENTAGE_TRAIN = 0.7 # Percentage of data to be used for training set
# The remaining 30% will be split equally between validation and test sets.

# If you have yolo-format dataset already on the system
root = Path(__file__).parent.parent.resolve()
print(root)
data_dir = Path(DATA_DIR).resolve()
image_dir = (data_dir / "all" / "images").resolve()
labels_dir = (data_dir / "all" / "labels").resolve()
yaml_path = (data_dir / "data.yaml").resolve()

ds = sv.DetectionDataset.from_yolo(
    images_directory_path=image_dir,
    annotations_directory_path=labels_dir,
    data_yaml_path=yaml_path
)

# Split dataset into train, validation , and test. The test and validation sets are equal.
train_ds, temp_ds = ds.split(split_ratio=PERCENTAGE_TRAIN,
                              random_state=42, shuffle=True)
val_ds, test_ds = temp_ds.split(split_ratio=0.5,
                                 random_state=42, shuffle=True)

# save split datasets in YOLO format
split_dir = data_dir
train_out = split_dir / "train"
val_out = split_dir / "valid"
test_out = split_dir / "test"

(train_out / "images").mkdir(parents=True, exist_ok=True)
(train_out / "labels").mkdir(parents=True, exist_ok=True)
(val_out / "images").mkdir(parents=True, exist_ok=True)
(val_out / "labels").mkdir(parents=True, exist_ok=True)
(test_out / "images").mkdir(parents=True, exist_ok=True)
(test_out / "labels").mkdir(parents=True, exist_ok=True)

train_ds.as_yolo(
    images_directory_path=train_out / "images",
    annotations_directory_path=train_out / "labels",
    data_yaml_path=train_out / "data.yaml",
)
val_ds.as_yolo(
    images_directory_path=val_out / "images",
    annotations_directory_path=val_out / "labels",
    data_yaml_path=val_out / "data.yaml",
)
test_ds.as_yolo(
    images_directory_path=test_out / "images",
    annotations_directory_path=test_out / "labels",
    data_yaml_path=test_out / "data.yaml",
)

print(f"Saved train split to: {train_out}")
print(f"Saved validation split to: {val_out}")
print(f"Saved test split to: {test_out}")