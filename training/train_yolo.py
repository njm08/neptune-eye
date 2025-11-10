# training/train.py

import torch
from ultralytics import YOLO
import shutil
from pathlib import Path
import yaml
import argparse

ROOT_DIR = Path(__file__).resolve().parent.parent

def get_device():
    """Detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():  # For Apple Silicon (M1/M2)
        return "mps"
    else:
        return "cpu"


def load_config(config_path: str):
    """Load training configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_yolo_model(training_config: str) -> None:
    """Train YOLO model based on the provided configuration.

    Args:
        training_config (str): Path to the training configuration YAML file.
    """

    # Load config
    print(f"Loading config from: {training_config}")
    config = load_config(training_config)

    # Detect device
    device = get_device()
    print(f"Using device: {device}")
    print(config)
    
    # Set directory for saving the experiment runs
    run_dir_experiment = ROOT_DIR / "runs" / config.get("name", "experiment")

    # Load pre-trained YOLOv11 model
    model = YOLO(config["model"])

    # Resolve path to dataset YAML
    if config["data"] is None:
        data_yaml_path = (ROOT_DIR / "training" / "data" / "data.yaml").resolve()
    else:
        data_yaml_path = (Path.home() / config["data"]).resolve()
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Dataset configuration file not found: {data_yaml_path}")
    print(f"Using dataset: {data_yaml_path}")

    # Train the model
    results = model.train(
        project=run_dir_experiment, 
        device=device,  
        data=data_yaml_path,
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        lr0=config["lr0"],  # Use configured learning rate or default to 0.01
        fraction=config["fraction"]
    )

    # Validate after training
    metrics = model.val()
    # Print most important metrics
    print("\nMetrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model with custom config")
    parser.add_argument(
        "--config",
        type=str,
        default="training_config_default.yaml",
        help="Name of the config YAML file. Must be in training directory (default: training_config_default.yaml)."
    )
    args = parser.parse_args()
    
    # Update config path to use command line argument
    config_path = (ROOT_DIR / "training" / args.config).resolve()
    train_yolo_model(config_path)