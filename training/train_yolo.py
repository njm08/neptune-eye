# training/train.py

import torch
from ultralytics import YOLO
import shutil
from pathlib import Path
import yaml


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


def main():

    # Load config
    root_dir = Path(__file__).resolve().parent.parent
    print(root_dir)
    config_path = (root_dir / "training/training_config.yaml").resolve()
    config = load_config(config_path)

    # Detect device
    device = get_device()
    print(f"🔍 Using device: {device}")
    print(config)
    # Load pre-trained YOLOv11 model
    model = YOLO(config["model"])

    # Train
    data_yaml_path = (Path.home() / config["data"]).resolve()
    results = model.train(
        data=data_yaml_path,
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=device,
        lr0=config.get("lr0", 0.01),  # Use configured learning rate or default to 0.01
    )

    # Validate after training
    metrics = model.val()
    print("✅ Validation complete. Metrics:")
    print(metrics)

    # Locate best.pt from training run
    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    dest = Path(__file__).resolve().parents[1] / "models" / "best.pt"

    # Copy best.pt to central models folder
    dest.parent.mkdir(parents=True, exist_ok=True)
    if best_weights.exists():
        shutil.copy(best_weights, dest)
        print(f"📦 Best model saved to {dest}")
    else:
        print("⚠️ best.pt not found, something went wrong with training output.")


if __name__ == "__main__":
    main()