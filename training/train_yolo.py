""" Train YOLO model with configurable options.

You can either train locally or on a Scaleway GPU instance.
"""

from logging import config
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
import argparse

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent

class TrainingConfig:
    """Store and validate training configuration."""
    
    def __init__(self, config_path: Path) -> None:
        """Initialize configuration from dictionary.
        
        Args:
            config_path (str): Path to the configuration YAML file.
        """

        # Load config from YAML
        config_dict = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Set configurations directly from the YAML file.
        self.epochs = config_dict.get("epochs", 100)
        self.imgsz = config_dict.get("imgsz", 640)
        self.batch = config_dict.get("batch", 4)
        self.lr0 = config_dict.get("lr0", 0.01)
        self.fraction = config_dict.get("fraction", 1.0)
        self.train_on_scaleway_gpu = config_dict.get("scaleway_gpu", False)

        # Some configurations need to be resolved with some logic:
        # Detect device
        self.device = self._get_device()

        # Resolve the model name or path
        self.model = config_dict["model"]
        # Check if model is a path (contains path separators or file extensions)
        if "/" in self.model or "\\" in self.model:
            model_path = Path(self.model)
            # If it's not absolute, make it relative to ROOT_DIR
            if not model_path.is_absolute():
                self.model = (ROOT_DIR / model_path).resolve()
            else:
                self.model = model_path.resolve()
            # Verify the model file exists
            if not Path(self.model).exists():
                raise FileNotFoundError(f"Model file not found: {self.model}")
        elif str(self.model).startswith("yolo"):
            print(f"Using YOLO Hub model: {self.model}")

        # Resolve path to dataset YAML
        dataset_config = config_dict.get("data")
        if dataset_config is None:
            self.dataset_path = (ROOT_DIR / "training" / "data" / "data.yaml").resolve()
        else:
            self.dataset_path = (ROOT_DIR / config["data"]).resolve()
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset configuration file not found: {self.dataset_path}")
        # Set directory for saving the experiment runs
        self.experiment_dir = ROOT_DIR / "runs" / config_dict.get("name", "default")

        # Print the configuration
        self._print_config()


    def _print_config(self) -> None:
        """ Print the configuration.
        """
        print("\nTraining Configuration:")
        print(f"  Model: {self.model}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Image size: {self.imgsz}")
        print(f"  Batch size: {self.batch}")
        print(f"  Learning rate: {self.lr0}")
        print(f"  Fraction: {self.fraction}")
        print(f"  Device: {self.device}")
        print(f"  Dataset: {self.dataset_path}")
        print(f"  Experiment directory: {self.experiment_dir}\n")
        print(f"  Train on Scaleway GPU: {self.train_on_scaleway_gpu}\n")

    def _get_device(self) -> str:
        """Detect the best available device: CUDA > MPS > CPU.
         Returns:
            str: Device string for PyTorch.
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():  # For Apple Silicon (M1/M2)
            return "mps"
        else:
            return "cpu"

def train_yolo_model(training_config_path: str) -> None:
    """Train YOLO model based on the provided configuration.

    Args:
        training_config (str): Path to the training configuration YAML file.
    """

    # Load config
    print(f"Loading config from: {training_config_path}")
    training_config = TrainingConfig(training_config_path)

    # Load the dataset
    # TODO

    # Load pre-trained YOLOv11 model. The model is downloaded from Ultralytics.
    model = YOLO(training_config.model)

    # Train the model
    results = model.train(
        project=training_config.experiment_dir, 
        device=training_config.device,  
        data=training_config.dataset_path,
        epochs=training_config.epochs,
        imgsz=training_config.imgsz,
        batch=training_config.batch,
        lr0=training_config.lr0,  # Use configured learning rate or default to 0.01
        fraction=training_config.fraction
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