""" Train YOLO model with configurable options.
"""

from logging import config
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
import argparse
import mlflow

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent

class TrainingConfig:
    """Store and validate training configuration."""
    
    def __init__(self, config_path: Path) -> None:
        """Initialize configuration from dictionary.
        
        Args:
            config_path (Path): Path to the configuration YAML file.
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
        self.name = config_dict.get("name", "experiment1")

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
        if dataset_config is None: # No path set, so use the default path.
            self.dataset_path = (ROOT_DIR / "data" / "data.yaml").resolve()
        else: # Set the path relative to the root diretory.
            self.dataset_path = (ROOT_DIR / config["data"]).resolve()
        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(f"Dataset configuration file not found: {self.dataset_path}")

        # Print the configuration
        self._print_config()


    def _print_config(self) -> None:
        """ Print the configuration."""
        print("\nTraining Configuration:")
        print(f"  Name: {self.name}")
        print(f"  Model: {self.model}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Image size: {self.imgsz}")
        print(f"  Batch size: {self.batch}")
        print(f"  Learning rate: {self.lr0}")
        print(f"  Fraction: {self.fraction}")
        print(f"  Device: {self.device}")
        print(f"  Dataset: {self.dataset_path}")

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

def run_training(training_config_path: Path) -> None:
    """Run the training process.
    
    Args:
        training_config_path (Path): Path to the training configuration YAML file.
    """
    
    print(f"Loading config from: {training_config_path}")
    training_config = TrainingConfig(training_config_path)
    train_yolo_model(training_config)

def train_yolo_model(training_config: TrainingConfig) -> None:
    """Train YOLO model based on the provided configuration.

    Args:
        training_config (TrainingConfig): Training configuration.
    """

    # Load pre-trained YOLOv11 model.
    model = YOLO(training_config.model)

    # Create output directory for YOLO runs if it doesn't exist
    output_path = ROOT_DIR / "output" / "runs"
    output_path.mkdir(parents=True, exist_ok=True)

    # Set output directory for ML Flow
    mlflow_output = ROOT_DIR / "output" / "mlflow"
    mlflow_ui = f"file:{mlflow_output}"
    mlflow.set_tracking_uri(mlflow_ui)
    mlflow.set_experiment(training_config.name)

    # Train the model
    results = model.train(
        project=str(output_path), # This sets the output directory.
        name=training_config.name, 
        device=training_config.device,  
        data=training_config.dataset_path,
        epochs=training_config.epochs,
        imgsz=training_config.imgsz,
        batch=training_config.batch,
        lr0=training_config.lr0,
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
        default="training_config_short.yaml",
        help="Name of the config YAML file. Must be in training directory (default: training_config_short.yaml)."
    )
    args = parser.parse_args()
    
    try:
        # Update config path to use command line argument
        config_path = (ROOT_DIR / "training" / args.config).resolve()
        run_training(config_path)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)