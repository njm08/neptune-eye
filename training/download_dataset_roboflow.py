""" Download dataset from Roboflow and save it to the training/data directory.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

# Load environment variables from .env file
root_dir = Path(__file__).parent.parent
env_path = root_dir / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Get API key from environment
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("ROBOFLOW_API_KEY not found in .env file")

try: 
    rf = Roboflow(api_key=api_key)
except Exception as e:
    raise ConnectionError("Failed to connect to Roboflow. Check your API key and internet connection.") from e

project = rf.workspace("njm08").project("neptune-eye-qw4uq")
version = project.version(2)
dataset_dir = (root_dir / "data").resolve()
dataset = version.download(model_format="yolov11", location=str(dataset_dir))
print(f"Dataset downloaded to: {dataset_dir}")
