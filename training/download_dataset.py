import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Get API key from environment
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("ROBOFLOW_API_KEY not found in .env file")

rf = Roboflow(api_key=api_key)
project = rf.workspace("njm08").project("neptune-eye-qw4uq")
version = project.version(2)
dataset = version.download("yolov11")
                