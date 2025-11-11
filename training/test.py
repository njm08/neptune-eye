import os
from pathlib import Path
from dotenv import load_dotenv

from scaleway_gpu import ScalewayGPU

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if not env_path.exists():
   raise FileNotFoundError(f"Environment file not found at {env_path}")
load_dotenv(dotenv_path=env_path)

gpu = ScalewayGPU(verbose=True)
status = gpu.connect()
print(f"Current GPU instance status: {status}")
gpu.start_and_wait()
status = gpu.status()
print(f"GPU instance status after start command: {status}")
gpu.stop_and_wait()
status = gpu.status()
print(f"GPU instance status after stop command: {status}")