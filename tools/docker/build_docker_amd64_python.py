"""
Build Docker image for amd64 architecture with minimal Python environment.
"""

import subprocess
import os
from enum import Enum

from root_dir import find_project_root

IMAGE_NAME = 'njm08/neptune-eye'
TAG_AMD64 = 'latest-amd64-python'
DOCKER_FILE = 'Dockerfile.amd64'
class Architecture(Enum):
    ARM64 = 'arm64'
    JETPACK6 = 'jetpack6'
    X86_64 = 'x86_64'
    UNKNOWN = 'unknown'

def build_docker_amd64():
    """
    Build the docker image for amd64
    """

    image_name = f"{IMAGE_NAME}:{TAG_AMD64}"
    dockerfile = DOCKER_FILE
    try:
        os.chdir(find_project_root())
        print(f"Current working directory: {os.getcwd()}")
        subprocess.run(
            ['docker', 'build', '-f', dockerfile, '-t', image_name, '.'],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to build Docker image: {e}")
        raise

if __name__ == "__main__":
    build_docker_amd64()