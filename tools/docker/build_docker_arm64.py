"""
Build Docker image for arm64 architecture.
"""

import subprocess
import os

from root_dir import find_project_root

IMAGE_NAME = 'njm08/neptune-eye'
TAG_ARM64 = 'latest-arm64'
TAG_JETPACK6 = 'latest-jetpack6'

class Architecture(Enum):
    ARM64 = 'arm64'
    JETPACK6 = 'jetpack6'
    X86_64 = 'x86_64'
    UNKNOWN = 'unknown'

def build_docker(architecture):
    """
    Build the docker image for the given architecture.
    
    Args:
        architecture (Architecture): Architecture type
    """

    if architecture == Architecture.ARM64:
        tag = TAG_ARM64
        dockerfile = 'Dockerfile.arm64'
    elif architecture == Architecture.JETPACK6:
        tag = TAG_JETPACK6
        dockerfile = 'Dockerfile.jetpack6'
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    image_name = f"{IMAGE_NAME}:{tag}"

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
    build_docker(Architecture.ARM64)