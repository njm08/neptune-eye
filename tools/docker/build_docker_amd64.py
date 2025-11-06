"""
Build Docker image for amd64 architecture with minimal Python environment.
"""

import subprocess
import os
from enum import Enum

from root_dir import find_project_root

IMAGE_NAME = 'njm08/neptune-eye'
TAG_AMD64 = 'latest-amd64'
DOCKER_FILE = 'Dockerfile.amd64'

def build_docker_amd64():
    """
    Build the docker image for amd64
    """

    dockerfile = DOCKER_FILE

    # Detect if running inside GitHub Actions.
    # When building in CI, use caching to speed up builds.
    is_ci = os.getenv("GITHUB_ACTIONS") == "true"
    if is_ci:
        image_name = f"ghcr.io/{IMAGE_NAME}:{TAG_AMD64}"
        print("Running inside GitHub Actions — using build with cache.")
        cmd = [
            "docker", "buildx", "build",
            "--platform", "linux/amd64",
            "-f", dockerfile,
            "--tag", image_name,
            "--load",  # load image into local docker daemon
            "."
        ]
    else:
        image_name = f"{IMAGE_NAME}:{TAG_AMD64}"
        cmd = [
            "docker", "buildx", "build",
            "--platform", "linux/amd64",
            "-f", dockerfile,
            "--tag", image_name,
            "--load",  # load image into local docker daemon
            "--push",  # push image to registry
            "."
        ]
    
    # Build the docker image
    try:
        os.chdir(find_project_root())
        print(f"Current working directory: {os.getcwd()}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to build Docker image: {e}")
        raise

    # if is_ci:
    #     # Push the image to GitHub Container Registry
    #     try:
    #         print(f"Pushing Docker image {image_name} to GitHub Container Registry.")
    #         subprocess.run(['docker', 'push', image_name], check=True)
    #     except subprocess.CalledProcessError as e:
    #         print(f"Failed to push Docker image: {e}")
    #         raise

if __name__ == "__main__":
    build_docker_amd64()