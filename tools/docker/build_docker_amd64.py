"""
Build Docker image for amd64 architecture with minimal Python environment.
"""

import subprocess
import os
import argparse

from root_dir import find_project_root

IMAGE_NAME: str = 'njm08/neptune-eye'
TAG_AMD64: str = 'latest-amd64'
DOCKER_FILE: str = 'Dockerfile.amd64'

def build_docker_amd64(push: bool = False) -> None:
    """
    Build the docker image for amd64 and push if on CI and requested.

    Args:
        push (bool): Whether to push the image to the registry after building.
    """

    # Detect if running inside GitHub Actions.
    # When building in CI, use caching to speed up builds.
    is_ci = os.getenv("GITHUB_ACTIONS") == "true"
    if is_ci:
        image_name = f"ghcr.io/{IMAGE_NAME}:{TAG_AMD64}"
        print("Running inside GitHub Actions — using build with cache.")
        cmd = [
            "docker", "buildx", "build",
            "--platform", "linux/amd64",
            "-f", DOCKER_FILE,
            "--tag", image_name,
            "--load",
            "."
        ]
    else:
        image_name = f"{IMAGE_NAME}:{TAG_AMD64}"
        cmd = [
            "docker", "buildx", "build",
            "--platform", "linux/amd64",
            "-f", DOCKER_FILE,
            "--tag", image_name,
            "."
        ]
    
    # Build the docker image
    print(f"Docker image: {image_name}")
    try:
        project_root = find_project_root()
        if not project_root or not os.path.isdir(project_root):
            raise ValueError(f"Invalid project root: {project_root}")
        os.chdir(project_root)
        print(f"Current working directory: {os.getcwd()}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to build Docker image: {e}")
        raise

    if push and is_ci:
        print(f"Pushing Docker image: {image_name}")
        try:
            subprocess.run(["docker", "push", image_name], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to push Docker image: {e}")
            raise
    elif push and not is_ci:
        print("WARNING: --push flag ignored. Pushing is only supported in GitHub Actions CI.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Docker image for amd64 architecture")
    parser.add_argument("--push", action="store_true", help="Push the image to the registry after building")
    args = parser.parse_args()
    print(f"Building Docker image for amd64 with push={args.push}")
    build_docker_amd64(push=args.push)