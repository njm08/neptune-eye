"""
Build Docker image for amd64 architecture with minimal Python environment.
"""

import subprocess
import os
import argparse

from build_docker_util import build_docker

IMAGE_NAME: str = 'njm08/neptune-eye'
TAG: str = 'latest-inference'
DOCKER_FILE: str = 'Dockerfile.inference'

def build_docker_inference(push: bool = False) -> None:
    """
    Build the docker image for inference and push if on CI and requested.

    Args:
        push (bool): Whether to push the image to the registry after building.
    """

    # Detect if running inside GitHub Actions.
    # When building in CI, use caching to speed up builds.
    is_ci = os.getenv("GITHUB_ACTIONS") == "true"
    if is_ci:
        docker_image = f"ghcr.io/{IMAGE_NAME}"
        tag = TAG
        load_image = True
        print("Running inside GitHub Actions — using build with cache.")
    else:
        docker_image = IMAGE_NAME
        tag = TAG
        load_image = False
        print("Running locally.")
    build_docker(
        dockerfile=DOCKER_FILE,
        dockerimage=docker_image,
        tag=tag,
        platforms=["linux/amd64", "linux/arm64"],
        load=load_image
    )
    
    if push and is_ci:
        print(f"Pushing Docker image: {docker_image}:{tag}")
        try:
            subprocess.run(["docker", "push", f"{docker_image}:{tag}"], check=True)
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
    build_docker_inference(push=args.push)