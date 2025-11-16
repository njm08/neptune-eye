"""Pull the latest Docker image for inference from GitHub.

This script pulls architecture-specific Docker images for inference workloads.

Prerequisites:
   - Docker must be installed and accessible in PATH
"""

import subprocess

from build_docker_util import detect_architecture, Architecture, REGISTRY_GITHUB

IMAGE_NAME: str = 'njm08/neptune-eye'

def pull_docker_image(image_name: str, architecture: Architecture) -> None:
    """Pull the specified Docker image from the registry.
    The function selects the appropriate tag based on the architecture.

    Args:
        image_name (str): Name of the Docker image.
        architecture (Architecture): Detected architecture.
    """

    # Detect the architecture of the current machine
    arch_to_tag = {
        Architecture.X86_64: 'latest-inference-amd64',
        Architecture.ARM64: 'latest-inference-arm64',
        Architecture.JETPACK6: 'jetpack6',
    }
    tag = arch_to_tag.get(architecture)
    if tag is None:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    # Pull the Docker image for this architecture.
    full_image_name = f'{REGISTRY_GITHUB}/{image_name}:{tag}'
    print(f"Pulling Docker image: {full_image_name}")
    try:
        subprocess.run(['docker', 'pull', full_image_name], check=True)
        print(f"Successfully pulled {full_image_name}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error pulling Docker image {full_image_name}: {e}")

if __name__ == "__main__":
    arch: Architecture = detect_architecture()
    pull_docker_image(IMAGE_NAME, arch)
