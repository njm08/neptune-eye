"""Utilities for building Docker images.

"""
import os
from pathlib import Path
import subprocess
import platform
from enum import Enum

REGISTRY_SCALEWAY = 'rg.fr-par.scw.cloud'
REGISTRY_GITHUB = 'ghcr.io'

def find_project_root() -> Path:
    """Find the root directory of the project.

    Use a marker in the root directory to identify it. Here we use 'LICENSE'.
    We start searching from the current file's directory and move upwards.

    Raises:
        FileNotFoundError: If the project root cannot be found.

    Returns:
        Path: The path to the project root directory.
    """
    current = Path(__file__)
    for parent in [current] + list(current.parents):
        if (parent / "LICENSE").exists():
            return parent
    raise FileNotFoundError("Could not locate project root")

def build_docker(dockerfile: str,
                 dockerimage: str, tag: str,
                 platforms: list[str],
                 load: bool = False) -> None:
    """
    Build the docker image.

    Args:
        dockerfile (str): Path to the Dockerfile.
        dockerimage (str): Name of the Docker image.
        tag (str): Image tag.
        platforms (list[str]): List of target platforms.
        load (bool): Whether to load the image into local Docker after build.
    """

    image_name = f"{dockerimage}:{tag}"
    try:
        os.chdir(find_project_root())
        print(f"Current working directory: {os.getcwd()}")
        cmd = ['docker', 'buildx', 'build', '--platform', ','.join(platforms), '-f', dockerfile, '-t', image_name]
        if load:
            cmd.append('--load')
        cmd.append('.')
        print(cmd)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to build Docker image: {e}")
        raise


class Architecture(Enum):
    ARM64 = 'arm64'
    JETPACK6 = 'jetpack6'
    X86_64 = 'x86_64'
    UNKNOWN = 'unknown'

def detect_architecture() -> Architecture:
    """
    Detect the underlying architecture.
    
    Returns:
        Architecture: Architecture type
    """
    machine = platform.machine().lower()
    architecture = Architecture.UNKNOWN
    # Check if running on Jetson (presence of jetson-specific files)
    if os.path.exists('/etc/nv_tegra_release'):
        # Parse JetPack version from the release file
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                content = f.read()
                # JetPack 6.x detection
                if 'R36' in content or 'R37' in content:
                    return Architecture.JETPACK6
        except (IOError, OSError):
            pass
        # Default to JETPACK6 if we can't determine version
        architecture = Architecture.JETPACK6    
    
    # Check for ARM64 architecture
    elif machine in ['arm64', 'aarch64']:
        architecture = Architecture.ARM64
    # Check for x86_64/AMD64
    elif machine in ['x86_64', 'amd64']:
        architecture = Architecture.X86_64

    print(f"Detected architecture: {architecture.value}")
    
    return architecture

def tag_and_push_image(dockerimage: str, tag: str, registry: str) -> None:
    """Tag and push the image to a registry (e.g. Github or Scaleway).

       Args:
          dockerimage (str): Name of the Docker image.
          tag (str): Image tag.
          registry (str): Registry URL (e.g., 'ghcr.io', 'rg.fr-par.scw.cloud').
     """
    registry_image_name = f'{registry}/{dockerimage}:{tag}'
    print(f"Tagging image as {registry_image_name}")
    subprocess.run(['docker', 'tag', f'{dockerimage}:{tag}', registry_image_name], check=True)
    
    print(f"Pushing image {registry_image_name} to Scaleway registry")
    subprocess.run(['docker', 'push', registry_image_name], check=True)

def parse_for_push() -> bool:
    """Parse command line arguments for push option.

    Returns:
        bool: True if --push is specified, False otherwise.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Build Docker image and push if specified")
    parser.add_argument("--push", action="store_true", help="Push the image to the registry after building")
    args = parser.parse_args()
    return args.push

def is_github_ci() -> bool:
    """Check if running in CI environment.

    Returns:
        bool: True if running in CI, False otherwise.
    """
    is_ci = os.getenv("GITHUB_ACTIONS") == "true"
    return is_ci
