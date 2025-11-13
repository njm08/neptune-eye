"""Utilities for building Docker images.

"""
import os
from pathlib import Path
import subprocess
import platform
from enum import Enum

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

def build_docker(dockerfile: str, dockerimage:str, tag: str, platforms: list[str], load: bool = False) -> None:
    """
    Build the docker image.
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
                    return 'jetpack6'
        except:
            pass
        architecture = Architecture.JETPACK6
    
    # Check for ARM64 architecture
    if machine in ['arm64', 'aarch64']:
        architecture = Architecture.ARM64

    # Check for x86_64/AMD64
    if machine in ['x86_64', 'amd64']:
        architecture = Architecture.X86_64
    
    print(f"Detected architecture: {architecture.value}")
    
    return architecture

