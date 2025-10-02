"""
Utility functions for the Neptune Eye project.
"""
from pathlib import Path

def find_project_root() -> Path:
    """Find the root directory of the project.

    Use a marker in the root directory to identify it. Here we use 'Dockerfile'.
    We start searching from the current file's directory and move upwards.

    Raises:
        FileNotFoundError: If the project root cannot be found.

    Returns:
        Path: The path to the project root directory.
    """
    current = Path(__file__)
    for parent in [current] + list(current.parents):
        if (parent / "Dockerfile").exists():
            return parent
    raise FileNotFoundError("Could not locate project root")
