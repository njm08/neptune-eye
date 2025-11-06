"""
Script to run all tests.
"""
import subprocess
import sys
from pathlib import Path
from typing import Optional


def main():
    """Run the tests with pytest."""
    
    # Check if pytest is installed
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print(f"Pytest: {result.stdout.strip()}")
        else:
           print("Pytest not found. Please install pytest.")
           return 1          
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        print(f"Error: Pytest not found. {e}")
        return 1

    # Run the tests
    project_root = Path(__file__).parent
    
    cmd = [
        sys.executable, "-m", "pytest",
        str(project_root),
        "-v", # Verbose output
        "-s" # Disable output capturing
    ]
    
    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
