"""
Script to run all tests.
"""
import subprocess
import sys
from pathlib import Path


def main():
    """Run the tests with pytest."""
    
    # Check if pytest is installed
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            text=True
        )
        print(f"Pytest: {result.stdout.strip()}")
    except Exception as e:
        print("Pytest not found.")
    
    # Run the tests
    project_root = Path(__file__).parent
    
    cmd = [
        sys.executable, "-m", "pytest",
        str(project_root),
        "-v",
    ]
    
    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
