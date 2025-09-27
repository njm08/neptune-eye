#!/bin/bash
# Build and run Neptune Eye in Docker

set -e

echo "Building Neptune Eye Docker image..."
docker build -t neptune-eye:latest .

echo "Running Neptune Eye container..."
if [ $# -eq 0 ]; then
    # No arguments provided, run with default command
    docker run --rm --name neptune-eye-run neptune-eye:latest
else
    # Arguments provided, run with custom command
    docker run --rm --name neptune-eye-run neptune-eye:latest python -m neptune_eye.neptune_eye "$@"
fi