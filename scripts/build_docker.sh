#!/bin/bash
# Build Neptune Eye Docker image

set -e

echo "Building Neptune Eye Docker image..."
docker build -t neptune-eye:latest .

echo "Docker image built successfully!"
echo "To run: docker run --rm neptune-eye:latest"
echo "Or use: docker-compose up"