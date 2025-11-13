"""
Build Docker image for training.
"""

from build_docker_util import build_docker

IMAGE_NAME: str = 'njm08/neptune-eye'
TAG: str = 'latest-training'
DOCKER_FILE: str = 'Dockerfile.training'

if __name__ == "__main__":
        build_docker(
        docker_file=DOCKER_FILE,
        dockerimage=IMAGE_NAME,
        tag=TAG,
        platforms=["linux/amd64", "linux/arm64"])