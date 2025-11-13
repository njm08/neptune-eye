"""
Build Docker image for arm64 architecture with minimal Python environment.
"""

from build_docker_util import build_docker

IMAGE_NAME: str = 'njm08/neptune-eye'
TAG: str = 'latest-inference-arm64'
DOCKER_FILE: str = 'Dockerfile.inference-arm64'

if __name__ == "__main__":
        build_docker(
        dockerfile=DOCKER_FILE,
        dockerimage=IMAGE_NAME,
        tag=TAG,
        platforms=["linux/arm64"])