"""
Build Docker image for JetPack 6 architecture.
"""

from build_docker_util import build_docker

from build_docker_util import build_docker

IMAGE_NAME: str = 'njm08/neptune-eye'
TAG: str = 'latest-jetpack6'
DOCKER_FILE: str = 'Dockerfile.jetpack6'

if __name__ == "__main__":
        build_docker(
        dockerfile=DOCKER_FILE,
        dockerimage=IMAGE_NAME,
        tag=TAG,
        platforms=["linux/arm64"])