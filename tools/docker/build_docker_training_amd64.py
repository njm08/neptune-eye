"""
Build Docker image for training.
"""

from build_docker_util import build_docker, tag_and_push_image, parse_for_push, is_github_ci, REGISTRY_SCALEWAY

IMAGE_NAME: str = 'njm08/neptune-eye'
TAG: str = 'latest-training-amd64'
DOCKER_FILE: str = 'Dockerfile.training-amd64'
PLATFORM = ["linux/amd64"]

if __name__ == "__main__":
        push = parse_for_push()
        build_docker(dockerfile=DOCKER_FILE, dockerimage=IMAGE_NAME, tag=TAG, platforms=PLATFORM)
        if push:
            tag_and_push_image(dockerimage=IMAGE_NAME, tag=TAG, registry=REGISTRY_SCALEWAY)