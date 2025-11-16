"""
Build Docker image for amd64 architecture with minimal Python environment.
"""

from build_docker_util import build_docker, tag_and_push_image, parse_for_push, is_github_ci, REGISTRY_GITHUB

IMAGE_NAME: str = 'njm08/neptune-eye'
TAG: str = 'latest-inference-amd64'
DOCKER_FILE: str = 'Dockerfile.inference-amd64'
PLATFORM = ["linux/amd64"]

if __name__ == "__main__":
    push = parse_for_push()
    load = False
    if push and is_github_ci():
        load = True
    build_docker(dockerfile=DOCKER_FILE, dockerimage=IMAGE_NAME, tag=TAG, platforms=PLATFORM, load=load)
    if push and is_github_ci():
        tag_and_push_image(dockerimage=IMAGE_NAME, tag=TAG, registry=REGISTRY_GITHUB)