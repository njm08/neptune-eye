"""
Build Docker image for JetPack 6 architecture.
"""

from build_docker_arm64 import build_docker, Architecture

if __name__ == "__main__":
    build_docker(Architecture.JETPACK6)