# Neptune Eye

- AI-powered maritime object detection system for sailboats
- Real-time detection of boats, buoys and other objects
- Detection ob objects that are blocked by large sails
- Stay safe even when away from the helm
- Open-source and low cost

**Follow the development journey:** <https://njm08.github.io/>

![Neptune Eye Detecting Sailboats](/res/gifs/detection_sailboat_svendborg.gif) ![Neptune Eye Detecting Ferry](/res/gifs/detection_buoys_one.gif)\
_Neptune Eye detecting other sailboats and buoys_

## Features

- **Real-world data**: Trained on imagery collected onboard a sailboat across varied maritime conditions
- **Real-time performance**: ~10 ms inference on NVIDIA Jetson Orin Nano
- **Cloud training**: Scalable model training and evaluation on Cloud GPU instances
- **Multi-architecture support**: Docker images for inference and training (ARM64 and x86), including specialized NVIDIA Jetpack 6 image
- **Modular architecture**: Configurable frame sources (camera/video/streaming) and detection models

## Cloud Training

Training is fully containerized and runs on Scaleway Cloud GPU instances. Models are trained using specialized Docker images, with experiment tracking and performance monitoring via MLflow.

## Multi-Architecture Support

Multiple Docker images are available for different architectures and purposes, based on _Ultralytics_ base images:

- **Jetpack 6**: Optimized for NVIDIA Jetson Orin Nano with GPU support for training and inference
- **Inference (AMD64/ARM64)**: Minimal Python environments for ARM64 (Mac M1/M2/M3, Raspberry Pi) and x86 (Intel/AMD) architectures. Used in CI pipelines for testing.
- **Training (AMD64)**: Full-featured image for GPU-accelerated training on cloud instances

### Pull Docker Image

The following script automatically detects your architecture and pulls the appropriate inference image:

```shell
python3 tools/docker/pull_docker_image_inference.py
```

### Build Docker Image

Build images using the appropriate script for your target platform:

```shell
python3 tools/docker/build_docker_jetpack6.py
```

```shell
python3 tools/docker/build_docker_inference_amd64.py
```

```shell
python3 tools/docker/build_docker_inference_arm64.py
```

```shell
python3 tools/docker/build_docker_training_amd64.py
```

### Limitations

- **GUI support**: Limited in Docker containers. Use headless mode for stable operation.
- **Mac GPU support**: M1/M2/M3 GPUs not supported in Docker. For GPU acceleration, install dependencies locally per the [Ultralytics installation guide](https://docs.ultralytics.com/quickstart/#custom-installation-methods).

## Roadmap

- Integration with outdoor marine surveillance camera
- Onboard installation and real-world testing (Spring 2026)

## License

MIT License — see `LICENSE` for details and third-party notices.
