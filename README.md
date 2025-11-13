# Neptune Eye

You are cruising on your sailboat for hours in the vast blue sea.\
You haven't seen a boat for a while.\
You go under deck and make yourself a snack.\
You come back to the cockpit and a boat passes you a lot closer than you are comfortable with.\
_Neptune Eye_ prevents these scary situations.

It is an AI-powered object detection system constantly looking out for you and your crew,\
_so you can relax and stay safe_!

![Neptune Eye Detecting Sailboats](/res/gifs/detection_sailboat_svendborg.gif) ![Neptune Eye Detecting Ferry](/res/gifs/detection_buoys_one.gif)\
_Neptune Eye detecting other sailboats and buoys_

## Features

- __Real-world data__: Collected onboard a sailboat across varied conditions
- __Real-time performance__: ~10 ms inference on an NVIDIA Jetson Orin Nano
- __YOLOv11 models__: Nano, Small, and Medium with FP16/FP32 precision options
- __Modular__:  Frame sources (camera/video/streaming) and detection model are easy to change
- __Easy integration__: Docker containers for easy integration across platforms

## Coming Soon

- Integration of an outdoor surveillance camera
- Cloud integration for model training, evaluation, and performance tracking

## Follow the Journey

- Blog: <https://njm08.github.io/>

## Docker

The application can be run in Docker images. This way you do not need to install the dependencies locally.
There are several Docker images provided. They are based on the Docker images provided by _Ultralytics_.

- __Jetpack6__: Specialized Docker image for the NVIDIA Jetson Orin Nano running Jetpack 6. This image has GPU support.

```shell
python3 tools/docker/build_docker_jetpack6.py
```

- __Inference__: Image for running inference with a minimal python environment. Available for ARM64 architectures such as Mac M1/M2/M3 or Raspberry Pi or x86 architectures (Intel and AMD CPUs).

```shell
python3 tools/docker/build_docker_inference.py
```

- __Training__: Full blown image for running training on GPU. It is used for running the training on Cloud GPUs.

```shell
python3 tools/docker/build_docker_training.py
```

__Limitations__:

- GUI is tricky when running in Docker image. Use the default headless mode for a stable run.
- No GPU support for Mac M1/M2/M3 when running in Docker container. For GPU support you will need to install the dependencies locally according to the [installation guide](https://docs.ultralytics.com/quickstart/#custom-installation-methods).
  
## License

MIT License — see `LICENSE` for details and third‑party notices.
