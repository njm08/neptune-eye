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

## Docker Setup

This project is configured to run in a Docker container. Here are the different ways to run it:

```bash
# Build and run the container
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# Stop the container
docker-compose down
```

## License

MIT License — see `LICENSE` for details and third‑party notices.
