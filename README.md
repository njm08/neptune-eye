# neptune-eye

You are crusing on your sailboat for hours in the vast blue see. You haven't seen a boat for a while.
You go under deck and make yourself a snack. You come back to the cockpit and a boat passes you closer than comfortable.
This scary situation can be prevented by having a camera on your mast and an AI-powered machine vision algorithm constantly looking for hazards,
so you don't have to!

## Goal

The goal is to develop a real-time AI-powered boat detection, to warn you from boats in your path.

## Project Features

- YOLO (You only look once) object detection running on NVIDIA Jetson Orin Nano with 10 ms inference.
- YOLO also running on MacBook with 20 ms inference for developing and testing the models.
- Docker container for seamless integration of the project on your platform.

![YoloV11 Pre-trained Model](/res/gifs/yolov11.gif)

## Coming Soon

- Trained model for detecting boats.
- Connection of outdoor surveillance camera.

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
