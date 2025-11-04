#!/bin/bash

# Detect if running on NVIDIA Jetson Nano
if [ -f /etc/nv_tegra_release ]; then
    echo "NVIDIA Jetson detected - running with Docker"
    docker run -it --ipc=host --runtime=nvidia --gpus all \
        -v .:/workspace  -w /workspace \
        -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
        njm08/neptune-eye:latest-jetpack6 \
        python3 app/src/neptune_eye/neptune_eye.py
else
    echo "Non-Jetson platform detected - running Python directly"
    python3 app/src/neptune_eye/neptune_eye.py
fi
