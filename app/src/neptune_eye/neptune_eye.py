#!/usr/bin/env python3
"""
Neptune Eye - YOLO Object Detection Runner

This module provides the main entry point for running YOLO object detection on maritime objects.
"""
from pathlib import Path
from enum import Enum, auto

import cv2
import torch
from ultralytics import YOLO

from frame_capture.video_file_capture import VideoFileCapture
from frame_capture.camera_capture import CameraCapture

# TODO Move to a config module.
class YoloModelSize(Enum):
    """Enum for different YOLO model sizes.
    """
    YOLO11N = auto()
    YOLO11S = auto()
    YOLO11M = auto()

class InferenceDevice(Enum):
    """Choose the device for inference.
    """
    NVIDIA_GPU = '0'
    M1_GPU = 'mps'
    CPU = 'cpu'

class InputSource(Enum):
    """Choose the input source for video processing.
    """
    CAMERA = auto()
    MOVIE = auto()

# *****************  Configuration  ************************
YOLO_MODEL_SIZE = YoloModelSize.YOLO11M     # Options: YoloModelSize.YOLO11N, YoloModelSize.YOLO11S
FP16 = False                                # True, use FP16 precision
OVERRIDE_DEVICE = None                      # Options: None, InferenceDevice.NVIDIA_GPU, InferenceDevice.M1_GPU, InferenceDevice.CPU
#OVERRIDE_MODEL_PATH = "models/pytorch/yolo11s_maritime_15.pt"                  # Options: None, path to model file
OVERRIDE_MODEL_PATH = None            # Options: None, path to model file

INPUT_SOURCE = InputSource.CAMERA           # Options: InputSource.CAMERA, InputSource.MOVIE
CAMERA_INDEX = 1                            # Camera index. 0 for default camera (usually built-in), 1 for external camera, etc.
MOVIE_PATH = "../neptune_add_ons/windraeder_segelboot.MOV" # Path to movie file (relative to project root)
LOOP_MOVIE = True                           # Whether to loop the movie when it ends
CONFIDENCE = 0.5
# ***********************************************************

def detect_device() -> InferenceDevice:
    """Detect the available hardware and return the device for inference.

    Returns:
        device: The device for inference.
    """
    
    if OVERRIDE_DEVICE is not None:
        print(f"Forcing device to: {OVERRIDE_DEVICE}")
        return InferenceDevice(OVERRIDE_DEVICE)
    
    if torch.backends.mps.is_available():
        device = InferenceDevice.M1_GPU
        print("M1 GPU detected. Using MPS for inference.")
    elif torch.cuda.is_available():
        device = InferenceDevice.NVIDIA_GPU
        print("NVIDIA GPU detected. Using CUDA for inference.")
    else:
        device = InferenceDevice.CPU
        print("No GPU detected. Using CPU for inference.")
    
    return device

def setup_inference() -> tuple[YOLO, InferenceDevice]:
    """Setup the YOLO model for inference depending on the detected hardware.

    Returns:
        model: The loaded YOLO model.
        device: The device for inference.
    """

    # Detect device
    device = detect_device()

    # Determine model file path based on selected model size and device
    if OVERRIDE_MODEL_PATH is not None:
        yolo_model_fp = OVERRIDE_MODEL_PATH
        print(f"Overriding model path to: {yolo_model_fp}")
    else:
        if device == InferenceDevice.NVIDIA_GPU:
            if YOLO_MODEL_SIZE == YoloModelSize.YOLO11N:
                if FP16:
                    yolo_model_fp = "models/gpu_engine/yolo11n_16fp_gpu.engine"
                else:
                    yolo_model_fp = "models/gpu_engine/yolo11n_32fp_gpu.engine"
            elif YOLO_MODEL_SIZE == YoloModelSize.YOLO11S:
                if FP16:
                    yolo_model_fp = "models/gpu_engine/yolo11s_16fp_gpu.engine"
                else:
                    yolo_model_fp = "models/gpu_engine/yolo11s_32fp_gpu.engine"
            elif YOLO_MODEL_SIZE == YoloModelSize.YOLO11M:
                raise RuntimeError("YOLO11M model not yet supported on NVIDIA GPU.")
        elif device == InferenceDevice.M1_GPU or device == InferenceDevice.CPU:
            if YOLO_MODEL_SIZE == YoloModelSize.YOLO11N:
                yolo_model_fp = "models/pytorch/yolo11n.pt"
            elif YOLO_MODEL_SIZE == YoloModelSize.YOLO11S:
                yolo_model_fp = "models/pytorch/yolo11s.pt"
            elif YOLO_MODEL_SIZE == YoloModelSize.YOLO11M:
                yolo_model_fp = "models/pytorch/yolo11m.pt"
    
    # Combine root path with YOLO model file path
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    model_path = (root_dir / yolo_model_fp).resolve()
    # Load the model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}.")

    try:
        model = YOLO(model_path, task='detect')
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}") from e
    
    return model, device

def continuous_capture_and_inference() -> None:
    """Capture images from the webcam or movie file and run inference.
    """

    # Setup model and device
    model, device = setup_inference()
    
    try:
        # Initialize video capture based on input source
        if INPUT_SOURCE == InputSource.CAMERA:
            capture = CameraCapture(camera_index=CAMERA_INDEX)
        elif INPUT_SOURCE == InputSource.MOVIE:
            # Get absolute path to movie file
            root_dir = Path(__file__).resolve().parent.parent.parent.parent
            movie_path = (root_dir / MOVIE_PATH).resolve()
            capture = VideoFileCapture(video_path=str(movie_path))
        capture.open()

        print("Starting continuous capture and inference...")
        print("Press 'q' or 'ESC' in the video window to stop, or Ctrl+C in terminal")
            
        while True:
            # Capture frame
            success, frame = capture.read()           

            if success and frame is not None:
                # Run inference
                results = model(frame, device=device.value, conf=CONFIDENCE)

            # Draw results on frame
            annotated_frame = results[0].plot()

            # Display the frame with detections
            cv2.imshow(f'Neptune Eye - YOLO Detection', annotated_frame)

            # Check for exit condition
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is the Escape key
                print("Stopping capture and inference...")
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping capture and inference...")
    except Exception as e:
        print(f"Error during capture and inference: {e}")
        raise
    finally:
        # Clean up
        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("\n \
                        _._\n \
                          :.\n \
                          : :\n \
                          :  .\n \
                         .:   :\n \
                        : :    .\n \
                       :  :     :\n \
                      .   :      .\n \
                     :    :       :\n \
                    :     :        .\n \
                   .      :         :\n \
                  :       :          .\n \
                 :        :           :\n \
                .=w=w=w=w=:            .\n \
                          :=w=w=w=w=w=w=.   ....\n \
           <--._______:U~~~~~~~~\_________.:---/\n \
            \      ____===================____/\n \
.,-~^~-,.,-~^~-,.,-~^~-,.,-~^~-,.,-~^~-,.,-~^~-,.,-~^~-,.\n \
.,-~^~-,.,-~^~-,.,-~^~-,.,-~^~-,.,-~^~-,.,-~^~-,.,-~^~-,.\n")
    print(f"Neptune Eye - YOLO Object Detection\n\n")

    continuous_capture_and_inference()