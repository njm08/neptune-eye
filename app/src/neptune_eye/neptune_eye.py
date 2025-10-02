#!/usr/bin/env python3
"""
Neptune Eye - YOLO Object Detection Runner

This module provides the main entry point for running YOLO object detection on maritime objects.
"""
from pathlib import Path
from enum import Enum, auto

import cv2

from frame_capture.video_file_capture import VideoFileCapture
from frame_capture.camera_capture import CameraCapture
from object_detection.yolo_object_detection import Yolo11ObjectDetection, YoloModelSize, InferenceDevice

# TODO Move to a config module.
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

INPUT_SOURCE = InputSource.MOVIE           # Options: InputSource.CAMERA, InputSource.MOVIE
CAMERA_INDEX = 1                            # Camera index. 0 for default camera (usually built-in), 1 for external camera, etc.
MOVIE_PATH = "../neptune_add_ons/windraeder_segelboot.MOV" # Path to movie file (relative to project root)
CONFIDENCE = 0.5
# ***********************************************************

def continuous_capture_and_inference() -> None:
    """Capture images from the webcam or movie file and run inference.
    """

    # Setup the object detection model
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    model_dir = (root_dir / "models").resolve()
    model = Yolo11ObjectDetection(
        model_dir=model_dir,
        model_size=YOLO_MODEL_SIZE,
        model_path=OVERRIDE_MODEL_PATH,
        device=OVERRIDE_DEVICE,
        confidence=CONFIDENCE)
    model.setup()

    # Initialize video capture based on input source
    if INPUT_SOURCE == InputSource.CAMERA:
        capture = CameraCapture(camera_index=CAMERA_INDEX)
    elif INPUT_SOURCE == InputSource.MOVIE:
        # Get absolute path to movie file
        movie_path = (root_dir / MOVIE_PATH).resolve()
        capture = VideoFileCapture(video_path=str(movie_path))
    capture.open()
   
    print("Starting continuous capture and inference...")
    print("Press 'q' or 'ESC' in the video window to stop, or Ctrl+C in terminal")

    try:
        while True:
            # Capture frame
            success, frame = capture.read()           

            if success and frame is not None:
                # Run inference
                results = model.detect(frame)

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