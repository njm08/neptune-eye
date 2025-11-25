#!/usr/bin/env python3
"""
Neptune Eye - YOLO Object Detection Runner

This module provides the main entry point for running YOLO object detection on maritime objects.
"""
from pathlib import Path

from cv2 import __version__ as cv2_version
from frame_capture.camera_capture import CameraCapture
from frame_capture.image_capture import ImageCapture
from frame_capture.video_file_capture import VideoFileCapture
from object_detection.yolo_object_detection import Yolo11ObjectDetection
from config import NeptuneEyeConfig
from utilites import find_project_root
from result_display import ResultDisplay

def continuous_capture_and_inference() -> None:
    """Capture images from the webcam or movie file and run inference.
    """
    
    # Find project root directory
    root_dir = find_project_root()
    if root_dir is None:
        raise RuntimeError("Could not find project root directory")
    
    # Load and validate configuration. If no configuration is found, a default one is created.
    config_path = root_dir / "app" / "src" / "neptune_eye" / "neptune_eye_config.yaml"
    config: NeptuneEyeConfig = NeptuneEyeConfig(config_path=config_path)

    # Initialize the YOLO model
    try:
        model: Yolo11ObjectDetection = Yolo11ObjectDetection(
            model_path=config.get_model_path(),
            device=config.get_device(),
            confidence=config.get_confidence(),
            iou=config.get_iou_threshold(),
            imgsz=config.get_image_size(),
            half_precision=config.get_fp16())
        model.setup()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize YOLO model: {e}") from e

    # Initialize the input source (camera or movie file) and the result display (GUI or console)
    if config.is_input_camera():
        capture_source = CameraCapture(camera_index=config.get_camera_index())
    elif config.is_input_movie():
        capture_source = VideoFileCapture(config.get_movie_path())
    elif config.is_input_images():
        capture_source = ImageCapture(image_folder=config.get_image_folder_path())
    else:
        raise RuntimeError("No valid input source configured.")
    with capture_source as capture, ResultDisplay(headless=config.get_headless()) as result_display:   

        # Enter continuous capture and inference loop
        print("Starting continuous capture and inference...")
        try:
            while True:
                # Capture frame
                success, frame = capture.read()           
                if not success or frame is None:
                   continue # This skips to the next iteration of the loop
                
                # Run inference
                results = model.detect(frame)
                # Display results
                exit_display = result_display.display(frame, results)
                if exit_display:
                    break
        except KeyboardInterrupt:
            print("Interrupted by user.\nExiting Neptune Eye.")
        except Exception as e:
            print(f"Error during capture and inference: {e}")
            raise
        finally:
            # Resources cleanup is handled by context managers
            pass

if __name__ == "__main__":
    print(
        r"""
                        _._
                          :.
                          : :
                          :  .
                         .:   :
                        : :    .
                       :  :     :
                      .   :      .
                     :    :       :
                    :     :        .
                   .      :         :
                  :       :          .
                 :        :           :
                .=w=w=w=w=:            .
                          :=w=w=w=w=w=w=.   ....
           <--._______:U~~~~~~~~\_________.:---/
            \      ____===================____/
.,-~^~-,.,-~^~-,.,-~^~-,.,-~^~-,.,-~^~-,.,-~^~-,.,-~^~-,.
.,-~^~-,.,-~^~-,.,-~^~-,.,-~^~-,.,-~^~-,.,-~^~-,.,-~^~-,.
"""
    )
    print("Neptune Eye - YOLO Object Detection\n\n")
    print(f"OpenCV version: {cv2_version}")
    continuous_capture_and_inference()