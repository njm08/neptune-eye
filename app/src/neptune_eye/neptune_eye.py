#!/usr/bin/env python3
"""
Neptune Eye - YOLO Object Detection Runner

This module provides the main entry point for running YOLO object detection on maritime objects.
"""
from pathlib import Path

from cv2 import __version__ as cv2_version
from frame_capture.video_file_capture import VideoFileCapture
from frame_capture.camera_capture import CameraCapture
from object_detection.yolo_object_detection import Yolo11ObjectDetection
from config import NeptuneEyeConfig, InputSource
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
    config = NeptuneEyeConfig.load(config_path)

    # Initialize the YOLO model
    try:
        model = Yolo11ObjectDetection(
            model_path=config.expert.model_path,
            device=config.expert.device,
            confidence=config.general.confidence,
            iou=config.expert.iou_threshold,
            imgsz=config.expert.image_size,
            half_precision=config.expert.fp16)
        model.setup()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize YOLO model: {e}") from e

    # Initialize the input source (camera or movie file) and the result display (GUI or console)
    movie_path = (root_dir / config.general.movie_path).resolve()
    with (CameraCapture(camera_index=config.general.camera_index) 
          if config.general.source == InputSource.CAMERA 
          else VideoFileCapture(str(movie_path))) as capture, \
          ResultDisplay(headless=config.general.headless) as result_display:   

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