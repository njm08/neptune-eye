#!/usr/bin/env python3
"""
Neptune Eye - YOLO Object Detection Runner

This module provides the main entry point for running YOLO object detection on maritime objects.
"""
from pathlib import Path

import cv2

from frame_capture.video_file_capture import VideoFileCapture
from frame_capture.camera_capture import CameraCapture
from object_detection.yolo_object_detection import Yolo11ObjectDetection
from config import load_config, validate_config, InputSource
from utilites import find_project_root

def continuous_capture_and_inference() -> None:
    """Capture images from the webcam or movie file and run inference.
    """
    # Load configuration from YAML file
    config = load_config()
    validate_config(config)
    
    print(f"   Configuration loaded successfully")
    print(f"   Model: {config.model.size.name}, FP16: {config.model.fp16}")
    print(f"   Device: {config.model.override_device.name if config.model.override_device else 'Auto-detect'}")
    print(f"   Input: {config.input.source.value}")
    print(f"   Confidence: {config.model.confidence}")

    # Setup the object detection model
    root_dir = find_project_root()
    model_dir = (root_dir / "models").resolve()
    model = Yolo11ObjectDetection(
        model_dir=model_dir,
        model_size=config.model.size,
        model_path=config.model.override_model_path,
        device=config.model.override_device,
        confidence=config.model.confidence)
    model.setup()

    # Get absolute path to movie file
    movie_path = (root_dir / config.input.movie_path).resolve()
    
    with (CameraCapture(camera_index=config.input.camera_index) if config.input.source == InputSource.CAMERA 
        else VideoFileCapture(str(movie_path))) as capture:
    
        print("Starting continuous capture and inference...")
        print("Press 'q' or 'ESC' in the video window to stop, or Ctrl+C in terminal")

        try:
            while True:
                # Capture frame
                success, frame = capture.read()           
                if not success or frame is None:
                   continue # This skips to the next iteration of the loop
                
                # Run inference
                results = model.detect(frame)
                # Draw results on frame
                annotated_frame = results[0].plot()

                # Display the frame with detections
                cv2.imshow("Neptune Eye", annotated_frame)

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