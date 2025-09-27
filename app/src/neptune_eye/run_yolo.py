import cv2
from ultralytics import YOLO
from pathlib import Path
from enum import Enum, auto
import torch
from . import __version__

# TODO Move to a config module.
class YoloModelSize(Enum):
    """Enum for different YOLO model sizes.
    """
    YOLO11N = auto()
    YOLO11S = auto()

class InferenceDevice(Enum):
    """Chose the device for inference.
    """
    NVIDIA_GPU = '0'
    M1_GPU = 'mps'
    CPU = 'cpu'

# *****************  Configuration  ************************
YOLO_MODEL_SIZE = YoloModelSize.YOLO11N     # Options: YoloModelSize.YOLO11N, YoloModelSize.YOLO11S
FP16 = True                                 # True, use FP16 precision
OVERRIDE_DEVICE = InferenceDevice.CPU       # Options: None, InferenceDevice.NVIDIA_GPU, InferenceDevice.M1_GPU, InferenceDevice.CPU
OVERRIDE_MODEL_PATH = "models/m1/yolo11n_cpu.onnx"  # Options: None, path to model file
CAMERA_INDEX = 1                            # Camera index. 0 for default camera (usually built-in), 1 for external camera, etc.
# ***********************************************************

def detect_device():
    """Detect the available hardware and return the device for inference.

    Returns:
        device: The device for inference.
    """
    
    if OVERRIDE_DEVICE is not None:
        print(f"Forcing device to: {OVERRIDE_DEVICE.name}")
        return OVERRIDE_DEVICE
    
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

def setup_inference():
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
                    yolo_model_fp = "models/jetson/yolo11n_16fpu_gpu.engine"
                else:
                    yolo_model_fp = "models/jetson/yolo11n_gpu.engine"
            elif YOLO_MODEL_SIZE == YoloModelSize.YOLO11S:
                if FP16:
                    yolo_model_fp = "models/jetson/yolo11s_16fpu_gpu.engine"
                else:
                    yolo_model_fp = "models/jetson/yolo11s_gpu.engine"
        elif device == InferenceDevice.M1_GPU or device == InferenceDevice.CPU:
            if YOLO_MODEL_SIZE == YoloModelSize.YOLO11N:
                yolo_model_fp = "models/m1/yolo11n.pt"
            elif YOLO_MODEL_SIZE == YoloModelSize.YOLO11S:
                yolo_model_fp = "models/m1/yolo11s.pt"
    
    # Combine root path with YOLO model file path
    root_dir = Path(__file__).resolve().parent.parent.parent
    model_path = (root_dir / yolo_model_fp).resolve()
    print(f"Using YOLO model at: {model_path}")
    
    # Load the model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}.")
    model = YOLO(model_path, task='detect')
    
    return model, device

def continuous_capture_and_inference():
    """Capture images from the webcam and run inference.
    """

    # Setup model and device
    model, device = setup_inference()
        
    # Initialize webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    
    print("Starting continuous capture and inference...")
    print("Press 'q' or 'ESC' in the video window to stop, or Ctrl+C in terminal")
        
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame, retrying...")
                continue
            
            # Run inference
            results = model(frame, device=device.value)

            # Draw results on frame
            annotated_frame = results[0].plot()
            
            # Display the frame with detections
            cv2.imshow('Neptune Eye - YOLO Detection', annotated_frame)
            
            # Check for exit condition
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is the Escape key
                print("Stopping capture and inference...")
                break
                
    except KeyboardInterrupt:
        print("\nStopping capture and inference...")

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print(f"Neptune Eye v{__version__}")
    continuous_capture_and_inference()