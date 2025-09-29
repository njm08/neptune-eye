import cv2
from ultralytics import YOLO
from pathlib import Path
from enum import Enum, auto
import torch

# TODO Move to a config module.
class YoloModelSize(Enum):
    """Enum for different YOLO model sizes.
    """
    YOLO11N = auto()
    YOLO11S = auto()
    YOLO11M = auto()

class InferenceDevice(Enum):
    """Chose the device for inference.
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
#OVERRIDE_MODEL_PATH = "models/pytorch/yolo11n_maritime_best_30.pt"                  # Options: None, path to model file
OVERRIDE_MODEL_PATH = None            # Options: None, path to model file

INPUT_SOURCE = InputSource.MOVIE           # Options: InputSource.CAMERA, InputSource.MOVIE
CAMERA_INDEX = 0                            # Camera index. 0 for default camera (usually built-in), 1 for external camera, etc.
MOVIE_PATH = "res/movies/windraeder_segelboot.MOV" # Path to movie file (relative to project root)
LOOP_MOVIE = True                           # Whether to loop the movie when it ends
CONFIDENCE = 0.5
# ***********************************************************

def detect_device():
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
                    yolo_model_fp = "models/gpu_engine/yolo11n_16fp_gpu.engine"
                else:
                    yolo_model_fp = "models/gpu_engine/yolo11n_32fp_gpu.engine"
            elif YOLO_MODEL_SIZE == YoloModelSize.YOLO11S:
                if FP16:
                    yolo_model_fp = "models/gpu_engine/yolo11s_16fp_gpu.engine"
                else:
                    yolo_model_fp = "models/gpu_engine/yolo11s_32fp_gpu.engine"
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
    print(f"Using YOLO model at: {model_path}")
    
    # Load the model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}.")
    model = YOLO(model_path, task='detect')
    
    return model, device

def continuous_capture_and_inference():
    """Capture images from the webcam or movie file and run inference.
    """

    # Setup model and device
    model, device = setup_inference()
    
    # Initialize video capture based on input source
    if INPUT_SOURCE == InputSource.CAMERA:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera with index {CAMERA_INDEX}.")
        print(f"Using camera input (index: {CAMERA_INDEX})")
        source_name = "Camera"
    elif INPUT_SOURCE == InputSource.MOVIE:
        # Get absolute path to movie file
        root_dir = Path(__file__).resolve().parent.parent.parent.parent
        movie_path = (root_dir / MOVIE_PATH).resolve()
        
        if not movie_path.exists():
            raise FileNotFoundError(f"Movie file not found at {movie_path}")
            
        cap = cv2.VideoCapture(str(movie_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open movie file: {movie_path}")
        print(f"Using movie input: {movie_path}")
        source_name = "Movie"
        
        # Get total frame count for movie looping
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Movie info: {total_frames} frames at {fps} FPS")
    
    print("Starting continuous capture and inference...")
    print("Press 'q' or 'ESC' in the video window to stop, or Ctrl+C in terminal")
        
    try:
        frame_count = 0
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            # Handle end of movie file
            if not ret:
                if INPUT_SOURCE == InputSource.MOVIE and LOOP_MOVIE:
                    print("Movie ended, restarting from beginning...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    continue
                elif INPUT_SOURCE == InputSource.MOVIE:
                    print("Movie ended, stopping...")
                    break
                else:
                    print("Failed to capture frame from camera, retrying...")
                    continue
            
            frame_count += 1
            
            # Run inference
            results = model(frame, device=device.value, conf=CONFIDENCE)

            # Draw results on frame
            annotated_frame = results[0].plot()
            
            # Add frame counter for movies
            if INPUT_SOURCE == InputSource.MOVIE:
                cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame with detections
            cv2.imshow(f'Neptune Eye - YOLO Detection ({source_name})', annotated_frame)
            
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
    print(f"Neptune Eye - YOLO Object Detection")
    print(f"Input Source: {INPUT_SOURCE.name}")
    if INPUT_SOURCE == InputSource.MOVIE:
        print(f"Movie Path: {MOVIE_PATH}")
        print(f"Loop Movie: {LOOP_MOVIE}")
    elif INPUT_SOURCE == InputSource.CAMERA:
        print(f"Camera Index: {CAMERA_INDEX}")
    print("-" * 50)
    continuous_capture_and_inference()