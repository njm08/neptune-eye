from enum import Enum, auto

import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Any, List

from .object_detection_interface import ObjectDetectionInterface

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

class Yolo11ObjectDetection(ObjectDetectionInterface):
    """
    Object detection implementation backed by Ultralytics YOLO11.

    Arguments:
        model_dir (str): Directory where the model files are located. The model files must follow a specific naming convention based on the model size and type (e.g., models/pytorch/yolo11s.pt for YOLO11 Small PyTorch model).
        model_size (YoloModelSize): The size of the YOLO11 model to use. Default is YoloModelSize.YOLO11S. The model will be picked to have the best performance on the device.
        model_path (str | None): Optional path to a custom YOLO11 model file. This will override the model_size parameter if provided.
        device (InferenceDevice | None): The device to run inference on. If none is chosen, the implementation will attempt to select the best available device (CUDA, MPS, CPU).
        confidence (float): Confidence threshold for detections. Default is 0.25.
        iou (float): IoU threshold for NMS. Default is 0.45
        imgsz (int | Tuple[int, int]): Inference image size. Fore square images only one parameter is necessary. Default is 640.
        half_precision (bool): Whether to use half precision (FP16). Default is False.
    """

    def __init__(
        self,
        model_dir: str,
        model_size: YoloModelSize = YoloModelSize.YOLO11S,
        model_path: str | None = None,
        device: InferenceDevice | None = None,
        confidence: float = 0.25,
        iou: float = 0.45,
        imgsz: int | tuple[int, int] = 640,
        half_precision: bool = False,
    ) -> None:

        self.model_dir = model_dir
        self.model_size = model_size
        self.model_path = model_path
        self.device = device
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.imgsz = imgsz
        self.device = self.device
        self.model_path = self.model_path
        self.half_precision = bool(half_precision)
        self._model = None

    def setup(self) -> None:
        """ Set up the object detection model or environment.

        This method should be called before detect().
        No warm-up is done, since the first inference call is usually not time-critical.
        """

        # Set the device and model path. These settings can be overridden by the user in the constructor.
        self.device = self._set_device()
        self.model_path = self._set_model_path()

        # Combine root path with YOLO model file path
        absolute_model_path = (Path(self.model_dir) / Path(self.model_path)).resolve()
        # Load the model
        if not absolute_model_path.exists():
            raise FileNotFoundError(f"Model not found at {absolute_model_path}.")
        
        try:
            self._model = YOLO(absolute_model_path, task='detect')
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {absolute_model_path}: {e}") from e
        
    def detect(self, frame: Any) -> List:
        """
       
        """
        if self._model is None:
            raise RuntimeError("Model not set up. Call setup() before detect().")

        results = self._model(frame, device=self.device.value, conf=self.confidence, half=self.half_precision)

        return results

    # ----- Private Functions -----

    def _set_device(self) -> InferenceDevice:
        """Set the available hardware to run the inference on.

        Use the best available hardware if it is not overridden by the user in the constructor.

        Returns:
            device: The device for inference.
        """
    
        if self.device is not None:
            print(f"User defined device: {self.device.value}")
            return self.device
        
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

    def _set_model_path(self) -> str:
        """Set the model file path based on the selected model size and device.

        The file path is from the root of the project.

        Returns:
            str: The model file path.
        """
        # Determine model file path based on selected model size and device
        if self.model_path is not None:
            model_path = self.model_path
            print(f"User defined model path: {model_path}")
        else:
            if self.device is None:
                raise RuntimeError("Device must be set before determining model path.")
            if self.device == InferenceDevice.NVIDIA_GPU: # NVIDIA GPU uses TensorRT engines for best performance.
                model_path = self._get_nvidia_gpu_model_path()
            else:
                model_path = self._get_pytorch_model_path() # All other devices use PyTorch models.

        return model_path

    def _get_nvidia_gpu_model_path(self) -> str:
        """Get model path for NVIDIA GPU inference.
        """
        precision_suffix = "16fp" if self.half_precision else "32fp"
        
        model_paths = {
            YoloModelSize.YOLO11N: f"engine/yolo11n_{precision_suffix}.engine",
            YoloModelSize.YOLO11S: f"engine/yolo11s_{precision_suffix}.engine",
        }

        if self.model_size not in model_paths:
            raise RuntimeError(f"Unsupported model size for NVIDIA GPU: {self.model_size}")
        
        return model_paths[self.model_size]

    def _get_pytorch_model_path(self) -> str:
        """Get model path for PyTorch inference (M1 GPU or CPU).
        """
        model_paths = {
            YoloModelSize.YOLO11N: "pytorch/yolo11n.pt",
            YoloModelSize.YOLO11S: "pytorch/yolo11s.pt",
            YoloModelSize.YOLO11M: "pytorch/yolo11m.pt",
        }

        # Error checks.
        if self.model_size not in model_paths:
            raise RuntimeError(f"Unsupported model size for PyTorch: {self.model_size}")
        
        return model_paths[self.model_size]