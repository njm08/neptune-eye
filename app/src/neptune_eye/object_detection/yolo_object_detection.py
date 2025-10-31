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
        model_path (str): Absolute path to the YOLO model file (.pt, .engine, etc.).
        device (InferenceDevice | None): The device to run inference on. If none is chosen, the implementation will attempt to select the best available device (CUDA, MPS, CPU).
        confidence (float): Confidence threshold for detections. Default is 0.25.
        iou (float): IoU threshold for NMS. Default is 0.45
        imgsz (int | Tuple[int, int]): Inference image size. For square images only one parameter is necessary. Default is 640.
        half_precision (bool): Whether to use half precision (FP16). Default is False.
    """

    def __init__(
        self,
        model_path: str,
        device: InferenceDevice | None = None,
        confidence: float = 0.25,
        iou: float = 0.45,
        imgsz: int | tuple[int, int] = 640,
        half_precision: bool = False,
    ) -> None:

        self.model_path = model_path
        self.device = device
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = imgsz
        self.half_precision = bool(half_precision)
        self._model = None

    def setup(self) -> None:
        """ Set up the object detection model or environment.

        This method should be called before detect().
        No warm-up is done, since the first inference call is usually not time-critical.
        """

        # Set the device if not provided
        self.device = self._set_device()

        # Validate model path exists
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}.")
        
        try:
            self._model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}") from e
        
    def detect(self, frame: Any) -> List:
        """ Detect objects in the given frame.

        Args:
            frame (Any): The input frame for object detection.

        Raises:
            RuntimeError: If the model is not set up before.

        Returns:
            List: The detection results.
        """
        if self._model is None:
            raise RuntimeError("Model not set up. Call setup() before detect().")

        results = self._model.track(frame, device=self.device.value, conf=self.confidence, half=self.half_precision, imgsz=self.image_size, iou=self.iou, verbose=False)

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

