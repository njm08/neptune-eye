"""Module for capturing frames from a camera using OpenCV.
"""
from typing import Tuple, Optional
import cv2
import numpy as np

from .frame_capture_interface import FrameCaptureInterface


class CameraCapture(FrameCaptureInterface):
    def __init__(self, camera_index: int = 0) -> None:
        """Constructor for CameraSource.

        Args:
            camera_index (int): Camera index. 0 is usually the built-in camera. Defaults to 0.
        """
        super().__init__()
        self.camera_index: int = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.open()

    def __del__(self) -> None:
        """Destructor to ensure the camera is released.
        """
        self.release()

    def __enter__(self):
        """Enter context manager."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and release resources."""
        self.release()
        return False

    def open(self) -> None:
        """Open the camera for capturing.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera with index {self.camera_index}.")
        print(f"Using camera index: {self.camera_index}")


    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture a frame from the camera.

        Raises:
            RuntimeError: If the camera is not opened.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: A tuple containing a success flag and the captured frame.
        """
        if not self.is_open:
            raise RuntimeError("Camera not opened. Call open() first.")
        return self.cap.read()

    def release(self) -> None:
        """Release the camera resource.
        """
        if self.cap:
            self.cap.release()
        self.cap = None

    @property
    def is_open(self) -> bool:
        """Check if the camera is opened.

        Returns:
            bool: True if the camera is opened, False otherwise.
        """
        is_open = self.cap is not None and self.cap.isOpened()
        return is_open