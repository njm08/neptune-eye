"""Capture images from camera or video file.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional

class FrameCaptureInterface(ABC):
    """Abstract base class for capturing frames from different sources (camera, video file, real-time-streaming, etc).

    """
    def __init__(self) -> None:
        super().__init__()

    def __del__(self) -> None:
        self.release()

    @abstractmethod
    def open(self) -> bool:
        """Open the frame source (e.g. camera or video file).

        Returns: 
        bool: True if the source was opened successfully, False otherwise.
        """

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Check if the source is opened.

        Returns:
            bool: True if the source is opened, False otherwise.
        """

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the next frame from the source.
        
        Returns:
            bool: Whether a frame was read.
            Optional[np.ndarray]: The image frame, or None if reading failed.
        """

    @abstractmethod
    def release(self) -> None:
        """Release the source (close video/camera)."""
