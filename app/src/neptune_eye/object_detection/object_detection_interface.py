from abc import ABC, abstractmethod
from typing import Any, List, Tuple

class ObjectDetectionInterface(ABC):
    @abstractmethod
    def setup(self) -> None:
        """
        Set up the object detection model or environment.
        This method should be called before detect().
        """

    @abstractmethod
    def detect(self, frame: Any) -> List:
        """
        Detect objects in the given frame.

        Args:
            frame: The input frame in a format suitable for the implementation.

        Returns:
            A list of detections. The format of each detection is implementation-specific.
        """