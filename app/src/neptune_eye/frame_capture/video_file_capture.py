"""Module to capture frames from a video file using OpenCV.
"""
from typing import Tuple, Optional
from pathlib import Path

import cv2
import numpy as np

from .frame_capture_interface import FrameCaptureInterface

class VideoFileCapture(FrameCaptureInterface):
    """Capture frames from a video file using OpenCV.

    Args:
        video_path (str): Path to the video file.
    """
    def __init__(self, video_path: str) -> None:
        """Initializer for VideoFileCapture.
        """
        super().__init__()
        self.video_path: Path = Path(video_path)
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 0.0
        self.total_frames: int = 0

    def __del__(self) -> None:
        """Destructor
        """
        self.release()

    def __enter__(self) -> 'VideoFileCapture':
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.release()

    def open(self) -> None:
        """Open the video file.

        Raises:
            FileNotFoundError: If the video file path does not exist.
            RuntimeError: If the video file cannot be opened.
            
        """
        if not self.video_path.exists():
            raise FileNotFoundError(f"Movie file not found at {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path)) # Ensure string path for OpenCV
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file {self.video_path}")
        
        # Get total frame count for movie looping
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        if self.fps <= 0:
            raise RuntimeError(f"Invalid FPS ({self.fps}) for video {self.video_path}. "
+                "The video file may be corrupted or use an unsupported format.")


    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the video file.
        
            Automatically loops back to the beginning when the end of the video is reached,
            enabling continuous frame capture.

        Raises:
            RuntimeError: If the video file cannot be read.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: A tuple containing a success flag and the captured frame.
        """
        if not self.is_open:
            raise RuntimeError("Video not opened. Call open() first.")
        success, frame = self.cap.read()
        
        # Handle end of movie file
        if not success:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.cap.read() # Try reading again after going back to start
            if not success:
                raise RuntimeError(f"Failed to read from video {self.video_path} even after seeking to start. "
                                    "The video file may be corrupted.")


        return success, frame

    def release(self) -> None:
        """Release the video capture object.
        """
        if self.cap:
            self.cap.release()
        self.cap = None
        self.fps = 0.0
        self.total_frames = 0

    @property
    def is_open(self) -> bool:
        """Check if the video is opened.

        Returns:
            bool: True if the video is opened, False otherwise.
        """
        is_open = self.cap is not None and self.cap.isOpened()
        return is_open