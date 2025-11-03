"""Module to capture frames from a video file using OpenCV.
"""
from typing import Tuple, Optional
from pathlib import Path
import subprocess
import json

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
        self.rotation_code: Optional[int] = None  # Store rotation transformation code

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

    def _get_video_rotation(self) -> Optional[int]:
        """Extract rotation metadata from the video file using ffprobe.
        
        Returns:
            Optional[int]: OpenCV rotation code (cv2.ROTATE_*) or None if no rotation needed.
        """
        try:
            # Use ffprobe to extract rotation metadata (both tags and side_data)
            cmd = [
                'ffprobe',
                '-loglevel', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream_tags=rotate:stream_side_data=rotation',
                '-of', 'json',
                str(self.video_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
                        
        except FileNotFoundError:
            # ffprobe is not installed
            print("Warning: ffprobe not found. Video rotation metadata cannot be read.")
            print("Install FFmpeg to enable automatic video rotation: sudo apt-get install ffmpeg")
            return None
        except subprocess.CalledProcessError as e:
            # ffprobe command failed
            print(f"Warning: ffprobe failed to read video metadata: {e.stderr}")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            # Failed to parse ffprobe output
            print(f"Warning: Failed to parse video metadata: {e}")
            return None
                
        # Extract rotation value from metadata
        rotation_degrees = None
        if 'streams' in data and len(data['streams']) > 0:
            stream = data['streams'][0]
            
            # Check for rotation in tags (older format)
            if 'tags' in stream and 'rotate' in stream['tags']:
                rotation_degrees = int(stream['tags']['rotate'])
            
            # Check for rotation in side_data (newer format, takes precedence)
            elif 'side_data_list' in stream:
                for side_data in stream['side_data_list']:
                    if 'rotation' in side_data:
                        rotation_degrees = int(side_data['rotation'])
                        break
                    
        rotation_code_open_cv = None
        if rotation_degrees is not None:
            # Map rotation degrees to OpenCV rotation codes
            # Video metadata rotation indicates how much to rotate the video for correct display
            # Positive values = counterclockwise, Negative values = clockwise
            rotation_map = {
                90: cv2.ROTATE_90_COUNTERCLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_CLOCKWISE,
                -90: cv2.ROTATE_90_CLOCKWISE,
                -180: cv2.ROTATE_180,
                -270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                0: None
                }
            rotation_code_open_cv = rotation_map.get(rotation_degrees, None)
            if rotation_code_open_cv is not None:
                print(f"Video rotation detected: {rotation_degrees}° - will be corrected during playback")
        
        return rotation_code_open_cv

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
                "The video file may be corrupted or use an unsupported format.")
        
        # Extract rotation metadata
        self.rotation_code = self._get_video_rotation()


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
            
        # Apply rotation to the looped frame as well
        if success:
            frame = self._rotate_frame(frame)

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
    
    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Rotate the given frame based on the rotation code.

        Args:
            frame (np.ndarray): The frame to rotate.

        Returns:
            np.ndarray: The rotated frame.
        """
        if self.rotation_code is not None and frame is not None:
            frame = cv2.rotate(frame, self.rotation_code)
        return frame