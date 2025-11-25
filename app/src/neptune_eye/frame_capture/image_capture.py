"""Capture frames from a folder of images at a configurable delay.
Created by ChatGPT-5-Codex
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .frame_capture_interface import FrameCaptureInterface


class ImageCapture(FrameCaptureInterface):
    """Capture frames by iterating over images in a directory."""

    _SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    _SLEEP_CHUNK_SECONDS = 0.05  # Sleep in short slices to stay responsive while waiting

    def __init__(self, image_folder: str, delay_ms: float = 2000.0) -> None:
        """Initialise the image folder capture.

        Args:
            image_folder: Directory containing images to stream.
            delay_ms: Delay between individual frames in milliseconds.
        """
        super().__init__()
        self._image_folder = Path(image_folder)
        self._delay_ms = max(0.0, delay_ms)
        self._delay_seconds = self._delay_ms / 1000.0 if self._delay_ms > 0 else 0.0
        self._image_paths: List[Path] = []
        self._current_index = 0
        self._last_frame_time: Optional[float] = None
        self._is_open = False

    def __enter__(self) -> "ImageCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def open(self) -> None:
        """Prepare the image list for iteration."""
        if self._is_open:
            return

        if not self._image_folder.exists() or not self._image_folder.is_dir():
            raise FileNotFoundError(f"Image folder not found: {self._image_folder}")

        images = [
            path
            for path in sorted(self._image_folder.iterdir())
            if path.is_file() and path.suffix.lower() in self._SUPPORTED_EXTENSIONS
        ]

        if not images:
            raise RuntimeError(f"No supported images found in folder: {self._image_folder}")

        self._image_paths = images
        self._current_index = 0
        self._last_frame_time = None
        self._is_open = True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the next frame in the folder sequence."""
        if not self.is_open:
            raise RuntimeError("Image folder not opened. Call open() first.")

        # Pace the frame delivery to respect the configured delay
        if self._delay_seconds > 0:
            if self._last_frame_time is not None:
                target_time = self._last_frame_time + self._delay_seconds
                while True:
                    remaining = target_time - time.time()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, self._SLEEP_CHUNK_SECONDS))
            self._last_frame_time = time.time()

        for _ in range(len(self._image_paths)):
            image_path = self._image_paths[self._current_index]
            frame = cv2.imread(str(image_path))
            self._advance_index()

            if frame is not None:
                return True, frame

        # If we reach here we failed to load any image in a full cycle
        return False, None

    def release(self) -> None:
        """Release resources held by the capture."""
        self._image_paths = []
        self._current_index = 0
        self._last_frame_time = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    def set_delay_ms(self, delay_ms: float) -> None:
        """Update the frame delay at runtime."""
        self._delay_ms = max(0.0, delay_ms)
        self._delay_seconds = self._delay_ms / 1000.0 if self._delay_ms > 0 else 0.0

    def _advance_index(self) -> None:
        """Advance the index and wrap around to keep streaming."""
        if not self._image_paths:
            return
        self._current_index = (self._current_index + 1) % len(self._image_paths)