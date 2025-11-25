"""
This module provides functions for displaying the results of object detection.
"""

import sys
import time
from typing import Optional, Tuple

import cv2

class ResultDisplay:
    """Handles the display of detection results.
    """

    def __init__(self, headless: bool = True, web_server=None) -> None:
        """Initialize the ResultDisplay.

        Args:
            headless (bool): If True, run in headless mode without displaying images.
            web_server (WebServer): Optional web server instance to update with frames.
        """
        self._headless = headless
        self._web_server = web_server
        self._previous_line_count = 0
        self._last_detection_time = None
        self._window_name = "Neptune Eye"
        self._window_initialized = False
        self._screen_size = self._detect_screen_size() if not self._headless else None
        self._max_screen_coverage = 0.92  # Keep a small margin around the window
        if self._headless: 
            print("Press Ctrl+C in terminal to stop.")   
        else:
            print("Press 'q' or 'ESC' in the video window or Ctrl+C in terminal to stop.")

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and release resources."""
        self.release()
        return False
    
    def display(self, frame: any, results: any) -> bool:
        """Display the given frame with detection results.

        Args:
            frame (any): The frame to display.
            results (any): The detection results to overlay on the frame.

        Returns:
            bool: True if the display should be exited. False otherwise.
        """
        exit = False

        if self._web_server:
            annotated_frame = results[0].plot()
            if annotated_frame is not None:
                self._web_server.update_frame(annotated_frame)

        if self._headless:
            self._display_headless(results)
        else:
            exit = self._display_gui(results)

        return exit

    def _display_headless(self, results: any) -> None:
        """Display results in headless mode (e.g., print to console).
           The detections are cleared after 1 second of no new detections.

        Args:
            results (any): The detection results to display.
        """
        # Get the detection results and format them for printing
        detections = results[0].summary(decimals=2)
        lines_to_print = ([f"Class: {d['name']} {d['track_id']} - Confidence: {d['confidence']}" for d in detections])
        # Print the new detection results. Leave the old detection results for a while if there are no new detections.
        current_time = time.monotonic()
        if lines_to_print:
            # Clear lines from previous detection
            for _ in range(self._previous_line_count):
                sys.stdout.write("\033[F\033[K")
            # Print new detection results
            for line in lines_to_print:
                print(line)
            self._previous_line_count = len(lines_to_print)
            self._last_detection_time = current_time # Store the time of the this detection
        # Clear the detections after 1 second of no new detections
        elif self._last_detection_time is not None and self._previous_line_count > 0 and current_time - self._last_detection_time > 1.0:
            for _ in range(self._previous_line_count):
                sys.stdout.write("\033[F\033[K")
            sys.stdout.flush()
            self._previous_line_count = 0
            self._last_detection_time = None
    
    def _display_gui(self, results: any) -> bool:
        """Display results in GUI mode (e.g., show images with overlays).

        Args:
            results (any): The detection results to display.

        Returns:
            bool: True if the display should be exited. False otherwise.
        """

        # Draw results on frame
        annotated_frame = results[0].plot()
        if annotated_frame is None:
            return True # Nothing to draw but we want to continue in case the next frame is valid.

        # Display the frame on the screen. Ensure the frame fits on the screen.
        # imshow can behave differently on various platforms, so we go through resizing explicitly.
        self._initialize_window() # Initialize window if not already done
        display_frame = self._resize_to_screen(annotated_frame) # Resize frame to fit screen if needed
        exit = False
        try:
            cv2.imshow(self._window_name, display_frame)
            
            # Check for exit condition
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is the Escape key
                exit = True
        except cv2.error as e:
            print(f"OpenCV error during imshow or waitKey: {e}")
            exit = True

        return exit
    
    def release(self) -> None:
        """Close all window resources.
        """
        if not self._headless:
            cv2.destroyAllWindows()

    def _initialize_window(self) -> None:
        """Create an OpenCV window that can be resized while preserving aspect ratio.

        Does nothing if already initialized or in headless mode.
        """
        # Leave if already initialized or in headless mode
        if self._headless or self._window_initialized:
            return

        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            # Ensure the window keeps the aspect ratio of the displayed frames when resized
            try:
                cv2.setWindowProperty(self._window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
            except (cv2.error, AttributeError):
                # Some OpenCV builds do not expose WND_PROP_ASPECT_RATIO; ignore if unavailable
                pass
            self._window_initialized = True
        except cv2.error as exc:
            print(f"OpenCV error creating display window: {exc}")

    def _resize_to_screen(self, frame: any) -> any:
        """Scale the frame so it comfortably fits on the screen while keeping aspect ratio.
        
        Args:
            frame (any): The frame to resize.

        Returns:
            any: The resized frame.
        """
        # Leave if headless or screen size is unknown
        if self._headless or self._screen_size is None:
            return frame

        # Get frame and screen dimensions
        frame_height, frame_width = frame.shape[:2]
        screen_width, screen_height = self._screen_size

        # Leave a bit of margin so OS window decorations fit on-screen as well
        max_width = int(screen_width * self._max_screen_coverage)
        max_height = int(screen_height * self._max_screen_coverage)

        width_scale = max_width / frame_width
        height_scale = max_height / frame_height
        scale = min(1.0, width_scale, height_scale)

        if scale < 1.0:
            new_width = max(1, int(frame_width * scale))
            new_height = max(1, int(frame_height * scale))
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            if self._window_initialized:
                cv2.resizeWindow(self._window_name, new_width, new_height)
            return resized

        # Ensure window matches the original frame size if scaling isn't needed
        if self._window_initialized:
            cv2.resizeWindow(self._window_name, frame_width, frame_height)
        return frame

    def _detect_screen_size(self) -> Optional[Tuple[int, int]]:
        """Best-effort screen size detection for dynamic scaling.

        Returns:
            Optional[Tuple[int, int]]: (width, height) of the screen in pixels, or None if detection fails.
        """
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return width, height
        except Exception as exc:  # noqa: BLE001 - broad catch to stay resilient on headless systems
            print(f"Warning: Unable to detect screen size automatically ({exc}). Using 1920x1080 fallback.")
            return 1920, 1080