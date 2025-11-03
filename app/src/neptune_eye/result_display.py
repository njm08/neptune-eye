"""
This module provides functions for displaying the results of object detection.
"""

import sys
import time
import cv2

class ResultDisplay:
    """Handles the display of detection results.
    """

    def __init__(self, headless: bool = True) -> None:
        """Initialize the ResultDisplay.

        Args:
            headless (bool): If True, run in headless mode without displaying images.
        """
        self.headless = headless
        self.previous_line_count = 0
        self.last_detection_time = None
        if self.headless: 
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
        if self.headless:
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
            for _ in range(self.previous_line_count):
                sys.stdout.write("\033[F\033[K")
            # Print new detection results
            for line in lines_to_print:
                print(line)
            self.previous_line_count = len(lines_to_print)
            self.last_detection_time = current_time # Store the time of the this detection
        # Clear the detections after 1 second of no new detections
        elif self.last_detection_time is not None and self.previous_line_count > 0 and current_time - self.last_detection_time > 1.0:
            for _ in range(self.previous_line_count):
                sys.stdout.write("\033[F\033[K")
            sys.stdout.flush()
            self.previous_line_count = 0
            self.last_detection_time = None
    
    def _display_gui(self, results: any) -> bool:
        """Display results in GUI mode (e.g., show images with overlays).

        Args:
            results (any): The detection results to display.

        Returns:
            bool: True if the display should be exited. False otherwise.
        """

        # Draw results on frame
        annotated_frame = results[0].plot()
        exit = False
        try:
            cv2.imshow("Neptune Eye", annotated_frame)
            
            # Check for exit condition
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is the Escape key
                exit = True
        except cv2.error as e:
            print(f"OpenCV error during imshow or waitKey: {e}")
            print(f"Exiting display loop.")
            exit = True

        return exit
    
    def release(self) -> None:
        """Close all window resources.
        """
        if self.headless:
            cv2.destroyAllWindows()