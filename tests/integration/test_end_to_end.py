"""
End-to-End Integration Tests for Neptune Eye.

These tests run the actual neptune_eye.py script as a subprocess to validate
the complete application workflow in a real environment.
"""
import subprocess
import signal
import time
import sys
from pathlib import Path
import pytest

def test_end_to_end_video_input(neptune_eye_script, app_path, test_video_path, project_root):
    """Test end-to-end functionality of Neptune Eye with video input in headless mode.

    This test runs the neptune_eye.py script as a subprocess with a test video file
    and checks that a boat was detected in the output logs.
    """
   
    process = subprocess.Popen([sys.executable, "-u", str(neptune_eye_script)], # -u for unbuffered output so we can see stdout
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        cwd=str(app_path),
        text=True,
        bufsize=1,  # Line buffered
        )
        
    time.sleep(5)  # Let it run to process multiple frames

    # Gracefully terminate
    process.send_signal(signal.SIGINT)

    # Capture output to check for expected results afterwards
    try:
        stdout, _ = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, _ = process.communicate()
        pytest.fail("Process did not terminate gracefully after SIGINT")
    
    # Checks on output
    assert stdout is not None, "No output captured from the process"
    assert "Class: boat 1" in stdout, "Expected boat detection output not found"
    assert "Exiting Neptune Eye" in stdout or "Interrupted by user" in stdout, \
        "Graceful shutdown message not found"
    

def test_check_fail():
    """ Test always fails to check test framework
    """
    assert False, "Intentional failure for testing purposes"