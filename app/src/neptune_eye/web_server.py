"""
Neptune Eye - Web Server

This module provides a lightweight Flask web server to stream the live object detection feed.
It uses MJPEG streaming to display the annotated frames in a web browser.

Key Features:
- Runs in a separate thread to avoid blocking the main detection loop.
- Uses a background thread for JPEG encoding to decouple frame rate from client count.
- Provides a professional dark-mode UI for monitoring.

Thread Architecture:

+----------------+          +------------------+           +-------------------+
|  Main Thread   |          |  Encoding Thread |           | Web Server Thread |
| (Detection Loop)|          | (JPEG Converter) |           | (Flask / Clients) |
+----------------+          +------------------+           +-------------------+
        |                            |                               |
  [New Frame]                        |                               |
        |                            |                               |
  update_frame()                     |                               |
        |---(Raw Frame)------------->|                               |
        |   (Signal Event)           |                               |
        |                            |                               |
        |                      [Wait Event]                          |
        |                            |                               |
        |                     Encode to JPEG                         |
        |                            |                               |
        |                     (Signal Condition)                     |
        |                            |----(Encoded Frame)----------->|
        |                            |                               |
        |                            |                         [Wait Condition]
        |                            |                               |
        |                            |                          Send to Client
        v                            v                               v
"""

from flask import Flask, Response, render_template_string
import threading
import cv2
import time
import socket
from zeroconf import ServiceInfo, Zeroconf

class WebServer:
    """
    A threaded web server that streams video frames to a web browser.
    
    This class handles:
    1. Receiving raw frames from the main application.
    2. Encoding them to JPEG in a background thread (to save main thread CPU).
    3. Serving them via HTTP using MJPEG streaming.
    """
    def __init__(self, host='0.0.0.0', port=5005, name="neptune-eye"):
        """
        Initialize the web server.

        Args:
            host (str): The hostname to listen on. Defaults to '0.0.0.0' (all interfaces).
            port (int): The port to listen on. Defaults to 5005.
            name (str): The service name for mDNS. Defaults to "neptune-eye".
        """
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.name = name
        self.zeroconf = None
        self.service_info = None
        
        # Frame management
        self.frame = None
        self.encoded_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.frame_condition = threading.Condition()
        
        # Background encoding thread
        self.raw_frame = None
        self.new_raw_frame_event = threading.Event()
        self.encoding_thread = threading.Thread(target=self._encoding_loop, daemon=True)
        self.encoding_thread.start()

        # Define routes
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)

    def index(self):
        """
        Render the main page.
        
        Returns:
            str: The HTML content of the main page.
        """
        return render_template_string("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Neptune Eye | Live Monitor</title>
                <style>
                    :root {
                        --bg-color: #121212;
                        --card-bg: #1e1e1e;
                        --text-color: #e0e0e0;
                        --accent-color: #007acc;
                        --border-color: #333;
                    }
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                        background-color: var(--bg-color);
                        color: var(--text-color);
                        margin: 0;
                        padding: 0;
                        height: 100vh;
                        display: flex;
                        flex-direction: column;
                    }
                    header {
                        background-color: var(--card-bg);
                        padding: 1rem 2rem;
                        border-bottom: 1px solid var(--border-color);
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    }
                    .brand {
                        font-size: 1.25rem;
                        font-weight: 600;
                        color: #fff;
                        display: flex;
                        align-items: center;
                        gap: 10px;
                    }
                    .status-badge {
                        background-color: #28a745;
                        color: white;
                        padding: 4px 8px;
                        border-radius: 4px;
                        font-size: 0.8rem;
                        font-weight: 500;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    }
                    main {
                        flex: 1;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        padding: 20px;
                        overflow: hidden;
                    }
                    .video-container {
                        background-color: #000;
                        border: 1px solid var(--border-color);
                        border-radius: 8px;
                        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
                        padding: 4px;
                        max-width: 100%;
                        max-height: 100%;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }
                    img {
                        max-width: 100%;
                        max-height: 80vh;
                        width: auto;
                        height: auto;
                        display: block;
                        border-radius: 4px;
                    }
                    footer {
                        text-align: center;
                        padding: 10px;
                        font-size: 0.8rem;
                        color: #666;
                    }
                </style>
            </head>
            <body>
                <header>
                    <div class="brand">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>
                        Neptune Eye
                    </div>
                    <div class="status-badge">Live</div>
                </header>
                <main>
                    <div class="video-container">
                        <img src="{{ url_for('video_feed') }}" alt="Live Feed">
                    </div>
                </main>
                <footer>
                    Neptune Eye Object Detection System
                </footer>
            </body>
            </html>
        """)

    def video_feed(self):
        """
        Route for the video stream.
        
        Returns:
            Response: A Flask Response object containing the multipart MJPEG stream.
        """
        return Response(self.generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate(self):
        """
        Generator function that yields JPEG frames for the MJPEG stream.
        
        This function waits for new encoded frames to be available and yields them
        as multipart HTTP responses. It handles multiple clients efficiently by
        waiting on a condition variable.
        """
        while True:
            with self.frame_condition:
                self.frame_condition.wait()
                if self.encoded_frame is None:
                    continue
                current_bytes = self.encoded_frame

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + current_bytes + b'\r\n')

    def update_frame(self, frame):
        """
        Update the current frame to be displayed.
        
        This method is called by the main application loop. It copies the frame
        and signals the background encoding thread. It is designed to be non-blocking
        to minimize impact on the object detection loop.

        Args:
            frame (numpy.ndarray): The new video frame (BGR format).
        """
        # Just store the raw frame and signal the encoder thread
        # This is non-blocking for the main detection loop
        with self.frame_lock:
            self.raw_frame = frame.copy()
        self.new_raw_frame_event.set()

    def _encoding_loop(self):
        """
        Background thread to encode frames to JPEG.
        
        This loop waits for new raw frames, encodes them to JPEG, and then
        notifies all connected clients via the condition variable.
        """
        while True:
            self.new_raw_frame_event.wait()
            self.new_raw_frame_event.clear()
            
            with self.frame_lock:
                if self.raw_frame is None:
                    continue
                frame_to_encode = self.raw_frame
            
            # Encode to JPEG (CPU intensive operation)
            # We lower quality to 70% to save bandwidth
            (flag, encoded_image) = cv2.imencode(".jpg", frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            
            if flag:
                with self.frame_condition:
                    self.encoded_frame = bytearray(encoded_image)
                    self.frame_condition.notify_all()

    def _register_service(self):
        """Register the service via mDNS."""
        try:
            self.zeroconf = Zeroconf()
            
            # Get local IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # doesn't even have to be reachable
                s.connect(('10.255.255.255', 1))
                ip_address = s.getsockname()[0]
            except Exception:
                ip_address = '127.0.0.1'
            finally:
                s.close()

            desc = {'path': '/'}
            
            self.service_info = ServiceInfo(
                "_http._tcp.local.",
                f"{self.name}._http._tcp.local.",
                addresses=[socket.inet_aton(ip_address)],
                port=self.port,
                properties=desc,
                server=f"{self.name}.local.",
            )
            
            print(f"Registering mDNS service: {self.name}.local at {ip_address}:{self.port}")
            self.zeroconf.register_service(self.service_info)
            
        except Exception as e:
            print(f"Failed to register mDNS service: {e}")

    def start(self):
        """
        Start the web server in a separate daemon thread.
        """
        self.running = True
        # Disable Flask banner
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        kwargs = {'host': self.host, 'port': self.port, 'debug': False, 'use_reloader': False}
        t = threading.Thread(target=self.app.run, kwargs=kwargs)
        t.daemon = True
        t.start()
        print(f"Web server started at http://{self.host}:{self.port}")
        
        # Register mDNS service
        self._register_service()
