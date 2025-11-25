from flask import Flask, Response, render_template_string
import threading
import cv2
import time

class WebServer:
    def __init__(self, host='0.0.0.0', port=5005):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        
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
        return Response(self.generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate(self):
        while True:
            with self.frame_condition:
                self.frame_condition.wait()
                if self.encoded_frame is None:
                    continue
                current_bytes = self.encoded_frame

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + current_bytes + b'\r\n')

    def update_frame(self, frame):
        # Just store the raw frame and signal the encoder thread
        # This is non-blocking for the main detection loop
        with self.frame_lock:
            self.raw_frame = frame.copy()
        self.new_raw_frame_event.set()

    def _encoding_loop(self):
        """Background thread to encode frames to JPEG."""
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

    def start(self):
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
