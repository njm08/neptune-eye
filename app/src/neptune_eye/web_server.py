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
        self.frame_event = threading.Event()

        # Define routes
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)

    def index(self):
        return render_template_string("""
            <html>
            <head>
                <title>Neptune Eye Live Feed</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; background-color: #f0f0f0; }
                    h1 { color: #333; }
                    img { border: 5px solid #333; border-radius: 10px; max-width: 100%; height: auto; }
                </style>
            </head>
            <body>
                <h1>Neptune Eye Live Feed</h1>
                <img src="{{ url_for('video_feed') }}">
            </body>
            </html>
        """)

    def video_feed(self):
        return Response(self.generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate(self):
        while True:
            # Wait for a new frame to be available
            self.frame_event.wait()
            
            with self.frame_lock:
                if self.encoded_frame is None:
                    continue
                current_bytes = self.encoded_frame

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + current_bytes + b'\r\n')
            
            # Simple rate limiting to avoid sending the same frame too many times 
            # if the client is faster than the producer
            time.sleep(0.01) 

    def update_frame(self, frame):
        # Encode the frame immediately in the update thread (or offload if needed)
        # Encoding once here is more efficient than encoding per-client in generate()
        # We also lower quality to 70% to save bandwidth
        (flag, encoded_image) = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        
        if flag:
            with self.frame_lock:
                self.encoded_frame = bytearray(encoded_image)
            # Notify all waiting clients that a new frame is ready
            self.frame_event.set()
            self.frame_event.clear()

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
