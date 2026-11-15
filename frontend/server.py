"""Serve Touchline locally and run YOLO video inference."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import time
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / "model" / "weights_better_model.pt"
MAX_VIDEO_BYTES = 512 * 1024 * 1024
MODEL_LOCK = threading.Lock()
MODEL_INSTANCE = None


def analyze(path: str, progress=lambda **kwargs: None) -> dict:
    global MODEL_INSTANCE
    # Keep Ultralytics configuration local to a writable temporary directory.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "touchline-ultralytics"))
    import cv2
    from ultralytics import YOLO

    progress(stage="waiting")
    with MODEL_LOCK:
        progress(stage="loading")
        if MODEL_INSTANCE is None:
            if not MODEL.is_file():
                raise FileNotFoundError(f"Model not found: {MODEL}")
            MODEL_INSTANCE = YOLO(str(MODEL))
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            raise ValueError("Could not open the video. Try a supported MP4 file.")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Could not determine the video's frame rate.")
            total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            frames = []
            started = time.monotonic()
            progress(stage="processing", processed=0, total=total, elapsed=0)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                result = MODEL_INSTANCE.predict(source=frame, conf=0.25, imgsz=1280, verbose=False)[0]
                objects = []
                if result.boxes is not None:
                    for box in result.boxes:
                        label = str(result.names[int(box.cls[0])]).lower().strip()
                        kind = "ball" if "ball" in label else "player" if label in ("player", "person") else "other"
                        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                        x1, x2 = sorted((max(0., min(1., x1 / width)), max(0., min(1., x2 / width))))
                        y1, y2 = sorted((max(0., min(1., y1 / height)), max(0., min(1., y2 / height))))
                        objects.append({"type": kind, "label": label, "confidence": float(box.conf[0]), "bbox": [x1, y1, x2-x1, y2-y1]})
                frames.append({"time": len(frames) / fps, "objects": objects})
                progress(stage="processing", processed=len(frames), total=total, elapsed=time.monotonic() - started)
            if not frames:
                raise ValueError("The video contains no readable frames.")
            return {"name": "YOLO analysis", "duration": len(frames) / fps, "frames": frames}
        finally:
            capture.release()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_POST(self):
        if self.path != "/api/analyze":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > MAX_VIDEO_BYTES:
                raise ValueError("Choose a video up to 512 MB.")
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as upload:
                path = upload.name
                remaining = size
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("The video upload was incomplete.")
                    upload.write(chunk)
                    remaining -= len(chunk)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            stream_started = True
            def send(payload):
                self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
                self.wfile.flush()
            result = analyze(path, lambda **status: send({"event": "progress", **status}))
            send({"event": "complete", "result": result})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            if locals().get("stream_started"):
                try:
                    send({"event": "error", "error": str(exc)})
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.respond(400, {"error": str(exc)})
        finally:
            if 'path' in locals():
                Path(path).unlink(missing_ok=True)

    def respond(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("Touchline is running at http://127.0.0.1:8000 (Ctrl+C to stop)", flush=True)
    server.serve_forever()
