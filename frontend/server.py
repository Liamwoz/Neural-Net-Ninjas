"""Serve Touchline locally and run YOLO video inference."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import time
import threading
import sys
from types import SimpleNamespace
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / "model" / "weights_better_model.pt"
POSE_MODEL = ROOT.parent / "model" / "pitch_keypoints.pt"
sys.path.insert(0, str(ROOT.parent / "scripts"))
MAX_VIDEO_BYTES = 512 * 1024 * 1024
MODEL_LOCK = threading.Lock()
MODEL_INSTANCE = None
POSE_INSTANCE = None


def analyze(path: str, progress=lambda **kwargs: None) -> dict:
    global MODEL_INSTANCE, POSE_INSTANCE
    # Keep Ultralytics configuration local to a writable temporary directory.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "touchline-ultralytics"))
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from pitch_keypoints import KEYPOINT_NAMES, PITCH_LENGTH_M, PITCH_WIDTH_M
    from pitch_homography_video import CameraMotion, camera_transforms, smooth_keypoints, fit_homographies, smooth_homographies
    import run_tracking_with_jersey as tracking
    from jersey_classifier import JerseyClassifier, TEAM_RED, TEAM_GREEN, TEAM_REFEREE

    progress(stage="waiting")
    with MODEL_LOCK:
        progress(stage="loading")
        if MODEL_INSTANCE is None:
            if not MODEL.is_file():
                raise FileNotFoundError(f"Model not found: {MODEL}")
            MODEL_INSTANCE = YOLO(str(MODEL))
        if POSE_INSTANCE is None:
            if not POSE_MODEL.is_file():
                raise FileNotFoundError(f"Pitch keypoint model not found: {POSE_MODEL}")
            POSE_INSTANCE = YOLO(str(POSE_MODEL))
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            raise ValueError("Could not open the video. Try a supported MP4 file.")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Could not determine the video's frame rate.")
            total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise ValueError("Could not determine the video dimensions.")
            frames = []
            started = time.monotonic()
            motion = CameraMotion()
            raw_xy, raw_conf, steps = [], [], []
            progress(stage="keypoints", processed=0, total=total, elapsed=0)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                step = motion.update(frame)
                steps.append(np.full((3, 3), np.nan) if step is None else step)
                result = POSE_INSTANCE.predict(frame, imgsz=960, verbose=False)[0]
                xy, conf = np.zeros((len(KEYPOINT_NAMES), 2)), np.zeros(len(KEYPOINT_NAMES))
                if result.keypoints is not None and result.boxes is not None:
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    confidences = result.boxes.conf.cpu().numpy()
                    points = result.keypoints.xy[:, 0].cpu().numpy()
                    for cls, score, point in zip(classes, confidences, points):
                        if 0 <= cls < len(conf) and score > conf[cls]:
                            xy[cls], conf[cls] = point, score
                raw_xy.append(xy)
                raw_conf.append(conf)
                progress(stage="keypoints", processed=len(raw_xy), total=total, elapsed=time.monotonic() - started)
            if not raw_xy:
                raise ValueError("The video contains no readable frames.")
            progress(stage="projection", processed=len(raw_xy), total=len(raw_xy), elapsed=time.monotonic() - started)
            steps = np.asarray(steps)
            cumulative, segments = camera_transforms(steps)
            settings = SimpleNamespace(kpt_conf=0.4, max_reproj_m=2.0, gate_m=4.0, keypoint_weight=3)
            xy, conf = smooth_keypoints(np.asarray(raw_xy), np.asarray(raw_conf), cumulative, segments, settings.kpt_conf, 3)
            homographies, statuses, _ = fit_homographies(xy, conf, steps, (width, height), settings)
            homographies = smooth_homographies(homographies, cumulative, segments, (width, height), 2)
            capture.release()
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                raise ValueError("Could not reopen the video for detection.")
            total = len(homographies)
            # Each analysis starts with fresh identity and team state.
            tracking.logical_tracks.clear()
            tracking.next_display_id = 1
            classifier = JerseyClassifier(min_samples=6, max_samples=20, min_players_for_clustering=6)
            detection_started = time.monotonic()
            progress(stage="processing", processed=0, total=total, elapsed=0)
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("The video could not be read consistently during detection.")
                height, width = frame.shape[:2]
                result = MODEL_INSTANCE.predict(source=frame, conf=0.25, imgsz=1280, verbose=False)[0]
                objects = []
                player_objects, player_boxes = [], []
                if result.boxes is not None:
                    for box in result.boxes:
                        label = str(result.names[int(box.cls[0])]).lower().strip()
                        kind = "ball" if "ball" in label else "player" if label in ("player", "person") else "other"
                        if kind == "player" and float(box.conf[0]) < 0.60:
                            continue
                        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                        x1, x2 = sorted((max(0., min(1., x1 / width)), max(0., min(1., x2 / width))))
                        y1, y2 = sorted((max(0., min(1., y1 / height)), max(0., min(1., y2 / height))))
                        obj = {"type": kind, "label": label, "confidence": float(box.conf[0]), "bbox": [x1, y1, x2-x1, y2-y1]}
                        h = homographies[len(frames)]
                        if h is not None and kind in ("player", "ball"):
                            point = np.array([[[((x1 + x2) / 2) * width, ((y1 + y2) / 2 if kind == "ball" else y2) * height]]], dtype=np.float64)
                            px, py = cv2.perspectiveTransform(point, h)[0, 0]
                            if np.isfinite([px, py]).all() and 0 <= px <= PITCH_LENGTH_M and 0 <= py <= PITCH_WIDTH_M:
                                obj.update(x=float(px / PITCH_LENGTH_M * 100), y=float(py / PITCH_WIDTH_M * 100))
                        objects.append(obj)
                        if kind == "player":
                            player_objects.append(obj)
                            player_boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
                frame_index = len(frames)
                frame_number = frame_index + 1
                detections = []
                for i, player_box in enumerate(player_boxes):
                    x1, y1, x2, y2 = player_box
                    detections.append({"box": player_box, "match_position": ((x1+x2)/2, (y1+y2)/2),
                        "appearance": tracking.extract_appearance(frame, player_box),
                        "overlapping": tracking.detection_is_overlapping(i, player_boxes),
                        "good_view": tracking.is_good_appearance_view(player_box),
                        "jersey_occluded": tracking.jersey_region_is_occluded(i, player_boxes)})
                assignments = tracking.assign_stable_ids(detections, frame_number) if detections else {}
                for i, det in enumerate(detections):
                    player_id = assignments[i]
                    tracking.update_track(player_id, det["match_position"], det["appearance"], frame_number,
                                          det["good_view"] and not det["overlapping"])
                    if det["good_view"]:
                        classifier.observe_referee(player_id, frame, det["box"])
                        if not det["jersey_occluded"]:
                            classifier.update_player(player_id, frame, det["box"])
                    player_objects[i]["id"] = player_id
                classifier.end_frame(frame_number, list(assignments.values()))
                if frame_number % 10 == 0:
                    classifier.try_fit_team_model()
                for obj in player_objects:
                    team = classifier.get_team(obj["id"])
                    obj["team"] = {TEAM_RED: "team_a", TEAM_GREEN: "team_b", TEAM_REFEREE: "referee"}.get(team, "unknown")
                    obj["team_label"] = classifier.get_team_name(obj["id"]) or "Unknown team"
                frames.append({"time": frame_index / fps, "objects": objects, "projection_status": statuses[frame_index]})
                progress(stage="processing", processed=len(frames), total=total, elapsed=time.monotonic() - detection_started)
                if len(frames) >= total:
                    break
            if not frames:
                raise ValueError("The video contains no readable frames.")
            # Fill early unknown labels from confirmed identities after the full clip is analyzed.
            for item in frames:
                for obj in item["objects"]:
                    if obj.get("type") == "player" and obj.get("team") == "unknown":
                        team = classifier.get_team(obj["id"])
                        obj["team"] = {TEAM_RED: "team_a", TEAM_GREEN: "team_b", TEAM_REFEREE: "referee"}.get(team, "unknown")
                        obj["team_label"] = classifier.get_team_name(obj["id"]) or "Unknown team"
            return {"name": "YOLO and pitch projection", "duration": len(frames) / fps, "frames": frames,
                    "projected_frames": sum(h is not None for h in homographies)}
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
