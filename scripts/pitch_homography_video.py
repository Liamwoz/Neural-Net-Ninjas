"""Project YOLO detections onto a 2D pitch map using predicted pitch keypoints.

The video is processed offline in two passes:
1. Pass 1: the trained pitch pose model predicts the pitch keypoints in every frame and
   the camera motion between consecutive frames is estimated with optical flow.
2. Keypoint smoothing: each keypoint is replaced by the median of its detections in
   neighbouring frames, after moving them into the current frame with the camera motion.
   Single-frame "twitches" are outvoted without adding lag.
3. A homography (image pixels -> pitch metres) is fitted per frame from the smoothed
   keypoints, fused with the motion-propagated homography of the previous frame. This
   bridges frames where only collinear keypoints are visible (e.g. just the penalty-box line).
4. Homography smoothing: a centred, motion-compensated weighted average over neighbouring
   frames removes the remaining small jitter.
5. Pass 2: the tracked players, referee and ball from ``scripts/tracking_output.csv``
   (ground point of each box) are projected onto the map image in their team colours and
   the annotated video is written.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm
from ultralytics import YOLO

from pitch_keypoints import (
    KEYPOINT_NAMES,
    KEYPOINT_PITCH_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    has_non_degenerate_quad,
    pitch_m_to_map_matrix,
)

# Team ids from the tracking model; colours (BGR) match the jerseys in video/121364_0.mp4.
TEAM_STYLES = {
    0: ("Red team", (40, 40, 230)),
    1: ("Green team", (90, 255, 150)),
    2: ("Referee", (30, 30, 30)),
    -1: ("Unknown", (170, 170, 170)),
}
BALL_COLOR = (0, 140, 255)


@dataclass
class TrackedObject:
    object_type: str
    track_id: int | None
    team: int
    ground: tuple[float, float]
    box: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, nargs="?", default=Path("video/121364_0.mp4"))
    parser.add_argument("--pose-model", type=Path, default=Path("model/pitch_keypoints.pt"))
    parser.add_argument(
        "--tracks",
        type=Path,
        default=Path("scripts/tracking_output.csv"),
        help="Tracking CSV with players (id, team) and ball per frame",
    )
    parser.add_argument(
        "--tracks-frame-offset",
        type=int,
        default=1,
        help="Subtract this from the tracking CSV frame numbers to get 0-based video frames",
    )
    parser.add_argument("--map-image", type=Path, default=Path("scripts/images.png"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/yolo-video"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--kpt-conf", type=float, default=0.4, help="Minimum landmark detection confidence")
    parser.add_argument(
        "--max-reproj-m",
        type=float,
        default=2.0,
        help="Reject keypoint homographies whose mean reprojection error exceeds this (metres)",
    )
    parser.add_argument(
        "--keypoint-weight",
        type=int,
        default=3,
        help="How many times each keypoint is counted against the 12 motion-prior points (higher = less smoothing)",
    )
    parser.add_argument(
        "--gate-m",
        type=float,
        default=4.0,
        help="Ignore keypoints further than this (metres) from the motion prediction when they cannot be fitted alone",
    )
    parser.add_argument(
        "--keypoint-smooth-radius",
        type=int,
        default=3,
        help="Frames on each side used for the motion-compensated keypoint median (0 = off)",
    )
    parser.add_argument(
        "--homography-smooth-radius",
        type=int,
        default=2,
        help="Frames on each side used for the final homography smoothing (0 = off)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Optional limit for quick tests"
    )
    return parser.parse_args()


def normalize(h: np.ndarray) -> np.ndarray:
    return h / h[2, 2]


def load_tracks(path: Path, frame_offset: int) -> dict[int, list[TrackedObject]]:
    """Load tracked players/ball per 0-based video frame.

    The tracker reports team -1 while a track's team is not yet decided (mostly its first
    frames), so each track gets the team it is assigned most often.
    """
    tracks: dict[int, list[TrackedObject]] = defaultdict(list)
    if not path.is_file():
        print(f"Tracking CSV not found ({path}); only the homography will be visualised.")
        return tracks
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    team_votes: dict[int, Counter] = defaultdict(Counter)
    for row in rows:
        if row["id"] and row["team"] and int(float(row["team"])) >= 0:
            team_votes[int(float(row["id"]))][int(float(row["team"]))] += 1

    for row in rows:
        track_id = int(float(row["id"])) if row["id"] else None
        team = -1
        if track_id is not None and team_votes[track_id]:
            team = team_votes[track_id].most_common(1)[0][0]
        tracks[int(row["frame"]) - frame_offset].append(
            TrackedObject(
                object_type=row["object_type"],
                track_id=track_id,
                team=team,
                ground=(float(row["ground_x"]), float(row["ground_y"])),
                box=tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2")),
            )
        )
    return tracks


def reprojection_errors(h: np.ndarray, image_points: np.ndarray, pitch_points: np.ndarray) -> np.ndarray:
    projected = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), h).reshape(-1, 2)
    return np.linalg.norm(projected - pitch_points, axis=1)


def fit_keypoint_homography(
    image_points: np.ndarray, pitch_points: np.ndarray, max_reproj_m: float
) -> tuple[np.ndarray | None, np.ndarray]:
    """Least-squares homography image->pitch from keypoints alone.

    Drops the worst point once if the fit is poor. Returns the homography (or None if the
    keypoints are too few / degenerate) and a mask of the keypoints that were kept.
    """
    keep = np.ones(len(image_points), dtype=bool)
    for _ in range(2):
        if not has_non_degenerate_quad(pitch_points[keep]):
            break
        h, _ = cv2.findHomography(image_points[keep], pitch_points[keep], 0)
        if h is None:
            break
        errors = reprojection_errors(h, image_points[keep], pitch_points[keep])
        if errors.mean() <= max_reproj_m:
            return normalize(h), keep
        keep[np.flatnonzero(keep)[int(np.argmax(errors))]] = False
    return None, keep


def fuse_with_prior(
    predicted: np.ndarray,
    image_points: np.ndarray,
    pitch_points: np.ndarray,
    frame_size: tuple[int, int],
    keypoint_weight: int,
) -> np.ndarray:
    """Fit a homography to keypoints plus a grid of correspondences from the predicted one.

    The prior grid supplies any degrees of freedom the keypoints cannot (e.g. when all
    visible landmarks lie on the penalty-box line), while the keypoints, repeated
    ``keypoint_weight`` times, pull the result back onto the actual pitch lines.
    """
    width, height = frame_size
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 4) * width, np.linspace(0.35, 0.95, 3) * height)
    grid_image = np.stack([gx.ravel(), gy.ravel()], axis=1)
    grid_pitch = cv2.perspectiveTransform(grid_image.reshape(-1, 1, 2), predicted).reshape(-1, 2)
    src = np.vstack([grid_image] + [image_points] * keypoint_weight)
    dst = np.vstack([grid_pitch] + [pitch_points] * keypoint_weight)
    h, _ = cv2.findHomography(src, dst, 0)
    return normalize(h) if h is not None else predicted


class CameraMotion:
    """Estimates the homography mapping the previous frame to the current frame."""

    def __init__(self, scale: float = 0.5) -> None:
        self.scale = scale
        self.prev_gray: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=self.scale, fy=self.scale), cv2.COLOR_BGR2GRAY)
        prev, self.prev_gray = self.prev_gray, gray
        if prev is None:
            return None
        points = cv2.goodFeaturesToTrack(prev, maxCorners=600, qualityLevel=0.01, minDistance=8)
        if points is None or len(points) < 8:
            return None
        moved, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, points, None)
        good = status.reshape(-1) == 1
        if good.sum() < 8:
            return None
        f, _ = cv2.findHomography(points[good], moved[good], cv2.RANSAC, 2.0)
        if f is None:
            return None
        s = np.diag([self.scale, self.scale, 1.0])
        return normalize(np.linalg.inv(s) @ f @ s)


def camera_transforms(steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chain per-frame camera steps into cumulative transforms.

    Returns ``cumulative[t]`` (segment start frame -> frame t) and ``segment[t]``; a new
    segment starts wherever camera motion could not be estimated, and points are never
    transferred across segments.
    """
    cumulative = np.empty_like(steps)
    segment = np.zeros(len(steps), dtype=int)
    for t, step in enumerate(steps):
        if t == 0 or np.isnan(step).any():
            cumulative[t] = np.eye(3)
            segment[t] = segment[t - 1] + 1 if t else 0
        else:
            cumulative[t] = normalize(step @ cumulative[t - 1])
            segment[t] = segment[t - 1]
    return cumulative, segment


def transfer(points: np.ndarray, cumulative: np.ndarray, source: int, target: int) -> np.ndarray:
    """Move image points from frame ``source`` into frame ``target`` (same segment)."""
    m = cumulative[target] @ np.linalg.inv(cumulative[source])
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float64), m).reshape(-1, 2)


def smooth_keypoints(
    xy: np.ndarray,
    conf: np.ndarray,
    cumulative: np.ndarray,
    segment: np.ndarray,
    conf_threshold: float,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Motion-compensated temporal median of each keypoint.

    A keypoint is kept in frame t only if it was detected in at least half of the frames
    of its window, so one-off false detections and single-frame jumps are rejected.
    """
    detected = conf >= conf_threshold
    if radius <= 0:
        return xy.copy(), np.where(detected, conf, 0.0)
    smoothed_xy = np.zeros_like(xy)
    smoothed_conf = np.zeros_like(conf)
    frames = len(xy)
    for t in range(frames):
        window = [s for s in range(max(0, t - radius), min(frames, t + radius + 1)) if segment[s] == segment[t]]
        for j in range(xy.shape[1]):
            support = [s for s in window if detected[s, j]]
            if len(support) * 2 < len(window):
                continue
            moved = np.array([transfer(xy[s, j], cumulative, s, t)[0] for s in support])
            smoothed_xy[t, j] = np.median(moved, axis=0)
            smoothed_conf[t, j] = conf[support, j].mean()
    return smoothed_xy, smoothed_conf


def smooth_homographies(
    homographies: list[np.ndarray | None],
    cumulative: np.ndarray,
    segment: np.ndarray,
    frame_size: tuple[int, int],
    radius: int,
) -> list[np.ndarray | None]:
    """Centred, motion-compensated Gaussian average of the image->pitch homographies.

    For frame t, a grid of image points is moved into each neighbouring frame s, projected
    to the pitch with H_s, and the weighted mean pitch positions are refitted into H_t.
    """
    if radius <= 0:
        return homographies
    width, height = frame_size
    gx, gy = np.meshgrid(np.linspace(0.05, 0.95, 5) * width, np.linspace(0.3, 0.95, 4) * height)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    sigma = max(radius / 2.0, 0.5)
    smoothed: list[np.ndarray | None] = []
    for t, h_t in enumerate(homographies):
        if h_t is None:
            smoothed.append(None)
            continue
        total = np.zeros_like(grid)
        weight_sum = 0.0
        for s in range(max(0, t - radius), min(len(homographies), t + radius + 1)):
            h_s = homographies[s]
            if h_s is None or segment[s] != segment[t]:
                continue
            w = float(np.exp(-0.5 * ((s - t) / sigma) ** 2))
            pitch = cv2.perspectiveTransform(transfer(grid, cumulative, t, s).reshape(-1, 1, 2), h_s)
            total += w * pitch.reshape(-1, 2)
            weight_sum += w
        h, _ = cv2.findHomography(grid, total / weight_sum, 0)
        smoothed.append(normalize(h) if h is not None else h_t)
    return smoothed


def fit_homographies(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    steps: np.ndarray,
    frame_size: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray | None], list[str], list[int]]:
    """Per-frame keypoint homography, fused with the motion-propagated previous one."""
    homographies: list[np.ndarray | None] = []
    statuses: list[str] = []
    used_counts: list[int] = []
    image_to_pitch: np.ndarray | None = None
    for t in range(len(keypoints_xy)):
        predicted = None
        if image_to_pitch is not None and not np.isnan(steps[t]).any():
            predicted = normalize(image_to_pitch @ np.linalg.inv(steps[t]))

        keep = keypoints_conf[t] >= args.kpt_conf
        image_points = keypoints_xy[t][keep].astype(np.float64)
        pitch_points = KEYPOINT_PITCH_M[keep]

        measured, used = fit_keypoint_homography(image_points, pitch_points, args.max_reproj_m)
        if measured is None and predicted is not None and len(image_points):
            # Too few / collinear keypoints: gate them against the motion prediction instead.
            used = reprojection_errors(predicted, image_points, pitch_points) <= args.gate_m
        image_points, pitch_points = image_points[used], pitch_points[used]

        if predicted is not None and len(image_points):
            image_to_pitch = fuse_with_prior(
                predicted, image_points, pitch_points, frame_size, args.keypoint_weight
            )
            status = "keypoints+motion" if measured is not None else "partial-keypoints+motion"
        elif measured is not None:
            image_to_pitch = measured
            status = "keypoints"
        elif predicted is not None:
            image_to_pitch = predicted
            status = "motion"
        else:
            image_to_pitch = None
            status = "none"
        homographies.append(image_to_pitch)
        statuses.append(status)
        used_counts.append(len(image_points))
    return homographies, statuses, used_counts


def collect_measurements(
    video: Path, model: YOLO, imgsz: int, total_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pass 1: raw keypoints (best detection per landmark class) and camera steps per frame."""
    capture = cv2.VideoCapture(str(video))
    motion = CameraMotion()
    count = len(KEYPOINT_NAMES)
    all_xy, all_conf, steps = [], [], []
    for _ in tqdm(range(total_frames), desc="Pass 1: keypoints + camera motion", unit="frame"):
        ok, frame = capture.read()
        if not ok:
            break
        step = motion.update(frame)
        steps.append(np.full((3, 3), np.nan) if step is None else step)

        result = model.predict(frame, imgsz=imgsz, verbose=False)[0]
        xy = np.zeros((count, 2))
        conf = np.zeros(count)
        if result.keypoints is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            box_conf = result.boxes.conf.cpu().numpy()
            points = result.keypoints.xy[:, 0].cpu().numpy()
            for cls, c, point in zip(classes, box_conf, points):
                if c > conf[cls]:
                    conf[cls] = c
                    xy[cls] = point
        all_xy.append(xy)
        all_conf.append(conf)
    capture.release()
    return np.array(all_xy), np.array(all_conf), np.array(steps)


def draw_pitch_overlay(frame: np.ndarray, lines_mask: np.ndarray, image_to_map: np.ndarray) -> None:
    map_to_image = np.linalg.inv(image_to_map)
    warped = cv2.warpPerspective(lines_mask, map_to_image, (frame.shape[1], frame.shape[0]))
    frame[warped > 0] = (255, 0, 255)


def text_color_for(background: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = background
    return (0, 0, 0) if 0.114 * b + 0.587 * g + 0.299 * r > 140 else (255, 255, 255)


def draw_player(frame: np.ndarray, obj: TrackedObject) -> None:
    color = TEAM_STYLES[obj.team][1]
    x1, y1, x2, y2 = (int(round(v)) for v in obj.box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if obj.track_id is not None:
        label = f"#{obj.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color_for(color), 1,
                    cv2.LINE_AA)


def draw_ball(frame: np.ndarray, obj: TrackedObject) -> None:
    """Ball gets a padded box, a ring and a pointer above it so it stands out from players."""
    x1, y1, x2, y2 = obj.box
    cx, cy = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
    pad = 6
    cv2.rectangle(frame, (int(x1) - pad, int(y1) - pad), (int(x2) + pad, int(y2) + pad), BALL_COLOR, 3)
    cv2.circle(frame, (cx, cy), 24, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 27, BALL_COLOR, 3, cv2.LINE_AA)
    tip_y = cy - 32
    pointer = np.array([[cx, tip_y], [cx - 12, tip_y - 20], [cx + 12, tip_y - 20]], dtype=np.int32)
    cv2.fillPoly(frame, [pointer], BALL_COLOR)
    cv2.polylines(frame, [pointer], True, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "BALL", (cx - 24, tip_y - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, "BALL", (cx - 24, tip_y - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BALL_COLOR, 2, cv2.LINE_AA)


def draw_map_marker(minimap: np.ndarray, point: np.ndarray, obj: TrackedObject) -> None:
    center = (int(round(point[0])), int(round(point[1])))
    if obj.object_type == "ball":
        cv2.circle(minimap, center, 11, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(minimap, center, 9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(minimap, center, 6, BALL_COLOR, -1, cv2.LINE_AA)
        cv2.circle(minimap, center, 6, (0, 0, 0), 1, cv2.LINE_AA)
        return
    color = TEAM_STYLES[obj.team][1]
    outline = (255, 255, 255) if obj.team == 2 else (0, 0, 0)
    cv2.circle(minimap, center, 5, color, -1, cv2.LINE_AA)
    cv2.circle(minimap, center, 5, outline, 1, cv2.LINE_AA)


def build_legend(width: int, teams: list[int]) -> np.ndarray:
    legend = np.full((28, width, 3), 95, dtype=np.uint8)
    x = 10
    entries = [(TEAM_STYLES[t][0], TrackedObject("player", None, t, (0, 0), (0, 0, 0, 0))) for t in teams]
    entries.append(("Ball", TrackedObject("ball", None, -1, (0, 0), (0, 0, 0, 0))))
    for name, obj in entries:
        draw_map_marker(legend, np.array([x + 8, 14]), obj)
        cv2.putText(legend, name, (x + 22, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        x += 22 + cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] + 16
    return legend


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {args.video}")
    if not args.pose_model.is_file():
        raise FileNotFoundError(
            f"Pose model not found: {args.pose_model}. Train it with scripts/train_pitch_pose.py."
        )
    tracks = load_tracks(args.tracks, args.tracks_frame_offset)

    map_image = cv2.imread(str(args.map_image))
    if map_image is None:
        raise FileNotFoundError(f"Could not read map image: {args.map_image}")
    lines_mask = cv2.dilate(
        (map_image.min(axis=2) > 200).astype(np.uint8) * 255, np.ones((2, 2), np.uint8)
    )
    pitch_to_map = pitch_m_to_map_matrix()
    teams_present = sorted(
        {obj.team for objs in tracks.values() for obj in objs if obj.object_type == "player"},
        key=lambda t: (t < 0, t),
    )
    legend = build_legend(map_image.shape[1], teams_present)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_video = args.output_dir / f"{args.video.stem}_homography.mp4"
    positions_csv = args.output_dir / f"{args.video.stem}_pitch_positions.csv"
    homography_csv = args.output_dir / f"{args.video.stem}_homographies.csv"

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if args.max_frames is not None:
        total_frames = min(total_frames, args.max_frames)

    model = YOLO(str(args.pose_model))
    raw_xy, raw_conf, steps = collect_measurements(args.video, model, args.imgsz, total_frames)
    cumulative, segment = camera_transforms(steps)
    keypoints_xy, keypoints_conf = smooth_keypoints(
        raw_xy, raw_conf, cumulative, segment, args.kpt_conf, args.keypoint_smooth_radius
    )
    homographies, statuses, used_counts = fit_homographies(
        keypoints_xy, keypoints_conf, steps, (width, height), args
    )
    homographies = smooth_homographies(
        homographies, cumulative, segment, (width, height), args.homography_smooth_radius
    )

    capture = cv2.VideoCapture(str(args.video))
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_video}")

    with positions_csv.open("w", newline="", encoding="utf-8") as pos_file, homography_csv.open(
        "w", newline="", encoding="utf-8"
    ) as h_file:
        pos_writer = csv.writer(pos_file)
        pos_writer.writerow(
            ["frame", "object_type", "track_id", "team", "team_name", "image_x", "image_y",
             "pitch_x_m", "pitch_y_m", "map_x", "map_y"]
        )
        h_writer = csv.writer(h_file)
        h_writer.writerow(
            ["frame", "status", "keypoints_used", "reproj_error_m"] + [f"h{i}{j}" for i in range(3) for j in range(3)]
        )

        for frame_number in tqdm(range(len(homographies)), desc="Pass 2: render", unit="frame"):
            ok, frame = capture.read()
            if not ok:
                break
            image_to_pitch = homographies[frame_number]
            status = statuses[frame_number]
            visible = keypoints_conf[frame_number] >= args.kpt_conf

            reproj = ""
            if image_to_pitch is not None and visible.any():
                errors = reprojection_errors(
                    image_to_pitch, keypoints_xy[frame_number][visible], KEYPOINT_PITCH_M[visible]
                )
                reproj = f"{errors.mean():.3f}"
            h_values = image_to_pitch.reshape(-1).tolist() if image_to_pitch is not None else [""] * 9
            h_writer.writerow(
                [frame_number, status, used_counts[frame_number], reproj]
                + [f"{v:.8g}" if v != "" else "" for v in h_values]
            )

            minimap = map_image.copy()
            if image_to_pitch is not None:
                draw_pitch_overlay(frame, lines_mask, pitch_to_map @ image_to_pitch)
            # Players first, ball last so it is drawn on top.
            objects = sorted(tracks.get(frame_number, []), key=lambda o: o.object_type == "ball")
            for obj in objects:
                if obj.object_type == "ball":
                    draw_ball(frame, obj)
                else:
                    draw_player(frame, obj)

            if image_to_pitch is not None:
                image_to_map = pitch_to_map @ image_to_pitch
                if objects:
                    ground = np.array([obj.ground for obj in objects], dtype=np.float64).reshape(-1, 1, 2)
                    pitch_xy = cv2.perspectiveTransform(ground, image_to_pitch).reshape(-1, 2)
                    map_xy = cv2.perspectiveTransform(ground, image_to_map).reshape(-1, 2)
                    for obj, p, m in zip(objects, pitch_xy, map_xy):
                        on_pitch = -3 <= p[0] <= PITCH_LENGTH_M + 3 and -3 <= p[1] <= PITCH_WIDTH_M + 3
                        if not on_pitch:
                            continue
                        is_ball = obj.object_type == "ball"
                        pos_writer.writerow(
                            [frame_number, obj.object_type, "" if obj.track_id is None else obj.track_id,
                             "" if is_ball else obj.team, "Ball" if is_ball else TEAM_STYLES[obj.team][0],
                             f"{obj.ground[0]:.1f}", f"{obj.ground[1]:.1f}",
                             f"{p[0]:.2f}", f"{p[1]:.2f}", f"{m[0]:.1f}", f"{m[1]:.1f}"]
                        )
                        draw_map_marker(minimap, m, obj)

                # Camera field of view on the minimap.
                corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
                fov = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), image_to_map).reshape(-1, 2)
                cv2.polylines(minimap, [fov.astype(np.int32)], True, (255, 0, 255), 1)

            # Raw detections as small hollow circles, smoothed keypoints as filled dots.
            for (x, y), c in zip(raw_xy[frame_number], raw_conf[frame_number]):
                if c >= args.kpt_conf:
                    cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 255), 2)
            for name, (x, y), c in zip(KEYPOINT_NAMES, keypoints_xy[frame_number], keypoints_conf[frame_number]):
                if c >= args.kpt_conf:
                    cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), -1)
                    cv2.putText(frame, f"{name} {c:.2f}", (int(x) + 10, int(y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(frame, f"frame {frame_number} | H: {status}", (20, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            panel = np.vstack([minimap, legend])
            mh, mw = panel.shape[:2]
            x0, y0 = width - mw - 20, 20
            frame[y0 - 2 : y0 + mh + 2, x0 - 2 : x0 + mw + 2] = 0
            frame[y0 : y0 + mh, x0 : x0 + mw] = panel
            writer.write(frame)

    capture.release()
    writer.release()
    print("Homography source per frame:", dict(Counter(statuses)))
    print(f"Annotated video: {output_video}")
    print(f"Pitch positions CSV: {positions_csv}")
    print(f"Homographies CSV: {homography_csv}")


if __name__ == "__main__":
    main()
