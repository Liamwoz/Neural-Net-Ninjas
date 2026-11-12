import cv2
import numpy as np


def draw_player_box(frame, box):
    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (255, 255, 255),
        1
    )


def draw_track_id(frame, track_id, box):
    x1, y1, _, _ = box

    cv2.putText(
        frame,
        f"ID {track_id}",
        (int(x1), int(y1) - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


def draw_position(frame, position):
    x, y = position

    cv2.circle(
        frame,
        (int(x), int(y)),
        3,
        (255, 255, 255),
        -1
    )


def draw_trajectory(frame, history):
    if len(history) < 2:
        return

    pts = np.array(
        history,
        dtype=np.int32
    ).reshape((-1, 1, 2))

    cv2.polylines(
        frame,
        [pts],
        False,
        (255, 255, 255),
        1
    )