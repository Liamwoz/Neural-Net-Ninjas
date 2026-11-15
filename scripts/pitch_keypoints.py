"""Shared pitch keypoint definitions for the YOLO pose homography pipeline.

Keypoint order matches the Roboflow COCO export in ``data/coco homography data``.
Pitch coordinates are in metres on a 105 x 68 m pitch with the origin at the
top-left corner flag (far touchline, left goal line), matching
``runs/pitch-template/pitch_landmarks.csv``.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
CENTRE_CIRCLE_RADIUS_M = 9.15
PENALTY_ARC_HALF_HEIGHT_M = float(np.sqrt(CENTRE_CIRCLE_RADIUS_M**2 - (16.5 - 11.0) ** 2))

KEYPOINT_NAMES = [
    "GoalCornerB",
    "GoalCircleB",
    "GoalCircleT",
    "GoalCornerT",
    "MiddleCircleT",
    "MiddleCircleB",
    "CircleEdge",
]

# Left penalty box front line (x=16.5), penalty arc intersections, centre circle
# intersections with the halfway line, and the left-most point of the centre circle.
KEYPOINT_PITCH_M = np.array(
    [
        [16.5, 54.16],
        [16.5, 34.0 + PENALTY_ARC_HALF_HEIGHT_M],
        [16.5, 34.0 - PENALTY_ARC_HALF_HEIGHT_M],
        [16.5, 13.84],
        [52.5, 34.0 - CENTRE_CIRCLE_RADIUS_M],
        [52.5, 34.0 + CENTRE_CIRCLE_RADIUS_M],
        [52.5 - CENTRE_CIRCLE_RADIUS_M, 34.0],
    ],
    dtype=np.float64,
)

# Pixel positions of the pitch outline in scripts/images.png (measured from the
# white touchlines/goal lines).
MAP_IMAGE_PITCH_BOUNDS_PX = {"left": 18.0, "right": 542.0, "top": 9.0, "bottom": 346.0}


def pitch_m_to_map_matrix(bounds: dict[str, float] = MAP_IMAGE_PITCH_BOUNDS_PX) -> np.ndarray:
    """Return a 3x3 affine matrix converting pitch metres to map-image pixels."""
    sx = (bounds["right"] - bounds["left"]) / PITCH_LENGTH_M
    sy = (bounds["bottom"] - bounds["top"]) / PITCH_WIDTH_M
    return np.array(
        [[sx, 0.0, bounds["left"]], [0.0, sy, bounds["top"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def has_non_degenerate_quad(pitch_points: np.ndarray, min_area: float = 1.0) -> bool:
    """True if some 4 points contain no 3 collinear points (needed for a homography)."""
    if len(pitch_points) < 4:
        return False

    def tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return abs(float(np.cross(b - a, c - a))) / 2.0

    for quad in combinations(pitch_points, 4):
        if all(tri_area(*tri) > min_area for tri in combinations(quad, 3)):
            return True
    return False
