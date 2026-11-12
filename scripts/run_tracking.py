from ultralytics import YOLO

import cv2
import numpy as np

from collections import defaultdict

from scipy.optimize import linear_sum_assignment

import math

from jersey_classifier import JerseyClassifier, TEAM_UNKNOWN


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "../model/weights.pt"
VIDEO_PATH = "../data/121364_0.mp4"

PLAYER_CLASS = 1

TRACKER_CONFIG = "custom_botsort.yaml"


# =========================================================
# TRACKING SETTINGS
# =========================================================

MAX_HISTORY = 8

# How many frames a stable player can disappear for
MAX_LOST_FRAMES = 30

# Maximum initial spatial search radius
BASE_MATCH_DISTANCE = 90

# Velocity smoothing
VELOCITY_SMOOTHING = 0.7


# =========================================================
# MATCHING WEIGHTS
# =========================================================

# Position still matters
POSITION_WEIGHT = 0.45

# Appearance is more important during tackles / crossings
APPEARANCE_WEIGHT = 0.55

# Reject a match if appearance is extremely different
MAX_APPEARANCE_DISTANCE = 0.75

# Cost used for impossible matches
INVALID_COST = 100000.0


# =========================================================
# APPEARANCE SETTINGS
# =========================================================

# EMA update of stored jersey appearance
APPEARANCE_UPDATE_RATE = 0.10

# Don't update appearance if current crop is wildly
# different from the stored player appearance
MAX_APPEARANCE_UPDATE_DISTANCE = 0.45


# =========================================================
# LOAD YOLO
# =========================================================

model = YOLO(MODEL_PATH)

print(model.names)

# Expected:
# {0: 'ball', 1: 'player', 2: 'soccer ball'}


# =========================================================
# JERSEY CLASSIFIER
# =========================================================

jersey_classifier = JerseyClassifier(
    min_samples=6,
    max_samples=20,
    min_players_for_clustering=6
)


# =========================================================
# STATE
# =========================================================

next_display_id = 1

frame_number = 0


# =========================================================
# STABLE PLAYER MEMORY
#
# IMPORTANT:
#
# These IDs are OUR IDs.
# They are independent from BoT-SORT IDs.
# =========================================================

logical_tracks = {}


# Example:
#
# logical_tracks[12] = {
#
#     "position": (cx, cy),
#
#     "velocity": (vx, vy),
#
#     "last_seen": 250,
#
#     "appearance": histogram
# }
