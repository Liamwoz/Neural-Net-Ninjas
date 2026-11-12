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


# =========================================================
# TRAJECTORY
# =========================================================

track_history = defaultdict(list)


# =========================================================
# COLORS
# =========================================================

def get_track_color(track_id):

    colors = [
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
        (0, 255, 0),
        (255, 128, 0),
        (128, 0, 255),
        (0, 128, 255),
    ]

    return colors[
        track_id % len(colors)
    ]


# =========================================================
# DISTANCE
# =========================================================

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        +
        (p1[1] - p2[1]) ** 2
    )


# =========================================================
# TRAJECTORY SMOOTHING
# =========================================================

def smooth_history(
    history,
    window=3
):

    if len(history) < window:
        return history

    smoothed = []

    for i in range(
        len(history)
    ):

        start = max(
            0,
            i - window + 1
        )

        subset = history[
            start:i + 1
        ]

        avg_x = (
            sum(p[0] for p in subset)
            /
            len(subset)
        )

        avg_y = (
            sum(p[1] for p in subset)
            /
            len(subset)
        )

        smoothed.append(
            (
                avg_x,
                avg_y
            )
        )

    return smoothed


# =========================================================
# IOU
# =========================================================

def calculate_iou(
    box_a,
    box_b
):

    x1a, y1a, x2a, y2a = box_a
    x1b, y1b, x2b, y2b = box_b

    inter_x1 = max(
        x1a,
        x1b
    )

    inter_y1 = max(
        y1a,
        y1b
    )

    inter_x2 = min(
        x2a,
        x2b
    )

    inter_y2 = min(
        y2a,
        y2b
    )


    inter_w = max(
        0,
        inter_x2 - inter_x1
    )

    inter_h = max(
        0,
        inter_y2 - inter_y1
    )


    intersection = (
        inter_w * inter_h
    )


    area_a = max(
        0,
        x2a - x1a
    ) * max(
        0,
        y2a - y1a
    )


    area_b = max(
        0,
        x2b - x1b
    ) * max(
        0,
        y2b - y1b
    )


    union = (
        area_a
        +
        area_b
        -
        intersection
    )


    if union <= 0:
        return 0.0


    return (
        intersection
        /
        union
    )


# =========================================================
# CHECK OVERLAP
# =========================================================

def detection_is_overlapping(
    index,
    boxes,
    threshold=0.10
):

    current_box = boxes[
        index
    ]


    for other_index, other_box in enumerate(
        boxes
    ):

        if (
            other_index
            ==
            index
        ):
            continue


        if (
            calculate_iou(
                current_box,
                other_box
            )
            >
            threshold
        ):

            return True


    return False


# =========================================================
# CHECK GOOD APPEARANCE VIEW
# =========================================================

def is_good_appearance_view(
    box
):

    x1, y1, x2, y2 = box

    width = (
        x2 - x1
    )

    height = (
        y2 - y1
    )


    if (
        width <= 0
        or
        height <= 0
    ):

        return False


    if height < 30:
        return False


    # Do not LEARN appearance while player is
    # completely horizontal.
    #
    # We can still USE appearance from that frame
    # for matching.
    if (
        height / width
        <
        1.15
    ):

        return False


    return True


# =========================================================
# APPEARANCE CROP
# =========================================================

def get_appearance_crop(
    frame,
    box
):

    x1, y1, x2, y2 = map(
        int,
        box
    )


    width = (
        x2 - x1
    )

    height = (
        y2 - y1
    )


    if (
        width <= 5
        or
        height <= 5
    ):

        return None


    # -----------------------------------------------------
    # PLAYER IS MOSTLY UPRIGHT
    # -----------------------------------------------------

    if (
        height / width
        >=
        1.15
    ):

        crop_x1 = (
            x1
            +
            int(
                width * 0.18
            )
        )

        crop_x2 = (
            x2
            -
            int(
                width * 0.18
            )
        )

        # Upper / middle body
        crop_y1 = (
            y1
            +
            int(
                height * 0.10
            )
        )

        crop_y2 = (
            y1
            +
            int(
                height * 0.62
            )
        )


    # -----------------------------------------------------
    # FALLEN PLAYER
    #
    # Don't assume "top half = jersey" anymore.
    # Use more of the box.
    # -----------------------------------------------------

    else:

        crop_x1 = (
            x1
            +
            int(
                width * 0.08
            )
        )

        crop_x2 = (
            x2
            -
            int(
                width * 0.08
            )
        )

        crop_y1 = (
            y1
            +
            int(
                height * 0.08
            )
        )

        crop_y2 = (
            y2
            -
            int(
                height * 0.08
            )
        )


    crop_x1 = max(
        0,
        crop_x1
    )

    crop_y1 = max(
        0,
        crop_y1
    )


    crop_x2 = min(
        frame.shape[1],
        crop_x2
    )

    crop_y2 = min(
        frame.shape[0],
        crop_y2
    )


    crop = frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]


    if crop.size == 0:
        return None


    return crop


# =========================================================
# HSV APPEARANCE HISTOGRAM
# =========================================================

def extract_appearance(
    frame,
    box
):
    """
    Creates a color fingerprint.

    This is NOT team classification.

    It represents what the player's clothing
    currently looks like.
    """

    crop = get_appearance_crop(
        frame,
        box
    )


    if crop is None:
        return None


    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV
    )


    # -----------------------------------------------------
    # Ignore extremely dark / grey pixels
    #
    # Keeps stronger clothing colors.
    # -----------------------------------------------------

    saturation = (
        hsv[:, :, 1]
    )

    brightness = (
        hsv[:, :, 2]
    )


    mask = np.uint8(
        (
            (saturation > 40)
            &
            (brightness > 40)
        )
        *
        255
    )


    if (
        cv2.countNonZero(mask)
        <
        5
    ):

        return None


    # =====================================================
    # 2D HUE + SATURATION HISTOGRAM
    # =====================================================

    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        mask,
        [18, 8],
        [
            0,
            180,
            0,
            256
        ]
    )


    histogram = cv2.normalize(
        histogram,
        histogram
    )


    return histogram.astype(
        np.float32
    )


# =========================================================
# APPEARANCE DISTANCE
# =========================================================

def appearance_distance(
    appearance_a,
    appearance_b
):

    if (
        appearance_a is None
        or
        appearance_b is None
    ):

        # Neutral / unknown
        return 0.5


    d = cv2.compareHist(
        appearance_a,
        appearance_b,
        cv2.HISTCMP_BHATTACHARYYA
    )


    return float(
        d
    )


# =========================================================
# PREDICT POSITION
# =========================================================

def predicted_position(
    track,
    current_frame
):

    gap = (
        current_frame
        -
        track["last_seen"]
    )


    px, py = (
        track["position"]
    )


    vx, vy = (
        track["velocity"]
    )


    predicted_x = (
        px
        +
        vx * gap
    )


    predicted_y = (
        py
        +
        vy * gap
    )


    return (
        predicted_x,
        predicted_y
    )


# =========================================================
# CREATE STABLE TRACK
# =========================================================

def create_track(
    position,
    appearance,
    current_frame
):

    global next_display_id


    display_id = (
        next_display_id
    )


    next_display_id += 1


    logical_tracks[
        display_id
    ] = {

        "position":
            position,

        "velocity":
            (
                0.0,
                0.0
            ),

        "last_seen":
            current_frame,

        "appearance":
            appearance
    }


    print(
        f"Created stable "
        f"player {display_id}"
    )


    return display_id


# =========================================================
# UPDATE STABLE TRACK
# =========================================================

def update_track(
    display_id,
    position,
    current_appearance,
    current_frame,
    allow_appearance_update
):

    track = logical_tracks[
        display_id
    ]


    old_x, old_y = (
        track["position"]
    )


    new_x, new_y = (
        position
    )


    frame_gap = max(
        1,
        current_frame
        -
        track["last_seen"]
    )


    measured_vx = (
        new_x - old_x
    ) / frame_gap


    measured_vy = (
        new_y - old_y
    ) / frame_gap


    old_vx, old_vy = (
        track["velocity"]
    )


    vx = (
        VELOCITY_SMOOTHING
        * old_vx
        +
        (
            1
            -
            VELOCITY_SMOOTHING
        )
        * measured_vx
    )


    vy = (
        VELOCITY_SMOOTHING
        * old_vy
        +
        (
            1
            -
            VELOCITY_SMOOTHING
        )
        * measured_vy
    )


    track["position"] = (
        position
    )


    track["velocity"] = (
        vx,
        vy
    )


    track["last_seen"] = (
        current_frame
    )


    # =====================================================
    # UPDATE APPEARANCE REFERENCE
    # =====================================================

    if (
        allow_appearance_update
        and
        current_appearance
        is not None
    ):


        old_appearance = (
            track["appearance"]
        )


        if old_appearance is None:

            track[
                "appearance"
            ] = (
                current_appearance.copy()
            )


        else:

            app_dist = (
                appearance_distance(
                    old_appearance,
                    current_appearance
                )
            )


            # Only update if new appearance is
            # reasonably consistent.
            #
            # This prevents a tackle / overlap from
            # corrupting the saved player appearance.
            if (
                app_dist
                <=
                MAX_APPEARANCE_UPDATE_DISTANCE
            ):

                blended = (

                    (
                        1
                        -
                        APPEARANCE_UPDATE_RATE
                    )
                    *
                    old_appearance

                    +

                    APPEARANCE_UPDATE_RATE
                    *
                    current_appearance
                )


                blended = cv2.normalize(
                    blended,
                    blended
                )


                track[
                    "appearance"
                ] = blended


# =========================================================
# GLOBAL STABLE-ID ASSIGNMENT
# =========================================================

def assign_stable_ids(
    detections,
    current_frame
):
    """
    THIS is now the identity authority.

    We intentionally ignore BoT-SORT's identity here.

    All current detections are globally matched against
    all recently seen stable players.

    Cost =
        predicted position
        +
        appearance similarity
    """

    assignments = {}


    # =====================================================
    # CANDIDATE EXISTING PLAYERS
    # =====================================================

    candidate_ids = []


    for (
        display_id,
        track
    ) in logical_tracks.items():


        gap = (
            current_frame
            -
            track["last_seen"]
        )


        if (
            gap >= 1
            and
            gap <= MAX_LOST_FRAMES
        ):

            candidate_ids.append(
                display_id
            )


    # =====================================================
    # FIRST FRAME / NO OLD TRACKS
    # =====================================================

    if len(candidate_ids) == 0:

        for detection_index, det in enumerate(
            detections
        ):

            display_id = create_track(
                det["match_position"],
                det["appearance"],
                current_frame
            )


            assignments[
                detection_index
            ] = display_id


        return assignments


    # =====================================================
    # COST MATRIX
    # =====================================================

    cost_matrix = np.full(
        (
            len(detections),
            len(candidate_ids)
        ),
        INVALID_COST,
        dtype=np.float32
    )


    for detection_index, det in enumerate(
        detections
    ):


        detection_position = (
            det["match_position"]
        )


        detection_appearance = (
            det["appearance"]
        )


        for track_index, display_id in enumerate(
            candidate_ids
        ):


            track = logical_tracks[
                display_id
            ]


            gap = (
                current_frame
                -
                track["last_seen"]
            )


            predicted = predicted_position(
                track,
                current_frame
            )


            position_distance = distance(
                detection_position,
                predicted
            )


            allowed_distance = (
                BASE_MATCH_DISTANCE
                +
                gap * 4
            )


            # =================================================
            # TOO FAR = IMPOSSIBLE
            # =================================================

            if (
                position_distance
                >
                allowed_distance
            ):

                continue


            # =================================================
            # APPEARANCE
            # =================================================

            app_distance = (
                appearance_distance(
                    track["appearance"],
                    detection_appearance
                )
            )


            # =================================================
            # HARD APPEARANCE REJECTION
            #
            # This is what should prevent:
            #
            # red player  <-> green player
            #
            # during a tackle.
            # =================================================

            if (
                track["appearance"]
                is not None
                and
                detection_appearance
                is not None
                and
                app_distance
                >
                MAX_APPEARANCE_DISTANCE
            ):

                continue


            # Normalize spatial distance to 0..1
            position_cost = (
                position_distance
                /
                allowed_distance
            )


            # =================================================
            # TOTAL COST
            # =================================================

            total_cost = (

                POSITION_WEIGHT
                *
                position_cost

                +

                APPEARANCE_WEIGHT
                *
                app_distance

                +

                # tiny preference for recently seen tracks
                gap * 0.005
            )


            cost_matrix[
                detection_index,
                track_index
            ] = total_cost


    # =====================================================
    # HUNGARIAN GLOBAL MATCHING
    # =====================================================

    rows, cols = (
        linear_sum_assignment(
            cost_matrix
        )
    )


    matched_detections = set()


    for row, col in zip(
        rows,
        cols
    ):


        cost = (
            cost_matrix[
                row,
                col
            ]
        )


        if (
            cost
            >=
            INVALID_COST
        ):

            continue


        display_id = (
            candidate_ids[
                col
            ]
        )


        assignments[
            row
        ] = display_id


        matched_detections.add(
            row
        )


    # =====================================================
    # UNMATCHED DETECTIONS = NEW STABLE PLAYERS
    # =====================================================

    for detection_index, det in enumerate(
        detections
    ):


        if (
            detection_index
            in
            matched_detections
        ):

            continue


        display_id = create_track(
            det["match_position"],
            det["appearance"],
            current_frame
        )


        assignments[
            detection_index
        ] = display_id


    return assignments


# =========================================================
# DRAW PLAYER
# =========================================================

def draw_player(
    frame,
    box,
    display_id,
    history
):

    x1, y1, x2, y2 = map(
        int,
        box
    )


    team = jersey_classifier.get_team(
        display_id
    )

    if team == TEAM_UNKNOWN:
        color = get_track_color(
            display_id
        )
    else:
        color = jersey_classifier.get_draw_color(
            display_id
        )


    # =====================================================
    # BOX
    # =====================================================

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        1
    )


    # =====================================================
    # ID LABEL
    # =====================================================

    team_name = jersey_classifier.get_team_name(
        display_id
    )

    if team_name:
        label = f"{display_id} {team_name}"
    else:
        label = str(
            display_id
        )


    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )


    font_scale = 0.45

    thickness = 1


    (
        text_width,
        text_height
    ), _ = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness
    )


    label_top = max(
        0,
        y1 - text_height - 8
    )


    cv2.rectangle(
        frame,

        (
            x1,
            label_top
        ),

        (
            x1
            +
            text_width
            +
            8,

            y1
        ),

        color,

        -1
    )


    cv2.putText(
        frame,

        label,

        (
            x1 + 4,
            y1 - 4
        ),

        font,

        font_scale,

        (
            0,
            0,
            0
        ),

        thickness,

        cv2.LINE_AA
    )


    # =====================================================
    # BOTTOM CENTER / GROUND POINT
    # =====================================================

    ground_x = int(
        (
            x1 + x2
        )
        / 2
    )


    ground_y = (
        y2
    )


    cv2.circle(
        frame,

        (
            ground_x,
            ground_y
        ),

        3,

        color,

        -1
    )


    # =====================================================
    # TRAJECTORY
    # =====================================================

    points = smooth_history(
        history[
            -MAX_HISTORY:
        ],
        window=3
    )


    if len(points) > 1:

        pts = np.array(
            points,
            dtype=np.int32
        )


        for i in range(
            1,
            len(pts)
        ):

            cv2.line(
                frame,

                tuple(
                    pts[
                        i - 1
                    ]
                ),

                tuple(
                    pts[i]
                ),

                color,

                2,

                cv2.LINE_AA
            )


# =========================================================
# VIDEO
# =========================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


while cap.isOpened():

    success, frame = (
        cap.read()
    )


    if not success:
        break


    frame_number += 1


    # =====================================================
    # YOLO + BOT-SORT
    #
    # BoT-SORT IDs are deliberately NOT trusted as the
    # stable identity anymore.
    # =====================================================

    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER_CONFIG,
        classes=[PLAYER_CLASS],
        verbose=False
    )


    result = results[0]


    # =====================================================
    # CURRENT DETECTIONS
    # =====================================================

    if (
        result.boxes is not None
        and
        len(result.boxes) > 0
    ):


        boxes = (
            result.boxes.xyxy
            .cpu()
            .tolist()
        )


        detections = []


        # =================================================
        # BUILD DETECTION FEATURES FIRST
        # =================================================

        for index, box in enumerate(
            boxes
        ):


            x1, y1, x2, y2 = (
                box
            )


            # -------------------------------------------------
            # MATCH POSITION
            #
            # IMPORTANT:
            #
            # Use BOUNDING BOX CENTRE for identity.
            #
            # This is more stable than bottom-center when
            # a player falls horizontally.
            # -------------------------------------------------

            match_position = (

                float(
                    (
                        x1 + x2
                    )
                    / 2
                ),

                float(
                    (
                        y1 + y2
                    )
                    / 2
                )
            )


            # -------------------------------------------------
            # Ground position for trajectory / tactical map
            # -------------------------------------------------

            ground_position = (

                float(
                    (
                        x1 + x2
                    )
                    / 2
                ),

                float(
                    y2
                )
            )


            # -------------------------------------------------
            # Appearance
            # -------------------------------------------------

            appearance = extract_appearance(
                frame,
                box
            )


            # -------------------------------------------------
            # Is player overlapping someone?
            # -------------------------------------------------

            overlapping = (
                detection_is_overlapping(
                    index,
                    boxes
                )
            )


            good_view = (
                is_good_appearance_view(
                    box
                )
            )


            detections.append(
                {
                    "box":
                        box,

                    "match_position":
                        match_position,

                    "ground_position":
                        ground_position,

                    "appearance":
                        appearance,

                    "overlapping":
                        overlapping,

                    "good_view":
                        good_view
                }
            )


        # =================================================
        # GLOBAL ID MATCHING
        # =================================================

        assignments = (
            assign_stable_ids(
                detections,
                frame_number
            )
        )


        # =================================================
        # UPDATE AFTER ASSIGNMENT
        # =================================================

        for detection_index, det in enumerate(
            detections
        ):


            display_id = (
                assignments[
                    detection_index
                ]
            )


            # -------------------------------------------------
            # Don't LEARN appearance during overlap/fall.
            #
            # But current appearance was still allowed to
            # help decide which old ID this detection matches.
            # -------------------------------------------------

            allow_appearance_update = (

                det["good_view"]

                and

                not det["overlapping"]
            )


            update_track(
                display_id,

                det["match_position"],

                det["appearance"],

                frame_number,

                allow_appearance_update
            )


            # =================================================
            # JERSEY / TEAM CLASSIFICATION
            #
            # Downstream only: it receives our already-resolved
            # stable display_id and cannot alter tracking identity.
            # =================================================

            jersey_classifier.update_player(
                display_id,
                frame,
                det["box"]
            )


            # =================================================
            # TRAJECTORY
            # =================================================

            track_history[
                display_id
            ].append(
                det[
                    "ground_position"
                ]
            )


            if (
                len(
                    track_history[
                        display_id
                    ]
                )
                >
                MAX_HISTORY
            ):

                track_history[
                    display_id
                ].pop(0)


            # =================================================
            # DRAW
            # =================================================

            draw_player(
                frame,

                det["box"],

                display_id,

                track_history[
                    display_id
                ]
            )


    # =====================================================
    # FIT JERSEY TEAM MODEL WHEN ENOUGH STABLE PLAYERS EXIST
    # =====================================================

    if frame_number % 10 == 0:
        jersey_classifier.try_fit_team_model()


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "Stable Tracking + KMeans Jersey Teams",
        frame
    )


    if (
        cv2.waitKey(1)
        &
        0xFF
        ==
        ord("q")
    ):

        break


cap.release()

cv2.destroyAllWindows()