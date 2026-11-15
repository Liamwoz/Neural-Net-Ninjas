
import cv2
import numpy as np

from collections import defaultdict, deque
from sklearn.cluster import KMeans


# =========================================================
# TEAM CONSTANTS
# =========================================================

TEAM_RED = 0
TEAM_GREEN = 1
TEAM_REFEREE = 2
TEAM_UNKNOWN = -1


TEAM_NAMES = {
    TEAM_RED: "RED",
    TEAM_GREEN: "GREEN",
    TEAM_REFEREE: "REFEREE",
    TEAM_UNKNOWN: ""
}


# OpenCV BGR
TEAM_COLORS = {
    TEAM_RED: (0, 0, 255),
    TEAM_GREEN: (0, 255, 0),
    TEAM_REFEREE: (0, 0, 0),
    TEAM_UNKNOWN: (255, 255, 255)
}


# =========================================================
# GOALKEEPER ROLE SETTINGS
#
# The goalkeeper's TEAM is match knowledge, not a color rule:
# in this match the yellow goalkeeper plays for RED.
#
# The goalkeeper is recognized from RAW torso pixels, not from
# the K-Means jersey color. The jersey K-Means uses the grass
# mask (hue 30..60), which removes both the yellow and the
# lime-green shirt pixels, so its output cannot separate them.
#
# Values measured on ../data/121364_0.mp4 (vivid torso pixels,
# S >= 90 and V >= 110, OpenCV hue 0..179):
#   yellow goalkeeper : per-frame median hue 33..36
#   lime-green players: per-frame median hue 44..48 (p10 >= 40)
#   red players       : median hue 8..14
#
# A single frame counts as "goalkeeper-like" when >= 60% of the
# vivid torso pixels are in the yellow band AND the yellow fraction
# is more than 2x the lime-green fraction.
#
# Re-measure these bands if using a different match/kit.
# =========================================================

GOALKEEPER_TEAM = TEAM_RED

GK_VIVID_MIN_S = 90
GK_VIVID_MIN_V = 110
GK_MIN_VIVID_PIXELS = 15

GK_YELLOW_HUE = (28, 39)
GK_LIME_HUE = (40, 60)
GK_FRAME_YELLOW_FRAC = 0.60

# Multi-frame confirmation: >= 12 goalkeeper-like frames among
# the last 15 checked frames AND >= 80% over the ID's lifetime.
GK_WINDOW = 15
GK_CONFIRM_POSITIVES = 12
GK_LIFETIME_MIN_RATIO = 0.80

# An ID with >= 3 goalkeeper-like frames in its window is a
# "suspect": it is kept out of RED/GREEN K-Means and out of
# RED/GREEN freezing until resolved.
GK_SUSPECT_POSITIVES = 3

# The confirmed goalkeeper counts as "currently visible" if he
# was seen within this many frames. While visible, nobody else
# can take the role.
GK_HOLDER_VISIBLE_GRACE = 5

# A returning goalkeeper (new ID) must also match the stored
# yellow hue signature within this tolerance.
GK_SIGNATURE_TOLERANCE = 4.0

# If the holder ID shows <= 3 goalkeeper-like frames in a full
# window, the ID most likely now belongs to someone else
# (tracking swap): the role is revoked from that ID.
GK_REVOKE_MAX_POSITIVES = 3

# Debug logging cadence
GK_LOST_LOG_FRAMES = 25
GK_LOG_EVERY = 25


# =========================================================
# JERSEY CLASSIFIER
#
# Keeps the GitHub-style K-Means approach:
# - top part of player crop
# - remove green pitch
# - K-Means for jersey color
# - several samples per stable player
# - global K-Means for RED/GREEN
#
# Performance/stability fixes:
# - expensive jersey K-Means only every few frames
# - cheap referee test (no extra K-Means every frame)
# - resize crop before K-Means
# - adapt n_clusters to number of unique colors
# - RED/GREEN decision is frozen once assigned
# - referee and confirmed goalkeeper roles are handled separately
# =========================================================

class JerseyClassifier:

    def __init__(
        self,
        min_samples=6,
        max_samples=20,
        min_players_for_clustering=6,
        sample_every=3,
        goalkeeper_team=GOALKEEPER_TEAM,
        gk_debug=False
    ):

        self.min_samples = min_samples
        self.max_samples = max_samples
        self.min_players_for_clustering = min_players_for_clustering
        self.sample_every = sample_every

        # Stable player ID -> jersey RGB samples
        self.player_color_samples = defaultdict(list)

        # Stable player ID -> final class/team
        self.player_team = {}

        # Stable player ID -> representative RGB
        self.player_color = {}

        # Global K-Means for RED/GREEN players
        self.team_kmeans = None
        self.cluster_to_team = {}

        # Per-player call counters so we don't run K-Means every frame
        self.call_counts = defaultdict(int)

        # Referee evidence
        self.referee_votes = defaultdict(int)
        self.referee_checks = defaultdict(int)

        # -------------------------------------------------
        # GOALKEEPER ROLE MEMORY
        #
        # Separate from RED/GREEN color classification.
        # The ROLE survives ID changes; IDs only hold it.
        # -------------------------------------------------

        self.goalkeeper_team = goalkeeper_team
        self.gk_debug = gk_debug

        # Per-ID evidence
        self.gk_window = defaultdict(
            lambda: deque(maxlen=GK_WINDOW)
        )
        self.gk_checks = defaultdict(int)
        self.gk_positives = defaultdict(int)
        self.gk_hues = defaultdict(
            lambda: deque(maxlen=50)
        )

        # Role state
        self.gk_id = None
        self.gk_last_seen = None
        self.gk_signature = None
        self.gk_former_ids = set()
        self.gk_lost_logged = False

        # Logging state (log changes, not every frame)
        self.gk_logged_suspects = set()
        self.gk_logged_rejections = set()
        self.frame_number = 0


    # =====================================================
    # GOALKEEPER: PER-FRAME EVIDENCE (cheap, no K-Means)
    #
    # Returns (is_goalkeeper_like, yellow_hue_median) or
    # None when the torso has too few vivid pixels.
    # =====================================================

    def measure_goalkeeper_evidence(
        self,
        frame,
        box
    ):

        crop = self.get_player_crop(
            frame,
            box
        )

        if crop is None:
            return None

        height, width, _ = crop.shape

        if height < 20 or width < 5:
            return None

        # Same central upper-torso region as the referee test
        torso = crop[
            int(height * 0.10):max(
                int(height * 0.10) + 1,
                int(height * 0.50)
            ),
            int(width * 0.20):max(
                int(width * 0.20) + 1,
                int(width * 0.80)
            )
        ]

        if torso.size == 0:
            return None

        hsv = cv2.cvtColor(
            torso,
            cv2.COLOR_BGR2HSV
        ).reshape(-1, 3)

        vivid = (
            (hsv[:, 1] >= GK_VIVID_MIN_S)
            &
            (hsv[:, 2] >= GK_VIVID_MIN_V)
        )

        hues = hsv[vivid, 0]

        if len(hues) < GK_MIN_VIVID_PIXELS:
            return None

        in_yellow = (
            (hues >= GK_YELLOW_HUE[0])
            &
            (hues <= GK_YELLOW_HUE[1])
        )

        in_lime = (
            (hues >= GK_LIME_HUE[0])
            &
            (hues <= GK_LIME_HUE[1])
        )

        yellow_frac = float(np.mean(in_yellow))
        lime_frac = float(np.mean(in_lime))

        is_gk_like = (
            yellow_frac >= GK_FRAME_YELLOW_FRAC
            and
            yellow_frac > 2.0 * lime_frac
        )

        yellow_hue = (
            float(np.median(hues[in_yellow]))
            if is_gk_like
            else None
        )

        return is_gk_like, yellow_hue


    def observe_goalkeeper(
        self,
        display_id,
        frame,
        box
    ):

        evidence = self.measure_goalkeeper_evidence(
            frame,
            box
        )

        if evidence is None:
            return

        is_gk_like, yellow_hue = evidence

        self.gk_window[display_id].append(
            1 if is_gk_like else 0
        )

        self.gk_checks[display_id] += 1

        if is_gk_like:
            self.gk_positives[display_id] += 1
            self.gk_hues[display_id].append(
                yellow_hue
            )


    # =====================================================
    # GOALKEEPER: STATE QUERIES
    # =====================================================

    def gk_window_positives(
        self,
        display_id
    ):

        if display_id not in self.gk_window:
            return 0

        return int(
            sum(self.gk_window[display_id])
        )


    def gk_lifetime_ratio(
        self,
        display_id
    ):

        return (
            self.gk_positives[display_id]
            /
            max(1, self.gk_checks[display_id])
        )


    def gk_hue(
        self,
        display_id
    ):

        hues = self.gk_hues.get(display_id)

        if not hues:
            return None

        return float(np.median(hues))


    def gk_is_suspect(
        self,
        display_id
    ):

        return (
            self.gk_window_positives(display_id)
            >=
            GK_SUSPECT_POSITIVES
        )


    def gk_blocks_team_assignment(
        self,
        display_id
    ):
        """
        True for IDs that must stay out of the RED/GREEN
        K-Means model and must not be frozen RED/GREEN:
        the goalkeeper, former goalkeeper IDs, and suspects.
        """

        return (
            display_id == self.gk_id
            or
            display_id in self.gk_former_ids
            or
            self.gk_is_suspect(display_id)
        )


    def gk_qualifies(
        self,
        display_id
    ):

        if (
            self.player_team.get(display_id)
            ==
            TEAM_REFEREE
        ):
            return False

        if self.referee_is_pending(display_id):
            return False

        if self.gk_window_positives(display_id) < GK_CONFIRM_POSITIVES:
            return False

        if self.gk_checks[display_id] < GK_CONFIRM_POSITIVES:
            return False

        # Lifetime ratio: an ID with a long history of normal
        # red/green crops can never qualify from a few
        # yellow-looking frames.
        if self.gk_lifetime_ratio(display_id) < GK_LIFETIME_MIN_RATIO:
            return False

        return True


    def debug_role(
        self,
        display_id
    ):

        if display_id == self.gk_id:
            return "GK"

        if self.gk_is_suspect(display_id):
            return "GK?"

        return ""


    def gk_log(
        self,
        message
    ):

        if self.gk_debug:
            print(
                f"[GK f{self.frame_number}] {message}"
            )


    def gk_evidence_text(
        self,
        display_id
    ):

        hue = self.gk_hue(display_id)

        return (
            f"window {self.gk_window_positives(display_id)}"
            f"/{len(self.gk_window[display_id])}, "
            f"lifetime {self.gk_positives[display_id]}"
            f"/{self.gk_checks[display_id]}, "
            f"hue {hue if hue is None else round(hue, 1)}"
        )


    def gk_reject(
        self,
        display_id,
        reason
    ):

        key = (display_id, reason)

        if key not in self.gk_logged_rejections:

            self.gk_logged_rejections.add(key)

            self.gk_log(
                f"rejected candidate {display_id}: {reason} "
                f"({self.gk_evidence_text(display_id)})"
            )


    # =====================================================
    # GOALKEEPER: ROLE STATE MACHINE
    #
    # Call ONCE per video frame, after all detections of the
    # frame were processed, with the stable IDs visible in
    # that frame. Never reassigns the role while the current
    # goalkeeper is visible, never gives it to two IDs.
    # =====================================================

    def end_frame(
        self,
        frame_number,
        visible_ids
    ):

        self.frame_number = frame_number

        visible = set(visible_ids)

        # -------------------------------------------------
        # CURRENT HOLDER: visibility, lost/return, revoke
        # -------------------------------------------------

        if self.gk_id is not None:

            if self.gk_id in visible:

                if self.gk_lost_logged:
                    self.gk_log(
                        f"goalkeeper {self.gk_id} visible again "
                        f"after {frame_number - self.gk_last_seen} frames"
                    )
                    self.gk_lost_logged = False

                self.gk_last_seen = frame_number

                hue = self.gk_hue(self.gk_id)

                if hue is not None:
                    self.gk_signature = hue

                window = self.gk_window[self.gk_id]

                if (
                    len(window) == GK_WINDOW
                    and
                    sum(window) <= GK_REVOKE_MAX_POSITIVES
                ):

                    self.gk_log(
                        f"REVOKED role from ID {self.gk_id}: recent crops "
                        f"no longer goalkeeper-like, likely an ID swap "
                        f"({self.gk_evidence_text(self.gk_id)})"
                    )

                    # Let this ID be re-classified from fresh
                    # color samples like a normal player.
                    self.gk_former_ids.discard(self.gk_id)
                    self.player_color_samples.pop(self.gk_id, None)
                    self.gk_window[self.gk_id].clear()
                    self.gk_id = None

            elif (
                not self.gk_lost_logged
                and
                frame_number - self.gk_last_seen
                >=
                GK_LOST_LOG_FRAMES
            ):

                self.gk_log(
                    f"goalkeeper {self.gk_id} not seen for "
                    f"{frame_number - self.gk_last_seen} frames "
                    f"(role kept, signature hue "
                    f"{round(self.gk_signature, 1)})"
                )

                self.gk_lost_logged = True

        # -------------------------------------------------
        # SUSPECT LOGGING (state changes + sampled checks)
        # -------------------------------------------------

        if self.gk_debug:

            for display_id in visible:

                if display_id == self.gk_id:
                    continue

                if not self.gk_is_suspect(display_id):
                    continue

                if display_id not in self.gk_logged_suspects:

                    self.gk_logged_suspects.add(display_id)

                    self.gk_log(
                        f"candidate {display_id} "
                        f"({self.gk_evidence_text(display_id)})"
                    )

                elif frame_number % GK_LOG_EVERY == 0:

                    self.gk_log(
                        f"candidate {display_id} check "
                        f"({self.gk_evidence_text(display_id)})"
                    )

        # -------------------------------------------------
        # CONFIRM / TRANSFER
        # -------------------------------------------------

        qualified = sorted(
            display_id
            for display_id in visible
            if display_id != self.gk_id
            and self.gk_qualifies(display_id)
        )

        if not qualified:
            return

        if len(qualified) > 1:

            for display_id in qualified:
                self.gk_reject(
                    display_id,
                    f"ambiguous, several qualifying IDs {qualified}"
                )

            return

        candidate = qualified[0]

        holder_visible = (
            self.gk_id is not None
            and
            frame_number - self.gk_last_seen
            <=
            GK_HOLDER_VISIBLE_GRACE
        )

        if holder_visible:

            self.gk_reject(
                candidate,
                f"goalkeeper {self.gk_id} is currently visible"
            )

            return

        if self.gk_signature is not None:

            candidate_hue = self.gk_hue(candidate)

            if (
                candidate_hue is None
                or
                abs(candidate_hue - self.gk_signature)
                >
                GK_SIGNATURE_TOLERANCE
            ):

                self.gk_reject(
                    candidate,
                    f"hue does not match goalkeeper signature "
                    f"{round(self.gk_signature, 1)}"
                )

                return

        previous = self.gk_id

        first_confirmation = self.gk_signature is None

        frozen = self.player_team.get(candidate)

        if frozen in (TEAM_RED, TEAM_GREEN):
            # Only reachable when the ID's whole history is
            # >= 80% goalkeeper-like (gk_qualifies), i.e. the
            # goalkeeper himself was frozen early.
            self.gk_log(
                f"ID {candidate} was frozen {TEAM_NAMES[frozen]} but its "
                f"entire history is goalkeeper-like"
            )

        if previous is not None:
            self.gk_former_ids.add(previous)

        self.gk_id = candidate
        self.gk_last_seen = frame_number
        self.gk_lost_logged = False

        hue = self.gk_hue(candidate)

        if hue is not None:
            self.gk_signature = hue

        if first_confirmation:

            self.gk_log(
                f"CONFIRMED ID {candidate} as goalkeeper "
                f"-> {TEAM_NAMES[self.goalkeeper_team]} "
                f"({self.gk_evidence_text(candidate)})"
            )

        else:

            self.gk_log(
                f"RE-IDENTIFIED goalkeeper as new ID {candidate} "
                f"(previous ID {previous}, "
                f"{self.gk_evidence_text(candidate)})"
            )

        print(
            f"Player {candidate} -> "
            f"{TEAM_NAMES[self.goalkeeper_team]} GK"
        )


    # =====================================================
    # VIEW CHECK
    # =====================================================

    def is_good_view(
        self,
        box
    ):

        x1, y1, x2, y2 = box

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return False

        if height < 30:
            return False

        # Do not learn jersey color from a player lying flat
        if height / width < 1.15:
            return False

        return True


    # =====================================================
    # PLAYER CROP
    # =====================================================

    def get_player_crop(
        self,
        frame,
        box
    ):

        x1, y1, x2, y2 = map(
            int,
            box
        )

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(
            frame.shape[1],
            x2
        )

        y2 = min(
            frame.shape[0],
            y2
        )

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            return None

        return crop


    # =====================================================
    # CHEAP REFEREE CHECK
    #
    # No K-Means here.
    # This is intentionally cheap so it does not kill FPS.
    # =====================================================

    def looks_like_referee(
        self,
        frame,
        box
    ):

        crop = self.get_player_crop(
            frame,
            box
        )

        if crop is None:
            return False

        height, width, _ = crop.shape

        if height < 20 or width < 5:
            return False

        # Central upper torso
        torso = crop[
            int(height * 0.08):max(
                int(height * 0.08) + 1,
                int(height * 0.50)
            ),
            int(width * 0.20):max(
                int(width * 0.20) + 1,
                int(width * 0.80)
            )
        ]

        if torso.size == 0:
            return False

        hsv = cv2.cvtColor(
            torso,
            cv2.COLOR_BGR2HSV
        )

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # Black/dark neutral shirt
        dark_neutral = (
            (value < 125)
            &
            (saturation < 150)
        )

        dark_ratio = float(
            np.mean(
                dark_neutral
            )
        )

        median_value = float(
            np.median(
                value
            )
        )

        return (
            dark_ratio >= 0.30
            and
            median_value < 150
        )


    # =====================================================
    # REFEREE OBSERVATION
    #
    # Runs independently from RED/GREEN jersey learning.
    # This means the referee can still be recognized while
    # close to another player or while jersey sampling is
    # temporarily skipped.
    # =====================================================

    def observe_referee(
        self,
        display_id,
        frame,
        box
    ):

        # Once referee, always referee.
        if (
            self.player_team.get(
                display_id
            )
            ==
            TEAM_REFEREE
        ):
            return True


        self.referee_checks[
            display_id
        ] += 1


        is_dark = self.looks_like_referee(
            frame,
            box
        )


        if is_dark:

            self.referee_votes[
                display_id
            ] += 1


        checks = self.referee_checks[
            display_id
        ]

        votes = self.referee_votes[
            display_id
        ]


        # Strong black-shirt evidence:
        # classify very early.
        if (
            is_dark
            and
            checks <= 4
            and
            votes >= 2
        ):

            self.player_team[
                display_id
            ] = TEAM_REFEREE

            print(
                f"Player {display_id} -> REFEREE"
            )

            return True


        # Normal repeated-evidence rule.
        if (
            checks >= 2
            and
            votes >= 2
            and
            votes
            /
            checks
            >=
            0.35
        ):

            self.player_team[
                display_id
            ] = TEAM_REFEREE

            print(
                f"Player {display_id} -> REFEREE"
            )

            return True


        return False


    # =====================================================
    # REFEREE PENDING STATE
    #
    # If an ID has early dark-shirt evidence, do not allow
    # RED/GREEN assignment yet. Keep it UNKNOWN/gray until
    # we are confident it is either REFEREE or a normal player.
    # =====================================================

    def referee_is_pending(
        self,
        display_id
    ):

        team = self.player_team.get(
            display_id,
            TEAM_UNKNOWN
        )

        if team == TEAM_REFEREE:
            return False

        checks = self.referee_checks[
            display_id
        ]

        votes = self.referee_votes[
            display_id
        ]

        if checks == 0:
            return False

        vote_ratio = (
            votes
            /
            max(
                1,
                checks
            )
        )

        # -------------------------------------------------
        # ONLY POSSIBLE REFEREES STAY GRAY
        #
        # Normal red/green players are no longer forced
        # through a universal warm-up period.
        # -------------------------------------------------

        if (
            votes >= 1
            and
            checks < 8
        ):
            return True

        if (
            votes >= 2
            and
            vote_ratio >= 0.30
        ):
            return True

        return False


    # =====================================================
    # GITHUB-STYLE JERSEY COLOR EXTRACTION
    # =====================================================

    def extract_jersey_color(
        self,
        frame,
        box
    ):

        player_crop = self.get_player_crop(
            frame,
            box
        )

        if player_crop is None:
            return None

        height, width, _ = (
            player_crop.shape
        )

        if height < 10 or width < 5:
            return None


        # -------------------------------------------------
        # TOP 40% OF PLAYER BOX
        # -------------------------------------------------

        jersey_crop = player_crop[
            0:max(
                1,
                int(
                    height * 0.40
                )
            ),
            :
        ]

        if jersey_crop.size == 0:
            return None


        # -------------------------------------------------
        # RESIZE BEFORE K-MEANS
        #
        # Same color-clustering idea, far fewer pixels.
        # -------------------------------------------------

        crop_h, crop_w = (
            jersey_crop.shape[:2]
        )

        scale = min(
            1.0,
            32.0 / max(
                crop_h,
                crop_w
            )
        )

        if scale < 1.0:

            jersey_crop = cv2.resize(
                jersey_crop,
                (
                    max(
                        2,
                        int(
                            crop_w * scale
                        )
                    ),
                    max(
                        2,
                        int(
                            crop_h * scale
                        )
                    )
                ),
                interpolation=cv2.INTER_AREA
            )


        # -------------------------------------------------
        # REMOVE GREEN PITCH (GitHub repo HSV range 30..60)
        # -------------------------------------------------

        hsv = cv2.cvtColor(
            jersey_crop,
            cv2.COLOR_BGR2HSV
        )

        lower_green = np.array(
            [30, 25, 25],
            dtype=np.uint8
        )

        upper_green = np.array(
            [60, 255, 255],
            dtype=np.uint8
        )

        grass_mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        keep_mask = (
            grass_mask == 0
        )

        non_grass_pixels = jersey_crop[
            keep_mask
        ]


        # Fallback if grass removal removed almost everything
        if len(non_grass_pixels) < 20:

            non_grass_pixels = (
                jersey_crop.reshape(
                    -1,
                    3
                )
            )


        if len(non_grass_pixels) < 3:
            return None


        # -------------------------------------------------
        # AVOID ConvergenceWarning
        #
        # If all/most pixels are identical, don't request
        # more clusters than unique colors actually present.
        # -------------------------------------------------

        unique_pixels = np.unique(
            non_grass_pixels,
            axis=0
        )

        unique_count = len(
            unique_pixels
        )

        if unique_count == 0:
            return None

        if unique_count == 1:

            dominant_bgr = unique_pixels[
                0
            ].astype(
                np.float32
            )

            return np.array(
                [
                    dominant_bgr[2],
                    dominant_bgr[1],
                    dominant_bgr[0]
                ],
                dtype=np.float32
            )


        n_clusters = min(
            3,
            unique_count
        )


        # -------------------------------------------------
        # K-MEANS
        # -------------------------------------------------

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=0,
            n_init=5,
            tol=1e-4
        )

        labels = kmeans.fit_predict(
            non_grass_pixels
        )

        centers_bgr = np.array(
            kmeans.cluster_centers_,
            dtype=np.float32
        )

        counts = np.bincount(
            labels,
            minlength=n_clusters
        ).astype(
            np.float32
        )

        percentages = counts / max(
            1,
            len(labels)
        )

        order = np.argsort(
            percentages
        )[::-1]


        # -------------------------------------------------
        # DOMINANT USABLE COLOR
        # -------------------------------------------------

        candidates = []

        for cluster_index in order:

            bgr = centers_bgr[
                cluster_index
            ]

            bgr_pixel = np.uint8(
                [[[
                    np.clip(
                        bgr[0],
                        0,
                        255
                    ),
                    np.clip(
                        bgr[1],
                        0,
                        255
                    ),
                    np.clip(
                        bgr[2],
                        0,
                        255
                    )
                ]]]
            )

            hsv_pixel = cv2.cvtColor(
                bgr_pixel,
                cv2.COLOR_BGR2HSV
            )[0][0]

            _, saturation, value = map(
                float,
                hsv_pixel
            )

            # Reject dark/grey/white clusters
            if value < 45:
                continue

            if saturation < 35:
                continue

            rgb = np.array(
                [
                    bgr[2],
                    bgr[1],
                    bgr[0]
                ],
                dtype=np.float32
            )

            candidates.append(
                (
                    percentages[
                        cluster_index
                    ],
                    rgb
                )
            )


        if not candidates:
            return None


        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        return candidates[0][1]


    # =====================================================
    # PER-SAMPLE RED/GREEN VOTE
    # =====================================================

    def classify_color_sample(
        self,
        color
    ):

        if color is None:
            return TEAM_UNKNOWN

        r, g, b = color

        pixel = np.uint8(
            [[[
                np.clip(r, 0, 255),
                np.clip(g, 0, 255),
                np.clip(b, 0, 255)
            ]]]
        )

        hsv = cv2.cvtColor(
            pixel,
            cv2.COLOR_RGB2HSV
        )[0][0]

        hue = float(
            hsv[0]
        )

        saturation = float(
            hsv[1]
        )

        value = float(
            hsv[2]
        )

        if saturation < 45 or value < 45:
            return TEAM_UNKNOWN

        if (
            hue <= 20
            or
            hue >= 160
        ):
            return TEAM_RED

        if (
            hue >= 28
            and
            hue <= 105
        ):
            return TEAM_GREEN

        return TEAM_UNKNOWN


    # =====================================================
    # MAJORITY SAMPLE VOTE
    # =====================================================

    def get_sample_vote_team(
        self,
        display_id
    ):

        samples = (
            self.player_color_samples.get(
                display_id,
                []
            )
        )

        red_votes = 0
        green_votes = 0

        for sample in samples:

            vote = self.classify_color_sample(
                sample
            )

            if vote == TEAM_RED:
                red_votes += 1

            elif vote == TEAM_GREEN:
                green_votes += 1


        valid_votes = (
            red_votes
            +
            green_votes
        )

        if valid_votes < 4:
            return TEAM_UNKNOWN


        # Strong majority only
        if (
            red_votes
            /
            valid_votes
            >=
            0.80
        ):
            return TEAM_RED


        if (
            green_votes
            /
            valid_votes
            >=
            0.80
        ):
            return TEAM_GREEN


        return TEAM_UNKNOWN


    # =====================================================
    # UPDATE PLAYER
    # =====================================================

    def update_player(
        self,
        display_id,
        frame,
        box
    ):

        if not self.is_good_view(
            box
        ):
            return


        self.call_counts[
            display_id
        ] += 1


        # Referee detection is handled separately by
        # observe_referee(), so jersey learning can be
        # skipped without delaying REFEREE recognition.

        if (
            self.player_team.get(
                display_id
            )
            ==
            TEAM_REFEREE
        ):
            return


        # -------------------------------------------------
        # GOALKEEPER EVIDENCE
        #
        # Cheap HSV count on the torso (no K-Means), collected
        # on every good, non-occluded view. Runs before the
        # sampling skip and before the RED/GREEN freeze, but
        # it only records evidence; it never sets a team here.
        # -------------------------------------------------

        self.observe_goalkeeper(
            display_id,
            frame,
            box
        )


        # -------------------------------------------------
        # PERFORMANCE:
        # Don't run jersey K-Means on every video frame.
        # -------------------------------------------------

        if (
            self.call_counts[
                display_id
            ]
            %
            self.sample_every
            !=
            0
        ):
            return


        color = self.extract_jersey_color(
            frame,
            box
        )


        if color is None:
            return


        self.player_color_samples[
            display_id
        ].append(
            color
        )


        if (
            len(
                self.player_color_samples[
                    display_id
                ]
            )
            >
            self.max_samples
        ):

            self.player_color_samples[
                display_id
            ].pop(0)


        # -------------------------------------------------
        # FREEZE TEAM ONCE ASSIGNED
        #
        # The exact GitHub grass-mask fix now gives the green
        # player the correct initial team. Do not allow later
        # overlap-contaminated samples to flip RED <-> GREEN.
        #
        # REFEREE remains the only later override.
        # -------------------------------------------------

        current_team = self.player_team.get(
            display_id,
            TEAM_UNKNOWN
        )

        if current_team in (
            TEAM_RED,
            TEAM_GREEN
        ):
            return


        if self.team_kmeans is not None:

            self.assign_player_team(
                display_id
            )


    # =====================================================
    # REPRESENTATIVE COLOR
    # =====================================================

    def get_representative_color(
        self,
        display_id
    ):

        samples = (
            self.player_color_samples[
                display_id
            ]
        )

        if len(samples) < self.min_samples:
            return None

        samples = np.array(
            samples,
            dtype=np.float32
        )

        return np.median(
            samples,
            axis=0
        )


    # =====================================================
    # MAP GLOBAL CLUSTERS TO RED/GREEN
    # =====================================================

    def determine_cluster_mapping(
        self
    ):

        centers = (
            self.team_kmeans.cluster_centers_
        )

        red_distances = []

        for center in centers:

            r, g, b = center

            pixel = np.uint8(
                [[[
                    np.clip(r, 0, 255),
                    np.clip(g, 0, 255),
                    np.clip(b, 0, 255)
                ]]]
            )

            hsv = cv2.cvtColor(
                pixel,
                cv2.COLOR_RGB2HSV
            )[0][0]

            hue = float(
                hsv[0]
            )

            red_distance = min(
                abs(hue),
                abs(hue - 179)
            )

            red_distances.append(
                red_distance
            )


        red_cluster = int(
            np.argmin(
                red_distances
            )
        )

        green_cluster = (
            1 - red_cluster
        )


        self.cluster_to_team = {
            red_cluster: TEAM_RED,
            green_cluster: TEAM_GREEN
        }


    # =====================================================
    # FIT GLOBAL RED/GREEN K-MEANS
    # =====================================================

    def try_fit_team_model(
        self
    ):

        if self.team_kmeans is not None:
            return


        player_ids = []
        colors = []


        for display_id in (
            self.player_color_samples.keys()
        ):

            # Keep possible referees completely out of the
            # RED/GREEN team model until resolved.
            if self.referee_is_pending(
                display_id
            ):
                continue

            # Keep the goalkeeper, former goalkeeper IDs and
            # goalkeeper suspects out of the two-team model.
            if self.gk_blocks_team_assignment(
                display_id
            ):
                continue

            # Exclude IDs already identified as referee
            if (
                self.player_team.get(
                    display_id
                )
                ==
                TEAM_REFEREE
            ):
                continue


            # Also exclude strong referee candidates
            checks = self.referee_checks[
                display_id
            ]

            votes = self.referee_votes[
                display_id
            ]


            if (
                checks >= 4
                and
                votes
                /
                max(
                    1,
                    checks
                )
                >=
                0.45
            ):
                continue


            color = self.get_representative_color(
                display_id
            )


            if color is None:
                continue


            player_ids.append(
                display_id
            )

            colors.append(
                color
            )


        if (
            len(colors)
            <
            self.min_players_for_clustering
        ):
            return


        X = np.array(
            colors,
            dtype=np.float32
        )


        self.team_kmeans = KMeans(
            n_clusters=2,
            n_init=20,
            random_state=42
        )


        self.team_kmeans.fit(
            X
        )


        self.determine_cluster_mapping()


        print(
            "\nJersey team model fitted"
        )


        for (
            display_id,
            color
        ) in zip(
            player_ids,
            colors
        ):

            self.assign_player_team(
                display_id,
                color
            )


    # =====================================================
    # ASSIGN TEAM ONCE
    # =====================================================

    def assign_player_team(
        self,
        display_id,
        color=None
    ):

        # Freeze RED/GREEN once decided.
        if display_id in self.player_team:
            return


        if self.team_kmeans is None:
            return

        # REFEREE has priority over RED/GREEN.
        # If black-shirt evidence exists, keep this ID gray
        # until the referee decision is resolved.
        if self.referee_is_pending(
            display_id
        ):
            return

        # Goalkeeper role is decided separately. While an ID
        # is (or may be) the goalkeeper, do not freeze it
        # RED/GREEN from its jersey color.
        if self.gk_blocks_team_assignment(
            display_id
        ):
            return

        # -------------------------------------------------
        # DON'T ASSIGN TOO EARLY
        #
        # The remaining wrong green player was most likely
        # frozen from a noisy early color estimate.
        # Wait for more K-Means jersey observations first.
        # -------------------------------------------------

        if (
            len(
                self.player_color_samples[
                    display_id
                ]
            )
            <
            8
        ):
            return


        if color is None:

            color = self.get_representative_color(
                display_id
            )


        if color is None:
            return


        cluster = int(
            self.team_kmeans.predict(
                color.reshape(
                    1,
                    -1
                )
            )[0]
        )


        team = (
            self.cluster_to_team[
                cluster
            ]
        )


        # Frame-level K-Means majority is a safeguard.
        vote_team = self.get_sample_vote_team(
            display_id
        )


        if vote_team in (
            TEAM_RED,
            TEAM_GREEN
        ):
            # Several independent frame-level K-Means
            # observations agree on this color, so prefer
            # that consensus over a noisy global boundary.
            team = vote_team


        self.player_team[
            display_id
        ] = team


        self.player_color[
            display_id
        ] = color


        print(
            f"Player {display_id} -> "
            f"{TEAM_NAMES[team]}"
        )


    # =====================================================
    # GET TEAM
    # =====================================================

    def get_team(
        self,
        display_id
    ):

        team = self.player_team.get(
            display_id,
            TEAM_UNKNOWN
        )

        # Once confirmed as referee, always show referee.
        if team == TEAM_REFEREE:
            return TEAM_REFEREE

        # Confirmed goalkeeper role: known team affiliation.
        # Only the single current role holder gets this.
        if display_id == self.gk_id:
            return self.goalkeeper_team

        # While referee status is unresolved, hide any
        # provisional RED/GREEN result and show UNKNOWN/gray.
        if self.referee_is_pending(
            display_id
        ):
            return TEAM_UNKNOWN

        return team


    # =====================================================
    # GET DRAW COLOR
    # =====================================================

    def get_draw_color(
        self,
        display_id
    ):

        team = self.get_team(
            display_id
        )

        return TEAM_COLORS[
            team
        ]


    # =====================================================
    # GET TEAM NAME
    # =====================================================

    def get_team_name(
        self,
        display_id
    ):

        team = self.get_team(
            display_id
        )

        if (
            display_id == self.gk_id
            and
            team == self.goalkeeper_team
        ):
            return f"{TEAM_NAMES[team]} GK"

        return TEAM_NAMES[
            team
        ]
