import cv2
import numpy as np

from collections import defaultdict
from sklearn.cluster import KMeans


# =========================================================
# TEAM CONSTANTS
# =========================================================

TEAM_RED = 0
TEAM_GREEN = 1
TEAM_UNKNOWN = -1


TEAM_NAMES = {
    TEAM_RED: "RED",
    TEAM_GREEN: "GREEN",
    TEAM_UNKNOWN: ""
}


# OpenCV BGR
TEAM_COLORS = {
    TEAM_RED: (0, 0, 255),
    TEAM_GREEN: (0, 255, 0),
    TEAM_UNKNOWN: (255, 255, 255)
}


# =========================================================
# JERSEY CLASSIFIER
# =========================================================

class JerseyClassifier:

    def __init__(
        self,
        min_samples=6,
        max_samples=20,
        min_players_for_clustering=6
    ):

        self.min_samples = (
            min_samples
        )

        self.max_samples = (
            max_samples
        )

        self.min_players_for_clustering = (
            min_players_for_clustering
        )


        # Stable player ID -> color samples
        self.player_color_samples = defaultdict(
            list
        )


        # Stable player ID -> team
        self.player_team = {}


        # Stable player ID -> representative RGB color
        self.player_color = {}


        # Global 2-team KMeans
        self.team_kmeans = None


        # KMeans cluster -> RED/GREEN
        self.cluster_to_team = {}


    # =====================================================
    # CHECK IF VIEW IS GOOD
    # =====================================================

    def is_good_view(
        self,
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


        # Too small
        if height < 30:
            return False


        # Fallen / horizontal
        if (
            height / width
            <
            1.25
        ):
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


        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            frame.shape[1],
            x2
        )

        y2 = min(
            frame.shape[0],
            y2
        )


        if (
            x2 <= x1
            or
            y2 <= y1
        ):

            return None


        crop = frame[
            y1:y2,
            x1:x2
        ]


        if crop.size == 0:
            return None


        return crop


    # =====================================================
    # EXTRACT JERSEY COLOR
    # =====================================================

    def extract_jersey_color(
        self,
        frame,
        box
    ):
        """
        K-Means jersey extraction:

        1. Crop player
        2. Use upper half
        3. K-Means = 2
        4. Determine background cluster from corners
        5. Other cluster = player/jersey color
        """

        player_crop = (
            self.get_player_crop(
                frame,
                box
            )
        )


        if player_crop is None:
            return None


        height, width, _ = (
            player_crop.shape
        )


        if (
            height < 10
            or
            width < 5
        ):

            return None


        # =================================================
        # UPPER HALF = JERSEY REGION
        # =================================================

        upper_half = (
            player_crop[
                0:int(height / 2),
                :
            ]
        )


        if upper_half.size == 0:
            return None


        # BGR -> RGB
        rgb = cv2.cvtColor(
            upper_half,
            cv2.COLOR_BGR2RGB
        )


        pixels = rgb.reshape(
            -1,
            3
        )


        if len(pixels) < 10:
            return None


        # =================================================
        # PIXEL K-MEANS
        # =================================================

        kmeans = KMeans(
            n_clusters=2,
            n_init=10,
            random_state=42
        )


        labels = (
            kmeans.fit_predict(
                pixels
            )
        )


        clustered_image = (
            labels.reshape(
                rgb.shape[0],
                rgb.shape[1]
            )
        )


        # =================================================
        # FIND BACKGROUND USING CORNERS
        # =================================================

        corner_clusters = [

            clustered_image[
                0,
                0
            ],

            clustered_image[
                0,
                -1
            ],

            clustered_image[
                -1,
                0
            ],

            clustered_image[
                -1,
                -1
            ]
        ]


        counts = np.bincount(
            corner_clusters,
            minlength=2
        )


        background_cluster = (
            int(
                np.argmax(
                    counts
                )
            )
        )


        player_cluster = (
            1
            -
            background_cluster
        )


        jersey_color = (
            kmeans.cluster_centers_[
                player_cluster
            ]
        )


        # =================================================
        # FILTER CLEARLY BAD EXTRACTIONS
        # =================================================

        rgb_pixel = np.uint8(
            [[[
                np.clip(
                    jersey_color[0],
                    0,
                    255
                ),

                np.clip(
                    jersey_color[1],
                    0,
                    255
                ),

                np.clip(
                    jersey_color[2],
                    0,
                    255
                )
            ]]]
        )


        hsv = cv2.cvtColor(
            rgb_pixel,
            cv2.COLOR_RGB2HSV
        )[0][0]


        h, s, v = map(
            float,
            hsv
        )


        # Dark / grey / white / poor crop
        if v < 45:
            return None


        if s < 40:
            return None


        return np.array(
            jersey_color,
            dtype=np.float32
        )


    # =====================================================
    # STORE PLAYER COLOR SAMPLE
    # =====================================================

    def update_player(
        self,
        display_id,
        frame,
        box
    ):
        """
        Called from main tracking loop.

        Once a team is known, we keep it fixed.
        """

        # Already classified
        if (
            display_id
            in
            self.player_team
        ):

            return


        if not self.is_good_view(
            box
        ):

            return


        color = (
            self.extract_jersey_color(
                frame,
                box
            )
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


        # Team model already exists?
        if (
            self.team_kmeans
            is not None
        ):

            self.assign_player_team(
                display_id
            )


    # =====================================================
    # REPRESENTATIVE PLAYER COLOR
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


        if (
            len(samples)
            <
            self.min_samples
        ):

            return None


        samples = np.array(
            samples,
            dtype=np.float32
        )


        # Median = robust against noisy frames
        return np.median(
            samples,
            axis=0
        )


    # =====================================================
    # DETERMINE RED VS GREEN CLUSTER
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
                    np.clip(
                        r,
                        0,
                        255
                    ),

                    np.clip(
                        g,
                        0,
                        255
                    ),

                    np.clip(
                        b,
                        0,
                        255
                    )
                ]]]
            )


            hsv = cv2.cvtColor(
                pixel,
                cv2.COLOR_RGB2HSV
            )[0][0]


            hue = float(
                hsv[0]
            )


            # Red is near 0 or 179 in OpenCV HSV
            red_distance = min(
                abs(
                    hue
                ),
                abs(
                    hue - 179
                )
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
            1
            -
            red_cluster
        )


        self.cluster_to_team = {

            red_cluster:
                TEAM_RED,

            green_cluster:
                TEAM_GREEN
        }


    # =====================================================
    # FIT TEAM MODEL
    # =====================================================

    def try_fit_team_model(
        self
    ):
        """
        Fit global KMeans once enough stable players
        have representative jersey colors.
        """

        if (
            self.team_kmeans
            is not None
        ):

            return


        player_ids = []

        colors = []


        for display_id in (
            self.player_color_samples.keys()
        ):

            color = (
                self.get_representative_color(
                    display_id
                )
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


        # =================================================
        # SECOND K-MEANS:
        # TWO TEAMS
        # =================================================

        self.team_kmeans = KMeans(
            n_clusters=2,
            n_init=30,
            random_state=42
        )


        self.team_kmeans.fit(
            X
        )


        self.determine_cluster_mapping()


        print(
            "\nJersey team model fitted"
        )


        # Assign the players used to build model
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
    # ASSIGN PLAYER TEAM
    # =====================================================

    def assign_player_team(
        self,
        display_id,
        color=None
    ):

        # Never flip an already assigned stable ID
        if (
            display_id
            in
            self.player_team
        ):

            return


        if (
            self.team_kmeans
            is None
        ):

            return


        if color is None:

            color = (
                self.get_representative_color(
                    display_id
                )
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


        self.player_team[
            display_id
        ] = team


        self.player_color[
            display_id
        ] = color


        print(
            f"Player "
            f"{display_id} -> "
            f"{TEAM_NAMES[team]}"
        )


    # =====================================================
    # GET TEAM
    # =====================================================

    def get_team(
        self,
        display_id
    ):

        return self.player_team.get(
            display_id,
            TEAM_UNKNOWN
        )


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


        return TEAM_NAMES[
            team
        ]