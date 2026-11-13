# Neural-Net-Ninjas

## Members
Lex Renders
Liam Wozniak
Rune Lemmens
Pamela Barbosa

## Project goal
Build a computer-vision pipeline that takes ~20 seconds of soccer game video and produces a 2D field map of player/ball positions.

## Week 1 scope
This week focuses on three setup milestones:
1. Select a suitable soccer clip (~20s)
2. Define labeling standards for consistent annotations
3. Run baseline experiments with the latest YOLO release

## Repository structure
- `docs/video-selection.md` – checklist for choosing the initial clip
- `docs/labeling-standards.md` – annotation rules and class definitions
- `docs/yolo-experiment-log.md` – repeatable template for YOLO experiments
- `docs/pitch-homography.md` – pitch keypoint model, homography pipeline and how it was improved
- `notebooks/Cv_Labeling.ipynb` – assisted labeling and dataset preparation
- `notebooks/Model_training_and_evaluation.ipynb` – YOLO26 training and evaluation
- `requirements-yolo.txt` – dependencies for the YOLO video baseline

## YOLO setup

Install the baseline dependencies from the repository root:

```powershell
python -m pip install -r requirements-yolo.txt
```

Run pretrained COCO detection on the supplied video:

```powershell
python scripts\yolo_video_experiment.py "data\Neural net ninja video.mp4"
```

The command writes an annotated MP4 and a per-detection CSV under `runs/yolo-video/`.

Use [the standard YOLO26 notebook](notebooks/yolo26_video_experiment.ipynb) for pretrained COCO detection, or [the Roboflow model notebook](notebooks/roboflow_yolo26_video.ipynb) to run `model/weights.pt` locally. Put the soccer video in `data/`, select the environment where the requirements are installed, and run the cells in order. Both notebooks show annotated frames and save an annotated video and detections CSV under `runs/`.

## Pitch keypoints and homography

A YOLO pose model is trained on the Roboflow keypoint annotations in `data/coco homography data/` (7 pitch landmarks around the left penalty box and centre circle, 31 frames of `video/121364_0.mp4`). Each landmark is its own class with one keypoint. The landmark positions in metres are defined in `scripts/pitch_keypoints.py`.

```powershell
# 1. Build the YOLO pose dataset (Roboflow images + converted labels) in runs/pitch-pose/dataset
python scripts\prepare_pitch_pose_dataset.py
# 2. Fine-tune yolo11n-pose; best weights are copied to model/pitch_keypoints.pt
python scripts\train_pitch_pose.py
# 3. Predict keypoints per frame, fit image->pitch homographies and project the
#    detections from runs/yolo-video/121364_0_detections.csv onto scripts/images.png
python scripts\pitch_homography_video.py video\121364_0.mp4
```

Step 3 writes `121364_0_homography.mp4` (pitch-line overlay, keypoints and a 2D minimap), `121364_0_pitch_positions.csv` (detections in pitch metres and map pixels) and `121364_0_homographies.csv` (per-frame 3x3 matrices) to `runs/yolo-video/`. When too few non-collinear keypoints are visible, camera motion from optical flow carries the homography forward.

To suppress single-frame keypoint "twitches", the video is processed offline: each keypoint is replaced by the median of its detections over neighbouring frames, after correcting for camera motion (`--keypoint-smooth-radius`, default 3 frames per side). The homographies then get a centred, motion-compensated average (`--homography-smooth-radius`, default 2). Neither adds lag. In the video, hollow yellow circles are the raw model detections and filled red dots are the smoothed keypoints. See [docs/pitch-homography.md](docs/pitch-homography.md) for the full write-up.

## Week 1 done criteria
- One video is selected and documented with source, timestamp, and rationale
- Label schema and annotation rules are agreed and documented
- At least one YOLO baseline run is documented with settings and observations
