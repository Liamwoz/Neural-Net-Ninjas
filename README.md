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

## Week 1 done criteria
- One video is selected and documented with source, timestamp, and rationale
- Label schema and annotation rules are agreed and documented
- At least one YOLO baseline run is documented with settings and observations

## Frontend

See [frontend/README.md](frontend/README.md) for the local dashboard and projected-data format.
