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

The same experiment is available as [`notebooks/yolo26_video_experiment.ipynb`](C:/Users/liamw/Documents/DeepLearning/Neural-Net-Ninjas/Neural-Net-Ninjas/notebooks/yolo26_video_experiment.ipynb). Open it in VS Code, select the environment where `requirements-yolo.txt` was installed, and run the cells. Its progress bar updates per frame while the video is processing. Set `MAX_FRAMES` in the configuration cell to a small number for a quick test.

For visible frame-by-frame output and an explicit later NMS stage, use [`notebooks/yolo26_visualize_nms.ipynb`](C:/Users/liamw/Documents/DeepLearning/Neural-Net-Ninjas/Neural-Net-Ninjas/notebooks/yolo26_visualize_nms.ipynb). It displays sampled annotated frames while running, saves an annotated MP4 and preview JPGs, and writes the boxes retained by the second-stage class-aware NMS to CSV. Ultralytics still performs its built-in NMS during inference; the notebook's NMS is an additional filter on the boxes returned by YOLO.

## Week 1 done criteria
- One video is selected and documented with source, timestamp, and rationale
- Label schema and annotation rules are agreed and documented
- At least one YOLO baseline run is documented with settings and observations
