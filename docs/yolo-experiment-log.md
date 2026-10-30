# YOLO experiment log template

Use this template for each baseline run with the latest YOLO version.

## Run metadata
- Date:
- YOLO version:
- Model size (e.g., n/s/m/l/x):
- Hardware:

## Data used
- Video clip source:
- Frame extraction method:
- Number of labeled frames:
- Train/val split:

## Training config
- Image size:
- Epochs:
- Batch size:
- Learning rate:
- Augmentations:

## Results
- mAP50:
- mAP50-95:
- Precision:
- Recall:

## Observations
- What worked well:
- Failure cases:
- Next changes to test:

## Inference baseline (2026-10-16)

This is a pretrained COCO inference run, not a trained run on the project's
six-class soccer dataset, so mAP and the training fields above are not
applicable yet.

- YOLO package: `ultralytics 8.4.178`
- Model: `yolo11n.pt`
- Input: `data/Neural net ninja video.mp4`
- Video: 1920x1080, 30 FPS, 601 frames (20.03 seconds)
- Inference image size: 640
- Confidence threshold: 0.25
- Output video: `runs/yolo-baseline/Neural net ninja video_annotated.mp4`
- Output detections: `runs/yolo-baseline/Neural net ninja video_detections.csv`
- Frames processed: 601
- Detections: 957 (`tv`: 601, `person`: 354, `bird`: 2)

The pretrained model detects generic `person` objects, but it does not
distinguish the project's team, goalkeeper, referee, or ball classes. The next
experiment should label representative frames using the project schema before
training a custom model.

## YOLO26 inference baseline (2026-10-16)

This run uses the pretrained COCO checkpoint and is directly comparable to the
YOLO11 baseline above. It is not a trained run on the project's six-class
soccer dataset, so mAP and the training fields above are not applicable yet.

- YOLO package: `ultralytics 8.4.178`
- Model: `yolo26n.pt`
- Input: `data/Neural net ninja video.mp4`
- Video: 1920x1080, 30 FPS, 601 frames (20.03 seconds)
- Inference image size: 640
- Confidence threshold: 0.25
- Output video: `runs/yolo26-baseline/Neural net ninja video_annotated.mp4`
- Output detections: `runs/yolo26-baseline/Neural net ninja video_detections.csv`
- Frames processed: 601
- Detections: 4,309 (`person`: 3,234, `tv`: 546, `sports ball`: 529)

Compared with YOLO11n on the same video and settings, YOLO26n detected many
more people and sports balls. These are still generic COCO classes and need
custom labels for team identity and the project's target schema.
