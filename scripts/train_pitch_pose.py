"""Train a YOLO pose model that predicts the pitch keypoints used for homography.

Each pitch landmark is a class with a single keypoint (see
``scripts/prepare_pitch_pose_dataset.py``, which must be run first).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("runs/pitch-pose/dataset/pitch_pose.yaml"))
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Pretrained pose weights to fine-tune")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu for CPU; auto by default")
    parser.add_argument("--project", type=Path, default=Path("runs/pitch-pose"))
    parser.add_argument("--name", default="train")
    parser.add_argument(
        "--export-weights",
        type=Path,
        default=Path("model/pitch_keypoints.pt"),
        help="Where to copy the best weights after training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(
            f"Dataset yaml not found: {args.data}. Run scripts/prepare_pitch_pose_dataset.py first."
        )

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        patience=50,
        # Very small single-video dataset: rely on geometric/colour augmentation, but
        # never flip (left/right pitch halves are different landmarks) and skip mosaic
        # (global pitch context helps tell e.g. top/bottom landmarks apart).
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        degrees=3.0,
        translate=0.15,
        scale=0.3,
        perspective=0.0005,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        plots=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    args.export_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.export_weights)
    print(f"Best weights: {best}")
    print(f"Copied to: {args.export_weights}")


if __name__ == "__main__":
    main()
