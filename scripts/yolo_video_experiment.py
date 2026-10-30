"""Run pretrained YOLO detection on a video and save an annotated sample."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/yolo-video"),
        help="Directory for the annotated video and detections CSV",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model weights, default: yolo11n.pt",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum confidence")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit for quick tests; process the whole video by default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {args.video}")
    if args.conf <= 0 or args.conf > 1:
        raise ValueError("--conf must be greater than 0 and at most 1")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_video = args.output_dir / f"{args.video.stem}_annotated.mp4"
    output_csv = args.output_dir / f"{args.video.stem}_detections.csv"

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {args.video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_video}")

    model = YOLO(args.model)
    frame_number = 0
    detection_count = 0
    try:
        with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                ["frame", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
            )

            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if args.max_frames is not None:
                total_frames = min(total_frames, args.max_frames)
            progress = tqdm(
                total=total_frames,
                desc="Processing video",
                unit="frame",
            )
            while args.max_frames is None or frame_number < args.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break

                result = model.predict(
                    source=frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    verbose=False,
                )[0]
                annotated_frame = result.plot()
                writer.write(annotated_frame)

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
                        csv_writer.writerow(
                            [
                                frame_number,
                                class_id,
                                result.names[class_id],
                                f"{confidence:.4f}",
                                f"{x1:.2f}",
                                f"{y1:.2f}",
                                f"{x2:.2f}",
                                f"{y2:.2f}",
                            ]
                        )
                        detection_count += 1

                frame_number += 1
                progress.update(1)
            progress.close()
    finally:
        capture.release()
        writer.release()

    print(f"Processed frames: {frame_number}")
    print(f"Detections: {detection_count}")
    print(f"Annotated video: {output_video}")
    print(f"Detections CSV: {output_csv}")


if __name__ == "__main__":
    main()
