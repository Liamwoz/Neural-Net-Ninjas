"""Convert the Roboflow COCO pitch-keypoint export into a YOLO pose dataset.

The export in ``data/coco homography data`` contains the full-resolution (1920x1080)
video frames, so its images are used as-is and only the labels are converted.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from pitch_keypoints import KEYPOINT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco-json",
        type=Path,
        default=Path("data/coco homography data/train/_annotations.coco.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pitch-pose/dataset"))
    parser.add_argument(
        "--val-every",
        type=int,
        default=5,
        help="Put every Nth annotated frame (sorted by original file name) in the validation split",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=80.0,
        help="Side length in pixels of the box drawn around each keypoint",
    )
    return parser.parse_args()


def yolo_label_lines(keypoints: list[float], width: int, height: int, box_size: float) -> list[str]:
    """One YOLO pose line per visible landmark: class = landmark, single keypoint.

    A single full-frame object with 7 keypoints makes YOLO regress keypoints far from their
    anchor, which was 50-150 px inaccurate; small boxes put an anchor right next to each keypoint.
    """
    half_w = box_size / 2 / width
    half_h = box_size / 2 / height
    lines = []
    for j in range(len(KEYPOINT_NAMES)):
        x, y, v = keypoints[3 * j : 3 * j + 3]
        if v <= 0:
            continue
        kx = min(max(x / width, 0.0), 1.0)
        ky = min(max(y / height, 0.0), 1.0)
        x1, x2 = max(kx - half_w, 0.0), min(kx + half_w, 1.0)
        y1, y2 = max(ky - half_h, 0.0), min(ky + half_h, 1.0)
        lines.append(
            f"{j} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f} {kx:.6f} {ky:.6f} 2"
        )
    return lines


def main() -> None:
    args = parse_args()
    coco = json.loads(args.coco_json.read_text(encoding="utf-8"))
    category = next(c for c in coco["categories"] if c.get("keypoints"))
    if category["keypoints"] != KEYPOINT_NAMES:
        raise ValueError(f"Unexpected keypoint order: {category['keypoints']}")

    annotations = {a["image_id"]: a for a in coco["annotations"] if a["category_id"] == category["id"]}
    images = sorted(
        (image for image in coco["images"] if image["id"] in annotations),
        key=lambda image: image.get("extra", {}).get("name", image["file_name"]),
    )

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    for split in ("train", "val"):
        (args.output_dir / "images" / split).mkdir(parents=True)
        (args.output_dir / "labels" / split).mkdir(parents=True)

    counts = {"train": 0, "val": 0}
    for index, image in enumerate(images):
        split = "val" if index % args.val_every == 0 and index > 0 else "train"
        source = args.coco_json.parent / image["file_name"]
        stem = Path(image.get("extra", {}).get("name", image["file_name"])).stem
        lines = yolo_label_lines(
            annotations[image["id"]]["keypoints"], image["width"], image["height"], args.box_size
        )
        shutil.copy2(source, args.output_dir / "images" / split / f"{stem}{source.suffix}")
        (args.output_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts[split] += 1

    # Horizontal flips would turn the left penalty box into the right one, which has no
    # classes of its own, so flipping is disabled during training instead of using flip_idx.
    yaml_text = (
        f"path: {args.output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "kpt_shape: [1, 3]\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(KEYPOINT_NAMES))
    )
    yaml_path = args.output_dir / "pitch_pose.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"Train images: {counts['train']}, val images: {counts['val']}")
    print(f"Dataset yaml: {yaml_path}")


if __name__ == "__main__":
    main()
