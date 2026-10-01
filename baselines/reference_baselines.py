"""
Off-the-shelf reference baseline for Tier 1. Runs a PRETRAINED fire/smoke
detector - no fine-tuning on FLAME 3 or Boreal - to establish what "an
existing open-source system, used as published" gets you before any of
our own training.

Suggested checkpoint: Abonia1/YOLOv8-Fire-and-Smoke-Detection (GitHub,
https://github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection) - a published
YOLOv8 model with fire/smoke classes matching our binary task. Download
the weights file from that repo yourself and point --weights at the local
path; this script doesn't fetch it automatically. Before citing this in
the milestone doc, check the repo directly for its license terms - it
isn't explicitly stated in the README as of this writing, which is worth
noting in your data/model card rather than assuming permissive use.

This script is split into two layers so it's testable without ultralytics
installed or any real weights downloaded:
  - detections_to_binary(): pure function, raw detections -> 0/1
  - run_reference_baseline(): takes any object with a .predict_image()
    method, so tests can inject a fake detector instead of a real model

Usage:
    pip install ultralytics
    python baselines/reference_baseline.py \
        --weights path/to/fire_smoke_yolov8.pt \
        --test data/splits/test.csv \
        --out results/predictions/reference_baseline.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def detections_to_binary(detections, fire_class_names=("fire", "smoke"), conf_threshold=0.25):
    """
    detections: list of (class_name: str, confidence: float) tuples for
    one image, as returned by a detector's predict_image().
    Returns 1 if any detection matches a fire-indicating class above the
    confidence threshold, else 0.
    """
    fire_class_names = {c.lower() for c in fire_class_names}
    for class_name, confidence in detections:
        if class_name.lower() in fire_class_names and confidence >= conf_threshold:
            return 1
    return 0


class YoloFireSmokeDetector:
    """Thin wrapper around ultralytics YOLO so run_reference_baseline()
    doesn't need to know about the underlying library. Lazily imports
    ultralytics so the rest of this module can be unit-tested without it
    installed."""

    def __init__(self, weights_path: str):
        from ultralytics import YOLO  # lazy import
        self.model = YOLO(weights_path)

    def predict_image(self, image_path: str):
        results = self.model.predict(source=image_path, verbose=False)
        detections = []
        for result in results:
            names = result.names  # class_id -> class_name
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                detections.append((names[class_id], confidence))
        return detections


def run_reference_baseline(test_df: pd.DataFrame, detector, fire_class_names, conf_threshold,
                            image_col: str = "rgb_path", label_col: str = "label"):
    """
    detector: any object with .predict_image(path) -> list[(class_name, confidence)].
    Returns a DataFrame with sample_id, label, prediction - same shape
    run_eval.py expects.
    """
    rows = []
    for _, row in test_df.iterrows():
        detections = detector.predict_image(row[image_col])
        prediction = detections_to_binary(detections, fire_class_names, conf_threshold)
        rows.append({
            "sample_id": row["sample_id"],
            "label": row[label_col],
            "prediction": prediction,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path,
                         help="local path to a pretrained fire/smoke YOLO checkpoint (.pt)")
    parser.add_argument("--test", required=True, type=Path,
                         help="test.csv from make_splits.py (needs sample_id, rgb_path, label columns)")
    parser.add_argument("--fire_class_names", nargs="+", default=["fire", "smoke"],
                         help="class names in the checkpoint that count as a fire detection")
    parser.add_argument("--conf_threshold", type=float, default=0.25)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(
            f"{args.weights} not found - download the checkpoint first "
            "(see module docstring for a suggested source) and pass its local path"
        )

    test_df = pd.read_csv(args.test)
    detector = YoloFireSmokeDetector(str(args.weights))

    preds_df = run_reference_baseline(test_df, detector, args.fire_class_names, args.conf_threshold)
    preds_df["strategy"] = f"reference_baseline:{args.weights.stem}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(args.out, index=False)
    print(f"wrote {len(preds_df)} predictions to {args.out}")


if __name__ == "__main__":
    main()