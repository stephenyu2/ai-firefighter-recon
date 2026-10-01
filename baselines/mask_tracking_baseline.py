"""
Deterministic Tier 2 baseline: no model, just geometry. Tracks how a fire
segmentation mask's area and centroid change across a sequence of ordered
frames, and classifies that into:
  - trend: "growing" / "stable" / "receding" (relative area change)
  - direction: one of 8 compass bins, or "no clear direction" if the
    centroid barely moves relative to frame size (noise, not real spread)

This operates on MASKS, not raw images - normally the masks produced by
the Tier 1 detector's segmentation output for a sequence of frames, so
Tier 2 errors will compound whatever Tier 1 got wrong. That's a known,
intentional property (see Milestone 1 "Tier 1 / Tier 2" framing) - this
script doesn't try to correct for it.

Two layers, same pattern as reference_baseline.py:
  - classify_trend() / classify_direction(): pure functions on plain
    Python/numpy data, fully testable with synthetic arrays
  - run_mask_tracking_baseline(): CLI-facing, reads real mask files from
    disk and optionally merges in human-labeled ground truth

Usage:
    python baselines/mask_tracking_baseline.py \
        --manifest data/tier2_eval_sequences.csv \
        --human_labels data/tier2_human_labels.csv \
        --out_prefix results/predictions/tier2_mask

    Writes results/predictions/tier2_mask_trend.csv and
    results/predictions/tier2_mask_direction.csv, each directly
    runnable by eval/run_tier2_eval.py.

Expected --manifest columns: sequence_id, frame_index, mask_path
  (one row per frame; frame_index determines temporal order within a
  sequence - sort ascending, doesn't need to be contiguous integers)
Expected --human_labels columns: sequence_id, trend_label, direction_label
  (optional - if omitted, output CSVs have predictions only, no label
  column, and you merge in ground truth yourself before running eval)
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

COMPASS_BINS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def mask_stats(mask: np.ndarray):
    """Returns (area_in_pixels, centroid_xy_or_None) for one binary mask."""
    area = int(mask.sum())
    if area == 0:
        return area, None
    ys, xs = np.nonzero(mask)
    centroid = (float(xs.mean()), float(ys.mean()))
    return area, centroid


def classify_trend(areas, growth_threshold: float = 0.15) -> str:
    """
    areas: ordered list of mask areas (pixels) across a sequence's frames.
    Compares the first and last frame with nonzero fire area; relative
    change beyond +/- growth_threshold counts as growing/receding,
    otherwise stable. Requires at least two nonzero-area frames to say
    anything beyond "stable" - a sequence with only one fire frame has no
    trend to measure.
    """
    nonzero = [a for a in areas if a > 0]
    if len(nonzero) < 2:
        return "stable"

    first, last = nonzero[0], nonzero[-1]
    rel_change = (last - first) / first
    if rel_change >= growth_threshold:
        return "growing"
    elif rel_change <= -growth_threshold:
        return "receding"
    return "stable"


def classify_direction(centroids, frame_width: int, frame_height: int,
                        min_displacement_frac: float = 0.03) -> str:
    """
    centroids: ordered list of (x, y) or None (for frames with no fire
    detected) across a sequence's frames. Direction is based on the
    straight-line displacement from the first to the last valid centroid,
    normalized against the frame diagonal so "clear direction" scales
    with image size rather than a fixed pixel count.
    """
    valid = [c for c in centroids if c is not None]
    if len(valid) < 2:
        return "no clear direction"

    x0, y0 = valid[0]
    x1, y1 = valid[-1]
    dx, dy = x1 - x0, y1 - y0

    diagonal = math.hypot(frame_width, frame_height)
    magnitude = math.hypot(dx, dy)
    if diagonal == 0 or magnitude / diagonal < min_displacement_frac:
        return "no clear direction"

    # image y increases downward, so flip dy to get standard math angle
    # (0deg = east, 90deg = north, measured counterclockwise)
    angle = math.degrees(math.atan2(-dy, dx)) % 360
    idx = int((angle + 22.5) // 45) % 8
    return COMPASS_BINS[idx]


def classify_sequence(mask_paths_in_order, growth_threshold=0.15, min_displacement_frac=0.03):
    """Loads masks from disk in order and returns {"trend": ..., "direction": ...}."""
    areas, centroids = [], []
    frame_width = frame_height = None

    for path in mask_paths_in_order:
        mask = np.array(Image.open(path).convert("L")) > 0  # treat any nonzero pixel as fire
        if frame_width is None:
            frame_height, frame_width = mask.shape
        area, centroid = mask_stats(mask)
        areas.append(area)
        centroids.append(centroid)

    return {
        "trend": classify_trend(areas, growth_threshold),
        "direction": classify_direction(centroids, frame_width, frame_height, min_displacement_frac),
    }


def run_mask_tracking_baseline(manifest_df: pd.DataFrame, growth_threshold=0.15, min_displacement_frac=0.03):
    """
    manifest_df: columns sequence_id, frame_index, mask_path.
    Returns a DataFrame: sequence_id, trend_prediction, direction_prediction.
    """
    rows = []
    for sequence_id, group in manifest_df.sort_values("frame_index").groupby("sequence_id"):
        result = classify_sequence(group["mask_path"].tolist(), growth_threshold, min_displacement_frac)
        rows.append({
            "sequence_id": sequence_id,
            "trend_prediction": result["trend"],
            "direction_prediction": result["direction"],
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path,
                         help="CSV with sequence_id, frame_index, mask_path columns")
    parser.add_argument("--human_labels", type=Path, default=None,
                         help="optional CSV with sequence_id, trend_label, direction_label - "
                              "if given, output is merged and ready for run_tier2_eval.py directly")
    parser.add_argument("--growth_threshold", type=float, default=0.15)
    parser.add_argument("--min_displacement_frac", type=float, default=0.03)
    parser.add_argument("--out_prefix", required=True, type=Path)
    args = parser.parse_args()

    manifest_df = pd.read_csv(args.manifest)
    preds_df = run_mask_tracking_baseline(manifest_df, args.growth_threshold, args.min_displacement_frac)

    if args.human_labels and args.human_labels.exists():
        labels_df = pd.read_csv(args.human_labels)
        merged = preds_df.merge(labels_df, on="sequence_id", how="inner")
        if len(merged) < len(preds_df):
            print(f"[warn] {len(preds_df) - len(merged)} sequence(s) had no matching human label and were dropped")

        trend_out = merged[["sequence_id", "trend_label", "trend_prediction"]].rename(
            columns={"trend_label": "label", "trend_prediction": "prediction"})
        direction_out = merged[["sequence_id", "direction_label", "direction_prediction"]].rename(
            columns={"direction_label": "label", "direction_prediction": "prediction"})
    else:
        trend_out = preds_df[["sequence_id", "trend_prediction"]].rename(columns={"trend_prediction": "prediction"})
        direction_out = preds_df[["sequence_id", "direction_prediction"]].rename(columns={"direction_prediction": "prediction"})

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    trend_path = Path(f"{args.out_prefix}_trend.csv")
    direction_path = Path(f"{args.out_prefix}_direction.csv")
    trend_out.to_csv(trend_path, index=False)
    direction_out.to_csv(direction_path, index=False)
    print(f"wrote {trend_path} and {direction_path}")


if __name__ == "__main__":
    main()