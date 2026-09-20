"""
Build a flat manifest (one row per sample) from raw FLAME 3 / Boreal folders.

Run this AFTER downloading the raw data and inspecting the actual folder
structure — the parsing logic below is a starting skeleton based on the
dataset pages' documented structure, not a verified match to what you'll
actually unzip. Print out a directory tree first and adjust the glob
patterns / regexes to match reality before trusting this output.

Usage:
    python build_manifest.py --dataset flame3 --root data/raw/flame3 --out data/manifests/flame3.csv
    python build_manifest.py --dataset boreal --root data/raw/boreal --out data/manifests/boreal.csv
"""
import argparse
import csv
import re
from pathlib import Path


def build_flame3_manifest(root: Path):
    """
    FLAME 3 CV subset (documented structure): each sample is an image
    "quartet" - raw RGB, raw thermal, corrected-FOV RGB, thermal TIFF -
    split across Fire/ and No_Fire/ folders (folder names TBD once you've
    unzipped - check and adjust).

    `group` is a placeholder time-block extracted from the filename order.
    FLAME 3 doesn't guarantee a parseable timestamp in the filename, so
    fall back to sequential frame index if none is found, then bin that
    index into fixed-size blocks. This assumes files sort in capture order
    when listed alphabetically - verify that assumption once you see the
    real filenames.
    """
    rows = []
    BLOCK_SIZE = 20  # frames per time-block group; tune after inspecting capture rate

    for label_name, label in [("Fire", 1), ("No_Fire", 0)]:
        label_dir = root / label_name
        if not label_dir.exists():
            print(f"[warn] expected folder not found: {label_dir} - adjust path")
            continue

        # group files into quartets by shared basename stem (adjust regex
        # once real filenames are visible, e.g. IMG_0001_rgb.jpg / _thermal.tiff)
        rgb_files = sorted(label_dir.glob("*rgb*.jpg")) or sorted(label_dir.glob("*RGB*.jpg"))

        for i, rgb_path in enumerate(rgb_files):
            stem = re.sub(r"(_rgb|_RGB).*", "", rgb_path.stem)
            thermal_tiff = next(label_dir.glob(f"{stem}*thermal*.tif*"), None)

            rows.append({
                "sample_id": stem,
                "dataset": "flame3",
                "burn_site": "sycan_marsh",  # single-burn subset; update if using full 6-burn set
                "group": i // BLOCK_SIZE,     # time-block for leakage-safe splitting
                "label": label,
                "rgb_path": str(rgb_path),
                "thermal_path": str(thermal_tiff) if thermal_tiff else "",
            })
    return rows


def build_boreal_manifest(root: Path):
    """
    Boreal Forest Fire dataset (documented structure): organized by burn
    site and video clip, with bounding box / segmentation annotations.
    `group` = clip ID, since frames within one clip are near-duplicates
    of each other just like FLAME 3.
    """
    rows = []

    for burn_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for clip_dir in sorted(p for p in burn_dir.iterdir() if p.is_dir()):
            for frame_path in sorted(clip_dir.glob("*.jpg")):
                # adjust: figure out whether label comes from a sidecar
                # annotation file (bbox/mask) per frame, or a clip-level
                # label - check the actual annotation format once inspected
                rows.append({
                    "sample_id": f"{burn_dir.name}_{clip_dir.name}_{frame_path.stem}",
                    "dataset": "boreal",
                    "burn_site": burn_dir.name,
                    "group": f"{burn_dir.name}_{clip_dir.name}",  # whole clip is one group
                    "label": None,  # fill in once annotation format is confirmed
                    "rgb_path": str(frame_path),
                    "thermal_path": "",  # Boreal has no thermal channel
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["flame3", "boreal"], required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = build_flame3_manifest(args.root) if args.dataset == "flame3" else build_boreal_manifest(args.root)

    if not rows:
        print("[warn] no rows built - inspect --root and fix the parsing logic above before proceeding")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()