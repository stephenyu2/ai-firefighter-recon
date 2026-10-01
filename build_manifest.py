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
    FLAME 3 CV subset - CONFIRMED structure (verified against a real
    unzipped copy, not guessed from the dataset page):

        Fire/
          RGB/
            Raw/
            Corrected FOV/      <- using this: aligned to thermal camera FOV
          Thermal/
            Raw JPG/
            Celsius TIFF/       <- using this: calibrated per-pixel temperature
        No Fire/                <- note the space, not an underscore
          (same subfolders)

    Filenames are plain sequential frame numbers (00001.JPG, 00001.TIFF,
    ...) shared between the RGB and Thermal side for the same frame - no
    suffix-stripping needed, just match on stem.

    `group` is a time-block built from that sequential frame order, since
    the numbering corresponds to capture order (confirmed - these aren't
    arbitrary filenames).
    """
    rows = []
    BLOCK_SIZE = 20  # frames per time-block group; tune after inspecting capture rate
    IMAGE_EXTENSIONS = {".jpg", ".jpeg"}

    for label_name, label in [("Fire", 1), ("No Fire", 0)]:
        label_dir = root / label_name
        rgb_dir = label_dir / "RGB" / "Corrected FOV"
        thermal_dir = label_dir / "Thermal" / "Celsius TIFF"

        if not rgb_dir.exists():
            print(f"[warn] expected folder not found: {rgb_dir} - adjust path")
            continue

        rgb_files = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

        for i, rgb_path in enumerate(rgb_files):
            stem = rgb_path.stem  # e.g. "00001"

            thermal_path = None
            if thermal_dir.exists():
                matches = list(thermal_dir.glob(f"{stem}.*"))
                thermal_path = matches[0] if matches else None
            if thermal_path is None:
                print(f"[warn] no thermal match for {rgb_path.name} in {label_name} - check {thermal_dir}")

            rows.append({
                "sample_id": f"{label_name.replace(' ', '')}_{stem}",
                "dataset": "flame3",
                "burn_site": "sycan_marsh",  # single-burn subset; update if using full 6-burn set
                "group": f"{label_name.replace(' ', '')}_{i // BLOCK_SIZE}",  # prefixed so Fire/No Fire group ids never collide
                "label": label,
                "rgb_path": str(rgb_path),
                "thermal_path": str(thermal_path) if thermal_path else "",
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