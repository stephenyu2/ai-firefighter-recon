"""
Export the real drone frames for the replay console.

For every sequence in export_demo.SELECTED this writes, under webapp/frames/<sequence_id>/:

  f01.jpg .. f08.jpg     the RGB frame the pipeline was given (640 px on the long side, metadata stripped)
  f01_hot.png ..         a transparent overlay of the thermal pixels at or above 80 C, same size as the frame
                         (only when the frame has a thermal image and at least one hot pixel)

and webapp/frames/manifest.json with, per frame, the file paths and the hot area measured from the thermal image:
its share of the frame, the hottest pixel, the hot-area center and its bounding box (all normalized 0..1, origin
top-left). The center is a measurement, so the viewer no longer has to place the hot area from the brief's
location word.

Frame ids (f01..) are the neutral ids the gateway shows to models, in capture-time order, so they line up with
frames[].id in data.js.

Two of the sequences are RGB only on purpose: the thermal images exist but were withheld from the pipeline to test
the hardest case. For those the overlay is written as f01_truth.png and listed as "truth_overlay". It shows where
the fire was. The pipeline never saw it, so the viewer should label it as ground truth, not as a pipeline output.

Needs the datasets under RECON_DATA_ROOT (default: the repo's data/ folder), laid out as in
experiments/gateway-model-trials/README.md. Run:  python3 webapp/export_frames.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
from export_demo import SELECTED  # noqa: E402
from recon_gateway.sequence import alias, from_flame_index, from_folder  # noqa: E402

OUT = HERE / "frames"
TRIALS = REPO / "experiments/gateway-model-trials"
SEQS = {s["id"]: s for s in json.loads((TRIALS / "sequences.json").read_text())}
INDEX = REPO / "training/flame3-lora/index.csv"
DATA = Path(os.environ.get("RECON_DATA_ROOT", REPO / "data"))
# Celsius TIFFs for the Florida frames whose thermal was withheld from the pipeline (optional).
WITHHELD_TIFFS = Path(os.environ.get("RECON_WITHHELD_TIFFS", DATA / "florida_celsius"))

HOT_C = 80.0     # same threshold as the gateway's thermal tools
MAX_SIDE = 640   # what the gateway sends to the models
SOURCES = {
    "FL": "FLAME 3 NADIR thermal plot subset (Hanna Hammock, Florida)",
    "SY": "FLAME 3 Computer Vision Subset (Sycan Marsh, Oregon)",
}


def load(spec):
    if spec["kind"] == "flame":
        return from_flame_index(INDEX, DATA / "flame3_cv", group=spec["group"], max_frames=8)
    tiffs = DATA / spec["tiff_dir"] if spec["tiff_dir"] else None
    return from_folder(DATA / spec["rgb_dir"], tiffs, seq_id=spec["id"], interval_s=spec["interval_s"])


def save_rgb(src, dst):
    im = Image.open(src).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    im.save(dst, format="JPEG", quality=90)  # a fresh save carries no EXIF
    return im.size


def hot_stats(celsius):
    mask = celsius >= HOT_C
    h, w = mask.shape
    stats = {"hot_percent": round(float(mask.mean()) * 100, 3), "max_c": round(float(celsius.max()), 1),
             "centroid": None, "bbox": None}
    if mask.any():
        ys, xs = np.nonzero(mask)
        stats["centroid"] = [round(float(xs.mean()) / w, 4), round(float(ys.mean()) / h, 4)]
        stats["bbox"] = [round(float(xs.min()) / w, 4), round(float(ys.min()) / h, 4),
                         round(float(xs.max() + 1) / w, 4), round(float(ys.max() + 1) / h, 4)]
    return mask, stats


def save_overlay(celsius, mask, size, dst):
    """Hot pixels coloured from orange (80 C) to pale yellow (400 C and above); everything else transparent."""
    t = np.clip((celsius - HOT_C) / (400.0 - HOT_C), 0.0, 1.0)
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = (90 + 150 * t).astype(np.uint8)
    rgba[..., 2] = (20 + 160 * t ** 2).astype(np.uint8)
    rgba[..., 3] = np.where(mask, 190, 0).astype(np.uint8)
    im = Image.fromarray(rgba, "RGBA")
    if im.size != size:
        im = im.resize(size, Image.NEAREST)
    im.save(dst, format="PNG", optimize=True)


def main():
    manifest = {"hot_threshold_c": HOT_C, "sequences": {}}
    for seq_id, _ in SELECTED:
        spec = SEQS[seq_id]
        seq, _ = alias(load(spec))
        folder = OUT / seq_id
        folder.mkdir(parents=True, exist_ok=True)
        frames = []
        for f in seq.frames:
            size = save_rgb(f.rgb_path, folder / f"{f.id}.jpg")
            row = {"id": f.id, "source_id": f.src_id, "rgb": f"frames/{seq_id}/{f.id}.jpg", "width": size[0], "height": size[1],
                   "overlay": None, "truth_overlay": None, "thermal": None}
            tiff, withheld = f.tiff_path, False
            if tiff is None and (WITHHELD_TIFFS / f"{f.src_id}.tiff").exists():
                tiff, withheld = WITHHELD_TIFFS / f"{f.src_id}.tiff", True
            if tiff is not None:
                celsius = tifffile.imread(tiff).astype(np.float32)
                mask, stats = hot_stats(celsius)
                row["thermal"] = dict(stats, seen_by_pipeline=not withheld)
                if mask.any():
                    name = f"{f.id}_truth.png" if withheld else f"{f.id}_hot.png"
                    save_overlay(celsius, mask, size, folder / name)
                    row["truth_overlay" if withheld else "overlay"] = f"frames/{seq_id}/{name}"
            frames.append(row)
        manifest["sequences"][seq_id] = {"source": SOURCES[seq_id[:2]], "thermal_given_to_pipeline": bool(spec["thermal"]),
                                         "frames": frames}
        hot = [r["thermal"]["hot_percent"] for r in frames if r["thermal"]]
        print(f"{seq_id:<34} {len(frames)} frames, {sum(1 for r in frames if r['overlay'] or r['truth_overlay'])} overlays"
              + (f", hot area {min(hot)} to {max(hot)} %" if hot else ""))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    js = "// Generated by export_frames.py. Do not edit by hand.\nwindow.FRAMES = " + json.dumps(manifest, indent=2) + ";\n"
    (OUT / "manifest.js").write_text(js)


if __name__ == "__main__":
    main()
