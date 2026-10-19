"""Builds a folder you can push to Baseten from the configs in deploy/ plus the shared code in serving/, training/ and
experiments/. Weights and frames are not in git, so they are copied in from your own paths.

Usage:
  python deploy/assemble.py NAME [--detector DIR] [--detector-giant DIR]
NAME is a folder under deploy/models (push with `truss push`) or deploy/training (push with `truss train push`).
--detector / --detector-giant are final_<variant>/ folders from training/flame3-lora/train_cv.py (adapter/, head.pt,
serving_config.json): final_dinov2b_lora_strongaug and final_dinov2g_qlora in our runs.
Frames are read from RECON_DATA_ROOT (default: the repo's data/ folder). Output: deploy/build/NAME/ (git-ignored).
"""
import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
DATA = Path(os.environ.get("RECON_DATA_ROOT", ROOT / "data"))
SYCAN = DATA / "flame3_cv/FLAME 3 CV Dataset (Sycan Marsh)"

# extra files per build: (source relative to the repo, destination inside the build folder)
CODE = {
    "fire-classifier": [("serving/fire_classifier/model.py", "model/model.py")],
    "clef": [("serving/clef/model.py", "model/model.py")],
    "d3": [("serving/d3/model.py", "model/model.py")],
    "precision-probe": [("experiments/full-burn-benchmark/precision_probe.py", "model/probe.py")],
    "flame3-lora": [("training/flame3-lora/train_cv.py", "train_cv.py"), ("training/flame3-lora/run.sh", "run.sh"),
                    ("training/flame3-lora/index.csv", "index.csv"),
                    ("training/flame3-lora/prior_scores_dinov2g_qlora_purged_cv.csv", "prior_scores_dinov2g_qlora_purged_cv.csv"),
                    ("deploy/training/run_baseten.sh", "run_baseten.sh")],
    "precision": [("training/precision/precision_eval.py", "precision_eval.py"), ("training/precision/run.sh", "run.sh"),
                  ("training/flame3-lora/index.csv", "index.csv"), ("deploy/training/run_baseten.sh", "run_baseten.sh")],
}


def kind(name):
    for k in ("fire-classifier", "clef", "precision-probe", "d3"):
        if name.startswith(k):
            return k
    return name if name in ("flame3-lora", "precision") else None


def sycan_frame(uid):
    lab, k = uid.split("_")
    return SYCAN / ("Fire" if lab == "Fire" else "No Fire") / "RGB/Corrected FOV" / f"{int(k):05d}.JPG"


def copy_detector(src, dst):
    src = Path(src)
    missing = [p for p in ("serving_config.json", "adapter", "head.pt") if not (src / p).exists()]
    if missing:
        sys.exit(f"{src} is missing {missing}; expected a final_<variant>/ folder from train_cv.py")
    shutil.copytree(src, dst)


def training_frames(out):
    """Flat <uid>.jpg copies of the 738 FLAME 3 CV frames, the layout run_baseten.sh points FLAME3_RGB_DIR at."""
    (out / "data/rgb").mkdir(parents=True)
    with open(ROOT / "training/flame3-lora/index.csv") as fh:
        uids = [r["uid"] for r in csv.DictReader(fh)]
    for u in uids:
        shutil.copy(sycan_frame(u), out / "data/rgb" / f"{u}.jpg")
    return len(uids)


def probe_frames(out):
    """The 64 probe frames listed in frames.txt, resized to the detector input (308 x 252)."""
    from PIL import Image
    (out / "data/frames").mkdir(parents=True)
    names = (DEPLOY / "models/precision-probe-rtx6000/frames.txt").read_text().split()
    for n in names:
        stem = n.removesuffix(".jpg")
        if stem.startswith("florida_"):
            src = DATA / "flame3_nadir_prepped/crop" / f"{stem.removeprefix('florida_')}.jpg"
        else:
            lab, k = stem.removeprefix("sycan_").split("_")
            src = sycan_frame(f"{'Fire' if lab == 'Fire' else 'NoFire'}_{k}")
        Image.open(src).convert("RGB").resize((308, 252), Image.BICUBIC).save(out / "data/frames" / n, quality=92)  # as the run used
    return len(names)


def main():
    p = argparse.ArgumentParser(description="Assemble a pushable Truss or training-job folder.")
    p.add_argument("name")
    p.add_argument("--detector", help="final_<variant>/ folder served as dinov2_lora (DINOv2-B)")
    p.add_argument("--detector-giant", help="final_<variant>/ folder served as dinov2g_qlora (DINOv2-g)")
    a = p.parse_args()
    k = kind(a.name)
    src = DEPLOY / ("training" if k in ("flame3-lora", "precision") else "models") / a.name
    if not src.is_dir():
        names = sorted(d.name for d in (DEPLOY / "models").iterdir() if d.is_dir()) + ["flame3-lora", "precision"]
        sys.exit(f"unknown build '{a.name}'; choose from {names}")
    if k in ("fire-classifier", "precision-probe") and not a.detector:
        sys.exit("--detector is required for this build")
    if (k == "precision-probe" or (k == "fire-classifier" and "nvfp4" in a.name)) and not a.detector_giant:
        sys.exit("--detector-giant is required for this build (its config or probe uses dinov2g_qlora)")
    final = DEPLOY / "build" / a.name
    out = final.with_name(f".{a.name}.partial")  # built here and moved into place only when complete
    if out.exists():
        shutil.rmtree(out)
    try:
        shutil.copytree(src, out, ignore=shutil.ignore_patterns("frames.txt"))
        for s, d in CODE.get(k, []):
            (out / d).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / s, out / d)
        notes = []
        if k in ("fire-classifier", "precision-probe"):
            copy_detector(a.detector, out / "data/detector")
            if a.detector_giant:
                copy_detector(a.detector_giant, out / "data/detector_giant")
        if k == "precision-probe":
            notes.append(f"{probe_frames(out)} probe frames")
        if k in ("flame3-lora", "precision"):
            notes.append(f"{training_frames(out)} training frames")
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)  # a failed run never leaves a half-built folder, and keeps the old build
        raise
    if final.exists():
        shutil.rmtree(final)
    out.rename(final)
    out = final
    print("built", out.relative_to(ROOT), *notes)
    if k in ("flame3-lora", "precision"):
        print(f"push: cd {out.relative_to(ROOT)} && truss train push config.py")
    else:
        print(f"push: truss push {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
