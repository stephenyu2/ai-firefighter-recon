"""Index and prepare the FLAME 3 NADIR plots (Hanna Hammock, FL) for the full-burn benchmark.

For every RGB frame (pre, during and post burn, all plots):
  - pair it with its raw thermal TIFF by frame number (during-burn only; pre/post have no thermal). Raw TIFFs are
    unwarped and cover every frame; the dataset README gives celsius = DN * 0.1 - 275. The geo TIFFs are
    stabilized and warped, and some frames are missing, so they are not used here.
  - label: pre-burn = 0, during-burn = 1 if the thermal max is at or above FIRE_C else 0. Post-burn frames have no
    thermal and some still show smoke or a live flame (plot 2 frame 1348), and plot 3 has RGB only and no phase in
    the README, so both are -1: timed and reported as a flag rate, not scored as hits or false alarms.
  - write two 640-wide copies: the thermal field of view cropped out of the RGB frame (what the detector was
    trained on) and the full RGB frame
Writes $RECON_DATA_ROOT/flame3_nadir_prepped/ (crops, full frames, index.csv, florida_crop.csv, florida_full.csv).
results/florida_index.csv is a copy of that index.csv (paths are relative to the NADIR root).
Usage: python prep_florida.py [NADIR_ROOT]"""
import os, re, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, tifffile
from PIL import Image, ExifTags

Image.MAX_IMAGE_PIXELS = None
DATA = Path(os.environ.get("RECON_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "flame3_nadir/Hanna Hammock"
OUT = DATA / "flame3_nadir_prepped"
FIRE_C = 80.0
# Thermal field of view as a fraction of the RGB frame (width, height), centered. Lens specs from EXIF (RGB 4.74 mm on
# a 6.4 mm sensor, thermal 13 mm on 640 x 12 um) give 0.438 x 0.467; the two frames where check_fov.py registered
# cleanly gave 0.44 x 0.469 at the center. Checked by eye: 80 C contours land on the visible fire front.
FOV_W, FOV_H = 0.44, 0.469


def phase_of(p):
    s = str(p)
    return "preburn" if "preburn" in s else "postburn" if "postburn" in s else "duringburn" if "duringburn" in s else "unknown"


def exif_time(im):
    ex = im.getexif().get_ifd(ExifTags.IFD.Exif)
    return ex.get(ExifTags.Base.DateTimeOriginal) or im.getexif().get(ExifTags.Base.DateTime)


def thermal_stats(tiff):
    a = tifffile.imread(tiff).astype(np.float32) * 0.1 - 275.0
    a = a[np.isfinite(a) & (a > -100) & (a < 2000)]
    if a.size == 0:
        return np.nan, np.nan
    return float(a.max()), float((a >= FIRE_C).mean())


def one(p, t):
    plot = p.parts[len(ROOT.parts)].replace(" ", "")
    ph, n = phase_of(p), int(re.search(r"(\d+)", p.stem).group(1))
    uid = f"{plot}_{ph}_{n:04d}"
    tmax, frac = thermal_stats(t) if t is not None else (np.nan, np.nan)
    if ph == "preburn":
        y = 0
    elif ph == "postburn":
        y = -1
    elif np.isfinite(tmax):
        y = int(tmax >= FIRE_C)
    else:
        y = -1
    im = Image.open(p)
    ts = exif_time(im)
    W, H = im.size
    im.draft("RGB", (W // 4, H // 4))  # JPEG DCT downscale on decode; the crop is still wider than 640 px
    im = im.convert("RGB")
    w, h = im.size
    cw, ch = int(w * FOV_W), int(h * FOV_H)
    im.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2)).resize((640, 512), Image.BICUBIC).save(OUT / "crop" / f"{uid}.jpg", quality=92)
    im.resize((640, round(640 * h / w)), Image.BICUBIC).save(OUT / "full" / f"{uid}.jpg", quality=92)
    return dict(uid=uid, plot=plot, phase=ph, frame=n, ts=ts, y=y, t_max=tmax, frac_gt80=frac,
                thermal=str(t.relative_to(ROOT)) if t is not None else "", rgb=str(p.relative_to(ROOT)))


def main():
    tiffs = {}
    for t in ROOT.rglob("*.TIFF"):
        if "raw_thermal_tiff" in str(t):
            m = re.search(r"IRX_(\d+)", t.name)
            tiffs[(t.parts[len(ROOT.parts)], int(m.group(1)))] = t
    for d in ("crop", "full"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    rgbs = sorted(p for p in ROOT.rglob("*") if p.suffix.upper() == ".JPG" and "rgb" in str(p.parent).lower())
    jobs = [(p, tiffs.get((p.parts[len(ROOT.parts)], int(re.search(r"(\d+)", p.stem).group(1))))) for p in rgbs]
    with Pool(8) as pool:
        rows = pool.starmap(one, jobs, chunksize=8)
    idx = pd.DataFrame(rows)
    idx["y"] = np.where(idx["plot"] == "plot3", -1, idx["y"])
    idx["slice"] = idx["plot"] + "_" + idx["phase"]
    idx.to_csv(OUT / "index.csv", index=False)
    for d in ("crop", "full"):
        m = idx[["uid", "y", "slice"]].copy()
        m["path"] = [str(OUT / d / f"{u}.jpg") for u in idx.uid]
        m.to_csv(OUT / f"florida_{d}.csv", index=False)
    print(idx.groupby(["slice", "y"]).size())


if __name__ == "__main__":
    main()
