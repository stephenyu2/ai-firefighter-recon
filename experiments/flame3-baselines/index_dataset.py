"""Index the FLAME 3 CV subset: labels, paths, capture metadata, thermal stats.

Output: cache/index.csv (one row per image quartet).
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

ROOT = Path(os.environ.get("FLAME3_ROOT", Path.home() / "school/5980/data/flame3_cv")) / "FLAME 3 CV Dataset (Sycan Marsh)"
OUT = Path(__file__).parent / "cache/index.csv"

XMP_KEYS = ["AbsoluteAltitude", "RelativeAltitude", "GimbalRollDegree", "GimbalYawDegree",
            "GimbalPitchDegree", "FlightRollDegree", "FlightYawDegree", "FlightPitchDegree",
            "GpsLatitude", "GpsLongitude", "GpsLongtitude"]


def xmp_fields(path):
    head = path.read_bytes()[:200_000].decode("latin-1", errors="ignore")
    out = {}
    for k in XMP_KEYS:
        m = re.search(rf'drone-dji:{k}="([^"]+)"', head) or re.search(rf"<drone-dji:{k}>([^<]+)<", head)
        if m:
            try:
                out[k] = float(m.group(1))
            except ValueError:
                pass
    return out


def exif_time(path):
    with Image.open(path) as im:
        ex = im.getexif()
        sub = ex.get_ifd(0x8769)
        return sub.get(36867) or ex.get(306), im.size


rows = []
for label in ["Fire", "No Fire"]:
    d = ROOT / label
    for cfov in sorted((d / "RGB/Corrected FOV").glob("*.JPG")):
        stem = cfov.stem
        raw = d / "RGB/Raw" / f"{stem}.JPG"
        tjpg = d / "Thermal/Raw JPG" / f"{stem}.JPG"
        tif = d / "Thermal/Celsius TIFF" / f"{stem}.TIFF"
        if not tif.exists():
            cands = list((d / "Thermal/Celsius TIFF").glob(f"{stem}.*"))
            tif = cands[0] if cands else tif
        t = tifffile.imread(tif)
        dto, raw_size = exif_time(raw)
        with Image.open(cfov) as im:
            cfov_size = im.size
        row = dict(
            label=label, y=int(label == "Fire"), num=int(stem), stem=stem,
            rgb=str(cfov), raw_rgb=str(raw), thermal_jpg=str(tjpg), tiff=str(tif),
            datetime=dto, raw_w=raw_size[0], raw_h=raw_size[1], cfov_w=cfov_size[0], cfov_h=cfov_size[1],
            tiff_dtype=str(t.dtype), tiff_h=t.shape[0], tiff_w=t.shape[1],
            t_min=float(np.nanmin(t)), t_mean=float(np.nanmean(t)), t_p99=float(np.nanpercentile(t, 99)),
            t_max=float(np.nanmax(t)),
            frac_gt80=float((t > 80).mean()), frac_gt150=float((t > 150).mean()), frac_gt200=float((t > 200).mean()),
        )
        row.update(xmp_fields(raw))
        rows.append(row)
    print(label, "done", flush=True)

df = pd.DataFrame(rows)
df["ts"] = pd.to_datetime(df["datetime"], format="%Y:%m:%d %H:%M:%S", errors="coerce")
df.to_csv(OUT, index=False)
print(df.shape)
