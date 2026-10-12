"""Builds the 20 test sequences for the Qwen3.8 / decider comparison and writes sequences.json.
Florida sequences get the thermal-view RGB crop (640x512) and, where listed, the raw thermal frame converted to a
float32 Celsius TIFF (C = DN * 0.1 - 275), so the gateway's thermal tools read them like the Sycan TIFFs.
Needs the NADIR set prepared by ../full-burn-benchmark/prep_florida.py. Folders are written under RECON_DATA_ROOT
(default: the repo's data/ folder) and sequences.json stores paths relative to it."""
import json, os, shutil
from pathlib import Path

import numpy as np, tifffile

HERE = Path(__file__).parent
DATA = Path(os.environ.get("RECON_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
PREP = DATA / "flame3_nadir_prepped"
RAW = DATA / "flame3_nadir/Hanna Hammock"
OUT = DATA / "recon_sequences"

FLORIDA = [  # id, plot, phase, first frame, thermal, truth, note
    ("FL-A_plot1_burn_thermal", "plot1", "duringburn", 600, True, "fire", "large active fire front"),
    ("FL-B_plot2_smallfire_thermal", "plot2", "duringburn", 900, True, "fire", "small fires the RGB detector misses; thermal sees them"),
    ("FL-C_plot2_burn_thermal", "plot2", "duringburn", 1250, True, "fire", "later plot 2 burn"),
    ("FL-D_plot1_preburn_rgb", "plot1", "preburn", 40, False, "no_fire", "pre-burn, RGB only"),
    ("FL-E_plot2_takeoff_smoke_thermal", "plot2", "duringburn", 825, True, "no_fire", "takeoff over smoke, no pixel at 80 C"),
    ("FL-F_plot1_burn_rgb", "plot1", "duringburn", 700, False, "fire", "large fire, thermal withheld"),
    ("FL-G_plot2_smallfire_rgb", "plot2", "duringburn", 900, False, "fire", "small fires, thermal withheld (hardest case)"),
    # added after the v2 review: more RGB-only sequences, because only FL-G separated the models
    ("FL-H_plot2_smallfire_rgb", "plot2", "duringburn", 841, False, "fire", "small fires, thermal withheld"),
    ("FL-I_plot2_smallfire_rgb", "plot2", "duringburn", 857, False, "fire", "small fires, thermal withheld"),
    ("FL-J_plot2_smallfire_rgb", "plot2", "duringburn", 873, False, "fire", "small fires, thermal withheld"),
    ("FL-K_plot2_smallfire_rgb", "plot2", "duringburn", 921, False, "fire", "small fires, thermal withheld"),
    ("FL-L_plot2_smallfire_rgb", "plot2", "duringburn", 945, False, "fire", "small fires, thermal withheld"),
    ("FL-M_plot2_smallfire_rgb", "plot2", "duringburn", 969, False, "fire", "small fires, thermal withheld"),
    ("FL-N_plot1_preburn_rgb", "plot1", "preburn", 10, False, "no_fire", "pre-burn, RGB only"),
    ("FL-O_plot2_preburn_rgb", "plot2", "preburn", 100, False, "no_fire", "pre-burn, RGB only"),
    ("FL-P_plot2_preburn_rgb", "plot2", "preburn", 140, False, "no_fire", "pre-burn, RGB only"),
]
SYCAN = [("SY-2_5", "2_5", "fire"), ("SY-15_0", "15_0", "fire"), ("SY-2_7", "2_7", "no_fire"), ("SY-13_0", "13_0", "fire")]

seqs = []
for sid, plot, phase, start, thermal, truth, note in FLORIDA:
    d = OUT / sid
    rel = d.relative_to(DATA)
    seqs.append(dict(id=sid, kind="folder", rgb_dir=str(rel / "rgb"), tiff_dir=str(rel / "tiff") if thermal else None, interval_s=5.0,
                     truth=truth, thermal=thermal, site="Florida", note=note))
    if (d / "rgb").exists() and len(list((d / "rgb").glob("*.jpg"))) == 8:
        continue  # never rebuild a folder a running experiment may be reading
    (d / "rgb").mkdir(parents=True, exist_ok=True)
    if thermal:
        (d / "tiff").mkdir(exist_ok=True)
    for n in range(start, start + 8):
        uid = f"{plot}_{phase}_{n:04d}"
        shutil.copy(PREP / "crop" / f"{uid}.jpg", d / "rgb" / f"{uid}.jpg")
        if thermal:
            dn = tifffile.imread(RAW / f"plot {plot[-1]}" / phase / "raw_thermal_tiff" / f"IRX_{n:04d}.TIFF").astype(np.float32)
            tifffile.imwrite(d / "tiff" / f"{uid}.tiff", dn * 0.1 - 275.0)
for sid, group, truth in SYCAN:
    seqs.append(dict(id=sid, kind="flame", group=group, truth=truth, thermal=True, site="Sycan", note="earlier demo sequence"))
(HERE / "sequences.json").write_text(json.dumps(seqs, indent=1))
print(len(seqs), "sequences ->", HERE / "sequences.json")
