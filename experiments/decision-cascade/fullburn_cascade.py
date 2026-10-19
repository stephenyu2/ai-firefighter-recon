"""The cascade over every frame of the Florida burn (FLAME 3 NADIR, Hanna Hammock).

The 1,353 thermal-view RGB crops (from ../full-burn-benchmark/prep_florida.py) are cut into consecutive 8-frame windows
per plot and phase; a trailing window of fewer than 4 frames is skipped (5 frames in all). Each frame carries its raw
thermal frame, converted to Celsius, wherever the drone recorded one. Window truth: fire if any frame is
thermal-labeled fire, no_fire if every frame is a clean negative, unscored if any frame is unlabeled (post-burn,
plot 3). Resumable: windows already in results/fullburn.jsonl are skipped.
Frames and thermal TIFFs are read from RECON_DATA_ROOT (default: the repo's data/ folder); converted Celsius TIFFs are
cached in RECON_DATA_ROOT/florida_celsius.
Usage: source ../gateway-model-trials/gateway_env.sh; python fullburn_cascade.py [DECIDER]   (default d3)"""
import json, os, sys, time, traceback
from pathlib import Path

import numpy as np, pandas as pd, tifffile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from recon_gateway import cascade  # noqa: E402
from recon_gateway.sequence import Frame, Sequence  # noqa: E402

DATA = Path(os.environ.get("RECON_DATA_ROOT", HERE.parents[1] / "data"))
INDEX = HERE.parent / "full-burn-benchmark/results/florida_index.csv"  # copy of prep_florida.py's index
NADIR = DATA / "flame3_nadir/Hanna Hammock"
CROPS = DATA / "flame3_nadir_prepped/crop"
CELSIUS = DATA / "florida_celsius"
LOG = HERE / "results" / "fullburn.jsonl"
WINDOW = 8


def celsius(row):
    if not isinstance(row.thermal, str) or not row.thermal:
        return None
    out = CELSIUS / f"{row.uid}.tiff"
    if not out.exists():
        tifffile.imwrite(out, tifffile.imread(NADIR / row.thermal).astype(np.float32) * 0.1 - 275.0)
    return out


def windows(index=INDEX):
    idx = pd.read_csv(index)
    for (plot, phase), g in idx.sort_values("frame").groupby(["plot", "phase"], sort=True):
        g = g.reset_index(drop=True)
        for start in range(0, len(g), WINDOW):
            w = g.iloc[start:start + WINDOW]
            if len(w) < 4:
                continue
            ys = set(w.y)
            truth = "unscored" if -1 in ys else ("fire" if 1 in ys else "no_fire")
            yield f"{plot}_{phase}_{int(w.frame.iloc[0]):04d}", plot, phase, truth, w


def main(decider):
    CELSIUS.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["window"] for l in LOG.read_text().splitlines() if l.strip()} if LOG.exists() else set()
    out = HERE / "results" / "runs" / "v3" / f"fullburn-{decider}"
    out.mkdir(parents=True, exist_ok=True)
    for wid, plot, phase, truth, w in windows():
        if wid in done:
            continue
        frames = [Frame(id=r.uid, rgb_path=CROPS / f"{r.uid}.jpg", tiff_path=celsius(r), t=5.0 * k)
                  for k, r in enumerate(w.itertuples())]
        t0 = time.time()
        try:
            r = cascade.run(Sequence(id=wid, frames=frames), decider=decider, run_tag=f"v3/fullburn-{decider}/{wid}")
            run = r.pop("run", None)
            if run:
                (out / f"{wid}.json").write_text(json.dumps(run, indent=1))
            row = dict(window=wid, plot=plot, phase=phase, truth=truth, n=len(w), fire_frames=int((w.y == 1).sum()),
                       min_hot=float(w.frac_gt80[w.y == 1].min()) if (w.y == 1).any() else None, ok=True, **r)
        except Exception as e:  # noqa: BLE001 - recorded, the next window still runs
            row = dict(window=wid, plot=plot, phase=phase, truth=truth, ok=False, error=f"{type(e).__name__}: {str(e)[:300]}",
                       trace=traceback.format_exc()[-600:], seconds=round(time.time() - t0, 1))
        with open(LOG, "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        print(wid, truth, row.get("path"), row.get("status"), row.get("measured_status"), row.get("seconds"), flush=True)
    print("FULLBURN_DONE", decider)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "d3")
