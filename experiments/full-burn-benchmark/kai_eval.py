"""Kai 0.6B as the fire decision layer, over every frame of each set.

Kai is text-only. Each frame's evidence is written in words the way recon_gateway/kai_decider.py does it (Kai reads
words far better than raw probabilities), and Kai answers the gateway's status question (fire / smoke only / none /
unclear) plus a yes/no "active fire" question in one pass. The decision is status == fire.
Evidence modes:
  rgb_b          DINOv2-B detector + CLIP, no thermal
  rgb_g          DINOv2-g detector + CLIP, no thermal
  rgb_b_thermal  DINOv2-B + CLIP + thermal hot-spot reading where the frame has one. On Florida the labels come from
                 the same thermal frames, so this mode checks that Kai uses thermal evidence, not accuracy.
  rgb_b_thermal_v2  same, with the hot area given as a pixel count and a percentage with two significant figures.
                 Version 1 printed the area to two decimals, so small fires read "covering 0.00% of the frame".
MODES (env, comma-separated) and DATASETS (env) limit what runs.
Usage: python kai_eval.py LABEL URL [CONCURRENCY]
Writes results/kai/kai_scores_<LABEL>.csv and appends to results/kai/kai_runs.jsonl"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd, requests
from sklearn.metrics import roc_auc_score

LABEL, URL = sys.argv[1], sys.argv[2]
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 8
HERE = Path(__file__).parent / "results" / "kai"
KEY = os.environ.get("RECON_API_KEY", "")  # direct to the endpoint, not via the gateway; unset for servers with no auth
H = {"Authorization": f"{os.environ.get('RECON_AUTH_SCHEME', 'Bearer')} {KEY}"} if KEY else {}
SERVER_TIME_HEADER = os.environ.get("SERVER_TIME_HEADER", "")  # response header with server-side ms; else the body's timing_ms
QUESTIONS = {
    "status": {"type": "choice", "instructions": "What does the evidence show about this frame?",
               "criteria": {"fire": "Active fire is present", "smoke_only": "Smoke is present but no fire",
                            "none": "No fire or smoke", "unclear": "The evidence conflicts or is too weak to tell"}},
    "fire": {"type": "noul", "instructions": "Is there active fire in this drone frame?"},
}


def words(p):  # same buckets as recon_gateway.kai_decider._words
    return ("very confident there is fire" if p > 0.9 else "leaning toward fire" if p > 0.5
            else "leaning toward no fire" if p > 0.1 else "very confident there is no fire")


def state(r, mode):
    det = r.p_g if mode == "rgb_g" else r.p_b
    s = ["A single wildfire drone frame was analyzed.", f"The trained fire detector is {words(det)}.",
         f"A general image model is {words(r.p_clip)}."]
    if mode == "rgb_b_thermal_v2" and np.isfinite(r.t_max):
        px = int(round(r.frac_gt80 * 640 * 512))
        if px > 0:
            s.append(f"The thermal camera detects fire-temperature heat: {px} pixels are above 80 C ({100 * r.frac_gt80:.2g}% of the frame), "
                     f"and the hottest pixel is {r.t_max:.0f} C.")
        else:
            s.append(f"The thermal camera shows no pixels above 80 C; the hottest pixel is {r.t_max:.0f} C.")
    elif mode == "rgb_b_thermal" and np.isfinite(r.t_max):
        if r.frac_gt80 > 0:
            s.append(f"The thermal camera shows hot spots above 80 C covering {100 * r.frac_gt80:.2f}% of the frame; the hottest pixel is {r.t_max:.0f} C.")
        else:
            s.append(f"The thermal camera shows no pixels above 80 C; the hottest pixel is {r.t_max:.0f} C.")
    else:
        s.append("No thermal camera data is available.")
    return " ".join(s)


def call(st):
    body = {"state": st, "questions": QUESTIONS}
    for attempt in range(60):
        t0 = time.perf_counter()
        try:
            r = requests.post(URL, headers=H, json=body, timeout=120)
            if r.status_code == 200:
                a = r.json()["answers"]
                srv = (r.headers.get(SERVER_TIME_HEADER) if SERVER_TIME_HEADER else None) or r.json().get("timing_ms")
                return dict(status=a["status"]["choice"], p_fire_status=a["status"]["probabilities"]["fire"],
                            p_fire_noul=a["fire"]["noul"], latency_ms=1000 * (time.perf_counter() - t0),
                            server_ms=float(srv) if srv else np.nan)
        except requests.RequestException:
            pass
        time.sleep(10)
    raise RuntimeError("Kai request failed 60 times")


# wake and warm up (not timed)
call("A single wildfire drone frame was analyzed. No evidence is available.")
# per-request latency, one request at a time, 200 requests
seq = [call(state(r, "rgb_b_thermal")) for r in pd.read_csv(HERE / "evidence_sycan_marsh.csv").head(200).itertuples()]
lat = dict(sequential_median_ms=round(float(np.median([q["latency_ms"] for q in seq])), 1),
           sequential_p90_ms=round(float(np.percentile([q["latency_ms"] for q in seq], 90)), 1),
           sequential_server_median_ms=round(float(np.nanmedian([q["server_ms"] for q in seq])), 1))
print(LABEL, "sequential latency", lat, flush=True)

all_rows = []
MODES = os.environ.get("MODES", "rgb_b,rgb_g,rgb_b_thermal").split(",")
for ds in os.environ.get("DATASETS", "sycan_marsh,florida_crop,florida_full").split(","):
    ev = pd.read_csv(HERE / f"evidence_{ds}.csv")
    for mode in MODES:
        states = [state(r, mode) for r in ev.itertuples()]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(CONC) as pool:
            res = list(pool.map(call, states))
        wall = time.perf_counter() - t0
        k = ev[["uid", "y", "slice", "fire_b", "fire_g"]].copy()
        k["state"] = states
        k = pd.concat([k, pd.DataFrame(res)], axis=1)
        k["kai_fire"] = k.status == "fire"
        k["dataset"], k["mode"] = ds, mode
        all_rows.append(k)
        fire, neg = k[k.y == 1], k[k.y == 0]
        det = "fire_g" if mode == "rgb_g" else "fire_b"
        out = dict(deploy=LABEL, dataset=ds, mode=mode, frames=len(k), unique_states=int(k.state.nunique()),
                   recall=round(fire.kai_fire.mean(), 4), missed=int((~fire.kai_fire).sum()), false_alarms=int(neg.kai_fire.sum()),
                   clean_negatives=len(neg), auc_status=round(roc_auc_score(k[k.y >= 0].y, k[k.y >= 0].p_fire_status), 4),
                   auc_noul=round(roc_auc_score(k[k.y >= 0].y, k[k.y >= 0].p_fire_noul), 4),
                   detector_recall=round(fire[det].mean(), 4), detector_false_alarms=int(neg[det].sum()),
                   agree_with_detector=round(float((k.kai_fire == k[det]).mean()), 4),
                   status_counts=k.status.value_counts().to_dict(),
                   wall_s=round(wall, 1), decisions_per_s=round(len(k) / wall, 1), concurrency=CONC,
                   client_latency_median_ms=round(float(k.latency_ms.median()), 1),
                   server_median_ms=round(float(k.server_ms.median()), 1), **lat)
        print(json.dumps(out), flush=True)
        with open(HERE / "kai_runs.jsonl", "a") as fh:
            fh.write(json.dumps(out) + "\n")
pd.concat(all_rows).to_csv(HERE / f"kai_scores_{LABEL}{os.environ.get('OUT_SUFFIX', '')}.csv", index=False)
print("ALL_KAI_DONE", LABEL)
