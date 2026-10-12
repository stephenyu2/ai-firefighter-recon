"""Full-burn benchmark: every frame in a set, one request at a time (no overlap), two separate metrics:
  end-to-end  = client wall time for the whole burn, network included
  server-side = time spent inside the server, split into decode / preprocess / GPU forward (GPU timed with cuda sync)
Usage: python bench_full.py LABEL URL MODEL [BATCH] [MANIFEST]
MANIFEST is a CSV with uid, path, y (1 fire, 0 no fire, -1 unlabeled) and slice. Default: the 738 Sycan Marsh frames.
CLIENT_RESIZE=WxH (env) resizes each frame on the client to the detector's input size before upload, which cuts the
payload; the label gets a "_clientresize" suffix."""
import base64, io, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests

LABEL, URL, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 32
MANIFEST = sys.argv[5] if len(sys.argv) > 5 else None
RESIZE = tuple(int(v) for v in os.environ["CLIENT_RESIZE"].split("x")) if os.environ.get("CLIENT_RESIZE") else None
if RESIZE:
    from PIL import Image
    LABEL += "_clientresize" + (f"_q{os.environ['CLIENT_QUALITY']}" if os.environ.get("CLIENT_QUALITY") else "") + ("_444" if os.environ.get("CLIENT_SUBSAMPLING") == "0" else "")


def payload(p):
    if not RESIZE:
        return Path(p).read_bytes()
    q = int(os.environ.get("CLIENT_QUALITY", 90)); sub = int(os.environ.get("CLIENT_SUBSAMPLING", 2))  # PIL: 0 = 4:4:4, 2 = 4:2:0
    buf = io.BytesIO(); Image.open(p).convert("RGB").resize(RESIZE, Image.BICUBIC).save(buf, "JPEG", quality=q, subsampling=sub)
    return buf.getvalue()
KEY = os.environ.get("RECON_API_KEY", "")  # direct to the endpoint, not via the gateway; unset for servers with no auth
H = {"Authorization": f"{os.environ.get('RECON_AUTH_SCHEME', 'Bearer')} {KEY}"} if KEY else {}
DATA = Path(os.environ.get("RECON_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
ROOT = DATA / "flame3_cv/FLAME 3 CV Dataset (Sycan Marsh)"
RESULTS = Path(__file__).parent / "results"
if MANIFEST:
    idx = pd.read_csv(MANIFEST); DATASET = Path(MANIFEST).stem
else:
    idx = pd.read_csv(Path(__file__).resolve().parents[2] / "training/flame3-lora/index.csv"); DATASET = "sycan_marsh"
    idx["path"] = [str(ROOT / ("Fire" if u.split("_")[0] == "Fire" else "No Fire") / "RGB/Corrected FOV" / f"{int(u.split('_')[1]):05d}.JPG") for u in idx.uid]
    idx["slice"] = "all"
frames = [{"id": u, "rgb_b64": base64.b64encode(payload(p)).decode()} for u, p in zip(idx.uid, idx.path)]
UPLOAD_MB = round(sum(len(f["rgb_b64"]) for f in frames) / 1e6, 1)
for _ in range(60):  # wake and warm up (not timed)
    try:
        if requests.post(URL, headers=H, json={"frames": frames[:BATCH], "models": [MODEL]}, timeout=600).status_code == 200: break
    except Exception: pass
    time.sleep(10)
rows, preds = [], []
t0 = time.perf_counter()
for i in range(0, len(frames), BATCH):
    b = frames[i:i + BATCH]; ts = time.perf_counter()
    r = requests.post(URL, headers=H, json={"frames": b, "models": [MODEL]}, timeout=900); r.raise_for_status(); j = r.json()
    tm = j["model_info"]["timing"]
    rows.append(dict(n=len(b), e2e_ms=1000 * (time.perf_counter() - ts), server_ms=1000 * j["model_info"]["elapsed_s"], decode_ms=tm["decode_ms"],
                     preprocess_ms=tm[MODEL]["preprocess_ms"], gpu_ms=tm[MODEL]["gpu_forward_ms"]))
    preds += [(f["id"], f[MODEL]["fire_recall_tuned"]) for f in j["frames"]]
wall = time.perf_counter() - t0
t = pd.DataFrame(rows); n = int(t.n.sum())
p = pd.DataFrame(preds, columns=["uid", "fire"]).merge(idx[["uid", "y", "slice"]])
p.to_csv(RESULTS / f"preds_{DATASET}_{LABEL}_{MODEL}.csv", index=False)
res = {"dataset": DATASET, "gpu": LABEL, "model": MODEL, "frames": n, "upload_mb": UPLOAD_MB, "batch": BATCH, "device": j["model_info"].get("device"),
       "e2e_total_s": round(wall, 2), "e2e_ms_per_frame": round(1000 * wall / n, 2),
       "server_total_s": round(t.server_ms.sum() / 1000, 2), "server_ms_per_frame": round(t.server_ms.sum() / n, 2),
       "decode_ms_per_frame": round(t.decode_ms.sum() / n, 2), "preprocess_ms_per_frame": round(t.preprocess_ms.sum() / n, 2),
       "gpu_forward_ms_per_frame": round(t.gpu_ms.sum() / n, 3), "gpu_forward_total_s": round(t.gpu_ms.sum() / 1000, 3),
       "network_and_overhead_s": round(wall - t.server_ms.sum() / 1000, 2),
       "recall": round(p.fire[p.y == 1].mean(), 4), "fire_frames": int((p.y == 1).sum()),
       "false_alarms": int(p.fire[p.y == 0].sum()), "no_fire_frames": int((p.y == 0).sum()),
       "by_slice": {k: {"n": len(g), "flagged_fire": int(g.fire.sum())} for k, g in p.groupby("slice")}}
print(json.dumps(res))
with open(RESULTS / "full_runs.jsonl", "a") as fh:
    fh.write(json.dumps(res) + "\n")
