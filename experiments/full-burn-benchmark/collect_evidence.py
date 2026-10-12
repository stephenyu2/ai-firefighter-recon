"""Per-frame detector and CLIP probabilities for a frame set, from the fire-classifier deployment (same inputs as
bench_full.py), plus thermal stats from the index. This is the evidence the Kai decision layer reads.
Usage: python collect_evidence.py URL [MANIFEST]   (default manifest: the 738 Sycan Marsh frames)
Writes results/kai/evidence_<dataset>.csv"""
import base64, os, sys, time
from pathlib import Path
import pandas as pd, requests

URL = sys.argv[1]
MANIFEST = sys.argv[2] if len(sys.argv) > 2 else None
HERE = Path(__file__).resolve().parent
KEY = os.environ.get("RECON_API_KEY", "")  # direct to the endpoint, not via the gateway; unset for servers with no auth
H = {"Authorization": f"{os.environ.get('RECON_AUTH_SCHEME', 'Bearer')} {KEY}"} if KEY else {}
MODELS = ["dinov2_lora", "dinov2g_qlora", "clip"]
if MANIFEST:
    idx = pd.read_csv(MANIFEST); DATASET = Path(MANIFEST).stem
    th = pd.read_csv(Path(MANIFEST).with_name("index.csv"))[["uid", "t_max", "frac_gt80"]]
else:
    ROOT = Path(os.environ.get("RECON_DATA_ROOT", HERE.parents[1] / "data")) / "flame3_cv/FLAME 3 CV Dataset (Sycan Marsh)"
    idx = pd.read_csv(HERE.parents[1] / "training/flame3-lora/index.csv"); DATASET = "sycan_marsh"
    idx["path"] = [str(ROOT / ("Fire" if u.split("_")[0] == "Fire" else "No Fire") / "RGB/Corrected FOV" / f"{int(u.split('_')[1]):05d}.JPG") for u in idx.uid]
    idx["slice"] = "all"
    th = idx[["uid", "t_max", "frac_gt80"]]
rows = []
for i in range(0, len(idx), 32):
    b = idx.iloc[i:i + 32]
    frames = [{"id": u, "rgb_b64": base64.b64encode(Path(p).read_bytes()).decode()} for u, p in zip(b.uid, b.path)]
    for attempt in range(60):
        try:
            r = requests.post(URL, headers=H, json={"frames": frames, "models": MODELS}, timeout=600)
            if r.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(10)
    r.raise_for_status()
    for f in r.json()["frames"]:
        rows.append(dict(uid=f["id"], p_b=f["dinov2_lora"]["p_fire"], fire_b=f["dinov2_lora"]["fire_recall_tuned"],
                         p_g=f["dinov2g_qlora"]["p_fire"], fire_g=f["dinov2g_qlora"]["fire_recall_tuned"], p_clip=f["clip"]["p_fire"]))
    print(DATASET, i + len(b), "of", len(idx), flush=True)
out = idx[["uid", "y", "slice"]].merge(pd.DataFrame(rows), on="uid").merge(th, on="uid", how="left")
(HERE / "results/kai").mkdir(parents=True, exist_ok=True)
out.to_csv(HERE / "results/kai" / f"evidence_{DATASET}.csv", index=False)
print("wrote", len(out), "rows")
