"""Per-frame vision check through the LiteLLM gateway: can each model tell, from one RGB frame, whether fire is visible?

LLM routes answer the gateway's own `flames_visible` question (log-probs over the answer letters, recon_gateway.tools.decide).
Clef and Clef-Flash get the frame as an image and a yes/no question in the SystemOne format.
Frames (seed 0): Florida thermal-view crops, 40 large fires (>= 0.5% of the thermal view at 80 C), 40 small fires (< 0.5%),
60 pre-burn and 20 burn frames with no hot pixel; Sycan Marsh, 40 fire and 40 no-fire. 240 frames.
Usage: source gateway_env.sh; python frame_eval.py NAME [NAME ...]   (NAME = an LLM route, clef or clef_flash)
GUIDED=1 adds the observer guidance (recon_gateway.prompts.WATCH_FOR): as a system message for LLM routes, appended to the
question instructions for Clef. SEED=N (N > 0) draws a confirmation sample that leaves out every seed-0 frame; groups whose
pool is too small keep what is left (the 21-frame "burn, no hot pixel" pool leaves 1 new frame). Both add a suffix to the name.
All names given on one command run in parallel, but each name sends one request at a time (vLLM log-probs can shift with
batch composition, so scored runs never batch on a deployment). Per-frame times are rough; use throughput.py for speed.
VERSION=v2 writes frames_v2.jsonl and frames_v2_<name>.csv. REP=N adds "#repN" to the name, to measure run-to-run noise.
AUC is computed on log-odds (from the raw top-20 log-probs for LLM routes, from p for Clef), not on rounded probabilities,
so confident "no" answers do not tie at exactly 0.
No resume: rerun only the names missing from frames.jsonl. summarize_v2.py keeps the last line per name.
Images are read from RECON_DATA_ROOT (default: the repo's data/ folder); the CSVs store paths relative to it.
Writes results/frames_<name>.csv and appends a summary line to results/frames.jsonl."""
import json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd, requests
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from recon_gateway import tools as T  # noqa: E402
from recon_gateway.sequence import Frame  # noqa: E402
from recon_gateway.prompts import WATCH_FOR  # noqa: E402

GUIDED = os.environ.get("GUIDED") == "1"
SEED = int(os.environ.get("SEED", "0"))
REP = os.environ.get("REP", "")
SUFFIX = ("+guide" if GUIDED else "") + (f"@seed{SEED}" if SEED else "") + (f"#rep{REP}" if REP else "")
VERSION = os.environ.get("VERSION", "")
PREFIX = f"frames_{VERSION}_" if VERSION else "frames_"
CONCURRENCY = int(os.environ.get("FRAME_CONCURRENCY", "1"))

DATA = Path(os.environ.get("RECON_DATA_ROOT", HERE.parents[1] / "data"))
RESULTS = HERE / "results"
FLORIDA_INDEX = HERE.parent / "full-burn-benchmark/results/florida_index.csv"  # copy of prep_florida.py's index.csv
SYCAN = "flame3_cv/FLAME 3 CV Dataset (Sycan Marsh)"
QUESTION = "Are flames or glowing fire visible in this aerial image?"


def frame_set(seed=None, exclude=()):
    seed = SEED if seed is None else seed
    if seed and not exclude:
        exclude = set(frame_set(seed=0).uid)  # confirmation samples never reuse a seed-0 frame
    rng = np.random.default_rng(seed)
    fl = pd.read_csv(FLORIDA_INDEX)
    fl = fl[~fl.uid.isin(exclude)]

    def pick(d, n):
        return d.iloc[rng.choice(len(d), min(n, len(d)), replace=False)]
    parts = [pick(fl[(fl.y == 1) & (fl.frac_gt80 >= 0.005)], 40).assign(group="Florida fire, large"),
             pick(fl[(fl.y == 1) & (fl.frac_gt80 < 0.005)], 40).assign(group="Florida fire, small"),
             pick(fl[(fl.y == 0) & (fl.phase == "preburn")], 60).assign(group="Florida pre-burn"),
             pick(fl[(fl.y == 0) & (fl.phase == "duringburn")], 20).assign(group="Florida burn, no hot pixel")]
    rows = [dict(uid=r.uid, y=int(r.y), group=r.group, path=f"flame3_nadir_prepped/crop/{r.uid}.jpg") for p in parts for r in p.itertuples()]
    sy = pd.read_csv(HERE.parents[1] / "training/flame3-lora/index.csv")
    sy = sy[~sy.uid.isin(exclude)]
    for y, n in ((1, 40), (0, 40)):
        for r in pick(sy[sy.y == y], n).itertuples():
            lab, k = r.uid.split("_")
            rows.append(dict(uid=r.uid, y=y, group=f"Sycan {'fire' if y else 'no fire'}",
                             path=f"{SYCAN}/{'Fire' if lab == 'Fire' else 'No Fire'}/RGB/Corrected FOV/{int(k):05d}.JPG"))
    return pd.DataFrame(rows)


def score_llm(route):
    ep = T.Endpoints(llm_model=route, run_tag=f"{VERSION or 'v1'}/frames/{route}{SUFFIX}")

    def one(r):
        t0 = time.perf_counter()
        d = T.decide(ep, Frame(id="frame", rgb_path=DATA / r.path), ["flames_visible"], system=WATCH_FOR if GUIDED else None)["flames_visible"]
        raw = d["raw_top_logprobs"]
        floor = min(lp for _, lp in raw) - 1.0 if raw else -30.0

        def lmass(letter):
            lps = [lp for tok, lp in raw if tok.strip().rstrip(".):") == letter]
            return float(np.logaddexp.reduce(lps)) if lps else floor
        return (d["probabilities"]["yes"], time.perf_counter() - t0, d["coverage"], d["margin"], d["answer"],
                lmass("A") - lmass("B"), json.dumps(raw))
    return one


def score_clef(name):
    url = os.environ["RECON_CLEF_URL" if name == "clef" else "RECON_CLEF_FLASH_URL"]
    h = {"Authorization": f"Bearer {os.environ['RECON_GATEWAY_KEY']}", "x-recon-run": f"{VERSION or 'v1'}/frames/{name}{SUFFIX}"}

    def one(r):
        body = {"model": name, "state": "One aerial frame from a wildfire reconnaissance drone flight.",
                "questions": {"flames": {"type": "noul", "instructions": QUESTION + ("\n\n" + WATCH_FOR if GUIDED else "")}},
                "images": [Frame(id=r.uid, rgb_path=DATA / r.path).rgb_b64(max_side=640)]}
        t0 = time.perf_counter()
        for _ in range(30):
            try:
                resp = requests.post(url, headers=h, json=body, timeout=600)
                if resp.status_code < 500:
                    break
            except requests.RequestException:
                pass
            time.sleep(10)
        resp.raise_for_status()
        p = resp.json()["answers"]["flames"]["noul"]
        q = min(max(p, 1e-12), 1 - 1e-12)
        return p, time.perf_counter() - t0, 1.0, abs(2 * p - 1), "yes" if p >= 0.5 else "no", float(np.log(q / (1 - q))), ""
    return one


def run(name):
    fs = frame_set()
    fn = score_clef(name) if name in ("clef", "clef_flash") else score_llm(name)
    fn(next(fs.itertuples()))  # warm-up, not timed
    t0 = time.perf_counter()
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        res = list(pool.map(fn, fs.itertuples()))
    wall = time.perf_counter() - t0
    fs["p_fire"] = [r[0] for r in res]; fs["latency_s"] = [r[1] for r in res]
    fs["coverage"] = [r[2] for r in res]; fs["margin"] = [r[3] for r in res]; fs["answer"] = [r[4] for r in res]
    fs["log_odds"] = [r[5] for r in res]; fs["raw_top_logprobs"] = [r[6] for r in res]
    fs["fire"] = fs.answer == "yes"  # unreliable answers (coverage < 0.5) count as not fire
    fs.to_csv(RESULTS / f"{PREFIX}{name}{SUFFIX}.csv", index=False)
    out = dict(name=name + SUFFIX, model=name, guided=GUIDED, seed=SEED, frames=len(fs), recall=round(fs.fire[fs.y == 1].mean(), 4), false_alarms=int(fs.fire[fs.y == 0].sum()),
               negatives=int((fs.y == 0).sum()), auc=round(roc_auc_score(fs.y, fs.log_odds), 4),
               auc_rounded_p=round(roc_auc_score(fs.y, fs.p_fire), 4), near_ties=int((fs.margin <= 0.0625).sum()),
               by_group={g: round(float(d.fire.mean()), 3) for g, d in fs.groupby("group")},
               median_latency_s=round(float(fs.latency_s.median()), 3), wall_s=round(wall, 1), frames_per_s=round(len(fs) / wall, 2),
               unreliable=int(fs.answer.isna().sum()), concurrency=CONCURRENCY, version=VERSION or "v1")
    with open(RESULTS / (f"frames_{VERSION}.jsonl" if VERSION else "frames.jsonl"), "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(json.dumps(out), flush=True)


threads = [threading.Thread(target=run, args=(n,)) for n in sys.argv[1:]]
[t.start() for t in threads]
[t.join() for t in threads]
print("ALL_FRAMES_DONE")
