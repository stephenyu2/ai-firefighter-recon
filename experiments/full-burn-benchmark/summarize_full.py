"""Tables for the full-burn runs in full_runs.jsonl (latest run per dataset, GPU and model).

Timing: end-to-end client wall time for the whole set vs server-side time (decode, preprocess, GPU forward).
Accuracy: recall on thermal-labeled fire frames, false alarms on clean negatives (pre-burn, and during-burn frames
with no pixel at or above 80 C), and the flag rate on unscored frames (post-burn, plot 3)."""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent / "results"
runs = pd.DataFrame([json.loads(l) for l in (HERE / "full_runs.jsonl").read_text().splitlines() if l.strip()])
runs["dataset"] = runs.get("dataset", pd.Series(dtype=str)).fillna("sycan_marsh")
runs = runs.drop_duplicates(["dataset", "gpu", "model"], keep="last")
rows = []
for r in runs.itertuples():
    pf = HERE / f"preds_{r.dataset}_{r.gpu}_{r.model}.csv"
    p = pd.read_csv(pf) if pf.exists() else None
    row = dict(dataset=r.dataset, gpu=r.gpu, model=r.model, frames=r.frames,
               e2e_total_s=r.e2e_total_s, server_total_s=r.server_total_s, network_and_overhead_s=r.network_and_overhead_s,
               server_ms_per_frame=r.server_ms_per_frame, decode_ms=r.decode_ms_per_frame, preprocess_ms=r.preprocess_ms_per_frame,
               gpu_forward_ms=r.gpu_forward_ms_per_frame)
    if p is not None:
        fire, neg = p[p.y == 1], p[p.y == 0]
        row.update(recall=round(fire.fire.mean(), 4), missed=int((1 - fire.fire).sum()), fire_frames=len(fire),
                   false_alarms=int(neg.fire.sum()), clean_negatives=len(neg))
        for sl, g in p[p.y == 1].groupby("slice"):
            row[f"recall_{sl}"] = round(g.fire.mean(), 4)
        for sl, g in p[p.y == -1].groupby("slice"):
            row[f"flag_rate_{sl}"] = round(g.fire.mean(), 4)
    rows.append(row)
out = pd.DataFrame(rows).sort_values(["dataset", "model", "gpu"])
out.to_csv(HERE / "full_runs_summary.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 40):
    for ds, g in out.groupby("dataset"):
        print(f"\n== {ds}")
        print(g.dropna(axis=1, how="all").drop(columns="dataset").to_string(index=False))
